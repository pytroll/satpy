#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""SEVIRI native format reader.

References:
    - `MSG Level 1.5 Native Format File Definition`_

.. _MSG Level 1.5 Native Format File Definition:
    https://www-cdn.eumetsat.int/files/2020-04/pdf_fg15_msg-native-format-15.pdf

"""

import logging
import warnings
from datetime import datetime

import dask.array as da
import numpy as np
import xarray as xr
from pyresample import geometry

from satpy import CHUNK_SIZE
from satpy._compat import cached_property
from satpy.readers._geos_area import get_area_definition, get_geos_area_naming
from satpy.readers.eum_base import get_service_mode, recarray2dict, time_cds_short
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.seviri_base import (
    CHANNEL_NAMES,
    HRV_NUM_COLUMNS,
    HRV_NUM_LINES,
    SATNUM,
    VISIR_NUM_COLUMNS,
    VISIR_NUM_LINES,
    NoValidOrbitParams,
    OrbitPolynomialFinder,
    SEVIRICalibrationHandler,
    add_scanline_acq_time,
    calculate_area_extent,
    create_coef_dict,
    dec10216,
    get_cds_time,
    get_satpos,
    pad_data_horizontally,
    pad_data_vertically,
)
from satpy.readers.seviri_l1b_native_hdr import (
    DEFAULT_15_SECONDARY_PRODUCT_HEADER,
    GSDTRecords,
    get_native_header,
    native_trailer,
)
from satpy.readers.utils import reduce_mda

logger = logging.getLogger('native_msg')


class NativeMSGFileHandler(BaseFileHandler):
    """SEVIRI native format reader.

    **Calibration**

    See :mod:`satpy.readers.seviri_base`.

    **Padding channel data to full disk**

    By providing the `fill_disk` as True in the `reader_kwargs`, the channel is loaded
    as full disk, padded with no-data where necessary. This is especially useful for the
    HRV channel, but can also be used for RSS and ROI data. By default the original,
    unpadded, data are loaded::

        scene = satpy.Scene(filenames,
                            reader='seviri_l1b_native',
                            reader_kwargs={'fill_disk': False})

    **Metadata**

    See :mod:`satpy.readers.seviri_base`.

    """

    def __init__(self, filename, filename_info, filetype_info,
                 calib_mode='nominal', fill_disk=False, ext_calib_coefs=None,
                 include_raw_metadata=False, mda_max_array_size=100):
        """Initialize the reader."""
        super(NativeMSGFileHandler, self).__init__(filename,
                                                   filename_info,
                                                   filetype_info)
        self.platform_name = None
        self.calib_mode = calib_mode
        self.ext_calib_coefs = ext_calib_coefs or {}
        self.fill_disk = fill_disk
        self.include_raw_metadata = include_raw_metadata
        self.mda_max_array_size = mda_max_array_size

        # Declare required variables.
        self.header = {}
        self.mda = {}
        self.trailer = {}

        # Read header, prepare dask-array, read trailer and initialize image boundaries
        # Available channels are known only after the header has been read
        self.header_type = get_native_header(self._has_archive_header())
        self._read_header()
        self.dask_array = da.from_array(self._get_memmap(), chunks=(CHUNK_SIZE,))
        self._read_trailer()
        self.image_boundaries = ImageBoundaries(self.header, self.trailer, self.mda)

    def _has_archive_header(self):
        """Check whether the file includes an ASCII archive header."""
        ascii_startswith = b'FormatName                  : NATIVE'
        with open(self.filename, mode='rb') as istream:
            return istream.read(36) == ascii_startswith

    @property
    def nominal_start_time(self):
        """Read the repeat cycle nominal start time from metadata."""
        return self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['TrueRepeatCycleStart']

    @property
    def nominal_end_time(self):
        """Read the repeat cycle nominal end time from metadata."""
        return self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['PlannedRepeatCycleEnd']

    @property
    def observation_start_time(self):
        """Read the repeat cycle sensing start time from metadata."""
        return self.trailer['15TRAILER']['ImageProductionStats'][
            'ActualScanningSummary']['ForwardScanStart']

    @property
    def observation_end_time(self):
        """Read the repeat cycle sensing end time from metadata."""
        return self.trailer['15TRAILER']['ImageProductionStats'][
            'ActualScanningSummary']['ForwardScanEnd']

    @property
    def start_time(self):
        """Get general start time for this file."""
        return self.nominal_start_time

    @property
    def end_time(self):
        """Get the general end time for this file."""
        return self.nominal_end_time

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels."""
        pkhrec = [
            ('GP_PK_HEADER', GSDTRecords.gp_pk_header),
            ('GP_PK_SH1', GSDTRecords.gp_pk_sh1)
        ]
        pk_head_dtype = np.dtype(pkhrec)

        def get_lrec(cols):
            lrec = [
                ("gp_pk", pk_head_dtype),
                ("version", np.uint8),
                ("satid", np.uint16),
                ("time", (np.uint16, 5)),
                ("lineno", np.uint32),
                ("chan_id", np.uint8),
                ("acq_time", time_cds_short),
                ("line_validity", np.uint8),
                ("line_rquality", np.uint8),
                ("line_gquality", np.uint8),
                ("line_data", (np.uint8, cols))
            ]

            return lrec

        # each pixel is 10-bits -> one line of data has 25% more bytes
        # than the number of columns suggest (10/8 = 1.25)
        visir_rec = get_lrec(int(self.mda['number_of_columns'] * 1.25))
        number_of_visir_channels = len(
            [s for s in self.mda['channel_list'] if not s == 'HRV'])
        drec = [('visir', (visir_rec, number_of_visir_channels))]

        if self.mda['available_channels']['HRV']:
            hrv_rec = get_lrec(int(self.mda['hrv_number_of_columns'] * 1.25))
            drec.append(('hrv', (hrv_rec, 3)))

        return np.dtype(drec)

    def _get_memmap(self):
        """Get the memory map for the SEVIRI data."""
        with open(self.filename) as fp:
            data_dtype = self._get_data_dtype()
            hdr_size = self.header_type.itemsize

            return np.memmap(fp, dtype=data_dtype,
                             shape=(self.mda['number_of_lines'],),
                             offset=hdr_size, mode="r")

    def _read_header(self):
        """Read the header info."""
        data = np.fromfile(self.filename,
                           dtype=self.header_type, count=1)

        self.header.update(recarray2dict(data))

        if '15_SECONDARY_PRODUCT_HEADER' not in self.header:
            # No archive header, that means we have a complete file
            # including all channels.
            self.header['15_SECONDARY_PRODUCT_HEADER'] = DEFAULT_15_SECONDARY_PRODUCT_HEADER

        data15hd = self.header['15_DATA_HEADER']
        sec15hd = self.header['15_SECONDARY_PRODUCT_HEADER']

        # Set the list of available channels:
        self.mda['available_channels'] = get_available_channels(self.header)
        self.mda['channel_list'] = [i for i in CHANNEL_NAMES.values()
                                    if self.mda['available_channels'][i]]

        self.platform_id = data15hd[
            'SatelliteStatus']['SatelliteDefinition']['SatelliteId']
        self.mda['platform_name'] = "Meteosat-" + SATNUM[self.platform_id]
        self.mda['offset_corrected'] = data15hd['GeometricProcessing'][
            'EarthModel']['TypeOfEarthModel'] == 2

        equator_radius = data15hd['GeometricProcessing'][
                             'EarthModel']['EquatorialRadius'] * 1000.
        north_polar_radius = data15hd[
                                 'GeometricProcessing']['EarthModel']['NorthPolarRadius'] * 1000.
        south_polar_radius = data15hd[
                                 'GeometricProcessing']['EarthModel']['SouthPolarRadius'] * 1000.
        polar_radius = (north_polar_radius + south_polar_radius) * 0.5
        ssp_lon = data15hd['ImageDescription'][
            'ProjectionDescription']['LongitudeOfSSP']

        self.mda['projection_parameters'] = {'a': equator_radius,
                                             'b': polar_radius,
                                             'h': 35785831.00,
                                             'ssp_longitude': ssp_lon}

        north = int(sec15hd['NorthLineSelectedRectangle']['Value'])
        east = int(sec15hd['EastColumnSelectedRectangle']['Value'])
        south = int(sec15hd['SouthLineSelectedRectangle']['Value'])
        west = int(sec15hd['WestColumnSelectedRectangle']['Value'])

        ncolumns = west - east + 1
        nrows = north - south + 1

        # check if the file has less rows or columns than
        # the maximum, if so it is a rapid scanning service
        # or region of interest file
        if (nrows < VISIR_NUM_LINES) or (ncolumns < VISIR_NUM_COLUMNS):
            self.mda['is_full_disk'] = False
        else:
            self.mda['is_full_disk'] = True

        # If the number of columns in the file is not divisible by 4,
        # UMARF will add extra columns to the file
        modulo = ncolumns % 4
        padding = 0
        if modulo > 0:
            padding = 4 - modulo
        cols_visir = ncolumns + padding

        # Check the VISIR calculated column dimension against
        # the header information
        cols_visir_hdr = int(sec15hd['NumberColumnsVISIR']['Value'])
        if cols_visir_hdr != cols_visir:
            logger.warning(
                "Number of VISIR columns from the header is incorrect!")
            logger.warning("Header: %d", cols_visir_hdr)
            logger.warning("Calculated: = %d", cols_visir)

        # HRV Channel - check if the area is reduced in east west
        # direction as this affects the number of columns in the file
        cols_hrv_hdr = int(sec15hd['NumberColumnsHRV']['Value'])
        if ncolumns < VISIR_NUM_COLUMNS:
            cols_hrv = cols_hrv_hdr
        else:
            cols_hrv = int(cols_hrv_hdr / 2)

        # self.mda represents the 16bit dimensions not 10bit
        self.mda['number_of_lines'] = int(sec15hd['NumberLinesVISIR']['Value'])
        self.mda['number_of_columns'] = cols_visir
        self.mda['hrv_number_of_lines'] = int(sec15hd["NumberLinesHRV"]['Value'])
        self.mda['hrv_number_of_columns'] = cols_hrv

        if self.header['15_MAIN_PRODUCT_HEADER']['QQOV']['Value'] == 'NOK':
            warnings.warn("The quality flag for this file indicates not OK. Use this data with caution!", UserWarning)

    def _read_trailer(self):

        hdr_size = self.header_type.itemsize
        data_size = (self._get_data_dtype().itemsize *
                     self.mda['number_of_lines'])

        with open(self.filename) as fp:
            fp.seek(hdr_size + data_size)
            data = np.fromfile(fp, dtype=native_trailer, count=1)

        self.trailer.update(recarray2dict(data))

    def get_area_def(self, dataset_id):
        """Get the area definition of the band.

        In general, image data from one window/area is available. For the HRV channel in FES mode, however,
        data from two windows ('Lower' and 'Upper') are available. Hence, we collect lists of area-extents
        and corresponding number of image lines/columns. In case of FES HRV data, two area definitions are
        computed, stacked and squeezed. For other cases, the lists will only have one entry each, from which
        a single area definition is computed.

        Note that the AreaDefinition area extents returned by this function for Native data will be slightly
        different compared to the area extents returned by the SEVIRI HRIT reader.
        This is due to slightly different pixel size values when calculated using the data available in the files. E.g.
        for the 3 km grid:

        ``Native: data15hd['ImageDescription']['ReferenceGridVIS_IR']['ColumnDirGridStep'] == 3000.4031658172607``
        ``HRIT:                            np.deg2rad(2.**16 / pdict['lfac']) * pdict['h'] == 3000.4032785810186``

        This results in the Native 3 km full-disk area extents being approx. 20 cm shorter in each direction.

        The method for calculating the area extents used by the HRIT reader (CFAC/LFAC mechanism) keeps the
        highest level of numeric precision and is used as reference by EUM. For this reason, the standard area
        definitions defined in the `areas.yaml` file correspond to the HRIT ones.

        """
        pdict = {}
        pdict['a'] = self.mda['projection_parameters']['a']
        pdict['b'] = self.mda['projection_parameters']['b']
        pdict['h'] = self.mda['projection_parameters']['h']
        pdict['ssp_lon'] = self.mda['projection_parameters']['ssp_longitude']

        area_naming_input_dict = {'platform_name': 'msg',
                                  'instrument_name': 'seviri',
                                  'resolution': int(dataset_id['resolution'])
                                  }
        area_naming = get_geos_area_naming({**area_naming_input_dict,
                                            **get_service_mode('seviri', pdict['ssp_lon'])})

        pdict['a_name'] = area_naming['area_id']
        pdict['a_desc'] = area_naming['description']
        pdict['p_id'] = ""

        area_extent = self.get_area_extent(dataset_id)
        areas = list()
        for aex, nlines, ncolumns in zip(area_extent['area_extent'], area_extent['nlines'], area_extent['ncolumns']):
            pdict['nlines'] = nlines
            pdict['ncols'] = ncolumns
            areas.append(get_area_definition(pdict, aex))

        if len(areas) == 2:
            area = geometry.StackedAreaDefinition(areas[0], areas[1])
            area = area.squeeze()
        else:
            area = areas[0]

        return area

    def get_area_extent(self, dataset_id):
        """Get the area extent of the file.

        Until December 2017, the data is shifted by 1.5km SSP North and West against the nominal GEOS projection. Since
        December 2017 this offset has been corrected. A flag in the data indicates if the correction has been applied.
        If no correction was applied, adjust the area extent to match the shifted data.

        For more information see Section 3.1.4.2 in the MSG Level 1.5 Image Data Format Description. The correction
        of the area extent is documented in a `developer's memo <https://github.com/pytroll/satpy/wiki/
        SEVIRI-georeferencing-offset-correction>`_.
        """
        data15hd = self.header['15_DATA_HEADER']

        # check for Earth model as this affects the north-south and
        # west-east offsets
        # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
        earth_model = data15hd['GeometricProcessing']['EarthModel'][
            'TypeOfEarthModel']
        if earth_model == 2:
            ns_offset = 0
            we_offset = 0
        elif earth_model == 1:
            ns_offset = -0.5
            we_offset = 0.5
            if dataset_id['name'] == 'HRV':
                ns_offset = -1.5
                we_offset = 1.5
        else:
            raise NotImplementedError(
                'Unrecognised Earth model: {}'.format(earth_model)
            )

        if dataset_id['name'] == 'HRV':
            grid_origin = data15hd['ImageDescription']['ReferenceGridHRV']['GridOrigin']
            center_point = (HRV_NUM_COLUMNS / 2) - 2
            column_step = data15hd['ImageDescription']['ReferenceGridHRV']['ColumnDirGridStep'] * 1000.0
            line_step = data15hd['ImageDescription']['ReferenceGridHRV']['LineDirGridStep'] * 1000.0
            nlines_fulldisk = HRV_NUM_LINES
            ncolumns_fulldisk = HRV_NUM_COLUMNS
        else:
            grid_origin = data15hd['ImageDescription']['ReferenceGridVIS_IR']['GridOrigin']
            center_point = VISIR_NUM_COLUMNS / 2
            column_step = data15hd['ImageDescription']['ReferenceGridVIS_IR']['ColumnDirGridStep'] * 1000.0
            line_step = data15hd['ImageDescription']['ReferenceGridVIS_IR']['LineDirGridStep'] * 1000.0
            nlines_fulldisk = VISIR_NUM_LINES
            ncolumns_fulldisk = VISIR_NUM_COLUMNS

        # Calculations assume grid origin is south-east corner
        # section 7.2.4 of MSG Level 1.5 Image Data Format Description
        origins = {0: 'NW', 1: 'SW', 2: 'SE', 3: 'NE'}
        if grid_origin != 2:
            msg = 'Grid origin not supported number: {}, {} corner'.format(
                grid_origin, origins[grid_origin]
            )
            raise NotImplementedError(msg)

        aex_data = {'area_extent': [], 'nlines': [], 'ncolumns': []}

        img_bounds = self.image_boundaries.get_img_bounds(dataset_id, self.is_roi())
        for south_bound, north_bound, east_bound, west_bound in zip(*img_bounds.values()):

            if self.fill_disk:
                east_bound, west_bound = 1, ncolumns_fulldisk
                if not self.mda['is_full_disk']:
                    south_bound, north_bound = 1, nlines_fulldisk

            nlines = north_bound - south_bound + 1
            ncolumns = west_bound - east_bound + 1

            area_dict = {'center_point': center_point,
                         'east': east_bound,
                         'west': west_bound,
                         'south': south_bound,
                         'north': north_bound,
                         'column_step': column_step,
                         'line_step': line_step,
                         'column_offset': we_offset,
                         'line_offset': ns_offset
                         }

            aex = calculate_area_extent(area_dict)

            aex_data['area_extent'].append(aex)
            aex_data['nlines'].append(nlines)
            aex_data['ncolumns'].append(ncolumns)

        return aex_data

    def is_roi(self):
        """Check if data covers a selected region of interest (ROI).

        Standard RSS data consists of 3712 columns and 1392 lines, covering the three northmost segements
        of the SEVIRI disk. Hence, if the data does not cover the full disk, nor the standard RSS region
        in RSS mode, it's assumed to be ROI data.
        """
        is_rapid_scan = self.trailer['15TRAILER']['ImageProductionStats']['ActualScanningSummary']['ReducedScan']

        # Standard RSS data is assumed to cover the three northmost segements, thus consisting of all 3712 columns and
        # the 1392 northmost lines
        nlines = int(self.mda['number_of_lines'])
        ncolumns = int(self.mda['number_of_columns'])
        north_bound = int(self.header['15_SECONDARY_PRODUCT_HEADER']['NorthLineSelectedRectangle']['Value'])

        is_top3segments = (ncolumns == VISIR_NUM_COLUMNS and nlines == 1392 and north_bound == VISIR_NUM_LINES)

        return not self.mda['is_full_disk'] and not (is_rapid_scan and is_top3segments)

    def get_dataset(self, dataset_id, dataset_info):
        """Get the dataset."""
        if dataset_id['name'] not in self.mda['channel_list']:
            raise KeyError('Channel % s not available in the file' % dataset_id['name'])
        elif dataset_id['name'] not in ['HRV']:
            data = self._get_visir_channel(dataset_id)
        else:
            data = self._get_hrv_channel()

        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0).astype(np.float32)

        if xarr is None:
            return None

        dataset = self.calibrate(xarr, dataset_id)
        self._add_scanline_acq_time(dataset, dataset_id)
        self._update_attrs(dataset, dataset_info)

        if self.fill_disk and not (dataset_id['name'] != 'HRV' and self.mda['is_full_disk']):
            padder = Padder(dataset_id,
                            self.image_boundaries.get_img_bounds(dataset_id, self.is_roi()),
                            self.mda['is_full_disk'])
            dataset = padder.pad_data(dataset)

        return dataset

    def _get_visir_channel(self, dataset_id):
        shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])
        # Check if there is only 1 channel in the list as a change
        # is needed in the arrray assignment ie channl id is not present
        if len(self.mda['channel_list']) == 1:
            raw = self.dask_array['visir']['line_data']
        else:
            i = self.mda['channel_list'].index(dataset_id['name'])
            raw = self.dask_array['visir']['line_data'][:, i, :]
        data = dec10216(raw.flatten())
        data = data.reshape(shape)
        return data

    def _get_hrv_channel(self):
        shape = (self.mda['hrv_number_of_lines'], self.mda['hrv_number_of_columns'])
        shape_layer = (self.mda['number_of_lines'], self.mda['hrv_number_of_columns'])

        data_list = []
        for i in range(3):
            raw = self.dask_array['hrv']['line_data'][:, i, :]
            data = dec10216(raw.flatten())
            data = data.reshape(shape_layer)
            data_list.append(data)

        return np.stack(data_list, axis=1).reshape(shape)

    def calibrate(self, data, dataset_id):
        """Calibrate the data."""
        tic = datetime.now()
        channel_name = dataset_id['name']
        calib = SEVIRICalibrationHandler(
            platform_id=self.platform_id,
            channel_name=channel_name,
            coefs=self._get_calib_coefs(channel_name),
            calib_mode=self.calib_mode,
            scan_time=self.start_time
        )
        res = calib.calibrate(data, dataset_id['calibration'])
        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

    def _get_calib_coefs(self, channel_name):
        """Get coefficients for calibration from counts to radiance."""
        # even though all the channels may not be present in the file,
        # the header does have calibration coefficients for all the channels
        # hence, this channel index needs to refer to full channel list
        band_idx = list(CHANNEL_NAMES.values()).index(channel_name)

        coefs_nominal = self.header['15_DATA_HEADER'][
            'RadiometricProcessing']['Level15ImageCalibration']
        coefs_gsics = self.header['15_DATA_HEADER'][
            'RadiometricProcessing']['MPEFCalFeedback']
        radiance_types = self.header['15_DATA_HEADER']['ImageDescription'][
                'Level15ImageProduction']['PlannedChanProcessing']
        return create_coef_dict(
            coefs_nominal=(
                coefs_nominal['CalSlope'][band_idx],
                coefs_nominal['CalOffset'][band_idx]
            ),
            coefs_gsics=(
                coefs_gsics['GSICSCalCoeff'][band_idx],
                coefs_gsics['GSICSOffsetCount'][band_idx]
            ),
            ext_coefs=self.ext_calib_coefs.get(channel_name, {}),
            radiance_type=radiance_types[band_idx]
        )

    def _add_scanline_acq_time(self, dataset, dataset_id):
        """Add scanline acquisition time to the given dataset."""
        if dataset_id['name'] == 'HRV':
            tline = self._get_acq_time_hrv()
        else:
            tline = self._get_acq_time_visir(dataset_id)
        acq_time = get_cds_time(days=tline['Days'], msecs=tline['Milliseconds'])
        add_scanline_acq_time(dataset, acq_time)

    def _get_acq_time_hrv(self):
        """Get raw acquisition time for HRV channel."""
        tline = self.dask_array['hrv']['acq_time']
        tline0 = tline[:, 0]
        tline1 = tline[:, 1]
        tline2 = tline[:, 2]
        return da.stack((tline0, tline1, tline2), axis=1).reshape(
            self.mda['hrv_number_of_lines']).compute()

    def _get_acq_time_visir(self, dataset_id):
        """Get raw acquisition time for VIS/IR channels."""
        # Check if there is only 1 channel in the list as a change
        # is needed in the arrray assignment ie channl id is not present
        if len(self.mda['channel_list']) == 1:
            return self.dask_array['visir']['acq_time'].compute()
        i = self.mda['channel_list'].index(dataset_id['name'])
        return self.dask_array['visir']['acq_time'][:, i].compute()

    def _update_attrs(self, dataset, dataset_info):
        """Update dataset attributes."""
        dataset.attrs['units'] = dataset_info['units']
        dataset.attrs['wavelength'] = dataset_info['wavelength']
        dataset.attrs['standard_name'] = dataset_info['standard_name']
        dataset.attrs['platform_name'] = self.mda['platform_name']
        dataset.attrs['sensor'] = 'seviri'
        dataset.attrs['georef_offset_corrected'] = self.mda[
            'offset_corrected']
        dataset.attrs['time_parameters'] = {
            'nominal_start_time': self.nominal_start_time,
            'nominal_end_time': self.nominal_end_time,
            'observation_start_time': self.observation_start_time,
            'observation_end_time': self.observation_end_time,
        }
        dataset.attrs['orbital_parameters'] = self._get_orbital_parameters()
        if self.include_raw_metadata:
            dataset.attrs['raw_metadata'] = reduce_mda(
                self.header, max_size=self.mda_max_array_size
            )

    def _get_orbital_parameters(self):
        orbital_parameters = {
            'projection_longitude': self.mda['projection_parameters'][
                'ssp_longitude'],
            'projection_latitude': 0.,
            'projection_altitude': self.mda['projection_parameters']['h'],
            'satellite_nominal_longitude': self.header['15_DATA_HEADER'][
                'SatelliteStatus']['SatelliteDefinition'][
                'NominalLongitude'],
            'satellite_nominal_latitude': 0.0
        }
        try:
            actual_lon, actual_lat, actual_alt = self.satpos
            orbital_parameters.update({
                'satellite_actual_longitude': actual_lon,
                'satellite_actual_latitude': actual_lat,
                'satellite_actual_altitude': actual_alt
            })
        except NoValidOrbitParams as err:
            logger.warning(err)
        return orbital_parameters

    @cached_property
    def satpos(self):
        """Get actual satellite position in geodetic coordinates (WGS-84).

        Evaluate orbit polynomials at the start time of the scan.

        Returns: Longitude [deg east], Latitude [deg north] and Altitude [m]
        """
        poly_finder = OrbitPolynomialFinder(self.header['15_DATA_HEADER'][
            'SatelliteStatus']['Orbit']['OrbitPolynomial'])
        orbit_polynomial = poly_finder.get_orbit_polynomial(self.start_time)
        return get_satpos(
            orbit_polynomial=orbit_polynomial,
            time=self.observation_start_time,
            semi_major_axis=self.mda['projection_parameters']['a'],
            semi_minor_axis=self.mda['projection_parameters']['b']
        )


class ImageBoundaries:
    """Collect image boundary information."""

    def __init__(self, header, trailer, mda):
        """Initialize the class."""
        self._header = header
        self._trailer = trailer
        self._mda = mda

    def get_img_bounds(self, dataset_id, is_roi):
        """Get image line and column boundaries.

        returns:
            Dictionary with the four keys 'south_bound', 'north_bound', 'east_bound' and 'west_bound',
            each containing a list of the respective line/column numbers of the image boundaries.

        Lists (rather than scalars) are returned since the HRV data in FES mode contain data from two windows/areas.
        """
        if dataset_id['name'] == 'HRV' and not is_roi:
            img_bounds = self._get_hrv_actual_img_bounds()
        else:
            img_bounds = self._get_selected_img_bounds(dataset_id)

        self._check_for_valid_bounds(img_bounds)

        return img_bounds

    def _get_hrv_actual_img_bounds(self):
        """Get HRV (if not ROI) image boundaries from the ActualL15CoverageHRV information stored in the trailer."""
        hrv_bounds = self._trailer['15TRAILER']['ImageProductionStats']['ActualL15CoverageHRV']

        img_bounds = {'south_bound': [], 'north_bound': [], 'east_bound': [], 'west_bound': []}
        for hrv_window in ['Lower', 'Upper']:
            img_bounds['south_bound'].append(hrv_bounds['%sSouthLineActual' % hrv_window])
            img_bounds['north_bound'].append(hrv_bounds['%sNorthLineActual' % hrv_window])
            img_bounds['east_bound'].append(hrv_bounds['%sEastColumnActual' % hrv_window])
            img_bounds['west_bound'].append(hrv_bounds['%sWestColumnActual' % hrv_window])

            # Data from the upper hrv window are only available in FES mode
            if not self._mda['is_full_disk']:
                break

        return img_bounds

    def _get_selected_img_bounds(self, dataset_id):
        """Get VISIR and HRV (if ROI) image boundaries from the SelectedRectangle information stored in the header."""
        sec15hd = self._header['15_SECONDARY_PRODUCT_HEADER']
        south_bound = int(sec15hd['SouthLineSelectedRectangle']['Value'])
        east_bound = int(sec15hd['EastColumnSelectedRectangle']['Value'])

        if dataset_id['name'] == 'HRV':
            nlines, ncolumns = self._get_hrv_img_shape()
            south_bound = self._convert_visir_bound_to_hrv(south_bound)
            east_bound = self._convert_visir_bound_to_hrv(east_bound)
        else:
            nlines, ncolumns = self._get_visir_img_shape()

        north_bound = south_bound + nlines - 1
        west_bound = east_bound + ncolumns - 1

        img_bounds = {'south_bound': [south_bound], 'north_bound': [north_bound],
                      'east_bound': [east_bound], 'west_bound': [west_bound]}

        return img_bounds

    def _get_hrv_img_shape(self):
        nlines = int(self._mda['hrv_number_of_lines'])
        ncolumns = int(self._mda['hrv_number_of_columns'])
        return nlines, ncolumns

    def _get_visir_img_shape(self):
        nlines = int(self._mda['number_of_lines'])
        ncolumns = int(self._mda['number_of_columns'])
        return nlines, ncolumns

    @staticmethod
    def _convert_visir_bound_to_hrv(bound):
        return 3 * bound - 2

    @staticmethod
    def _check_for_valid_bounds(img_bounds):
        len_img_bounds = [len(bound) for bound in img_bounds.values()]

        same_lengths = (len(set(len_img_bounds)) == 1)
        no_empty = (min(len_img_bounds) > 0)

        if not (same_lengths and no_empty):
            raise ValueError('Invalid image boundaries')


class Padder:
    """Padding of HRV, RSS and ROI data to full disk."""

    def __init__(self, dataset_id, img_bounds, is_full_disk):
        """Initialize the padder."""
        self._img_bounds = img_bounds
        self._is_full_disk = is_full_disk

        if dataset_id['name'] == 'HRV':
            self._final_shape = (HRV_NUM_LINES, HRV_NUM_COLUMNS)
        else:
            self._final_shape = (VISIR_NUM_LINES, VISIR_NUM_COLUMNS)

    def pad_data(self, dataset):
        """Pad data to full disk with empty pixels."""
        logger.debug('Padding data to full disk')

        data_list = []
        for south_bound, north_bound, east_bound, west_bound in zip(*self._img_bounds.values()):
            nlines = north_bound - south_bound + 1
            data = self._extract_data_to_pad(dataset, south_bound, north_bound)
            padded_data = pad_data_horizontally(data, (nlines, self._final_shape[1]), east_bound, west_bound)
            data_list.append(padded_data)

        padded_data = da.vstack(data_list)

        # If we're dealing with RSS or ROI data, we also need to pad vertically in order to form a full disk array
        if not self._is_full_disk:
            padded_data = pad_data_vertically(padded_data, self._final_shape, south_bound, north_bound)

        return xr.DataArray(padded_data, dims=('y', 'x'), attrs=dataset.attrs.copy())

    def _extract_data_to_pad(self, dataset, south_bound, north_bound):
        """Extract the data that shall be padded.

        In case of FES (HRV) data, 'dataset' contains data from twoseparate windows that
        are padded separately. Hence, we extract a subset of data.
        """
        if self._is_full_disk:
            data = dataset[south_bound - 1:north_bound, :].data
        else:
            data = dataset.data

        return data


def get_available_channels(header):
    """Get the available channels from the header information."""
    chlist_str = header['15_SECONDARY_PRODUCT_HEADER'][
        'SelectedBandIDs']['Value']
    retv = {}

    for idx, char in zip(range(12), chlist_str):
        retv[CHANNEL_NAMES[idx + 1]] = (char == 'X')

    return retv
