#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Ulrich Hamann <ulrich.hamann@meteoswiss.ch>
#   Sauli Joro <sauli.joro@eumetsat.int>
#   Colin Duff <colin.duff@external.eumetsat.int>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""SEVIRI native format reader.

References:
    MSG Level 1.5 Native Format File Definition
    https://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_FG15_MSG-NATIVE-FORMAT-15&RevisionSelectionMethod=LatestReleased&Rendition=Web
    MSG Level 1.5 Image Data Format Description
    https://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_TEN_05105_MSG_IMG_DATA&RevisionSelectionMethod=LatestReleased&Rendition=Web
"""

import os
import logging
from datetime import datetime
import numpy as np

import xarray as xr
import dask.array as da

from satpy import CHUNK_SIZE
from pyresample import geometry

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.eum_base import recarray2dict
from satpy.readers.msg_base import (SEVIRICalibrationHandler,
                                    CHANNEL_NAMES, CALIB, SATNUM,
                                    dec10216)
from satpy.readers.native_msg_hdr import (GSDTRecords, native_header,
                                          native_trailer)


logger = logging.getLogger('native_msg')


class NativeMSGFileHandler(BaseFileHandler, SEVIRICalibrationHandler):
    """SEVIRI native format reader.
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(NativeMSGFileHandler, self).__init__(filename,
                                                   filename_info,
                                                   filetype_info)
        self.platform_name = None

        # to be (re)set in read_header in order to keep track whether
        # we're dealing with a file with an area of interest
        self.roi = False

        # The available channels are only known after the header
        # has been read, after that we know what the indices are for each channel
        self.header = {}
        self.available_channels = {}
        self.mda = {}
        self._read_header()

        # Prepare dask-array
        self.dask_array = da.from_array(self._get_memmap(), chunks=(CHUNK_SIZE,))

        # Read trailer
        self.trailer = {}
        self._read_trailer()

        data15hd = self.header['15_DATA_HEADER']['ImageDescription']['PlannedCoverageHRV']
        trail = self.trailer['15TRAILER']['ImageProductionStats']['ActualL15CoverageHRV']

    @property
    def start_time(self):
        return self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['TrueRepeatCycleStart']

    @property
    def end_time(self):
        return self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['PlannedRepeatCycleEnd']

    def _get_data_dtype(self):
        """Get the dtype of the file based on the actual available channels"""

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
                ("acq_time", (np.uint16, 3)),
                ("line_validity", np.uint8),
                ("line_rquality", np.uint8),
                ("line_gquality", np.uint8),
                ("line_data", (np.uint8, cols))
            ]

            return lrec

        # each pixel is 10-bits -> one line of data has 25% more bytes
        # than the number of columns suggest (10/8 = 1.25)
        visir_rec = get_lrec(int(self.mda['number_of_columns'] * 1.25))
        number_of_lowres_channels = len(
            [s for s in self._channel_list if not s == 'HRV'])
        drec = [('visir', (visir_rec, number_of_lowres_channels))]

        if self.available_channels['HRV']:
            hrv_rec = get_lrec(int(self.mda['hrv_number_of_columns'] * 1.25))
            drec.append(('hrv', (hrv_rec, 3)))

        return np.dtype(drec)

    def _get_memmap(self):
        """Get the memory map for the SEVIRI data"""

        with open(self.filename) as fp:

            data_dtype = self._get_data_dtype()
            hdr_size = native_header.itemsize

            return np.memmap(fp, dtype=data_dtype,
                             shape=(self.mda['number_of_lines'],),
                             offset=hdr_size, mode="r")

    def _read_header(self):
        """Read the header info"""

        data = np.fromfile(self.filename,
                           dtype=native_header, count=1)

        self.header.update(recarray2dict(data))

        data15hd = self.header['15_DATA_HEADER']
        sec15hd = self.header['15_SECONDARY_PRODUCT_HEADER']

        # Set the list of available channels:
        self.available_channels = get_available_channels(self.header)
        self._channel_list = [i for i in CHANNEL_NAMES.values()
                              if self.available_channels[i]]

        self.platform_id = data15hd[
            'SatelliteStatus']['SatelliteDefinition']['SatelliteId']
        self.platform_name = "Meteosat-" + SATNUM[self.platform_id]

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

        # check if the file has less rows or columns than
        # the maximum, if so it is an area of interest file
        # columns and rows start at 1 -> 3712 - 1 = 3711
        if (north - south < 3711) or (west - east < 3711):
            self.roi = True

        # If the number of columns in the file is not divisible by 4,
        # UMARF will add extra columns to the file
        modulo = (west - east + 1) % 4
        padding = 0
        if modulo > 0:
            padding = 4 - modulo
        cols_visir = west - east + 1 + padding

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
        if (west - east) < 3711:
            cols_hrv = cols_hrv_hdr
        else:
            cols_hrv = int(cols_hrv_hdr / 2)

        # self.mda represents the 16bit dimensions not 10bit
        self.mda['number_of_lines'] = int(sec15hd['NumberLinesVISIR']['Value'])
        self.mda['number_of_columns'] = cols_visir
        self.mda['hrv_number_of_lines'] = int(sec15hd["NumberLinesHRV"]['Value'])
        self.mda['hrv_number_of_columns'] = cols_hrv

    def _read_trailer(self):

        hdr_size = native_header.itemsize
        data_size = (self._get_data_dtype().itemsize *
                     self.mda['number_of_lines'])

        with open(self.filename) as fp:

            fp.seek(hdr_size + data_size)
            data = np.fromfile(fp, dtype=native_trailer, count=1)

        self.trailer.update(recarray2dict(data))

    def get_area_def(self, dsid):

        a = self.mda['projection_parameters']['a']
        b = self.mda['projection_parameters']['b']
        h = self.mda['projection_parameters']['h']
        lon_0 = self.mda['projection_parameters']['ssp_longitude']

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        if dsid.name == 'HRV':
            nlines = self.mda['hrv_number_of_lines']
            ncols = self.mda['hrv_number_of_columns']
        else:
            nlines = self.mda['number_of_lines']
            ncols = self.mda['number_of_columns']

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosmsg',
            proj_dict,
            ncols,
            nlines,
            self.get_area_extent(dsid))

        return area

    def get_area_extent(self, dsid):

        data15hd = self.header['15_DATA_HEADER']
        sec15hd = self.header['15_SECONDARY_PRODUCT_HEADER']

        # check for Earth model as this affects the north-south and
        # west-east offsets
        # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
        earth_model = data15hd['GeometricProcessing']['EarthModel'][
            'TypeOfEarthModel']
        if earth_model not in [1,2]:
            raise NotImplementedError(
                'Unrecognised Earth model: {}'.format(earth_model)
            )
        else:
            # initialize offset assuming no correction needs to be done
            ns_offset = 0
            we_offset = 0

        # Calculations assume grid origin is south-east corner
        # section 7.2.4 of MSG Level 1.5 Image Data Format Description
        origins = {0: 'NW', 1: 'SW', 2: 'SE', 3: 'NE'}
        grid_origin = data15hd['ImageDescription'][
            "ReferenceGridVIS_IR"]["GridOrigin"]
        if grid_origin != 2:
            raise NotImplementedError(
                'Grid origin not supported number: {}, {} corner'
                .format(grid_origin, origins[grid_origin])
            )

        if dsid.name != 'HRV':

            center_point = 3712/2

            north = int(sec15hd['NorthLineSelectedRectangle']['Value'])
            east = int(sec15hd['EastColumnSelectedRectangle']['Value'])
            west = int(sec15hd['WestColumnSelectedRectangle']['Value'])
            south = int(sec15hd['SouthLineSelectedRectangle']['Value'])

            column_step = data15hd['ImageDescription'][
                'ReferenceGridVIS_IR']['ColumnDirGridStep'] * 1000.0
            line_step = data15hd['ImageDescription'][
                'ReferenceGridVIS_IR']['LineDirGridStep'] * 1000.0

            # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
            if earth_model == 1:
                ns_offset = -0.5  # north +ve
                we_offset = 0.5  # west +ve

            # section 3.1.5 of MSG Level 1.5 Image Data Format Description
            ll_c = (center_point - west - 0.5 + we_offset) * column_step
            ll_l = (south - center_point - 0.5 + ns_offset) * line_step
            ur_c = (center_point - east + 0.5 + we_offset) * column_step
            ur_l = (north - center_point + 0.5 + ns_offset) * line_step

            area_extent = (ll_c, ll_l, ur_c, ur_l)

        else:

            # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
            if earth_model == 1:
                ns_offset = -1.5  # north +ve
                we_offset = 1.5  # west +ve

            if self.roi:
                center_point = 11136/2

                north = int(sec15hd['NorthLineSelectedRectangle']['Value']) * 3
                east = int(sec15hd['EastColumnSelectedRectangle']['Value']) * 3
                west = int(sec15hd['WestColumnSelectedRectangle']['Value']) * 3
                south = int(sec15hd['SouthLineSelectedRectangle']['Value']) * 3

                column_step = data15hd['ImageDescription'][
                    'ReferenceGridHRV']['ColumnDirGridStep'] * 1000.0
                line_step = data15hd['ImageDescription'][
                    'ReferenceGridHRV']['LineDirGridStep'] * 1000.0

                # section 3.1.4.2 of MSG Level 1.5 Image Data Format Description
                if earth_model == 1:
                    ns_offset = -1.5  # north +ve
                    we_offset = 1.5  # west +ve

                # section 3.1.5 of MSG Level 1.5 Image Data Format Description
                ll_c = (center_point - west - 0.5 + we_offset) * column_step
                ll_l = (south - center_point - 0.5 + ns_offset) * line_step
                ur_c = (center_point - east + 0.5 + we_offset) * column_step
                ur_l = (north - center_point + 0.5 + ns_offset) * line_step

                area_extent = (ll_c, ll_l, ur_c, ur_l)

            else:
                raise NotImplementedError('HRV not supported!')



        return area_extent

    def get_dataset(self, dsid, info,
                    xslice=slice(None), yslice=slice(None)):

        channel = dsid.name
        channel_list = self._channel_list

        if channel not in channel_list:
            raise KeyError('Channel % s not available in the file' % channel)
        elif channel not in ['HRV']:
            shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])

            # Check if there is only 1 channel in the list as a change
            # is needed in the arrray assignment ie channl id is not present
            if len(channel_list) == 1:
                raw = self.dask_array['visir']['line_data']
            else:
                i = channel_list.index(channel)
                raw = self.dask_array['visir']['line_data'][:, i, :]

            data = dec10216(raw.flatten())
            data = da.flipud(da.fliplr((data.reshape(shape))))

        else:
            shape = (self.mda['hrv_number_of_lines'], self.mda['hrv_number_of_columns'])

            raw2 = self.dask_array['hrv']['line_data'][:, 2, :]
            raw1 = self.dask_array['hrv']['line_data'][:, 1, :]
            raw0 = self.dask_array['hrv']['line_data'][:, 0, :]

            shape_layer = (self.mda['number_of_lines'], self.mda['hrv_number_of_columns'])
            data2 = dec10216(raw2.flatten())
            data2 = da.flipud(da.fliplr((data2.reshape(shape_layer))))
            data1 = dec10216(raw1.flatten())
            data1 = da.flipud(da.fliplr((data1.reshape(shape_layer))))
            data0 = dec10216(raw0.flatten())
            data0 = da.flipud(da.fliplr((data0.reshape(shape_layer))))

            data = np.zeros(shape)
            idx = range(0, shape[0], 3)
            data[idx, :] = data2
            idx = range(1, shape[0], 3)
            data[idx, :] = data1
            idx = range(2, shape[0], 3)
            data[idx, :] = data0

        xarr = xr.DataArray(data, dims=['y', 'x']).where(data != 0).astype(np.float32)

        if xarr is None:
            dataset = None
        else:
            dataset = self.calibrate(xarr, dsid)
            dataset.attrs['units'] = info['units']
            dataset.attrs['wavelength'] = info['wavelength']
            dataset.attrs['standard_name'] = info['standard_name']
            dataset.attrs['platform_name'] = self.platform_name
            dataset.attrs['sensor'] = 'seviri'

        return dataset

    def calibrate(self, data, dsid):
        """Calibrate the data."""
        tic = datetime.now()

        data15hdr = self.header['15_DATA_HEADER']
        calibration = dsid.calibration
        channel = dsid.name

        # even though all the channels may not be present in the file,
        # the header does have calibration coefficients for all the channels
        # hence, this channel index needs to refer to full channel list
        i = list(CHANNEL_NAMES.values()).index(channel)

        if calibration == 'counts':
            return data

        if calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            # you cant apply GSICS values to the VIS channels
            visual_channels = ['HRV', 'VIS006', 'VIS008', 'IR_016']

            # determine the required calibration coefficients to use
            # for the Level 1.5 Header
            calMode = os.environ.get('CAL_MODE', 'NOMINAL')

            # NB GSICS doesn't have calibration coeffs for VIS channels
            if (calMode.upper() != 'GSICS' or channel in visual_channels):
                coeffs = data15hdr[
                    'RadiometricProcessing']['Level15ImageCalibration']
                gain = coeffs['CalSlope'][i]
                offset = coeffs['CalOffset'][i]
            else:
                coeffs = data15hdr[
                    'RadiometricProcessing']['MPEFCalFeedback']
                gain = coeffs['GSICSCalCoeff'][i]
                offset = coeffs['GSICSOffsetCount'][i]
                offset = offset * gain
            res = self._convert_to_radiance(data, gain, offset)

        if calibration == 'reflectance':
            solar_irradiance = CALIB[self.platform_id][channel]["F"]
            res = self._vis_calibrate(res, solar_irradiance)

        elif calibration == 'brightness_temperature':
            cal_type = data15hdr['ImageDescription'][
                'Level15ImageProduction']['PlannedChanProcessing'][i]
            res = self._ir_calibrate(res, channel, cal_type)

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res


def get_available_channels(header):
    """Get the available channels from the header information"""

    chlist_str = header['15_SECONDARY_PRODUCT_HEADER'][
        'SelectedBandIDs']['Value']
    retv = {}

    for idx, char in zip(range(12), chlist_str):
        retv[CHANNEL_NAMES[idx + 1]] = (char == 'X')

    return retv
