#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Ulrich Hamann <ulrich.hamann@meteoswiss.ch>
#   Sauli Joro <sauli.joro@icloud.com>

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

"""A reader for the EUMETSAT MSG native format

https://www.eumetsat.int/website/wcm/idc/idcplg?IdcService=GET_FILE&dDocName=PDF_FG15_MSG-NATIVE-FORMAT-15&RevisionSelectionMethod=LatestReleased&Rendition=Web

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
from satpy.readers.msg_base import SEVIRICalibrationHandler
from satpy.readers.msg_base import (CHANNEL_NAMES, CALIB, SATNUM, BTFIT)
from satpy.readers.native_msg_hdr import Msg15NativeHeaderRecord
import satpy.readers.msg_base as mb


class CalibrationError(Exception):
    pass


logger = logging.getLogger('native_msg')


class NativeMSGFileHandler(BaseFileHandler, SEVIRICalibrationHandler):

    """Native MSG format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(NativeMSGFileHandler, self).__init__(filename,
                                                   filename_info,
                                                   filetype_info)

        self.filename = filename
        self.platform_name = None

        # The available channels are only known after the header
        # has been read, after that we know what the indices are for each channel
        self.available_channels = {}
        self._get_header()

        self.dask_array = da.from_array(self._get_memmap(), chunks=(CHUNK_SIZE,))

    @property
    def start_time(self):
        tstart = self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['TrueRepeatCycleStart']
        return mb.get_cds_time(
            tstart['Day'][0], tstart['MilliSecsOfDay'][0])

    @property
    def end_time(self):
        tend = self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['PlannedRepeatCycleEnd']
        return mb.get_cds_time(
            tend['Day'][0], tend['MilliSecsOfDay'][0])

    def _get_memmap(self):
        """Get the memory map for the SEVIRI data"""

        with open(self.filename) as fp_:

            dt = self._get_filedtype()

            # Lazy reading:
            hdr_size = self.header.dtype.itemsize

            return np.memmap(fp_, dtype=dt, shape=(self.data_len,),
                             offset=hdr_size, mode="r")

    def _get_filedtype(self):
        """Get the dtype of the file based on the actual available channels"""

        pkhrec = [
            ('GP_PK_HEADER', self.header['GP_PK_HEADER'].dtype),
            ('GP_PK_SH1', self.header['GP_PK_SH1'].dtype)
        ]
        pk_head_dtype = np.dtype(pkhrec)

        # Create memory map for lazy reading of channel data:

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

        visir_rec = get_lrec(self._cols_visir)

        number_of_lowres_channels = len(
            [s for s in self._channel_index_list if not s == 'HRV'])
        drec = [('visir', (visir_rec, number_of_lowres_channels))]
        if self.available_channels['HRV']:
            hrv_rec = get_lrec(int(self.mda['hrv_number_of_columns'] * 1.25))
            drec.append(('hrv', (hrv_rec, 3)))

        return np.dtype(drec)

    def _get_header(self):
        """Read the header info"""

        hdrrec = Msg15NativeHeaderRecord().get()
        hd_dt = np.dtype(hdrrec)
        hd_dt = hd_dt.newbyteorder('>')
        self.header = np.fromfile(self.filename, dtype=hd_dt, count=1)

        # Set the list of available channels:
        self.available_channels = get_available_channels(self.header)
        self._channel_index_list = [i for i in CHANNEL_NAMES.values()
                                    if self.available_channels[i]]

        self.platform_id = self.header['15_DATA_HEADER'][
            'SatelliteStatus']['SatelliteDefinition']['SatelliteId'][0]
        self.platform_name = "Meteosat-" + SATNUM[self.platform_id]

        ssp_lon = self.header['15_DATA_HEADER']['ImageDescription'][
            'ProjectionDescription']['LongitudeOfSSP'][0]

        self.mda = {}
        equator_radius = self.header['15_DATA_HEADER']['GeometricProcessing'][
            'EarthModel']['EquatorialRadius'][0] * 1000.
        north_polar_radius = self.header['15_DATA_HEADER'][
            'GeometricProcessing']['EarthModel']['NorthPolarRadius'][0] * 1000.
        south_polar_radius = self.header['15_DATA_HEADER'][
            'GeometricProcessing']['EarthModel']['SouthPolarRadius'][0] * 1000.
        polar_radius = (north_polar_radius + south_polar_radius) * 0.5
        self.mda['projection_parameters'] = {'a': equator_radius,
                                             'b': polar_radius,
                                             'h': 35785831.00,
                                             'SSP_longitude': ssp_lon}

        sec15hd = self.header['15_SECONDARY_PRODUCT_HEADER']
        numlines_visir = int(sec15hd['NumberLinesVISIR']['Value'][0])

        self.mda['number_of_lines'] = numlines_visir

        west = int(sec15hd['WestColumnSelectedRectangle']['Value'][0])
        east = int(sec15hd['EastColumnSelectedRectangle']['Value'][0])
        north = int(sec15hd["NorthLineSelectedRectangle"]['Value'][0])
        south = int(sec15hd["SouthLineSelectedRectangle"]['Value'][0])

        # We suspect the UMARF will pad out any ROI colums that
        # arent divisible by 4 so here we work out how many pixels have
        # been added to the column.
        x = ((west - east + 1) * (10.0 / 8) % 1)
        y = int((1 - x) * 4)

        if y < 4:
            # column has been padded with y pixels
            self._cols_visir = int((west - east + 1 + y) * 1.25)
        else:
            # no padding has occurred
            self._cols_visir = int((west - east + 1) * 1.25)

        self.mda['number_of_columns'] = int(self._cols_visir / 1.25)
        self.mda['hrv_number_of_lines'] = int(sec15hd["NumberLinesHRV"]['Value'][0])
        # The number of HRV columns seem to be correct in the UMARF header:
        self.mda['hrv_number_of_columns'] = int(sec15hd["NumberColumnsHRV"]['Value'][0])

        coldir_step = self.header['15_DATA_HEADER']['ImageDescription'][
            "ReferenceGridVIS_IR"]["ColumnDirGridStep"][0] * 1000.0

        lindir_step = self.header['15_DATA_HEADER']['ImageDescription'][
            "ReferenceGridVIS_IR"]["LineDirGridStep"][0] * 1000.0

        # Check the calculated row,column dimensions against the header information:
        numcols_cal = self.mda['number_of_columns']
        numcols_hd = self.header['15_DATA_HEADER']['ImageDescription']['ReferenceGridVIS_IR']['NumberOfColumns'][0]
        if numcols_cal != numcols_hd:
            logger.warning("Number of (non HRV band) columns from header and derived from data are not consistent!")
            logger.warning("Number of columns read from header = %d", numcols_hd)
            logger.warning("Number of columns calculated from data = %d", numcols_cal)

        area_extent = ((1856 - west - 0.5) * coldir_step,
                       (1856 - north + 0.5) * lindir_step,
                       (1856 - east + 0.5) * coldir_step,
                       (1856 - south + 1.5) * lindir_step)

        self.area_extent = area_extent

        self.data_len = numlines_visir

    def get_area_def(self, dsid):

        a = self.mda['projection_parameters']['a']
        b = self.mda['projection_parameters']['b']
        h = self.mda['projection_parameters']['h']
        lon_0 = self.mda['projection_parameters']['SSP_longitude']

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
            self.area_extent)

        self.area = area
        return area

    def get_dataset(self, dataset_id, info,
                    xslice=slice(None), yslice=slice(None)):

        channel_name = dataset_id.name

        # import ipdb; ipdb.set_trace()
        if channel_name not in self._channel_index_list:
            raise KeyError('Channel % s not available in the file' % channel_name)
        elif channel_name not in ['HRV']:
            shape = (self.mda['number_of_lines'], self.mda['number_of_columns'])

            ch_idn = self._channel_index_list.index(channel_name)
            # Check if there is only 1 channel in the list as a change
            # is needed in the arrray assignment ie channl id is not present
            if len(self._channel_index_list) == 1:
                raw = self.dask_array['visir']['line_data']
            else:
                raw = self.dask_array['visir']['line_data'][:, ch_idn, :]

            data = mb.dec10216(raw.flatten())
            data = da.flipud(da.fliplr((data.reshape(shape))))

        else:
            shape = (self.mda['hrv_number_of_lines'], self.mda['hrv_number_of_columns'])

            raw2 = self.dask_array['hrv']['line_data'][:, 2, :]
            raw1 = self.dask_array['hrv']['line_data'][:, 1, :]
            raw0 = self.dask_array['hrv']['line_data'][:, 0, :]

            shape_layer = (self.mda['number_of_lines'], self.mda['hrv_number_of_columns'])
            data2 = mb.dec10216(raw2.flatten())
            data2 = da.flipud(da.fliplr((data2.reshape(shape_layer))))
            data1 = mb.dec10216(raw1.flatten())
            data1 = da.flipud(da.fliplr((data1.reshape(shape_layer))))
            data0 = mb.dec10216(raw0.flatten())
            data0 = da.flipud(da.fliplr((data0.reshape(shape_layer))))

            data = np.zeros(shape)
            idx = range(0, shape[0], 3)
            data[idx, :] = data2
            idx = range(1, shape[0], 3)
            data[idx, :] = data1
            idx = range(2, shape[0], 3)
            data[idx, :] = data0

        res = xr.DataArray(data, dims=['y', 'x']).where(data != 0).astype(np.float32)

        if res is not None:
            out = res
        else:
            return None

        out = self.calibrate(out, dataset_id)
        out.attrs['units'] = info['units']
        out.attrs['wavelength'] = info['wavelength']
        out.attrs['standard_name'] = info['standard_name']
        out.attrs['platform_name'] = self.platform_name
        out.attrs['sensor'] = 'seviri'

        return out

    def calibrate(self, data, dataset_id):
        """Calibrate the data."""
        tic = datetime.now()

        calibration = dataset_id.calibration
        channel_name = dataset_id.name
        channel_index = self._channel_index_list.index(channel_name)

        if calibration == 'counts':
            return

        if calibration in ['radiance', 'reflectance', 'brightness_temperature']:

            calMode = 'NOMINAL'
            # determine the required calibration coefficients to use
            # for the Level 1.5 Header
            # NB gsics doesnt apply to VIS channels so ignore them
            if (channel_index > 2):
                calMode = os.environ.get('CAL_MODE', 'NOMINAL')

            if (calMode.upper() != 'GSICS'):
                coeffs = self.header['15_DATA_HEADER'][
                    'RadiometricProcessing']['Level15ImageCalibration']
                gain = coeffs['CalSlope'][0][channel_index]
                offset = coeffs['CalOffset'][0][channel_index]
            else:
                coeffs = self.header['15_DATA_HEADER'][
                    'RadiometricProcessing']['MPEFCalFeedback']
                gain = coeffs['GSICSCalCoeff'][0][channel_index]
                offset = coeffs['GSICSOffsetCount'][0][channel_index]
                offset = offset * gain
            res = self._convert_to_radiance(data, gain, offset)

        if calibration == 'reflectance':
            solar_irradiance = CALIB[self.platform_id][channel_name]["F"]
            res = self._vis_calibrate(res, solar_irradiance)

        elif calibration == 'brightness_temperature':
            cal_type = self.header['15_DATA_HEADER']['ImageDescription'][
                'Level15ImageProduction']['PlannedChanProcessing'][0][channel_index]
            res = self._ir_calibrate(res, channel_name, cal_type)

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

    # def convert_to_radiance(self, data, channel_name):
    #     """Calibrate to radiance."""
        # all 12 channels are in calibration coefficients
        # regardless of how many channels are in file
        #channel_index = CHANNEL_LIST.index(channel_name)
        # channel_index = [key - 1 for key, value in CHANNEL_NAMES.items()
        #                  if value == channel_name][0]
        #
        # calMode = 'NOMINAL'
        # # determine the required calibration coefficients to use
        # # for the Level 1.5 Header
        # # NB gsics doesnt apply to VIS channels so ignore them
        # if (channel_index > 2):
        #     calMode = os.environ.get('CAL_MODE', 'NOMINAL')
        #
        # if (calMode.upper() != 'GSICS'):
        #     coeffs = self.header['15_DATA_HEADER'][
        #         'RadiometricProcessing']['Level15ImageCalibration']
        #     gain = coeffs['CalSlope'][0][channel_index]
        #     offset = coeffs['CalOffset'][0][channel_index]
        # else:
        #     coeffs = self.header['15_DATA_HEADER'][
        #         'RadiometricProcessing']['MPEFCalFeedback']
        #     gain = coeffs['GSICSCalCoeff'][0][channel_index]
        #     offset = coeffs['GSICSOffsetCount'][0][channel_index]
        #     offset = offset * gain
        #
        # return mb.convert_to_radiance(data, gain, offset)

    # def _vis_calibrate(self, data, channel_name):
    #     """Visible channel calibration only."""
    #     solar_irradiance = CALIB[self.platform_id][channel_name]["F"]
    #     return mb.vis_calibrate(data, solar_irradiance)

    # def _erads2bt(self, data, channel_name):
    #     """computation based on effective radiance."""
    #     cal_info = CALIB[self.platform_id][channel_name]
    #     alpha = cal_info["ALPHA"]
    #     beta = cal_info["BETA"]
    #     wavenumber = CALIB[self.platform_id][channel_name]["VC"]
    #     return mb.erads2bt(data, wavenumber, alpha, beta)

    # def _srads2bt(self, data, channel_name):
    #     """computation based on spectral radiance."""
    #     coef_a, coef_b, coef_c = BTFIT[channel_name]
    #     wavenumber = CALIB[self.platform_id][channel_name]["VC"]
    #
    #     return mb.srads2bt(data, wavenumber, coef_a, coef_b, coef_c)

    # def _ir_calibrate(self, data, channel_name):
    #     """IR calibration."""
    #     channel_index = self._channel_index_list.index(channel_name)
    #
    #     cal_type = self.header['15_DATA_HEADER']['ImageDescription'][
    #         'Level15ImageProduction']['PlannedChanProcessing'][0][channel_index]
    #
    #     if cal_type == 1:
    #         # spectral radiances
    #         return self._srads2bt(data, channel_name)
    #     elif cal_type == 2:
    #         # import ipdb; ipdb.set_trace()
    #         # effective radiances
    #         return self._erads2bt(data, channel_name)
    #     else:
    #         raise NotImplementedError('Unknown calibration type')


def get_available_channels(header):
    """Get the available channels from the header information"""

    try:
        chlist_str = header['15_SECONDARY_PRODUCT_HEADER'][
            'SelectedBandIDs'][0][-1].strip().decode()
    except AttributeError:
        # Strings have no deocde method in py3
        chlist_str = header['15_SECONDARY_PRODUCT_HEADER'][
            'SelectedBandIDs'][0][-1].strip()

    retv = {}


    for idx, chmark in zip(range(12), chlist_str):
        retv[CHANNEL_NAMES[idx + 1]] = (chmark == 'X')

    return retv
