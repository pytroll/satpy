#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Adam.Dybbroe

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

import logging
from datetime import datetime
import numpy as np

import xarray as xr

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hrit_msg import (CALIB, SATNUM, BTFIT)

from pyresample import geometry

from satpy.readers.native_msg_hdr import Msg15NativeHeaderRecord
from satpy.readers.msg_base import get_cds_time
from satpy.readers.msg_base import dec10216
import satpy.readers.msg_base as mb

import os


CHANNEL_LIST = ['VIS006', 'VIS008', 'IR_016', 'IR_039',
                'WV_062', 'WV_073', 'IR_087', 'IR_097',
                'IR_108', 'IR_120', 'IR_134', 'HRV']


class CalibrationError(Exception):
    pass


logger = logging.getLogger('native_msg')


class NativeMSGFileHandler(BaseFileHandler):

    """Native MSG format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(NativeMSGFileHandler, self).__init__(filename,
                                                   filename_info,
                                                   filetype_info)

        self.filename = filename
        self.platform_name = None
        self.available_channels = {}
        self.channel_order_list = []
        for item in CHANNEL_LIST:
            self.available_channels[item] = False

        self._get_header()
        for item in CHANNEL_LIST:
            if self.available_channels.get(item):
                self.channel_order_list.append(item)

        self.memmap = self._get_memmap()

    @property
    def start_time(self):
        tstart = self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['TrueRepeatCycleStart']
        return get_cds_time(
            tstart['Day'][0], tstart['MilliSecsOfDay'][0])

    @property
    def end_time(self):
        tend = self.header['15_DATA_HEADER']['ImageAcquisition'][
            'PlannedAcquisitionTime']['PlannedRepeatCycleEnd']
        return get_cds_time(
            tend['Day'][0], tend['MilliSecsOfDay'][0])

    def _get_memmap(self):
        """Get the numpy memory map for the SEVIRI data"""

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
            [s for s in self.channel_order_list if not s == 'HRV'])
        drec = [('visir', (visir_rec, number_of_lowres_channels))]
        if self.available_channels['HRV']:
            hrv_rec = get_lrec(self._cols_hrv)
            drec.append(('hrv', (hrv_rec, 3)))

        return np.dtype(drec)

    def _get_header(self):
        """Read the header info"""

        hdrrec = Msg15NativeHeaderRecord().get()
        hd_dt = np.dtype(hdrrec)
        hd_dt = hd_dt.newbyteorder('>')
        self.header = np.fromfile(self.filename, dtype=hd_dt, count=1)
        # Set the list of available channels:
        chlist_str = self.header['15_SECONDARY_PRODUCT_HEADER'][
            'SelectedBandIDs'][0][-1].strip().decode()

        for item, chmark in zip(CHANNEL_LIST, chlist_str):
            self.available_channels[item] = (chmark == 'X')

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
        self.mda['number_of_lines'] = self.header['15_DATA_HEADER'][
            'ImageDescription']['ReferenceGridVIS_IR']['NumberOfLines'][0]
        # The number of columns is incorrect - seems to be fixed at 3712
        # EUMETSAT will fix this
        self.mda['number_of_columns'] = self.header['15_DATA_HEADER'][
            'ImageDescription']['ReferenceGridVIS_IR']['NumberOfColumns'][0]

        sec15hd = self.header['15_SECONDARY_PRODUCT_HEADER']
        numlines_visir = int(sec15hd['NumberLinesVISIR']['Value'][0])

        west = int(sec15hd['WestColumnSelectedRectangle']['Value'][0])
        east = int(sec15hd['EastColumnSelectedRectangle']['Value'][0])
        north = int(sec15hd["NorthLineSelectedRectangle"]['Value'][0])
        south = int(sec15hd["SouthLineSelectedRectangle"]['Value'][0])

        numcols_hrv = int(sec15hd["NumberColumnsHRV"]['Value'][0])

        # We suspect the UMARF will pad out any ROI colums that
        # arent divisible by 4 so here we work out how many pixels have
        # been added to the column.
        x = ((west-east+1)*(10.0/8) % 1)
        y = int((1-x)*4)

        if y < 4:
            # column has been padded with y pixels
            self._cols_visir = int((west-east+1+y)*1.25)
        else:
            # no padding has occurred
            self._cols_visir = int((west-east+1)*1.25)

        if (west - east) < 3711:
            self._cols_hrv = int(np.ceil(numcols_hrv * 10.0 / 8))  # 6960
        else:
            self._cols_hrv = int(np.ceil(5568 * 10.0 / 8))  # 6960

        # 'WestColumnSelectedRectangle' - 'EastColumnSelectedRectangle'
        # 'NorthLineSelectedRectangle' - 'SouthLineSelectedRectangle'

        coldir_step = self.header['15_DATA_HEADER']['ImageDescription'][
            "ReferenceGridVIS_IR"]["ColumnDirGridStep"][0] * 1000.0

        lindir_step = self.header['15_DATA_HEADER']['ImageDescription'][
            "ReferenceGridVIS_IR"]["LineDirGridStep"][0] * 1000.0

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

        nlines = int(self.mda['number_of_lines'])
        ncols = int(self.mda['number_of_columns'])

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

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

    def get_dataset(self, key, info,
                    xslice=slice(None), yslice=slice(None)):

        if key.name not in self.channel_order_list:
            raise KeyError('Channel % s not available in the file' % key.name)
        elif key.name not in ['HRV']:
            ch_idn = self.channel_order_list.index(key.name)
            # Check if there is only 1 channel in the list as a change
            # is needed in the arrray assignment ie channl id is not present
            if len(self.channel_order_list) == 1:
                data = dec10216(
                        self.memmap['visir']['line_data'][:, :])[::-1, ::-1]
            else:
                data = dec10216(
                    self.memmap['visir']['line_data'][:, ch_idn, :])[::-1, ::-1]

        else:
            data2 = dec10216(
                self.memmap["hrv"]["line_data"][:, 2, :])[::-1, ::-1]
            data1 = dec10216(
                self.memmap["hrv"]["line_data"][:, 1, :])[::-1, ::-1]
            data0 = dec10216(
                self.memmap["hrv"]["line_data"][:, 0, :])[::-1, ::-1]
            # Make empty array:
            shape = data0.shape[0] * 3, data0.shape[1]
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

        out = self.calibrate(out, key)
        out.attrs['units'] = info['units']
        out.attrs['wavelength'] = info['wavelength']
        out.attrs['standard_name'] = info['standard_name']
        out.attrs['platform_name'] = self.platform_name
        out.attrs['sensor'] = 'seviri'

        return out

    def calibrate(self, data, key):
        """Calibrate the data."""
        tic = datetime.now()
        calibration = key.calibration
        if calibration == 'counts':
            return
        if calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            res = self.convert_to_radiance(data, key.name)
        if calibration == 'reflectance':
            res = self._vis_calibrate(res, key.name)
        elif calibration == 'brightness_temperature':
            res = self._ir_calibrate(res, key.name)

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return res

    def convert_to_radiance(self, data, key_name):
        """Calibrate to radiance."""
        # all 12 channels are in calibration coefficients
        # regardless of how many channels are in file
        channel_index = CHANNEL_LIST.index(key_name)
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
            offset = offset*gain

        return mb.convert_to_radiance(data, gain, offset)

    def _vis_calibrate(self, data, key_name):
        """Visible channel calibration only."""
        solar_irradiance = CALIB[self.platform_id][key_name]["F"]
        return mb.vis_calibrate(data, solar_irradiance)

    def _erads2bt(self, data, key_name):
        """computation based on effective radiance."""
        cal_info = CALIB[self.platform_id][key_name]
        alpha = cal_info["ALPHA"]
        beta = cal_info["BETA"]
        wavenumber = CALIB[self.platform_id][key_name]["VC"]
        return mb.erads2bt(data, wavenumber, alpha, beta)

    def _srads2bt(self, data, key_name):
        """computation based on spectral radiance."""
        coef_a, coef_b, coef_c = BTFIT[key_name]
        wavenumber = CALIB[self.platform_id][key_name]["VC"]

        return mb.srads2bt(data, wavenumber, coef_a, coef_b, coef_c)

    def _ir_calibrate(self, data, key_name):
        """IR calibration."""
        channel_index = self.channel_order_list.index(key_name)

        cal_type = self.header['15_DATA_HEADER']['ImageDescription'][
            'Level15ImageProduction']['PlannedChanProcessing'][0][channel_index]

        if cal_type == 1:
            # spectral radiances
            return self._srads2bt(data, key_name)
        elif cal_type == 2:
            # effective radiances
            return self._erads2bt(data, key_name)
        else:
            raise NotImplementedError('Unknown calibration type')
