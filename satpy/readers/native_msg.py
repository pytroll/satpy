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

https://www.google.se/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwi_pcrm1vjSAhUMS5oKHc3QADgQFggcMAA&url=http%3A%2F%2Fwww.eumetsat.int%2Fwebsite%2Fwcm%2Fidc%2Fidcplg%3FIdcService%3DGET_FILE%26dDocName%3DPDF_FG15_MSG-NATIVE-FORMAT-15%26RevisionSelectionMethod%3DLatestReleased%26Rendition%3DWeb&usg=AFQjCNE9YWoECpgBhGGSkPZkHBWB37pgUA&sig2=4TXWtAj-yFA6XCaTmhFN9w


"""

import logging
from datetime import datetime
import numpy as np

from satpy.dataset import Dataset, DatasetID
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.hrit_msg import (CALIB, SATNUM, C1, C2, BTFIT)

from pyresample import geometry
from satpy.readers.hrit_base import (HRITFileHandler, ancillary_text,
                                     annotation_header, base_hdr_map,
                                     image_data_function, make_time_cds_short,
                                     time_cds_short)

from satpy.readers.native_msg_hdr import Msg15NativeHeaderRecord
from satpy.readers.msg_base import get_cds_time
from satpy.readers.msg_base import dec10216
#from satpy.readers.hrit_base import dec10216

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

        self.filename = filename
        self.platform_name = None
        self.available_channels = {}
        self.channel_order_list = []
        for item in CHANNEL_LIST:
            self.available_channels[item] = False

        self._get_header()

        for item in CHANNEL_LIST:
            if item in self.available_channels and self.available_channels[item]:
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
            return np.memmap(
                fp_, dtype=dt, shape=(self.data_len, ), offset=hdr_size, mode="r")

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
            'SelectedBandIDs'][0][-1].strip()
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
        self.mda['number_of_columns'] = self.header['15_DATA_HEADER'][
            'ImageDescription']['ReferenceGridVIS_IR']['NumberOfColumns'][0]

        sec15hd = self.header['15_SECONDARY_PRODUCT_HEADER']
        numlines_visir = int(sec15hd['NumberLinesVISIR']['Value'][0])
        west = int(sec15hd['WestColumnSelectedRectangle']['Value'][0])
        east = int(sec15hd['EastColumnSelectedRectangle']['Value'][0])
        north = int(sec15hd["NorthLineSelectedRectangle"]['Value'][0])
        south = int(sec15hd["SouthLineSelectedRectangle"]['Value'][0])
        numcols_hrv = int(sec15hd["NumberColumnsHRV"]['Value'][0])

        # Subsetting doesn't work unless number of pixels on a line
        # divded by 4 is a whole number!
        # FIXME!
        if abs(int(numlines_visir / 4.) - numlines_visir / 4.) > 0.001:
            msgstr = (
                "Number of pixels in east-west direction needs to be a multiple of 4!" +
                "\nPlease get the full disk!")
            raise NotImplementedError(msgstr)

        # Data are stored in 10 bits!
        self._cols_visir = int(np.ceil(numlines_visir * 10.0 / 8))  # 4640
        if (west - east) < 3711:
            self._cols_hrv = int(np.ceil(numcols_hrv * 10.0 / 8))  # 6960
        else:
            self._cols_hrv = int(np.ceil(5568 * 10.0 / 8))  # 6960

        #'WestColumnSelectedRectangle' - 'EastColumnSelectedRectangle'
        #'NorthLineSelectedRectangle' - 'SouthLineSelectedRectangle'

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

    def get_dataset(self, key, info, out=None,
                    xslice=slice(None), yslice=slice(None)):

        if key.name not in self.channel_order_list:
            raise KeyError('Channel % s not available in the file' % key.name)
        elif key.name not in ['HRV']:
            ch_idn = self.channel_order_list.index(key.name)
            data = dec10216(
                self.memmap['visir']['line_data'][:, ch_idn, :])[::-1, ::-1]

            data = np.ma.masked_array(data, mask=(data == 0))
            res = Dataset(data, dtype=np.float32)
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

            data = np.ma.masked_array(data, mask=(data == 0))
            res = Dataset(data, dtype=np.float32)

        if res is not None:
            out = res
        else:
            return None

        self.calibrate(out, key)
        out.info['units'] = info['units']
        out.info['wavelength'] = info['wavelength']
        out.info['standard_name'] = info['standard_name']
        out.info['platform_name'] = self.platform_name
        out.info['sensor'] = 'seviri'

        return out

    def calibrate(self, data, key):
        """Calibrate the data."""
        tic = datetime.now()

        calibration = key.calibration
        if calibration == 'counts':
            return

        if calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            self.convert_to_radiance(data, key.name)
        if calibration == 'reflectance':
            self._vis_calibrate(data, key.name)
        elif calibration == 'brightness_temperature':
            self._ir_calibrate(data, key.name)

        logger.debug("Calibration time " + str(datetime.now() - tic))

    def convert_to_radiance(self, data, key_name):
        """Calibrate to radiance."""

        coeffs = self.header['15_DATA_HEADER'][
            'RadiometricProcessing']['Level15ImageCalibration']

        channel_index = self.channel_order_list.index(key_name)

        gain = coeffs['CalSlope'][0][channel_index]
        offset = coeffs['CalOffset'][0][channel_index]

        data.data[:] *= gain
        data.data[:] += offset
        data.data[data.data < 0] = 0

    def _vis_calibrate(self, data, key_name):
        """Visible channel calibration only."""
        solar_irradiance = CALIB[self.platform_id][key_name]["F"]
        data.data[:] *= 100 / solar_irradiance

    def _tl15(self, data, key_name):
        """Compute the L15 temperature."""
        wavenumber = CALIB[self.platform_id][key_name]["VC"]
        data.data[:] **= -1
        data.data[:] *= C1 * wavenumber ** 3
        data.data[:] += 1
        np.log(data.data, out=data.data)
        data.data[:] **= -1
        data.data[:] *= C2 * wavenumber

    def _erads2bt(self, data, key_name):
        """computation based on effective radiance."""
        cal_info = CALIB[self.platform_id][key_name]
        alpha = cal_info["ALPHA"]
        beta = cal_info["BETA"]

        self._tl15(data, key_name)

        data.data[:] -= beta
        data.data[:] /= alpha

    def _srads2bt(self, data, key_name):
        """computation based on spectral radiance."""
        coef_a, coef_b, coef_c = BTFIT[key_name]

        self._tl15(data, key_name)

        data.data[:] = (coef_a * data.data[:] ** 2 +
                        coef_b * data.data[:] +
                        coef_c)

    def _ir_calibrate(self, data, key_name):
        """IR calibration."""

        channel_index = self.channel_order_list.index(key_name)

        cal_type = self.header['15_DATA_HEADER']['ImageDescription'][
            'Level15ImageProduction']['PlannedChanProcessing'][0][channel_index]

        if cal_type == 1:
            # spectral radiances
            self._srads2bt(data, key_name)
        elif cal_type == 2:
            # effective radiances
            self._erads2bt(data, key_name)
        else:
            raise NotImplementedError('Unknown calibration type')
