#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2015, 2016 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>
#   Cooke, Michael.C, UK Met Office
#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""Advanced Himawari Imager (AHI) standard format data reader

http://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html

"""

AHI_CHANNEL_NAMES = ("1", "2", "3", "4", "5",
                     "6", "7", "8", "9", "10",
                     "11", "12", "13", "14", "15", "16")
import logging
from datetime import datetime, timedelta

import numpy as np

from pyresample import geometry
from satpy.dataset import Dataset
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.helper_functions import get_geostationary_angle_extent


class CalibrationError(Exception):
    pass

logger = logging.getLogger('ahi_hsd')

# Basic information block:
_BASIC_INFO_TYPE = np.dtype([("hblock_number", "u1"),
                             ("blocklength", "<u2"),
                             ("total_number_of_hblocks", "<u2"),
                             ("byte_order", "u1"),
                             ("satellite", "S16"),
                             ("proc_center_name", "S16"),
                             ("observation_area", "S4"),
                             ("other_observation_info", "S2"),
                             ("observation_timeline", "<u2"),
                             ("observation_start_time", "f8"),
                             ("observation_end_time", "f8"),
                             ("file_creation_time", "f8"),
                             ("total_header_length", "<u4"),
                             ("total_data_length", "<u4"),
                             ("quality_flag1", "u1"),
                             ("quality_flag2", "u1"),
                             ("quality_flag3", "u1"),
                             ("quality_flag4", "u1"),
                             ("file_format_version", "S32"),
                             ("file_name", "S128"),
                             ("spare", "S40"),
                             ])

# Data information block
_DATA_INFO_TYPE = np.dtype([("hblock_number", "u1"),
                            ("blocklength", "<u2"),
                            ("number_of_bits_per_pixel", "<u2"),
                            ("number_of_columns", "<u2"),
                            ("number_of_lines", "<u2"),
                            ("compression_flag_for_data", "u1"),
                            ("spare", "S40"),
                            ])

# Projection information block
# See footnote 2; LRIT/HRIT Global Specification Section 4.4, CGMS, 1999)
_PROJ_INFO_TYPE = np.dtype([("hblock_number", "u1"),
                            ("blocklength", "<u2"),
                            ("sub_lon", "f8"),
                            ("CFAC", "<u4"),
                            ("LFAC", "<u4"),
                            ("COFF", "f4"),
                            ("LOFF", "f4"),
                            ("distance_from_earth_center", "f8"),
                            ("earth_equatorial_radius", "f8"),
                            ("earth_polar_radius", "f8"),
                            ("req2_rpol2_req2", "f8"),
                            ("rpol2_req2", "f8"),
                            ("req2_rpol2", "f8"),
                            ("coeff_for_sd", "f8"),
                            # Note: processing center use only:
                            ("resampling_types", "<i2"),
                            # Note: processing center use only:
                            ("resampling_size", "<i2"),
                            ("spare", "S40"),
                            ])

# Navigation information block
_NAV_INFO_TYPE = np.dtype([("hblock_number", "u1"),
                           ("blocklength", "<u2"),
                           ("navigation_info_time", "f8"),
                           ("SSP_longitude", "f8"),
                           ("SSP_latitude", "f8"),
                           ("distance_earth_center_to_satellite", "f8"),
                           ("nadir_latitude", "f8"),
                           ("nadir_longitude", "f8"),
                           ("sun_position", "f8", (3,)),
                           ("moon_position", "f8", (3,)),
                           ("spare", "S40"),
                           ])

# Calibration information block
_CAL_INFO_TYPE = np.dtype([("hblock_number", "u1"),
                           ("blocklength", "<u2"),
                           ("band_number", "<u2"),
                           ("central_wave_length", "f8"),
                           ("valid_number_of_bits_per_pixel", "<u2"),
                           ("count_value_error_pixels", "<u2"),
                           ("count_value_outside_scan_pixels", "<u2"),
                           ("gain_count2rad_conversion", "f8"),
                           ("offset_count2rad_conversion", "f8"),
                           ])

# Infrared band (Band No. 7 – 16)
# (Band No. 2 – 5: backup operation (See Table 4 bb))
_IRCAL_INFO_TYPE = np.dtype([("c0_rad2tb_conversion", "f8"),
                             ("c1_rad2tb_conversion", "f8"),
                             ("c2_rad2tb_conversion", "f8"),
                             ("c0_tb2rad_conversion", "f8"),
                             ("c1_tb2rad_conversion", "f8"),
                             ("c2_tb2rad_conversion", "f8"),
                             ("speed_of_light", "f8"),
                             ("planck_constant", "f8"),
                             ("boltzmann_constant", "f8"),
                             ("spare", "S40"),
                             ])

# Visible, near-infrared band (Band No. 1 – 6)
# (Band No. 1: backup operation (See Table 4 bb))
_VISCAL_INFO_TYPE = np.dtype([("coeff_rad2albedo_conversion", "f8"),
                              ("spare", "S104"),
                              ])

# 6 Inter-calibration information block
_INTER_CALIBRATION_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("gsics_calibration_intercept", "f8"),
    ("gsics_calibration_intercept_stderr", "f8"),
    ("gsics_calibration_slope", "f8"),
    ("gsics_calibration_slope_stderr", "f8"),
    ("gsics_calibration_coeff_quadratic_term", "f8"),
    ("gsics_calibration_coeff_quadratic_term_stderr",
     "f8"),
    ("gsics_correction_starttime", "f8"),
    ("gsics_correction_endtime", "f8"),
    ("ancillary_text", "S64"),
    ("spare", "S128"),
])


# 7 Segment information block
_SEGMENT_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("total_number_of_segments", "u1"),
    ("segment_sequence_number", "u1"),
    ("first_line_number_of_image_segment", "u2"),
    ("spare", "S40"),
])

# 8 Navigation correction information block
_NAVIGATION_CORRECTION_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("center_column_of_rotation", "f4"),
    ("center_line_of_rotation", "f4"),
    ("amount_of_rotational_correction", "f8"),
    ("numof_correction_info_data", "<u2"),
])

# 9 Observation time information block
_OBS_TIME_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("number_of_observation_times", "<u2"),
])

# 10 Error information block
_ERROR_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("number_of_error_info_data", "<u2"),
])

# 11 Spare block
_SPARE_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("spare", "S256")
])


class AHIHSDFileHandler(BaseFileHandler):

    """AHI standard format reader
    """

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the reader."""
        super(AHIHSDFileHandler, self).__init__(filename, filename_info,
                                                filetype_info)

        self.channels = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self.units = dict([(i, 'counts') for i in AHI_CHANNEL_NAMES])

        self._data = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self._header = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self.lons = None
        self.lats = None
        self.segment_number = filename_info['segment_number']
        self.total_segments = filename_info['total_segments']

        with open(self.filename) as fd:
            self.basic_info = np.fromfile(fd,
                                          dtype=_BASIC_INFO_TYPE,
                                          count=1)
            self.data_info = np.fromfile(fd,
                                         dtype=_DATA_INFO_TYPE,
                                         count=1)
            self.proj_info = np.fromfile(fd,
                                         dtype=_PROJ_INFO_TYPE,
                                         count=1)[0]
            self.nav_info = np.fromfile(fd,
                                        dtype=_NAV_INFO_TYPE,
                                        count=1)[0]
        self.platform_name = self.basic_info['satellite'][0]
        self.sensor = 'ahi'

    def get_shape(self, dsid, ds_info):
        return int(self.data_info['number_of_lines']), int(self.data_info['number_of_columns'])

    @property
    def start_time(self):
        return (datetime(1858, 11, 17) +
                timedelta(days=float(self.basic_info['observation_start_time'])))

    @property
    def end_time(self):
        return (datetime(1858, 11, 17) +
                timedelta(days=float(self.basic_info['observation_end_time'])))

    def get_dataset(self, key, info, out=None, xslice=slice(None), yslice=slice(None)):
        to_return = out is None
        if out is None:
            nlines = int(self.data_info['number_of_lines'])
            ncols = int(self.data_info['number_of_columns'])
            out = Dataset(np.ma.empty((nlines, ncols), dtype=np.float32))

        self.read_band(key, info, out, xslice, yslice)

        if to_return:
            from satpy.yaml_reader import Shuttle
            return Shuttle(out.data, out.mask, out.info)

    def get_area_def(self, dsid):
        del dsid
        cfac = np.uint32(self.proj_info['CFAC'])
        lfac = np.uint32(self.proj_info['LFAC'])
        coff = np.float32(self.proj_info['COFF'])
        loff = np.float32(self.proj_info['LOFF'])
        a = float(self.proj_info['earth_equatorial_radius'] * 1000)
        h = float(self.proj_info['distance_from_earth_center'] * 1000 - a)
        b = float(self.proj_info['earth_polar_radius'] * 1000)
        lon_0 = float(self.proj_info['sub_lon'])
        nlines = int(self.data_info['number_of_lines'])
        ncols = int(self.data_info['number_of_columns'])

        # count starts at 1
        cols = 1 - 0.5
        lines = (self.total_segments - self.segment_number) * nlines + 1 - 0.5
        ll_x = (cols - coff) / cfac * 2**16
        ll_y = (lines - loff) / lfac * 2**16

        cols += ncols
        lines += nlines
        ur_x = (cols - coff) / cfac * 2**16
        ur_y = (lines - loff) / lfac * 2**16

        area_extent = (np.deg2rad(ll_x) * h, np.deg2rad(ll_y) * h,
                       np.deg2rad(ur_x) * h, np.deg2rad(ur_y) * h)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            'some_area_name',
            "On-the-fly area",
            'geosh8',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self.area = area
        return area

    def get_lonlats(self, key, info, lon_out, lat_out):
        logger.debug('Computing area for %s', str(key))
        lon_out[:], lat_out[:] = self.area.get_lonlats()

    def geo_mask(self, lineslice=None, colslice=None):
        """Masking the space pixels from geometry info."""
        cfac = np.uint32(self.proj_info['CFAC'])
        lfac = np.uint32(self.proj_info['LFAC'])
        coff = np.float32(self.proj_info['COFF'])
        loff = np.float32(self.proj_info['LOFF'])
        nlines = int(self.data_info['number_of_lines'])
        ncols = int(self.data_info['number_of_columns'])

        # count starts at 1
        local_coff = 1
        local_loff = (self.total_segments - self.segment_number) * nlines + 1

        xmax, ymax = get_geostationary_angle_extent(self.area)

        pixel_cmax = np.rad2deg(xmax) * cfac * 1.0 / 2**16
        pixel_lmax = np.rad2deg(ymax) * lfac * 1.0 / 2**16

        def ellipse(line, col):
            line /= pixel_lmax
            line *= line
            col /= pixel_cmax
            col *= col
            return (line + col) > 1

        cols_idx = np.arange(-(coff - local_coff),
                             ncols - (coff - local_coff),
                             dtype=np.float)[colslice]
        lines_idx = np.arange(nlines - (loff - local_loff),
                              -(loff - local_loff),
                              -1,
                              dtype=np.float)[lineslice]

        return ellipse(lines_idx[:, None], cols_idx[None, :])

    def read_band(self, key, info, out=None, xslice=slice(None), yslice=slice(None)):
        """Read the data"""
        tic = datetime.now()
        header = {}
        with open(self.filename, "rb") as fp_:

            header['block1'] = np.fromfile(
                fp_, dtype=_BASIC_INFO_TYPE, count=1)
            header["block2"] = np.fromfile(fp_, dtype=_DATA_INFO_TYPE, count=1)
            header["block3"] = np.fromfile(fp_, dtype=_PROJ_INFO_TYPE, count=1)
            header["block4"] = np.fromfile(fp_, dtype=_NAV_INFO_TYPE, count=1)
            header["block5"] = np.fromfile(fp_, dtype=_CAL_INFO_TYPE, count=1)
            logger.debug("Band number = " +
                         str(header["block5"]['band_number'][0]))
            logger.debug('Time_interval: %s - %s',
                         str(self.start_time), str(self.end_time))
            band_number = header["block5"]['band_number'][0]
            if band_number < 7:
                cal = np.fromfile(fp_, dtype=_VISCAL_INFO_TYPE, count=1)
            else:
                cal = np.fromfile(fp_, dtype=_IRCAL_INFO_TYPE, count=1)

            header['calibration'] = cal

            header["block6"] = np.fromfile(
                fp_, dtype=_INTER_CALIBRATION_INFO_TYPE, count=1)
            header["block7"] = np.fromfile(
                fp_, dtype=_SEGMENT_INFO_TYPE, count=1)
            header["block8"] = np.fromfile(
                fp_, dtype=_NAVIGATION_CORRECTION_INFO_TYPE, count=1)
            # 8 The navigation corrections:
            ncorrs = header["block8"]['numof_correction_info_data'][0]
            dtype = np.dtype([
                ("line_number_after_rotation", "<u2"),
                ("shift_amount_for_column_direction", "f4"),
                ("shift_amount_for_line_direction", "f4"),
            ])
            corrections = []
            for i in range(ncorrs):
                corrections.append(np.fromfile(fp_, dtype=dtype, count=1))
            fp_.seek(40, 1)
            header['navigation_corrections'] = corrections
            header["block9"] = np.fromfile(fp_,
                                           dtype=_OBS_TIME_INFO_TYPE,
                                           count=1)
            numobstimes = header["block9"]['number_of_observation_times'][0]

            dtype = np.dtype([
                ("line_number", "<u2"),
                ("observation_time", "f8"),
            ])
            lines_and_times = []
            for i in range(numobstimes):
                lines_and_times.append(np.fromfile(fp_,
                                                   dtype=dtype,
                                                   count=1))
            header['observation_time_information'] = lines_and_times
            fp_.seek(40, 1)

            header["block10"] = np.fromfile(fp_,
                                            dtype=_ERROR_INFO_TYPE,
                                            count=1)
            dtype = np.dtype([
                ("line_number", "<u2"),
                ("numof_error_pixels_per_line", "<u2"),
            ])
            num_err_info_data = header["block10"][
                'number_of_error_info_data'][0]
            err_info_data = []
            for i in range(num_err_info_data):
                err_info_data.append(np.fromfile(fp_, dtype=dtype, count=1))
            header['error_information_data'] = err_info_data
            fp_.seek(40, 1)

            dummy = np.fromfile(fp_, dtype=_SPARE_TYPE, count=1)

            nlines = int(header["block2"]['number_of_lines'][0])
            ncols = int(header["block2"]['number_of_columns'][0])
            out.data[:] = np.fromfile(
                fp_, dtype='<u2', count=nlines * ncols).reshape((nlines, ncols)).astype(np.float32)[yslice, xslice]

        self._header = header

        out.mask[header['block5']["count_value_outside_scan_pixels"]
                 [0] == out.data] = True
        out.mask[header['block5']["count_value_error_pixels"]
                 [0] == out.data] = True

        out.mask[self.geo_mask(yslice, xslice)] = True

        logger.debug("Reading time " + str(datetime.now() - tic))

        self.calibrate(out, key.calibration)

        new_info = dict(units=info['units'],
                        standard_name=info['standard_name'],
                        wavelength=info['wavelength'],
                        resolution='resolution',
                        id=key,
                        name=key.name,
                        platform_name=self.platform_name,
                        sensor=self.sensor,
                        satellite_longitude=float(
                            self.nav_info['SSP_longitude']),
                        satellite_latitude=float(
                            self.nav_info['SSP_latitude']),
                        satellite_altitude=float(self.nav_info['distance_earth_center_to_satellite'] -
                                                 self.proj_info['earth_equatorial_radius']) * 1000)
        out.info.update(new_info)

    def calibrate(self, data, calibration):
        """Calibrate the data"""
        tic = datetime.now()

        if calibration == 'counts':
            return

        if calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            self.convert_to_radiance(data)
        if calibration == 'reflectance':
            self._vis_calibrate(data)
        elif calibration == 'brightness_temperature':
            self._ir_calibrate(data)

        logger.debug("Calibration time " + str(datetime.now() - tic))

    def convert_to_radiance(self, data):
        """Calibrate to radiance.
        """

        gain = self._header["block5"]["gain_count2rad_conversion"][0]
        offset = self._header["block5"]["offset_count2rad_conversion"][0]

        data.data[:] *= gain
        data.data[:] += offset

    def _vis_calibrate(self, data):
        """Visible channel calibration only.
        """
        coeff = self._header["calibration"]["coeff_rad2albedo_conversion"]
        data.data[:] *= coeff * 100
        data.mask[data.data < 0] = True

    def _ir_calibrate(self, data):
        """IR calibration
        """

        cwl = self._header['block5']["central_wave_length"][0] * 1e-6
        c__ = self._header['calibration']["speed_of_light"][0]
        h__ = self._header['calibration']["planck_constant"][0]
        k__ = self._header['calibration']["boltzmann_constant"][0]
        a__ = (h__ * c__) / (k__ * cwl)

        #b__ = ((2 * h__ * c__ ** 2) / (1.0e6 * cwl ** 5 * data.data)) + 1

        data.data[:] *= 1.0e6 * cwl ** 5
        data.data[:] **= -1
        data.data[:] *= (2 * h__ * c__ ** 2)
        data.data[:] += 1

        #Te_ = a__ / np.log(b__)

        data.data[:] = a__ / np.log(data.data)

        c0_ = self._header['calibration']["c0_rad2tb_conversion"][0]
        c1_ = self._header['calibration']["c1_rad2tb_conversion"][0]
        c2_ = self._header['calibration']["c2_rad2tb_conversion"][0]

        #data.data[:] = c0_ + c1_ * Te_ + c2_ * Te_ ** 2

        data.data[:] = np.polyval([c2_, c1_, c0_], data.data)

        data.mask[data.data < 0] = True
        data.mask[np.isnan(data.data)] = True


def show(data, negate=False):
    """Show the stretched data.
    """
    from PIL import Image as pil
    data = np.array((data - data.min()) * 255.0 /
                    (data.max() - data.min()), np.uint8)
    if negate:
        data = 255 - data
    img = pil.fromarray(data)
    img.show()


if __name__ == "__main__":

    # TESTFILE = ("/media/My Passport/HIMAWARI-8/HISD/Hsfd/" +
    #            "201502/07/201502070200/00/B13/" +
    #            "HS_H08_20150207_0200_B13_FLDK_R20_S0101.DAT")
    TESTFILE = ("/local_disk/data/himawari8/testdata/" +
                "HS_H08_20130710_0300_B13_FLDK_R20_S1010.DAT")
    #"HS_H08_20130710_0300_B01_FLDK_R10_S1010.DAT")
    SCENE = ahisf([TESTFILE])
    SCENE.read_band(TESTFILE)
    SCENE.calibrate(['13'])
    # SCENE.calibrate(['13'], calibrate=0)

    # print SCENE._data['13']['counts'][0].shape

    show(SCENE.channels['13'], negate=False)

    import matplotlib.pyplot as plt
    plt.imshow(SCENE.channels['13'])
    plt.colorbar()
    plt.show()
