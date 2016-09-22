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
"""

AHI_CHANNEL_NAMES = ("1", "2", "3", "4", "5",
                     "6", "7", "8", "9", "10",
                     "11", "12", "13", "14", "15", "16")
import logging
from datetime import datetime

import numpy as np

from satpy.readers.file_handlers import BaseFileHandler


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
                           ("SSP_latitude", "f8"),
                           ("SSP_longitude", "f8"),
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
        super(VIIRSCompactFileHandler, self).__init__(filename, filename_info,
                                                      filetype_info)

        self.channels = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self.units = dict([(i, 'counts') for i in AHI_CHANNEL_NAMES])

        self._data = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self._header = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self.lons = None
        self.lats = None

    def read_band(self):
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

            nlines = header["block2"]['number_of_lines']
            ncols = header["block2"]['number_of_columns']
            dtype = np.dtype([('counts', '<u2', (nlines, ncols))])
            data = np.fromfile(fp_, dtype=dtype, count=1)

            self._header[str(band_number)] = header
            self._data[str(band_number)] = data

        logger.debug("Reading time " + str(datetime.now() - tic))

    def calibrate(self, chns=AHI_CHANNEL_NAMES, calibrate=1):
        """Calibrate the data"""
        tic = datetime.now()

        for chan in chns:

            if chan in ['1', '2', '3', '4', '5', '6']:
                self.channels[chan] = _vis_calibrate(self._header[chan],
                                                     self._data[chan],
                                                     calibrate)
                if calibrate == 0:
                    self.units[chan] = ''
                elif calibrate == 2:
                    self.units[chan] = 'W*m-2*sr-1*um-1'
                else:
                    self.units[chan] = '%'

            else:
                self.channels[chan] = _ir_calibrate(self._header[chan],
                                                    self._data[chan],
                                                    calibrate)
                if calibrate == 0:
                    self.units[chan] = ''
                elif calibrate == 2:
                    self.units[chan] = 'W*m-2*sr-1*um-1'
                else:
                    self.units[chan] = 'K'

        logger.debug("Calibration time " + str(datetime.now() - tic))


def _vis_calibrate(header, data, calib_type):
    """Visible channel calibration only.
    *calib_type* = 0: Counts
    *calib_type* = 1: Reflectances
    *calib_type* = 2: Radiances
    """
    # Calibration count to radiance or albedo.

    channel = data["counts"][0, :, :].astype(np.float)
    channel[channel == header['block5'][
        "count_value_error_pixels"][0]] = np.nan
    channel[channel == header['block5'][
        "count_value_outside_scan_pixels"][0]] = np.nan

    band_number = header['block5']['band_number'][0]
    gain = header["block5"]["gain_count2rad_conversion"][0]
    offset = header["block5"]["offset_count2rad_conversion"][0]
    coeff = header["calibration"]["coeff_rad2albedo_conversion"]
    if calib_type == 2:
        if band_number >= 7:
            raise CalibrationError("Band is not a SW Visible band and " +
                                   "cannot be converted to albedo!")
        channel = channel * gain + offset
    elif calib_type == 1:
        channel = (channel * gain + offset) * coeff

    channel[channel < 0] = np.nan
    return np.ma.masked_array(channel, np.isnan(channel))


def _ir_calibrate(header, data, calib_type):
    """IR calibration
    *calib_type* = 0: Counts
    *calib_type* = 1: BT
    *calib_type* = 2: Radiances
    """

    band_number = header['block5']['band_number'][0]
    if band_number < 7:
        raise CalibrationError("Band is not a AHI IR band and " +
                               "cannot be converted to brightness " +
                               "temperatures")

    channel = data["counts"][0, :, :].astype(np.float)
    channel[channel == header['block5'][
        "count_value_error_pixels"][0]] = np.nan
    channel[channel == header['block5'][
        "count_value_outside_scan_pixels"][0]] = np.nan

    gain = header["block5"]["gain_count2rad_conversion"][0]
    offset = header["block5"]["offset_count2rad_conversion"][0]

    if calib_type == 2 or calib_type == 1:
        channel = channel * gain + offset

    if calib_type == 1:
        # First calculate the effective brightness temperature using the
        # radiance, the central wavelength and the planck function:

        cwl = header['block5']["central_wave_length"][0] * 1e-6
        c__ = header['calibration']["speed_of_light"][0]
        h__ = header['calibration']["planck_constant"][0]
        k__ = header['calibration']["boltzmann_constant"][0]
        a__ = (h__ * c__) / (k__ * cwl)
        b__ = ((2 * h__ * c__ ** 2) / (1.0e6 * cwl ** 5 * channel)) + 1
        Te_ = a__ / np.log(b__)

        c0_ = header['calibration']["c0_rad2tb_conversion"][0]
        c1_ = header['calibration']["c1_rad2tb_conversion"][0]
        c2_ = header['calibration']["c2_rad2tb_conversion"][0]

        channel = c0_ + c1_ * Te_ + c2_ * Te_ ** 2

    channel[channel < 0] = np.nan
    return np.ma.masked_array(channel, np.isnan(channel))


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
    #SCENE.calibrate(['13'], calibrate=0)

    # print SCENE._data['13']['counts'][0].shape

    show(SCENE.channels['13'], negate=False)

    import matplotlib.pyplot as plt
    plt.imshow(SCENE.channels['13'])
    plt.colorbar()
    plt.show()
