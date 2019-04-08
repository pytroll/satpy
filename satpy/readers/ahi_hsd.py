#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2014-2019 PyTroll developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Advanced Himawari Imager (AHI) standard format data reader.

References:
    - Himawari-8/9 Himawari Standard Data User's Guide
    - http://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html

Time Information
****************

AHI observations use the idea of a "scheduled" time and an "observation time.
The "scheduled" time is when the instrument was told to record the data,
usually at a specific and consistent interval. The "observation" time is when
the data was actually observed. Scheduled time can be accessed from the
`scheduled_time` metadata key and observation time from the `start_time` key.

"""

import logging
from datetime import datetime, timedelta

import numpy as np
import dask.array as da
import xarray as xr
import warnings

from pyresample import geometry
from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import get_geostationary_mask, np2str

AHI_CHANNEL_NAMES = ("1", "2", "3", "4", "5",
                     "6", "7", "8", "9", "10",
                     "11", "12", "13", "14", "15", "16")

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
                              ("coeff_update_time", "f8"),
                              ("cali_gain_count2rad_conversion", "f8"),
                              ("cali_offset_count2rad_conversion", "f8"),
                              ("spare", "S80"),
                              ])

# 6 Inter-calibration information block
_INTER_CALIBRATION_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("gsics_calibration_intercept", "f8"),
    ("gsics_calibration_slope", "f8"),
    ("gsics_calibration_coeff_quadratic_term", "f8"),
    ("gsics_std_scn_radiance_bias", "f8"),
    ("gsics_std_scn_radiance_bias_uncertainty", "f8"),
    ("gsics_std_scn_radiance", "f8"),
    ("gsics_correction_starttime", "f8"),
    ("gsics_correction_endtime", "f8"),
    ("gsics_radiance_validity_upper_lim", "f4"),
    ("gsics_radiance_validity_lower_lim", "f4"),
    ("gsics_filename", "S128"),
    ("spare", "S56"),
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
    ("blocklength", "<u4"),
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

    The AHI sensor produces data for some pixels outside the Earth disk (i,e:
    atmospheric limb or deep space pixels).
    By default, these pixels are masked out as they contain data of limited
    or no value, but some applications do require these pixels.
    It is therefore possible to override the default behaviour and perform no
    masking of non-Earth pixels.

    In order to change the default behaviour, use the 'mask_space' variable
    as part of ``reader_kwargs`` upon Scene creation::

        import satpy
        import glob

        filenames = glob.glob('*FLDK*.dat')
        scene = satpy.Scene(filenames,
                            reader='ahi_hsd',
                            reader_kwargs={'mask_space':: False})
        scene.load([0.6])

    The AHI HSD data files contain multiple VIS channel calibration
    coefficients. By default, the standard coefficients in header block 5
    are used. If the user prefers the updated calibration coefficients then
    they can pass calib_mode='update' when creating a scene::

        import satpy
        import glob

        filenames = glob.glob('*FLDK*.dat')
        scene = satpy.Scene(filenames,
                            reader='ahi_hsd',
                            reader_kwargs={'calib_mode':: 'update'})
        scene.load([0.6])

    By default these updated coefficients are not used.

    """

    def __init__(self, filename, filename_info, filetype_info,
                 mask_space=True, calib_mode='nominal'):
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
        self.platform_name = np2str(self.basic_info['satellite'])
        self.observation_area = np2str(self.basic_info['observation_area'])
        self.sensor = 'ahi'
        self.mask_space = mask_space

        calib_mode_choices = ('NOMINAL', 'UPDATE')
        if calib_mode.upper() not in calib_mode_choices:
            raise ValueError('Invalid calibration mode: {}. Choose one of {}'.format(
                calib_mode, calib_mode_choices))
        self.calib_mode = calib_mode.upper()

    @property
    def start_time(self):
        return datetime(1858, 11, 17) + timedelta(days=float(self.basic_info['observation_start_time']))

    @property
    def end_time(self):
        return datetime(1858, 11, 17) + timedelta(days=float(self.basic_info['observation_end_time']))

    @property
    def scheduled_time(self):
        """Time this band was scheduled to be recorded."""
        timeline = "{:04d}".format(self.basic_info['observation_timeline'][0])
        return self.start_time.replace(hour=int(timeline[:2]), minute=int(timeline[2:4]), second=0, microsecond=0)

    def get_dataset(self, key, info):
        return self.read_band(key, info)

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
        left_x = (cols - coff) * (2.**16 / cfac)
        cols += ncols
        right_x = (cols - coff) * (2.**16 / cfac)

        lines = (self.segment_number - 1) * nlines + 1 - 0.5
        upper_y = -(lines - loff) * (2.**16 / lfac)
        lines += nlines
        lower_y = -(lines - loff) * (2.**16 / lfac)
        area_extent = (np.deg2rad(left_x) * h, np.deg2rad(lower_y) * h,
                       np.deg2rad(right_x) * h, np.deg2rad(upper_y) * h)

        proj_dict = {'a': float(a),
                     'b': float(b),
                     'lon_0': float(lon_0),
                     'h': float(h),
                     'proj': 'geos',
                     'units': 'm'}

        area = geometry.AreaDefinition(
            self.observation_area,
            "AHI {} area".format(self.observation_area),
            'geosh8',
            proj_dict,
            ncols,
            nlines,
            area_extent)

        self.area = area
        return area

    def _check_fpos(self, fp_, fpos, offset, block):
        """Check file position matches blocksize"""
        if (fp_.tell() + offset != fpos):
            warnings.warn("Actual "+block+" header size does not match expected")
        return

    def _read_header(self, fp_):
        """Read header"""
        header = {}

        fpos = 0
        header['block1'] = np.fromfile(
            fp_, dtype=_BASIC_INFO_TYPE, count=1)
        fpos = fpos + int(header['block1']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block1')
        fp_.seek(fpos, 0)
        header["block2"] = np.fromfile(fp_, dtype=_DATA_INFO_TYPE, count=1)
        fpos = fpos + int(header['block2']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block2')
        fp_.seek(fpos, 0)
        header["block3"] = np.fromfile(fp_, dtype=_PROJ_INFO_TYPE, count=1)
        fpos = fpos + int(header['block3']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block3')
        fp_.seek(fpos, 0)
        header["block4"] = np.fromfile(fp_, dtype=_NAV_INFO_TYPE, count=1)
        fpos = fpos + int(header['block4']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block4')
        fp_.seek(fpos, 0)
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
        fpos = fpos + int(header['block5']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block5')
        fp_.seek(fpos, 0)

        header['calibration'] = cal

        header["block6"] = np.fromfile(
            fp_, dtype=_INTER_CALIBRATION_INFO_TYPE, count=1)
        fpos = fpos + int(header['block6']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block6')
        fp_.seek(fpos, 0)
        header["block7"] = np.fromfile(
            fp_, dtype=_SEGMENT_INFO_TYPE, count=1)
        fpos = fpos + int(header['block7']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block7')
        fp_.seek(fpos, 0)
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
        fpos = fpos + int(header['block8']['blocklength'])
        self._check_fpos(fp_, fpos, 40, 'block8')
        fp_.seek(fpos, 0)
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
        fpos = fpos + int(header['block9']['blocklength'])
        self._check_fpos(fp_, fpos, 40, 'block9')
        fp_.seek(fpos, 0)

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
        fpos = fpos + int(header['block10']['blocklength'])
        self._check_fpos(fp_, fpos, 40, 'block10')
        fp_.seek(fpos, 0)

        header["block11"] = np.fromfile(fp_, dtype=_SPARE_TYPE, count=1)
        fpos = fpos + int(header['block11']['blocklength'])
        self._check_fpos(fp_, fpos, 0, 'block11')
        fp_.seek(fpos, 0)

        return header

    def _read_data(self, fp_, header):
        """Read data block"""
        nlines = int(header["block2"]['number_of_lines'][0])
        ncols = int(header["block2"]['number_of_columns'][0])
        return da.from_array(np.memmap(self.filename, offset=fp_.tell(),
                                       dtype='<u2', shape=(nlines, ncols), mode='r'),
                             chunks=CHUNK_SIZE)

    def _mask_invalid(self, data, header):
        """Mask invalid data"""
        invalid = da.logical_or(data == header['block5']["count_value_outside_scan_pixels"][0],
                                data == header['block5']["count_value_error_pixels"][0])
        return da.where(invalid, np.float32(np.nan), data)

    def _mask_space(self, data):
        """Mask space pixels"""
        return data.where(get_geostationary_mask(self.area))

    def read_band(self, key, info):
        """Read the data."""
        # Read data
        tic = datetime.now()
        with open(self.filename, "rb") as fp_:
            header = self._read_header(fp_)
            res = self._read_data(fp_, header)
        res = self._mask_invalid(data=res, header=header)
        self._header = header
        logger.debug("Reading time " + str(datetime.now() - tic))

        # Calibrate
        res = self.calibrate(res, key.calibration)

        # Update metadata
        new_info = dict(units=info['units'],
                        standard_name=info['standard_name'],
                        wavelength=info['wavelength'],
                        resolution='resolution',
                        id=key,
                        name=key.name,
                        scheduled_time=self.scheduled_time,
                        platform_name=self.platform_name,
                        sensor=self.sensor,
                        satellite_longitude=float(
                            self.nav_info['SSP_longitude']),
                        satellite_latitude=float(
                            self.nav_info['SSP_latitude']),
                        satellite_altitude=float(self.nav_info['distance_earth_center_to_satellite'] -
                                                 self.proj_info['earth_equatorial_radius']) * 1000)
        res = xr.DataArray(res, attrs=new_info, dims=['y', 'x'])

        # Mask space pixels
        if self.mask_space:
            res = self._mask_space(res)

        return res

    def calibrate(self, data, calibration):
        """Calibrate the data"""
        tic = datetime.now()

        if calibration == 'counts':
            return data

        if calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            data = self.convert_to_radiance(data)
        if calibration == 'reflectance':
            data = self._vis_calibrate(data)
        elif calibration == 'brightness_temperature':
            data = self._ir_calibrate(data)

        logger.debug("Calibration time " + str(datetime.now() - tic))
        return data

    def convert_to_radiance(self, data):
        """Calibrate to radiance."""

        bnum = self._header["block5"]['band_number'][0]
        # Check calibration mode and select corresponding coefficients
        if self.calib_mode == "UPDATE" and bnum < 7:
            gain = self._header['calibration']["cali_gain_count2rad_conversion"][0]
            offset = self._header['calibration']["cali_offset_count2rad_conversion"][0]
            if gain == 0 and offset == 0:
                logger.info(
                    "No valid updated coefficients, fall back to default values.")
                gain = self._header["block5"]["gain_count2rad_conversion"][0]
                offset = self._header["block5"]["offset_count2rad_conversion"][0]
        else:
            gain = self._header["block5"]["gain_count2rad_conversion"][0]
            offset = self._header["block5"]["offset_count2rad_conversion"][0]

        return (data * gain + offset).clip(0)

    def _vis_calibrate(self, data):
        """Visible channel calibration only."""
        coeff = self._header["calibration"]["coeff_rad2albedo_conversion"]
        return (data * coeff * 100).clip(0)

    def _ir_calibrate(self, data):
        """IR calibration."""
        # No radiance -> no temperature
        data = da.where(data == 0, np.float32(np.nan), data)

        cwl = self._header['block5']["central_wave_length"][0] * 1e-6
        c__ = self._header['calibration']["speed_of_light"][0]
        h__ = self._header['calibration']["planck_constant"][0]
        k__ = self._header['calibration']["boltzmann_constant"][0]
        a__ = (h__ * c__) / (k__ * cwl)

        b__ = ((2 * h__ * c__ ** 2) / (data * 1.0e6 * cwl ** 5)) + 1

        Te_ = a__ / da.log(b__)

        c0_ = self._header['calibration']["c0_rad2tb_conversion"][0]
        c1_ = self._header['calibration']["c1_rad2tb_conversion"][0]
        c2_ = self._header['calibration']["c2_rad2tb_conversion"][0]

        return (c0_ + c1_ * Te_ + c2_ * Te_ ** 2).clip(0)
