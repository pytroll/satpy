#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2019 Satpy developers
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
"""Advanced Himawari Imager (AHI) standard format data reader.

References:
    - Himawari-8/9 Himawari Standard Data User's Guide
    - http://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html

Time Information
****************

AHI observations use the idea of a "nominal" time and an "observation" time.
The "nominal" time or repeat cycle is the overall window when the instrument
can record data, usually at a specific and consistent interval. The
"observation" time is when the data was actually observed inside the nominal
window. These two times are stored in a sub-dictionary in the metadata calls
``time_parameters``. Nominal time can be accessed from the
``nominal_start_time`` and ``nominal_end_time`` metadata keys and
observation time from the ``observation_start_time`` and
``observation_end_time`` keys. Observation time can also be accessed from the
parent (``.attrs``) dictionary as the ``start_time`` and ``end_time`` keys.

Satellite Position
******************

As discussed in the :ref:`orbital_parameters` documentation, a satellite
position can be described by a specific "actual" position, a "nominal"
position, a "projection" position, or sometimes a "nadir" position. Not all
readers are able to produce all of these positions. In the case of AHI HSD data
we have an "actual" and "projection" position. For a lot of sensors/readers
though, the "actual" position values do not change between bands or segments
of the same time step (repeat cycle). AHI HSD files contain varying values for
the actual position.

Other components in Satpy use this actual satellite
position to generate other values (ex. sensor zenith angles). If these values
are not consistent between bands then Satpy (dask) will not be able to share
these calculations (generate one sensor zenith angle for band 1, another for
band 2, etc) even though there is rarely a noticeable difference. To deal with
this this reader has an option ``round_actual_position`` that defaults to
``True`` and will round the "actual" position (longitude, latitude, altitude)
in a way to produce as consistent a position between bands as possible.

"""

import logging
import os
import warnings
from datetime import datetime, timedelta

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy._compat import cached_property
from satpy.readers._geos_area import get_area_definition, get_area_extent
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.utils import (
    apply_rad_correction,
    get_earth_radius,
    get_geostationary_mask,
    get_user_calibration_factors,
    np2str,
    unzip_file,
)

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
                           ("nadir_longitude", "f8"),
                           ("nadir_latitude", "f8"),
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

# Navigation correction sub-info
_NAVIGATION_CORRECTION_SUBINFO_TYPE = np.dtype([
    ("line_number_after_rotation", "<u2"),
    ("shift_amount_for_column_direction", "f4"),
    ("shift_amount_for_line_direction", "f4"),
])

# 9 Observation time information block
_OBSERVATION_TIME_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("number_of_observation_times", "<u2"),
])

_OBSERVATION_LINE_TIME_INFO_TYPE = np.dtype([
    ("line_number", "<u2"),
    ("observation_time", "f8"),
])

# 10 Error information block
_ERROR_INFO_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u4"),
    ("number_of_error_info_data", "<u2"),
])

_ERROR_LINE_INFO_TYPE = np.dtype([
    ("line_number", "<u2"),
    ("numof_error_pixels_per_line", "<u2"),
])

# 11 Spare block
_SPARE_TYPE = np.dtype([
    ("hblock_number", "u1"),
    ("blocklength", "<u2"),
    ("spare", "S256")
])


class AHIHSDFileHandler(BaseFileHandler):
    """AHI standard format reader.

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
                            reader_kwargs={'mask_space': False})
        scene.load([0.6])

    The AHI HSD data files contain multiple VIS channel calibration
    coefficients. By default, the updated coefficients in header block 6
    are used. If the user prefers the default calibration coefficients from
    block 5 then they can pass calib_mode='nominal' when creating a scene::

        import satpy
        import glob

        filenames = glob.glob('*FLDK*.dat')
        scene = satpy.Scene(filenames,
                            reader='ahi_hsd',
                            reader_kwargs={'calib_mode': 'update'})
        scene.load([0.6])

    Alternative AHI calibrations are also available, such as GSICS
    coefficients. As such, you can supply custom per-channel correction
    by setting calib_mode='custom' and passing correction factors via::

        user_calibration={'chan': ['slope': slope, 'offset': offset]}

    Where slo and off are per-channel slope and offset coefficients defined by::

        rad_leo = (rad_geo - off) / slo

    If you do not have coefficients for a particular band, then by default the
    slope will be set to 1 .and the offset to 0.::

        import satpy
        import glob

        # Load bands 7, 14 and 15, but we only have coefs for 7+14
        calib_dict = {'B07': {'slope': 0.99, 'offset': 0.002},
                      'B14': {'slope': 1.02, 'offset': -0.18}}

        filenames = glob.glob('*FLDK*.dat')
        scene = satpy.Scene(filenames,
                            reader='ahi_hsd',
                            reader_kwargs={'user_calibration': calib_dict)
        # B15 will not have custom radiance correction applied.
        scene.load(['B07', 'B14', 'B15'])

    By default, user-supplied calibrations / corrections are applied to the
    radiance data in accordance with the GSICS standard defined in the
    equation above. However, user-supplied gain and offset values for
    converting digital number into radiance via Rad = DN * gain + offset are
    also possible. To supply your own factors, supply a user calibration dict
    using `type: 'DN'` as follows::

        calib_dict = {'B07': {'slope': 0.0037, 'offset': 18.5},
                      'B14': {'slope': -0.002, 'offset': 22.8},
                      'type': 'DN'}

    You can also explicitly select radiance correction with `'type': 'RAD'`
    but this is not necessary as it is the default option if you supply your
    own correction coefficients.

    """

    def __init__(self, filename, filename_info, filetype_info,
                 mask_space=True, calib_mode='update',
                 user_calibration=None, round_actual_position=True):
        """Initialize the reader."""
        super(AHIHSDFileHandler, self).__init__(filename, filename_info,
                                                filetype_info)

        self.is_zipped = False
        self._unzipped = unzip_file(self.filename, prefix=str(filename_info['segment']).zfill(2))
        # Assume file is not zipped
        if self._unzipped:
            # But if it is, set the filename to point to unzipped temp file
            self.is_zipped = True
            self.filename = self._unzipped

        self.channels = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self.units = dict([(i, 'counts') for i in AHI_CHANNEL_NAMES])

        self._data = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self._header = dict([(i, None) for i in AHI_CHANNEL_NAMES])
        self.lons = None
        self.lats = None
        self.segment_number = filename_info['segment']
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
        self.band_name = filetype_info['file_type'][4:].upper()
        calib_mode_choices = ('NOMINAL', 'UPDATE')
        if calib_mode.upper() not in calib_mode_choices:
            raise ValueError('Invalid calibration mode: {}. Choose one of {}'.format(
                calib_mode, calib_mode_choices))

        self.calib_mode = calib_mode.upper()
        self.user_calibration = user_calibration
        self._round_actual_position = round_actual_position

    def __del__(self):
        """Delete the object."""
        if self.is_zipped and os.path.exists(self.filename):
            os.remove(self.filename)

    @property
    def start_time(self):
        """Get the nominal start time."""
        return self.nominal_start_time

    @property
    def end_time(self):
        """Get the nominal end time."""
        return self.nominal_start_time

    @property
    def observation_start_time(self):
        """Get the observation start time."""
        return datetime(1858, 11, 17) + timedelta(days=float(self.basic_info['observation_start_time']))

    @property
    def observation_end_time(self):
        """Get the observation end time."""
        return datetime(1858, 11, 17) + timedelta(days=float(self.basic_info['observation_end_time']))

    @property
    def nominal_start_time(self):
        """Time this band was nominally to be recorded."""
        return self._modify_observation_time_for_nominal(self.observation_start_time)

    @property
    def nominal_end_time(self):
        """Get the nominal end time."""
        return self._modify_observation_time_for_nominal(self.observation_end_time)

    @staticmethod
    def _is_valid_timeline(timeline):
        """Check that the `observation_timeline` value is not a fill value."""
        if int(timeline[:2]) > 23:
            return False
        return True

    def _modify_observation_time_for_nominal(self, observation_time):
        """Round observation time to a nominal time based on known observation frequency.

        AHI observations are split into different sectors including Full Disk
        (FLDK), Japan (JP) sectors, and smaller regional (R) sectors. Each
        sector is observed at different frequencies (ex. every 10 minutes,
        every 2.5 minutes, and every 30 seconds). This method will take the
        actual observation time and round it to the nearest interval for this
        sector. So if the observation time is 13:32:48 for the "JP02" sector
        which is the second Japan observation where every Japan observation is
        2.5 minutes apart, then the result should be 13:32:30.

        """
        timeline = "{:04d}".format(self.basic_info['observation_timeline'][0])
        if not self._is_valid_timeline(timeline):
            warnings.warn("Observation timeline is fill value, not rounding observation time.")
            return observation_time

        if self.observation_area == 'FLDK':
            dt = 0
        else:
            observation_frequency_seconds = {'JP': 150, 'R3': 150, 'R4': 30, 'R5': 30}[self.observation_area[:2]]
            dt = observation_frequency_seconds * (int(self.observation_area[2:]) - 1)

        return observation_time.replace(
            hour=int(timeline[:2]), minute=int(timeline[2:4]) + dt//60,
            second=dt % 60, microsecond=0)

    def get_dataset(self, key, info):
        """Get the dataset."""
        return self.read_band(key, info)

    @cached_property
    def area(self):
        """Get AreaDefinition representing this file's data."""
        return self._get_area_def()

    def get_area_def(self, dsid):
        """Get the area definition."""
        del dsid
        return self.area

    def _get_area_def(self):
        pdict = {}
        pdict['cfac'] = np.uint32(self.proj_info['CFAC'])
        pdict['lfac'] = np.uint32(self.proj_info['LFAC'])
        pdict['coff'] = np.float32(self.proj_info['COFF'])
        pdict['loff'] = -np.float32(self.proj_info['LOFF']) + 1
        pdict['a'] = float(self.proj_info['earth_equatorial_radius'] * 1000)
        pdict['h'] = float(self.proj_info['distance_from_earth_center'] * 1000 - pdict['a'])
        pdict['b'] = float(self.proj_info['earth_polar_radius'] * 1000)
        pdict['ssp_lon'] = float(self.proj_info['sub_lon'])
        pdict['nlines'] = int(self.data_info['number_of_lines'])
        pdict['ncols'] = int(self.data_info['number_of_columns'])
        pdict['scandir'] = 'N2S'

        pdict['loff'] = pdict['loff'] + (self.segment_number * pdict['nlines'])

        aex = get_area_extent(pdict)

        pdict['a_name'] = self.observation_area
        pdict['a_desc'] = "AHI {} area".format(self.observation_area)
        pdict['p_id'] = f'geosh{self.basic_info["satellite"][0].decode()[-1]}'

        return get_area_definition(pdict, aex)

    def _check_fpos(self, fp_, fpos, offset, block):
        """Check file position matches blocksize."""
        if fp_.tell() + offset != fpos:
            warnings.warn(f"Actual {block} header size does not match expected")
        return

    def _read_header(self, fp_):
        """Read header."""
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
        corrections = []
        for _i in range(ncorrs):
            corrections.append(np.fromfile(fp_, dtype=_NAVIGATION_CORRECTION_SUBINFO_TYPE, count=1))
        fpos = fpos + int(header['block8']['blocklength'])
        self._check_fpos(fp_, fpos, 40, 'block8')
        fp_.seek(fpos, 0)
        header['navigation_corrections'] = corrections
        header["block9"] = np.fromfile(fp_,
                                       dtype=_OBSERVATION_TIME_INFO_TYPE,
                                       count=1)
        numobstimes = header["block9"]['number_of_observation_times'][0]

        lines_and_times = []
        for _i in range(numobstimes):
            lines_and_times.append(np.fromfile(fp_,
                                               dtype=_OBSERVATION_LINE_TIME_INFO_TYPE,
                                               count=1))
        header['observation_time_information'] = lines_and_times
        fpos = fpos + int(header['block9']['blocklength'])
        self._check_fpos(fp_, fpos, 40, 'block9')
        fp_.seek(fpos, 0)

        header["block10"] = np.fromfile(fp_,
                                        dtype=_ERROR_INFO_TYPE,
                                        count=1)
        num_err_info_data = header["block10"][
            'number_of_error_info_data'][0]
        err_info_data = []
        for _i in range(num_err_info_data):
            err_info_data.append(np.fromfile(fp_, dtype=_ERROR_LINE_INFO_TYPE, count=1))
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
        """Read data block."""
        nlines = int(header["block2"]['number_of_lines'][0])
        ncols = int(header["block2"]['number_of_columns'][0])
        return da.from_array(np.memmap(self.filename, offset=fp_.tell(),
                                       dtype='<u2', shape=(nlines, ncols), mode='r'),
                             chunks=CHUNK_SIZE)

    def _mask_invalid(self, data, header):
        """Mask invalid data."""
        invalid = da.logical_or(data == header['block5']["count_value_outside_scan_pixels"][0],
                                data == header['block5']["count_value_error_pixels"][0])
        return da.where(invalid, np.float32(np.nan), data)

    def _mask_space(self, data):
        """Mask space pixels."""
        return data.where(get_geostationary_mask(self.area, chunks=data.chunks))

    def read_band(self, key, ds_info):
        """Read the data."""
        with open(self.filename, "rb") as fp_:
            self._header = self._read_header(fp_)
            res = self._read_data(fp_, self._header)
        res = self._mask_invalid(data=res, header=self._header)
        res = self.calibrate(res, key['calibration'])

        new_info = self._get_metadata(key, ds_info)
        res = xr.DataArray(res, attrs=new_info, dims=['y', 'x'])
        if self.mask_space:
            res = self._mask_space(res)
        return res

    def _get_metadata(self, key, ds_info):
        # Get actual satellite position. For altitude use the ellipsoid radius at the SSP.
        actual_lon = float(self.nav_info['SSP_longitude'])
        actual_lat = float(self.nav_info['SSP_latitude'])
        re = get_earth_radius(lon=actual_lon, lat=actual_lat,
                              a=float(self.proj_info['earth_equatorial_radius'] * 1000),
                              b=float(self.proj_info['earth_polar_radius'] * 1000))
        actual_alt = float(self.nav_info['distance_earth_center_to_satellite']) * 1000 - re

        if self._round_actual_position:
            actual_lon = round(actual_lon, 3)
            actual_lat = round(actual_lat, 2)
            actual_alt = round(actual_alt / 150) * 150  # to the nearest 150m

        # Update metadata
        new_info = dict(
            units=ds_info['units'],
            standard_name=ds_info['standard_name'],
            wavelength=ds_info['wavelength'],
            resolution='resolution',
            id=key,
            name=key['name'],
            platform_name=self.platform_name,
            sensor=self.sensor,
            time_parameters=dict(
                nominal_start_time=self.nominal_start_time,
                nominal_end_time=self.nominal_end_time,
                observation_start_time=self.observation_start_time,
                observation_end_time=self.observation_end_time,
            ),
            orbital_parameters={
                'projection_longitude': float(self.proj_info['sub_lon']),
                'projection_latitude': 0.,
                'projection_altitude': float(self.proj_info['distance_from_earth_center'] -
                                             self.proj_info['earth_equatorial_radius']) * 1000,
                'satellite_actual_longitude': actual_lon,
                'satellite_actual_latitude': actual_lat,
                'satellite_actual_altitude': actual_alt,
                'nadir_longitude': float(self.nav_info['nadir_longitude']),
                'nadir_latitude': float(self.nav_info['nadir_latitude']),
            },
        )
        return new_info

    def calibrate(self, data, calibration):
        """Calibrate the data."""
        if calibration == 'counts':
            return data

        if calibration in ['radiance', 'reflectance', 'brightness_temperature']:
            data = self.convert_to_radiance(data)
        if calibration == 'reflectance':
            data = self._vis_calibrate(data)
        elif calibration == 'brightness_temperature':
            data = self._ir_calibrate(data)
        return data

    def convert_to_radiance(self, data):
        """Calibrate to radiance."""
        bnum = self._header["block5"]['band_number'][0]
        # Check calibration mode and select corresponding coefficients
        if self.calib_mode == "UPDATE" and bnum < 7:
            dn_gain = self._header['calibration']["cali_gain_count2rad_conversion"][0]
            dn_offset = self._header['calibration']["cali_offset_count2rad_conversion"][0]
            if dn_gain == 0 and dn_offset == 0:
                logger.info(
                    "No valid updated coefficients, fall back to default values.")
                dn_gain = self._header["block5"]["gain_count2rad_conversion"][0]
                dn_offset = self._header["block5"]["offset_count2rad_conversion"][0]
        else:
            dn_gain = self._header["block5"]["gain_count2rad_conversion"][0]
            dn_offset = self._header["block5"]["offset_count2rad_conversion"][0]

        # Assume no user correction
        correction_type = self._get_user_calibration_correction_type()
        if correction_type == 'DN':
            # Replace file calibration with user calibration
            dn_gain, dn_offset = get_user_calibration_factors(self.band_name,
                                                              self.user_calibration)

        data = (data * dn_gain + dn_offset)
        # If using radiance correction factors from GSICS or similar, apply here
        if correction_type == 'RAD':
            user_slope, user_offset = get_user_calibration_factors(self.band_name,
                                                                   self.user_calibration)
            data = apply_rad_correction(data, user_slope, user_offset)
        return data

    def _get_user_calibration_correction_type(self):
        correction_type = None
        if isinstance(self.user_calibration, dict):
            # Check if we have DN correction coeffs
            correction_type = self.user_calibration.get('type', 'RAD')
        return correction_type

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
