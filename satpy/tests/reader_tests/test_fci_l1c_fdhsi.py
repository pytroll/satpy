#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
#
# This file is part of the satpy.
#
# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for the 'fci_l1c_fdhsi' reader."""

import xarray as xr
import dask.array as da

from .netcdf_utils import NetCDF4FileHandler

try:
    from unittest import mock # Python 3.3 or newer
except ImportError:
    import mock # Python 2.7

class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    def _get_test_calib_for_channel_ir(self, chroot, meas):
        data = {}
        data[meas + "/radiance_unit_conversion_coefficient"] = 1
        data[chroot + "/central_wavelength_actual"] = 1
        for c in "ab":
            data[meas + "/radiance_to_bt_conversion_coefficient_" + c] = 1
        for c in "12":
            data[meas + "/radiance_to_bt_conversion_constant_c" + c] = 1
        return data

    def _get_test_calib_for_channel_vis(self, chroot, meas):
        data = {}
        data[meas + "/channel_effective_solar_irradiance"] = 42
        return data

    def _get_test_content_for_channel(self, pat, ch):
        nrows = 596
        ncols = 11136
        chroot = "data/{:s}"
        meas = chroot + "/measured"
        rad = meas + "/effective_radiance"
        pos_path = meas + "/{:s}_position_{:s}"
        data = {}
        ch_str = pat.format(ch)
        ch_path = rad.format(ch_str)
        da = xr.DataArray(
                da.ones(nrows, ncols),
                dtype="uint16",
                chunks=1024,
                attrs={
                    "valid_range": [0, 4095],
                    "scale_factor": 1,
                    "add_offset": 0,
                    }
                )
        data[ch_path] = da
        for (st, no) in (("start", 0), ("end", 100)):
            for rc in ("row", "column"):
                pos_path = pos.format(meas, st, rc)
                data[pos_path] = xr.DataArray(no)
        if "ir" in pat:
            data.extend(self._get_test_calib_for_channel_ir(self))
        elif "vis" in pat:
            data.extend(self._get_test_calib_for_channel_vis(self))
        return data

    def _get_test_content_all_channels(self):
        chan_patterns = {
                "vis_{:>02d}": (4, 5, 6, 8, 9),
                "nir_{:>02d}": (13, 16, 22),
                "ir_{:>02d}": (38, 87, 97, 105, 123, 133),
                "wv_{:>02d}": (63, 73),
                }
        data = {}
        for pat in chan_patterns.keys():
            for ch_num in chan_patterns[pat].values():
                data.extend(self._get_test_content_for_channel(ch_num))
        return data

    def _get_test_content_areadef(self):
        data = {}
        proc = "state/processor"
        for (lb, no) in (
                ("earth_equatorial_radius", 6378137),
                ("earth_polar_radius", 6356752),
                ("reference_altitude", 35786000),
                ("projection_origin_longitude", 0)):
            data[proc + "/" + lb] = xr.DataArray(no)
        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        # mock global attributes
        # - root groups global
        # - other groups global
        # mock data variables
        # mock dimensions
        #
        # ... but only what satpy is using ...

        return {
                **self._get_test_content_channels(),
                **self._get_test_content_areadef(),
                }

