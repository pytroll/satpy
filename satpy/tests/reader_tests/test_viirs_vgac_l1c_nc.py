#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023- Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""The viirs_vgac_l1b_nc reader tests package.

This version tests the readers for VIIIRS VGAC data preliminary version.

"""


import datetime as dt

import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset


@pytest.fixture
def nc_filename(tmp_path):
    """Create an nc test data file and return its filename."""
    now = dt.datetime.utcnow()
    filename = f"VGAC_VJ10XMOD_A{now:%Y%j_%H%M}_n004946_K005.nc"
    filename_str = str(tmp_path / filename)
    # Create test data
    with Dataset(filename_str, "w") as nc:
        nscn = 7
        npix = 800
        n_lut = 12000
        start_time_srting = "2023-03-28T09:08:07"
        end_time_string = "2023-03-28T10:11:12"
        nc.createDimension("npix", npix)
        nc.createDimension("nscn", nscn)
        nc.createDimension("n_lut", n_lut)
        nc.createDimension("one", 1)
        nc.StartTime = start_time_srting
        nc.EndTime = end_time_string
        for ind in range(1, 11, 1):
            ch_name = "M{:02d}".format(ind)
            r_a = nc.createVariable(ch_name, np.int16, dimensions=("nscn", "npix"))
            r_a[:] = np.ones((nscn, npix)) * 10
            attrs = {"scale_factor": 0.1, "units": "percent"}
            for attr in attrs:
                setattr(r_a, attr, attrs[attr])
        for ind in range(12, 17, 1):
            ch_name = "M{:02d}".format(ind)
            tb_b = nc.createVariable(ch_name, np.int16, dimensions=("nscn", "npix"))
            tb_b[:] = np.ones((nscn, npix)) * 800
            attrs = {"units": "radiances", "scale_factor": 0.002}
            for attr in attrs:
                setattr(tb_b, attr, attrs[attr])
            tb_lut = nc.createVariable(ch_name + "_LUT", np.float32, dimensions=("n_lut"))
            tb_lut[:] = np.array(range(0, n_lut)) * 0.5
            tb_lut.units = "Kelvin"
        reference_time = np.datetime64("2010-01-01T00:00:00")
        start_time = np.datetime64("2023-03-28T09:08:07") + np.timedelta64(123000, "us")
        delta_days = start_time - reference_time
        delta_full_days = delta_days.astype("timedelta64[D]")
        hidden_reference_time = reference_time + delta_full_days
        delta_part_of_days = start_time - hidden_reference_time
        proj_time0 = nc.createVariable("proj_time0", np.float64)
        proj_time0[:] = (delta_full_days.astype(np.int64) +
                         0.000001 * delta_part_of_days.astype("timedelta64[us]").astype(np.int64) / (60 * 60 * 24))
        proj_time0.units = "days since 01/01/2010T00:00:00"
        time_v = nc.createVariable("time", np.float64, ("nscn",))
        delta_h = np.datetime64(end_time_string) - start_time
        delta_hours = 0.000001 * delta_h.astype("timedelta64[us]").astype(np.int64) / (60 * 60)
        time_v[:] = np.linspace(0, delta_hours, num=nscn).astype(np.float64)
        time_v.units = "hours since proj_time0"

    return filename_str


class TestVGACREader:
    """Test the VGACFileHandler reader."""

    def test_read_vgac(self, nc_filename):
        """Test reading reflectances and BT."""
        from satpy.scene import Scene

        # Read data
        scn_ = Scene(
            reader="viirs_vgac_l1c_nc",
            filenames=[nc_filename])
        scn_.load(["M05", "M15", "scanline_timestamps"])
        diff_s = (scn_["scanline_timestamps"][0].values.astype("datetime64[us]") -
                  np.datetime64("2023-03-28T09:08:07.123000").astype("datetime64[us]"))
        diff_e = (np.datetime64("2023-03-28T10:11:12.000000").astype("datetime64[us]") -
                  scn_["scanline_timestamps"][-1].values.astype("datetime64[us]"))
        assert (diff_s < np.timedelta64(5, "us"))
        assert (diff_s > np.timedelta64(-5, "us"))
        assert (diff_e < np.timedelta64(5, "us"))
        assert (diff_e > np.timedelta64(-5, "us"))
        assert (scn_["M05"][0, 0] == 100)
        assert (scn_["M15"][0, 0] == 400)
        assert scn_.start_time == dt.datetime(year=2023, month=3, day=28,
                                              hour=9, minute=8, second=7)
        assert scn_.end_time == dt.datetime(year=2023, month=3, day=28,
                                            hour=10, minute=11, second=12)

    def test_decode_time_variable(self):
        """Test decode time variable branch."""
        from satpy.readers.viirs_vgac_l1c_nc import VGACFileHandler
        fh = VGACFileHandler(filename="",
                             filename_info={"start_time": "2023-03-28T09:08:07"},
                             filetype_info="")
        data = xr.DataArray(
            [[1, 2],
             [3, 4]],
            dims=("y", "x"),
            attrs={"units": "something not expected"})
        with pytest.raises(AttributeError):
            fh.decode_time_variable(data, "time", None)
