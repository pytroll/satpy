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


import datetime

import numpy as np
import pytest
from netCDF4 import Dataset


@pytest.fixture
def _nc_filename(tmp_path):
    now = datetime.datetime.utcnow()
    filename = f'VGAC_VJ10XMOD_A{now:%Y%j_%H%M}_n004946_K005.nc'
    filename_str = str(tmp_path / filename)
    # Create test data
    with Dataset(filename_str, 'w') as nc:
        nscn = 7
        npix = 800
        n_lut = 12000
        nc.createDimension('npix', npix)
        nc.createDimension('nscn', nscn)
        nc.createDimension('n_lut', n_lut)
        nc.StartTime = "2023-03-28T09:08:07"
        nc.EndTime = "2023-03-28T10:11:12"
        for ind in range(1, 11, 1):
            ch_name = "M{:02d}".format(ind)
            r_a = nc.createVariable(ch_name, np.int16, dimensions=('nscn', 'npix'))
            r_a[:] = np.ones((nscn, npix)) * 10
            attrs = {'scale_factor': 0.1, 'units': 'percent'}
            for attr in attrs:
                setattr(r_a, attr, attrs[attr])
        for ind in range(12, 17, 1):
            ch_name = "M{:02d}".format(ind)
            tb_b = nc.createVariable(ch_name, np.int16, dimensions=('nscn', 'npix'))
            tb_b[:] = np.ones((nscn, npix)) * 800
            attrs = {'units': 'radiances', 'scale_factor': 0.002}
            for attr in attrs:
                setattr(tb_b, attr, attrs[attr])
            tb_lut = nc.createVariable(ch_name + "_LUT", np.float32, dimensions=('n_lut'))
            tb_lut[:] = np.array(range(0, n_lut)) * 0.5
    return filename_str


class TestVGACREader:
    """Test the VGACFileHandler reader."""

    def test_read_vgac(self, _nc_filename):
        """Test reading reflectances and BT."""
        from satpy.scene import Scene

        # Read data
        scn_ = Scene(
            reader='viirs_vgac_l1c_nc',
            filenames=[_nc_filename])
        scn_.load(["M05", "M15"])
        assert (scn_["M05"][0, 0] == 100)
        assert (scn_["M15"][0, 0] == 400)
        assert scn_.start_time == datetime.datetime(year=2023, month=3, day=28,
                                                    hour=9, minute=8, second=7)
        assert scn_.end_time == datetime.datetime(year=2023, month=3, day=28,
                                                  hour=10, minute=11, second=12)
