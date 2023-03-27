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
import os
import unittest
import uuid

import dask.array as da
import numpy as np
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.viirs_vgac_l1c_nc import VGACFileHandler


TEST_FILE = 'test_VGAC_VJ102MOD_A2018305_1042_n004946_K005'

class TestVGACFileHandler(unittest.TestCase):
    """Test the VGACFileHandler reader."""

    def setUp(self):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        # uses a UUID to avoid permission conflicts during execution of tests in parallel
        self.test_file_name = TEST_FILE + str(uuid.uuid1()) + ".nc"

        with Dataset(self.test_file_name, 'w') as nc:
            nscn = 7
            npix = 800
            n_lut = 12000
            # Add dimensions to data group)
            nc.createDimension('npix', npix)
            nc.createDimension('nscn', nscn)
            nc.createDimension('n_lut', n_lut)

            # Add variables to data/calibration_data group
            for ind in range(1, 11, 1):
                ch_name  = "M{:02d}".format(ind)
                r_a = nc.createVariable(ch_name, np.int16, dimensions=('nscn','npix'))
                r_a[:] = np.ones((nscn, npix)) * 10
                attrs = {'scale_factor': 0.1, 'add_offset': 0., 'units': 'percent'}
                for attr in attrs:
                    setattr(r_a, attr, attrs[attr])

            for ind in range(12, 17, 1):
                ch_name  = "M{:02d}".format(ind)
                tb_b = nc.createVariable(ch_name, np.int16, dimensions=('nscn','npix'))
                tb_b[:] = np.ones((nscn, npix)) * 800
                attrs = {'add_offset': 0., 'units': 'radiances', 'scale_factor': 10., }
                for attr in attrs:
                    setattr(tb_b, attr, attrs[attr])
                tb_lut = nc.createVariable(ch_name + "_LUT", np.float32, dimensions=('n_lut'))
                tb_lut[:] = np.array(range(0, n_lut)) * 0.5


        self.reader = VGACFileHandler(
            filename=self.test_file_name,
            filename_info={
                'start_time': datetime.datetime(year=2017, month=9, day=20,
                                                hour=12, minute=30, second=30),
            },
            filetype_info={}
        )

    def tearDown(self):
        """Remove the previously created test file."""
        # Catch Windows PermissionError for removing the created test file.
        try:
            os.remove(self.test_file_name)
        except OSError:
            pass

    def test_reading_vis(self):
        """Test reading reflectances."""
        M05 = self.reader.get_dataset("M05", yaml_info={'name': 'M05',
                                                  'units': 'percent'})
        assert (M05[0,0] == 100)


    def test_reading_ir(self):
        """Test reading BTs."""
        M15 = self.reader.get_dataset("M15", yaml_info={'name': 'M15',
                                                        'nc_key': 'M15',
                                                        'scale_factor_nc': 10.,
                                                        'units': 'radiances'})
        assert (M15[0,0] == 400)


