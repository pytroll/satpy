# -*- coding: utf-8 -*-

# Copyright (c) 2017 Martin Raspaud

# Author(s):

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
"""The abi_l1b reader tests package.
"""

import os
import sys

import mock
import numpy as np

from satpy.readers.abi_l1b import NC_ABI_L1B

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class Test_NC_ABI_L1B(unittest.TestCase):
    """Test the NC_ABI_L1B reader."""
    @mock.patch('satpy.readers.abi_l1b.h5netcdf')
    def setUp(self, h5netcdf):
        h5netcdf.File.return_value = {
            'Rad': np.arange(10.).reshape((2, 5)),
            "planck_fk1": np.array(13432.1),
            "planck_fk2": np.array(1497.61),
            "planck_bc1": np.array(0.09102),
            "planck_bc2": np.array(0.99971),
            "esun": np.array(2017),
            "earth_sun_distance_anomaly_in_AU": np.array(0.99)}

        self.reader = NC_ABI_L1B('filename',
                                 {'platform_shortname': 'G16'},
                                 {'filetype': 'info'})

    def test_ir_calibrate(self):
        """Test IR calibration."""
        data = (np.ma.arange(10.).reshape((2, 5)) + 1) * 50

        self.reader._ir_calibrate(data)

        expected = np.ma.array([[267.55572248, 305.15576503, 332.37383249,
                                 354.73895301, 374.19710115],
                                [391.68679226, 407.74064808, 422.69329105,
                                 436.77021913, 450.13141732]])
        self.assertTrue(np.allclose(data, expected))

    def test_vis_calibrate(self):
        """Test VIS calibration."""
        data = (np.ma.arange(10.).reshape((2, 5)) + 1) * 100

        self.reader._vis_calibrate(data)

        expected = np.ma.array([[0.15265617, 0.30531234, 0.45796851,
                                 0.61062468, 0.76328085],
                                [0.91593702, 1.06859319, 1.22124936,
                                 1.37390553, 1.52656171]])
        self.assertTrue(np.allclose(data, expected))


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_NC_ABI_L1B))

    return mysuite
