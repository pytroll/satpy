#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Pytroll developers

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
"""Module for testing the satpy.readers.sar-c_safe module.
"""
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock


class TestSAFEGRD(unittest.TestCase):
    """Test various nc_slstr file handlers."""
    @mock.patch('rasterio.open')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.sar_c_safe import SAFEGRD

        filename_info = {'mission_id': 'S1A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'polarization': 'vv'}
        filetype_info = 'bla'
        noisefh = mock.MagicMock()
        calfh = mock.MagicMock()

        test = SAFEGRD('S1A_IW_GRDH_1SDV_20190201T024655_20190201T024720_025730_02DC2A_AE07.SAFE/measurement/s1a-iw-grd'
                       '-vv-20190201t024655-20190201t024720-025730-02dc2a-001.tiff',
                       filename_info, filetype_info, calfh, noisefh)
        assert(test._polarization == 'vv')
        assert(test.calibration == calfh)
        assert(test.noise == noisefh)
        mocked_dataset.assert_called()


def suite():
    """The test suite for test_sar_c_safe."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSAFEGRD))
    return mysuite


if __name__ == '__main__':
    unittest.main()
