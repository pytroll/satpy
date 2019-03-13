#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Pytroll developers

# Author(s):

#   Trygve Aspenes <trygveas@met.no>

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
"""Module for testing the satpy.readers.safe_rsae_l2_ocn module.
"""
import os
import sys
import numpy as np
import xarray as xr

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    import unittest.mock as mock
except ImportError:
    import mock


class TestSAFENC(unittest.TestCase):
    """Test various SAFE SAR L2 OCN file handlers."""

    @mock.patch('xarray.open_dataset')
    def test_init(self, mocked_dataset):
        """Test basic init with no extra parameters."""
        from satpy.readers.safe_sar_l2_ocn import SAFENC
        from satpy import DatasetID

        ds_id = DatasetID(name='foo')
        filename_info = {'mission_id': 'S3A', 'product_type': 'foo',
                         'start_time': 0, 'end_time': 0,
                         'fstart_time': 0, 'fend_time': 0,
                         'polarization': 'vv'}

        test = SAFENC('S1A_IW_OCN__2SDV_20190228T075834_20190228T075849_026127_02EA43_8846.SAFE/measurement/'
                      's1a-iw-ocn-vv-20190228t075741-20190228t075800-026127-02EA43-001.nc', filename_info, 'c')
        mocked_dataset.assert_called()


def suite():
    """The test suite for test_safe_sar_l2_ocn."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSAFENC))
    return mysuite


if __name__ == '__main__':
    unittest.main()
