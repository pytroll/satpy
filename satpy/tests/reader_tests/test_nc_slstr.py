#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Module for testing the satpy.readers.nc_slstr module.
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


class TestSLSTRReader(unittest.TestCase):
    """Test various nc_slstr file handlers."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.slstr_l1b import NCSLSTR1B, NCSLSTRGeo, NCSLSTRAngles, NCSLSTRFlag
        from satpy import DatasetID

        ds_id = DatasetID(name='foo')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0}

        test = NCSLSTR1B('somedir/S1_radiance_an.nc', filename_info, 'c')
        assert(test.view == 'n')
        assert(test.stripe == 'a')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCSLSTR1B('somedir/S1_radiance_co.nc', filename_info, 'c')
        assert(test.view == 'o')
        assert(test.stripe == 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCSLSTRGeo('somedir/S1_radiance_an.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCSLSTRAngles('somedir/S1_radiance_an.nc', filename_info, 'c')
        # TODO: Make this test work
        # test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCSLSTRFlag('somedir/S1_radiance_an.nc', filename_info, 'c')
        assert(test.view == 'n')
        assert(test.stripe == 'a')
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()


def suite():
    """The test suite for test_nc_slstr."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSLSTRReader))
    return mysuite
