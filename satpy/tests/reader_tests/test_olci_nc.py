#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Satpy developers
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
"""Module for testing the satpy.readers.olci_nc module.
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


class TestOLCIReader(unittest.TestCase):
    """Test various olci_nc filehandlers."""

    @mock.patch('xarray.open_dataset')
    def test_instantiate(self, mocked_dataset):
        """Test initialization of file handlers."""
        from satpy.readers.olci_nc import (NCOLCIBase, NCOLCICal, NCOLCIGeo,
                                           NCOLCIChannelBase, NCOLCI1B, NCOLCI2)
        from satpy import DatasetID

        ds_id = DatasetID(name='foo')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0}

        test = NCOLCIBase('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCICal('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCIGeo('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCIChannelBase('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCI1B('somedir/somefile.nc', filename_info, 'c', mock.Mock())
        test.get_dataset(ds_id, filename_info)
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        test = NCOLCI2('somedir/somefile.nc', filename_info, 'c')
        test.get_dataset(ds_id, {'nc_key': 'the_key'})
        mocked_dataset.assert_called()
        mocked_dataset.reset_mock()

        # ds_id = DatasetID(name='solar_azimuth_angle')
        # test = NCOLCIAngles('somedir/somefile.nc', filename_info, 'c')
        # test.get_dataset(ds_id, filename_info)
        # mocked_dataset.assert_called()
        # mocked_dataset.reset_mock()


class TestBitFlags(unittest.TestCase):
    """Test the bitflag reading."""

    def test_bitflags(self):
        """Test the BitFlags class."""
        import numpy as np
        from six.moves import reduce
        from satpy.readers.olci_nc import BitFlags
        flag_list = ['INVALID', 'WATER', 'LAND', 'CLOUD', 'SNOW_ICE',
                     'INLAND_WATER', 'TIDAL', 'COSMETIC', 'SUSPECT', 'HISOLZEN',
                     'SATURATED', 'MEGLINT', 'HIGHGLINT', 'WHITECAPS',
                     'ADJAC', 'WV_FAIL', 'PAR_FAIL', 'AC_FAIL', 'OC4ME_FAIL',
                     'OCNN_FAIL', 'Extra_1', 'KDM_FAIL', 'Extra_2',
                     'CLOUD_AMBIGUOUS', 'CLOUD_MARGIN', 'BPAC_ON',
                     'WHITE_SCATT', 'LOWRW', 'HIGHRW']

        bits = np.array([1 << x for x in range(len(flag_list))])

        bflags = BitFlags(bits)

        items = ["INVALID", "SNOW_ICE", "INLAND_WATER", "SUSPECT",
                 "AC_FAIL", "CLOUD", "HISOLZEN", "OCNN_FAIL",
                 "CLOUD_MARGIN", "CLOUD_AMBIGUOUS", "LOWRW", "LAND"]

        mask = reduce(np.logical_or, [bflags[item] for item in items])
        expected = np.array([True, False,  True,  True,  True,  True, False,
                             False,  True, True, False, False, False, False,
                             False, False, False,  True, False,  True, False,
                             False, False,  True,  True, False, False, True,
                             False])
        self.assertTrue(all(mask == expected))


def suite():
    """The test suite for test_nc_slstr."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestBitFlags))
    mysuite.addTest(loader.loadTestsFromTestCase(TestOLCIReader))
    return mysuite


if __name__ == '__main__':
    unittest.main()
