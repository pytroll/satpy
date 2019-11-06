#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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

"""Unittesting the SEVIRI L2 BUFR reader."""

import sys

import numpy as np
from datetime import datetime

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

FILETYPE_INFO = {'file_type':  'seviri_l2_bufr_csr'}

MPEF_PRODUCT_HEADER = {
    'NominalTime': datetime(2019, 11, 6, 18, 0),
    'SpacecraftName': '08',
    'RectificationLongitude': 'E0415'
}

DATASET_INFO = {
    'key': '#1#brightnessTemperature',
    'fill_value': 0
}

DATASET_ATTRS = {
    'spacecraft_name': 'MET08',
    'ssp_lon': 41.5,
    'seg_size': 16
}


class TestSeviriL2Bufr(unittest.TestCase):
    """Test NativeMSGBufrHandler."""

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def seviri_l2_bufr_test(self,):
        """Test the SEVIRI BUFR handler."""
        from satpy.readers.seviri_l2_bufr import SeviriL2BufrFileHandler
        import eccodes as ec
        buf1 = ec.codes_bufr_new_from_samples('BUFR4_local_satellite')
        ec.codes_set(buf1, 'unpac'
                           'k', 1)
        samp1 = np.random.uniform(low=250, high=350, size=(128,))
        samp2 = np.random.uniform(low=-60, high=60, size=(128,))
        samp3 = np.random.uniform(low=10, high=60, size=(128,))
        # write the bufr test data twice as we want to read in and the concatenate the data in the reader
        ec.codes_set_array(buf1, '#1#brightnessTemperature', samp1)
        ec.codes_set_array(buf1, '#1#brightnessTemperature', samp1)
        ec.codes_set_array(buf1, 'latitude', samp2)
        ec.codes_set_array(buf1, 'latitude', samp2)
        ec.codes_set_array(buf1, 'longitude', samp3)
        ec.codes_set_array(buf1, 'longitude', samp3)

        m = mock.mock_open()
        with mock.patch('satpy.readers.seviri_l2_bufr.np.fromfile') as fromfile:
            fromfile.return_value = MPEF_PRODUCT_HEADER
            with mock.patch('satpy.readers.seviri_l2_bufr.recarray2dict') as recarray2dict:
                recarray2dict.side_effect = (lambda x: x)
                fh = SeviriL2BufrFileHandler(None, {}, FILETYPE_INFO)
                fh.mpef_header = MPEF_PRODUCT_HEADER
                with mock.patch('satpy.readers.seviri_l2_bufr.open', m, create=True):
                    with mock.patch('eccodes.codes_bufr_new_from_file',
                                    side_effect=[buf1, buf1, None, buf1, buf1, None, buf1, buf1, None]) as ec1:
                        ec1.return_value = ec1.side_effect
                        with mock.patch('eccodes.codes_set') as ec2:
                            ec2.return_value = 1
                            with mock.patch('eccodes.codes_release') as ec5:
                                ec5.return_value = 1
                                z = fh.get_dataset(None, DATASET_INFO)
                                # concatenate the original test arrays as
                                # get dataset will have read and concatented the data
                                x1 = np.concatenate((samp1, samp1), axis=0)
                                x2 = np.concatenate((samp2, samp2), axis=0)
                                x3 = np.concatenate((samp3, samp3), axis=0)
                                np.testing.assert_array_equal(z.values, x1)
                                np.testing.assert_array_equal(z.coords['latitude'].values, x2)
                                np.testing.assert_array_equal(z.coords['longitude'].values, x3)
                                self.assertEqual(z.attrs['spacecraft_name'],
                                                 DATASET_ATTRS['spacecraft_name'])
                                self.assertEqual(z.attrs['ssp_lon'],
                                                 DATASET_ATTRS['ssp_lon'])
                                self.assertEqual(z.attrs['seg_size'],
                                                 DATASET_ATTRS['seg_size'])

    def test_seviri_l2_bufr(self):
        """Call the test function."""
        self.seviri_l2_bufr_test()


def suite():
    """Test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSeviriL2Bufr))
    return mysuite


if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
