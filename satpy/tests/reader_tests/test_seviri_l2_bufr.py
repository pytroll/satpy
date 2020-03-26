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
import unittest
from unittest import mock
import numpy as np
from datetime import datetime

FILETYPE_INFO = {'file_type':  'seviri_l2_bufr_csr'}

FILENAME_INFO = {'start_time': '20191112000000',
                 'spacecraft': 'MSG4'}
FILENAME_INFO2 = {'start_time': '20191112000000',
                  'spacecraft': 'MSG4',
                  'server': 'TESTSERVER'}
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
    'platform_name': 'MET08',
    'ssp_lon': 41.5,
    'seg_size': 16
}


class TestSeviriL2Bufr(unittest.TestCase):
    """Test NativeMSGBufrHandler."""

    @unittest.skipIf(sys.platform.startswith('win'), "'eccodes' not supported on Windows")
    def seviri_l2_bufr_test(self, filename):
        """Test the SEVIRI BUFR handler."""
        from satpy.readers.seviri_l2_bufr import SeviriL2BufrFileHandler
        import eccodes as ec
        buf1 = ec.codes_bufr_new_from_samples('BUFR4_local_satellite')
        ec.codes_set(buf1, 'unpack', 1)
        samp1 = np.random.uniform(low=250, high=350, size=(128,))
        # write the bufr test data twice as we want to read in and the concatenate the data in the reader
        # 55 id corresponds to METEOSAT 8
        ec.codes_set(buf1, 'satelliteIdentifier', 55)
        ec.codes_set_array(buf1, '#1#brightnessTemperature', samp1)
        ec.codes_set_array(buf1, '#1#brightnessTemperature', samp1)

        m = mock.mock_open()
        # only our offline product contain MPEF product headers so we get the metadata from there
        if ('BUFRProd' in filename):
            with mock.patch('satpy.readers.seviri_l2_bufr.np.fromfile') as fromfile:
                fromfile.return_value = MPEF_PRODUCT_HEADER
                with mock.patch('satpy.readers.seviri_l2_bufr.recarray2dict') as recarray2dict:
                    recarray2dict.side_effect = (lambda x: x)
                    fh = SeviriL2BufrFileHandler(filename, FILENAME_INFO2, FILETYPE_INFO)
                    fh.mpef_header = MPEF_PRODUCT_HEADER

        else:
            # No Mpef Header  so we get the metadata from the BUFR messages
            with mock.patch('satpy.readers.seviri_l2_bufr.open', m, create=True):
                with mock.patch('eccodes.codes_bufr_new_from_file',
                                side_effect=[buf1, None, buf1, None, buf1, None]) as ec1:
                    ec1.return_value = ec1.side_effect
                    with mock.patch('eccodes.codes_set') as ec2:
                        ec2.return_value = 1
                        with mock.patch('eccodes.codes_release') as ec5:
                            ec5.return_value = 1
                            fh = SeviriL2BufrFileHandler(filename, FILENAME_INFO, FILETYPE_INFO)

        with mock.patch('satpy.readers.seviri_l2_bufr.open', m, create=True):
            with mock.patch('eccodes.codes_bufr_new_from_file',
                            side_effect=[buf1, buf1, None]) as ec1:
                ec1.return_value = ec1.side_effect
                with mock.patch('eccodes.codes_set') as ec2:
                    ec2.return_value = 1
                    with mock.patch('eccodes.codes_release') as ec5:
                        ec5.return_value = 1
                        z = fh.get_dataset(None, DATASET_INFO)
                        # concatenate the original test arrays as
                        # get dataset will have read and concatented the data
                        x1 = np.concatenate((samp1, samp1), axis=0)
                        np.testing.assert_array_equal(z.values, x1)
                        self.assertEqual(z.attrs['platform_name'],
                                         DATASET_ATTRS['platform_name'])
                        self.assertEqual(z.attrs['ssp_lon'],
                                         DATASET_ATTRS['ssp_lon'])
                        self.assertEqual(z.attrs['seg_size'],
                                         DATASET_ATTRS['seg_size'])

    def test_seviri_l2_bufr(self):
        """Call the test function."""
        self.seviri_l2_bufr_test('GIIBUFRProduct_20191106130000Z_00_OMPEFS04_MET11_FES_E0000')
        self.seviri_l2_bufr_test('MSG4-SEVI-MSGGIIN-0101-0101-20191106130000.000000000Z-20191106131702-1362128.bfr')
        self.seviri_l2_bufr_test('MSG4-SEVI-MSGGIIN-0101-0101-20191106101500.000000000Z-20191106103218-1362148')
