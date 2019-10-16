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

"""Unittesting the  SEVIRI L2 Bufr reader.
"""

import sys

import numpy as np

from satpy.readers.seviri_l2_bufr import (
    MSGBUFRFileHandler,
)
import eccodes as ec

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestMSGBufr(unittest.TestCase):
    """Test NativeMSGFileHandler.get_area_extent
    The expected results have been verified by manually
    inspecting the output of geoferenced imagery.
    """
    @staticmethod
    def create_test_bufr_message():
        pass

    def msg_bufr_test(self,):
        buf1 = ec.codes_bufr_new_from_samples('BUFR4_local_satellite')
        ec.codes_set(buf1, 'masterTablesVersionNumber', 27)
        ec.codes_set(buf1, 'unpack', 1)
        sampl = np.random.randint(low=250, high=350, size=(128,))
        ec.codes_set_array(buf1, '#1#brightnessTemperature', sampl)
        ec.codes_set(buf1, '#1#orbitNumber', 900)
        # May be useful later if adding data to new keys
        # iterid = ec.codes_bufr_keys_iterator_new(buf1)
        # while ec.codes_bufr_keys_iterator_next(iterid):
        #    print(ec.codes_bufr_keys_iterator_get_name(iterid))

        info = {'satellite': 'meteosat9', 'subsat': 'E0000',
                'start_time': '201909180000',
                'key': '#1#brightnessTemperature', 'units': 'm',
                'wavelength': 10, 'standard_name': 'met9',
                }

        m = mock.mock_open()
        with mock.patch('satpy.readers.seviri_l2_bufr.open', m, create=True):
            with mock.patch('satpy.readers.seviri_l2_bufr.MSGBUFRFileHandler.get_attribute') as attr:
                attr.return_value = 9000
                with mock.patch('satpy.readers.seviri_l2_bufr.MSGBUFRFileHandler.get_array') as arr:
                    samplat = np.random.uniform(low=-60, high=60, size=(128,))
                    arr.return_value = samplat
                    fh = MSGBUFRFileHandler(None, info, None)

            with mock.patch('eccodes.codes_bufr_new_from_file',
                            side_effect=[buf1, buf1, buf1, None]) as ec1:
                ec1.return_value = ec1.side_effect
                with mock.patch('eccodes.codes_set') as ec2:
                    ec2.return_value = 1
                    with mock.patch('eccodes.codes_release') as ec5:
                        ec5.return_value = 1
                        x = fh.get_array('#1#orbitNumber', 1)
                        self.assertEqual(x, 900)
                        y = fh.get_attribute('masterTablesVersionNumber', 1)
                        self.assertEqual(y, 27)
                        z = fh.get_dataset(None, info)
                        self.assertTrue(abs(np.nanmean(z) - np.nanmean(sampl) < 1.5))

    def setUp(self):
        pass

    def test_msg_bufr(self):
        self.msg_bufr_test()

    def tearDown(self):
        pass


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMSGBufr))
    return mysuite


if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
