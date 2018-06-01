#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017-2018 PyTroll Community

# Author(s):

#   Adam.Dybbroe <adam.dybbroe@smhi.se>
#   Sauli Joro <sauli.joro@eumetsat.int>

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

"""Unittesting the native msg reader
"""

import sys

import numpy as np
from satpy.readers.native_msg import get_available_channels

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


CHANNEL_INDEX_LIST = ['VIS006', 'VIS008', 'IR_016', 'IR_039',
                      'WV_062', 'WV_073', 'IR_087', 'IR_097',
                      'IR_108', 'IR_120', 'IR_134', 'HRV']
AVAILABLE_CHANNELS = {}
for item in CHANNEL_INDEX_LIST:
    AVAILABLE_CHANNELS[item] = True


SEC15HDR = '15_SECONDARY_PRODUCT_HEADER'
IDS = 'SelectedBandIDs'

TEST1_HEADER_CHNLIST = {SEC15HDR: {IDS: {}}}
TEST1_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XX--XX--XX--'

TEST2_HEADER_CHNLIST = {SEC15HDR: {IDS: {}}}
TEST2_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XX-XXXX----X'

TEST3_HEADER_CHNLIST = {SEC15HDR: {IDS: {}}}
TEST3_HEADER_CHNLIST[SEC15HDR][IDS]['Value'] = 'XXXXXXXXXXXX'


# This should preferably be put in a helper-module
# Fixme!
def assertNumpyArraysEqual(self, other):
    if self.shape != other.shape:
        raise AssertionError("Shapes don't match")
    if not np.allclose(self, other):
        raise AssertionError("Elements don't match!")


class TestNativeMSGFileHandler(unittest.TestCase):

    """Test the NativeMSGFileHandler."""

    def setUp(self):
        pass

    def test_get_available_channels(self):
        """Test the derivation of the available channel list"""

        available_chs = get_available_channels(TEST1_HEADER_CHNLIST)
        trues = ['WV_062', 'WV_073', 'IR_108', 'VIS006', 'VIS008', 'IR_120']
        for bandname in AVAILABLE_CHANNELS.keys():
            if bandname in trues:
                self.assertTrue(available_chs[bandname])
            else:
                self.assertFalse(available_chs[bandname])

        available_chs = get_available_channels(TEST2_HEADER_CHNLIST)
        trues = ['VIS006', 'VIS008', 'IR_039', 'WV_062', 'WV_073', 'IR_087', 'HRV']
        for bandname in AVAILABLE_CHANNELS.keys():
            if bandname in trues:
                self.assertTrue(available_chs[bandname])
            else:
                self.assertFalse(available_chs[bandname])

        available_chs = get_available_channels(TEST3_HEADER_CHNLIST)
        for bandname in AVAILABLE_CHANNELS.keys():
            self.assertTrue(available_chs[bandname])

    def tearDown(self):
        pass


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNativeMSGFileHandler))
    return mysuite


if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
