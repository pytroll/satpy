#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c20671.ad.smhi.se>

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

"""Test the MSG common (native and hrit format) functionionalities
"""

import sys

import numpy as np

from satpy.readers.msg_base import dec10216

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestDec10216(unittest.TestCase):

    """Test the dec10216 function. Currently there is another 'replica' 
    function in hrit_base.py. FIXME!"""

    def test_dec10216(self):
        res = dec10216(np.array([255, 255, 255, 255, 255], dtype=np.uint8))
        exp = (np.ones((4, )) * 1023).astype(np.uint16)
        self.assertTrue(np.all(res == exp))
        res = dec10216(np.array([1, 1, 1, 1, 1], dtype=np.uint8))
        exp = np.array([4,  16,  64, 257], dtype=np.uint16)
        self.assertTrue(np.all(res == exp))


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestDec10216))
    return mysuite

if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
