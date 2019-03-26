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
from satpy.readers.seviri_base import dec10216, chebyshev

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestDec10216(unittest.TestCase):
    """Test the dec10216 function."""

    def test_dec10216(self):
        res = dec10216(np.array([255, 255, 255, 255, 255], dtype=np.uint8))
        exp = (np.ones((4, )) * 1023).astype(np.uint16)
        self.assertTrue(np.all(res == exp))
        res = dec10216(np.array([1, 1, 1, 1, 1], dtype=np.uint8))
        exp = np.array([4,  16,  64, 257], dtype=np.uint16)
        self.assertTrue(np.all(res == exp))


class TestChebyshev(unittest.TestCase):
    def chebyshev4(self, c, x, domain):
        """Evaluate 4th order Chebyshev polynomial"""
        start_x, end_x = domain
        t = (x - 0.5 * (end_x + start_x)) / (0.5 * (end_x - start_x))
        return c[0] + c[1]*t + c[2]*(2*t**2 - 1) + c[3]*(4*t**3 - 3*t) - 0.5*c[0]

    def test_chebyshev(self):
        coefs = [1, 2, 3, 4]
        time = 123
        domain = [120, 130]
        res = chebyshev(coefs=[1, 2, 3, 4], time=time, domain=domain)
        exp = self.chebyshev4(coefs, time, domain)
        self.assertTrue(np.allclose(res, exp))


def suite():
    """The test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    tests = [TestDec10216, TestChebyshev]
    for test in tests:
        mysuite.addTest(loader.loadTestsFromTestCase(test))
    return mysuite
