#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""Test the MSG common (native and hrit format) functionionalities
"""

import sys
import numpy as np
from satpy.readers.seviri_base import dec10216, chebyshev, get_cds_time

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def chebyshev4(c, x, domain):
    """Evaluate 4th order Chebyshev polynomial"""
    start_x, end_x = domain
    t = (x - 0.5 * (end_x + start_x)) / (0.5 * (end_x - start_x))
    return c[0] + c[1]*t + c[2]*(2*t**2 - 1) + c[3]*(4*t**3 - 3*t) - 0.5*c[0]


class SeviriBaseTest(unittest.TestCase):
    def test_dec10216(self):
        """Test the dec10216 function."""
        res = dec10216(np.array([255, 255, 255, 255, 255], dtype=np.uint8))
        exp = (np.ones((4, )) * 1023).astype(np.uint16)
        self.assertTrue(np.all(res == exp))
        res = dec10216(np.array([1, 1, 1, 1, 1], dtype=np.uint8))
        exp = np.array([4,  16,  64, 257], dtype=np.uint16)
        self.assertTrue(np.all(res == exp))

    def test_chebyshev(self):
        coefs = [1, 2, 3, 4]
        time = 123
        domain = [120, 130]
        res = chebyshev(coefs=[1, 2, 3, 4], time=time, domain=domain)
        exp = chebyshev4(coefs, time, domain)
        self.assertTrue(np.allclose(res, exp))

    def test_get_cds_time(self):
        # Scalar
        self.assertEqual(get_cds_time(days=21246, msecs=12*3600*1000),
                         np.datetime64('2016-03-03 12:00'))

        # Array
        days = np.array([21246, 21247, 21248])
        msecs = np.array([12*3600*1000, 13*3600*1000 + 1, 14*3600*1000 + 2])
        expected = np.array([np.datetime64('2016-03-03 12:00:00.000'),
                             np.datetime64('2016-03-04 13:00:00.001'),
                             np.datetime64('2016-03-05 14:00:00.002')])
        self.assertTrue(np.all(get_cds_time(days=days, msecs=msecs) == expected))


def suite():
    """The test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(SeviriBaseTest))
    return mysuite
