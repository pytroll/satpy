#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Geostationary project utility module tests package.
"""

import sys

from satpy.readers.geos_area import (get_xy_from_linecol,
                                     get_area_extent,
                                     get_area_definition)


if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestGEOSProjectionUtil(unittest.TestCase):
    """Tests for the area utilities"""

    # Create dict with test data
    pdict = {'a': 6378169.00,
             'b': 6356583.80,
             'h': 35785831.00,
             'ssp_lon': 0.0,
             'nlines': 3712,
             'ncols': 3712,
             'a_name': 'geostest',
             'a_desc': 'test area',
             'p_id': 'test_area',
             'cfac': -13642337,
             'lfac': -13642337,
             'coff': 1856}

    good_ext_s = (5567248.28340708,
                  5567248.28340708,
                  -5570248.686685662,
                  -5570248.686685662)

    good_ext = (5567248.28340708,
                5570248.686685662,
                -5570248.686685662,
                -5567248.28340708)

    def test_geos_area(self):
        """ Test area extent calculation with N->S scan then S->N scan"""
        # North -> South
        self.pdict['scandir'] = 1
        self.pdict['loff'] = 1856
        aex = get_area_extent(self.pdict)
        self.assertEqual(aex, self.good_ext_s)

        # South -> North
        self.pdict['scandir'] = -1
        self.pdict['loff'] = -1856
        aex = get_area_extent(self.pdict)
        self.assertEqual(aex, self.good_ext)

    def test_get_xy_from_linecol(self):
        """ Test the scan angle calculation """
        good_xy = [0.2690166648133674, -10.837528496767087]
        factors = (self.pdict['lfac'], self.pdict['cfac'])
        offsets = (self.pdict['loff'], self.pdict['coff'])
        x, y = get_xy_from_linecol(400, 1800, offsets, factors)
        self.assertEqual(x, good_xy[0])
        self.assertEqual(y, good_xy[1])

    def test_get_area_definition(self):
        """ Test the retrieval of the area definition """
        good_res = (-3000.4032785810186, -3000.4032785810186)

        a_def = get_area_definition(self.pdict, self.good_ext)
        self.assertEqual(a_def.area_id, self.pdict['a_name'])
        self.assertEqual(a_def.resolution, good_res)
        self.assertEqual(a_def.proj_dict['proj'], 'geos')
        self.assertEqual(a_def.proj_dict['units'], 'm')
        self.assertEqual(a_def.proj_dict['a'], 6378169)
        self.assertEqual(a_def.proj_dict['b'], 6356583.8)
        self.assertEqual(a_def.proj_dict['h'], 35785831)


def suite():
    """The test suite for test_geos_area."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGEOSProjectionUtil))
    return mysuite


if __name__ == "__main__":
    # So you can run tests from this module individually.
    unittest.main()
