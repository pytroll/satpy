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
"""Geostationary project utility module tests package."""

import unittest
from satpy.readers._geos_area import (get_xy_from_linecol,
                                      get_area_extent,
                                      get_area_definition)

import numpy as np


class TestGEOSProjectionUtil(unittest.TestCase):
    """Tests for the area utilities.

    TODO: Add some simple tests in addition to real world tests.

    """

    def make_pdict_ext(self, centered, scandir, area):
        """Create a dictionary and extents to use in testing."""
        if not centered:
            # SEVIRI like: Integer loff/coff and even number of lines -> Image datum is half
            # a pixel off the image center. That is why loff/coff are different for N2S and S2N
            # scanning directions.
            if area == 'full_disk':
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
                         'scandir': scandir}
                if scandir == 'N2S':
                    pdict['loff'] = 1857
                    pdict['coff'] = 1857
                    extent = (-5570248.686685662,
                              -5567248.28340708,
                              5567248.28340708,
                              5570248.686685662)
                elif scandir == 'S2N':
                    pdict['scandir'] = 'S2N'
                    pdict['loff'] = 1856
                    pdict['coff'] = 1856
                    extent = (5567248.28340708,
                              5570248.686685662,
                              -5570248.686685662,
                              -5567248.28340708)

            elif area == 'segment':
                pdict = {'a': 6378169.00,
                         'b': 6356583.80,
                         'h': 35785831.00,
                         'ssp_lon': 0.0,
                         'nlines': 464,
                         'ncols': 3712,
                         'a_name': 'geostest',
                         'a_desc': 'test area',
                         'p_id': 'test_area',
                         'cfac': -13642337,
                         'lfac': -13642337}
                if scandir == 'S2N':
                    pdict['scandir'] = 'S2N'
                    pdict['loff'] = 928  # 3rd segment from the top
                    pdict['coff'] = 1856
                    extent = (5567248.28340708, -1390686.9196223018,
                              -5570248.686685662, -2782874.0408838945)
                if scandir == 'N2S':
                    pdict['scandir'] = 'N2S'
                    pdict['loff'] = -463  # 3rd segment from the bottom
                    pdict['coff'] = 1857
                    extent = (-5570248.686685662, -2782874.0408838945,
                              5567248.28340708, -1390686.9196223018)
        else:
            # TODO
            raise NotImplementedError

        return pdict, extent

    def test_get_area_extent(self):
        """Test area extent calculation."""
        cases = [
            {'centered': False, 'scandir': 'N2S', 'area': 'full_disk'},
            {'centered': False, 'scandir': 'S2N', 'area': 'full_disk'},
            {'centered': False, 'scandir': 'N2S', 'area': 'segment'},
            {'centered': False, 'scandir': 'S2N', 'area': 'segment'},
        ]
        for case in cases:
            pdict, extent = self.make_pdict_ext(**case)
            aex = get_area_extent(pdict)
            np.testing.assert_allclose(aex, extent,
                                       err_msg='Incorrect area extent in case {}'.format(case))

    def test_get_xy_from_linecol(self):
        """Test the scan angle calculation."""
        good_xy = [0.2690166648133674, -10.837528496767087]
        factors = (-13642337, -13642337)
        offsets = (-1856, 1856)
        x, y = get_xy_from_linecol(400, 1800, offsets, factors)
        np.testing.assert_approx_equal(x, good_xy[0])
        np.testing.assert_approx_equal(y, good_xy[1])

        good_xy = [0.2690166648133674, 0.30744761692956274]
        factors = (-13642337, -13642337)
        offsets = (464, 1856)
        x, y = get_xy_from_linecol(400, 1800, offsets, factors)
        np.testing.assert_approx_equal(x, good_xy[0])
        np.testing.assert_approx_equal(y, good_xy[1])

    def test_get_area_definition(self):
        """Test the retrieval of the area definition."""
        from pyresample.utils import proj4_radius_parameters
        pdict, extent = self.make_pdict_ext(centered=False, scandir='N2S', area='full_disk')
        good_res = (3000.4032785810186, 3000.4032785810186)

        a_def = get_area_definition(pdict, extent)
        self.assertEqual(a_def.area_id, pdict['a_name'])
        self.assertEqual(a_def.resolution, good_res)
        self.assertEqual(a_def.proj_dict['proj'], 'geos')
        self.assertEqual(a_def.proj_dict['units'], 'm')
        a, b = proj4_radius_parameters(a_def.proj_dict)
        self.assertEqual(a, 6378169)
        self.assertEqual(b, 6356583.8)
        self.assertEqual(a_def.proj_dict['h'], 35785831)
