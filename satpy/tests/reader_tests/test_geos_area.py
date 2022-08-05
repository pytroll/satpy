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

import numpy as np

from satpy.readers._geos_area import (
    get_area_definition,
    get_area_extent,
    get_geos_area_naming,
    get_resolution_and_unit_strings,
    get_xy_from_linecol,
    sampling_to_lfac_cfac,
)


class TestGEOSProjectionUtil(unittest.TestCase):
    """Tests for the area utilities."""

    def make_pdict_ext(self, typ, scan):
        """Create a dictionary and extents to use in testing."""
        if typ == 1:  # Fulldisk
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
            if scan == 'N2S':
                pdict['scandir'] = 'N2S'
                pdict['loff'] = 1856
                extent = (5567248.28340708,
                          5567248.28340708,
                          -5570248.686685662,
                          -5570248.686685662)

            if scan == 'S2N':
                pdict['scandir'] = 'S2N'
                pdict['loff'] = -1856
                extent = (5567248.28340708,
                          5570248.686685662,
                          -5570248.686685662,
                          -5567248.28340708)

        if typ == 2:  # One sector
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
                     'lfac': -13642337,
                     'coff': 1856}
            if scan == 'N2S':
                pdict['scandir'] = 'N2S'
                pdict['loff'] = 464
                extent = (5567248.28340708,
                          1390686.9196223018,
                          -5570248.686685662,
                          -1500.2016392905093)

            if scan == 'S2N':
                pdict['scandir'] = 'S2N'
                pdict['loff'] = 464
                extent = (5567248.28340708,
                          -1390686.9196223018,
                          -5570248.686685662,
                          -2782874.0408838945)

        return pdict, extent

    def test_geos_area(self):
        """Test area extent calculation with N->S scan then S->N scan."""
        # North -> South full disk
        pdict, extent = self.make_pdict_ext(1, 'N2S')
        aex = get_area_extent(pdict)
        np.testing.assert_allclose(aex, extent)

        # South -> North full disk
        pdict, extent = self.make_pdict_ext(1, 'S2N')
        aex = get_area_extent(pdict)
        np.testing.assert_allclose(aex, extent)

        # North -> South one sector
        pdict, extent = self.make_pdict_ext(2, 'N2S')
        aex = get_area_extent(pdict)
        np.testing.assert_allclose(aex, extent)

        # South -> North one sector
        pdict, extent = self.make_pdict_ext(2, 'S2N')
        aex = get_area_extent(pdict)
        np.testing.assert_allclose(aex, extent)

    def test_get_xy_from_linecol(self):
        """Test the scan angle calculation."""
        pdict, extent = self.make_pdict_ext(1, 'S2N')
        good_xy = [0.2690166648133674, -10.837528496767087]
        factors = (pdict['lfac'], pdict['cfac'])
        offsets = (pdict['loff'], pdict['coff'])
        x, y = get_xy_from_linecol(400, 1800, offsets, factors)
        np.testing.assert_approx_equal(x, good_xy[0])
        np.testing.assert_approx_equal(y, good_xy[1])

        pdict, extent = self.make_pdict_ext(2, 'N2S')
        good_xy = [0.2690166648133674, 0.30744761692956274]
        factors = (pdict['lfac'], pdict['cfac'])
        offsets = (pdict['loff'], pdict['coff'])
        x, y = get_xy_from_linecol(400, 1800, offsets, factors)
        np.testing.assert_approx_equal(x, good_xy[0])
        np.testing.assert_approx_equal(y, good_xy[1])

    def test_get_area_definition(self):
        """Test the retrieval of the area definition."""
        from pyresample.utils import proj4_radius_parameters
        pdict, extent = self.make_pdict_ext(1, 'N2S')
        good_res = (-3000.4032785810186, -3000.4032785810186)

        a_def = get_area_definition(pdict, extent)
        self.assertEqual(a_def.area_id, pdict['a_name'])
        self.assertEqual(a_def.resolution, good_res)
        self.assertEqual(a_def.proj_dict['proj'], 'geos')
        self.assertEqual(a_def.proj_dict['units'], 'm')
        a, b = proj4_radius_parameters(a_def.proj_dict)
        self.assertEqual(a, 6378169)
        self.assertEqual(b, 6356583.8)
        self.assertEqual(a_def.proj_dict['h'], 35785831)

    def test_sampling_to_lfac_cfac(self):
        """Test conversion from angular sampling to line/column offset."""
        lfac = 13642337  # SEVIRI LFAC
        sampling = np.deg2rad(2 ** 16 / lfac)
        np.testing.assert_allclose(sampling_to_lfac_cfac(sampling), lfac)

    def test_get_geos_area_naming(self):
        """Test the geos area naming function."""
        input_dict = {'platform_name': 'testplatform',
                      'instrument_name': 'testinstrument',
                      'resolution': 1000,
                      'service_name': 'testservicename',
                      'service_desc': 'testdesc'}

        output_dict = get_geos_area_naming(input_dict)

        self.assertEqual(output_dict['area_id'], 'testplatform_testinstrument_testservicename_1km')
        self.assertEqual(output_dict['description'], 'TESTPLATFORM TESTINSTRUMENT testdesc area definition'
                                                     ' with 1 km resolution')

    def test_get_resolution_and_unit_strings_in_km(self):
        """Test the resolution and unit strings function for a km resolution."""
        out = get_resolution_and_unit_strings(1000)
        self.assertEqual(out['value'], '1')
        self.assertEqual(out['unit'], 'km')

    def test_get_resolution_and_unit_strings_in_m(self):
        """Test the resolution and unit strings function for a m resolution."""
        out = get_resolution_and_unit_strings(500)
        self.assertEqual(out['value'], '500')
        self.assertEqual(out['unit'], 'm')
