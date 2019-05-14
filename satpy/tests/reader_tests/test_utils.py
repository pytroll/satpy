#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi>
#   Martin Raspaud <martin.raspaud@smhi.se>
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
"""Testing of helper functions."""


import unittest

try:
    from unittest import mock
except ImportError:
    import mock

import numpy as np
import pyresample.geometry

from satpy.readers import utils as hf


class TestSatinHelpers(unittest.TestCase):
    """Class for testing satpy.satin."""

    def test_boundaries_to_extent(self):
        """Test conversion of area boundaries to area extent."""
        from satpy.satin.helper_functions import boundaries_to_extent

        # MSG3 proj4 string from
        #  xrit.sat.load(..., only_metadata=True).proj4_params
        proj4_str = 'proj=geos lon_0=0.00 lat_0=0.00 ' \
            'a=6378169.00 b=6356583.80 h=35785831.00'

        # MSG3 maximum extent
        msg_extent = [-5567248.07, -5570248.48, 5570248.48, 5567248.07]

        euro4_lons = [np.array([-47.45398384, -43.46278935,
                                -38.35946515, -31.73014962,
                                -23.05306111, 11.8361092,
                                1.9545262, 17.28655348,
                                32.17162432, 44.92350518,
                                55.01855232, 56.988557157486078]),
                      np.array([56.98855716, 50.26011569,
                                45.1592762, 41.21696892,
                                38.10602167, 35.60224391,
                                33.55098034, 31.8438098,
                                30.40324844, 29.17282762,
                                28.11061579, 27.886603224354555]),
                      np.array([27.88660322, 23.94855341,
                                19.91336672, 15.81854029,
                                11.70507781, 7.61511006,
                                3.58934937, -0.33524747,
                                -4.1272886, -7.76204144,
                                -11.2217833, -11.991484302295099]),
                      np.array([-11.9914843, -13.71190987,
                                -15.65433484, -17.8592324,
                                -20.37559742, -23.26235124,
                                -26.5893562, -30.43725577,
                                -34.8946782, -40.05040055,
                                -45.97725877, -47.453983842896925])
                      ]

        euro4_lats = [np.array([60.95152407, 64.07948755,
                                67.08804237, 69.89447062,
                                72.37400834, 74.34558786,
                                75.57997723, 75.8713547,
                                75.16167548, 73.58553666,
                                71.37260506, 70.797059167821104]),
                      np.array([70.79705917, 67.92687675,
                                64.85946318, 61.67911498,
                                58.44076323, 55.18141964,
                                51.92695755, 48.69607712,
                                45.50265971, 42.35720453,
                                39.26773508, 38.565754283815295]),
                      np.array([38.56575428, 39.21556029,
                                39.65166546, 39.86532337,
                                39.85213881, 39.61238514,
                                39.15098428, 38.47715262,
                                37.60377021, 36.54656798,
                                35.32324138, 35.020342638475668]),
                      np.array([35.02034264, 37.76813725,
                                40.533077, 43.300949,
                                46.05396441, 48.76986157,
                                51.42078481, 53.97194327,
                                56.38014919, 58.59254174,
                                60.54617556, 60.95152407157881])
                      ]

        # Correct extent values for these boundaries
        correct_values_euro4 = [-2041009.079233268, 3502723.3881863873,
                                2211266.5660426724, 5387911.4915445326]

        maximum_extent_euro4 = boundaries_to_extent(proj4_str,
                                                    None,
                                                    msg_extent,
                                                    euro4_lons, euro4_lats)

        for i in range(4):
            self.assertAlmostEqual(maximum_extent_euro4[i],
                                   correct_values_euro4[i], 2)

        # Two of the area corner points is outside the satellite view

        afgh_lons = [np.array([49.94506701, 52.14080597,
                               54.33654493, 56.53228389,
                               58.72802285, 60.92376181,
                               63.11950077, 65.31523973,
                               67.51097869, 69.70671766,
                               71.90245662, 74.09819558,
                               76.29393454, 78.4896735,
                               80.68541246, 82.88115142]),
                     np.array([85.05493299, 85.05493299,
                               85.05493299, 85.05493299,
                               85.05493299, 85.05493299,
                               85.05493299, 85.05493299,
                               85.05493299, 85.05493299,
                               85.05493299, 85.05493299,
                               85.05493299, 85.05493299,
                               85.05493299, 85.05493299]),
                     np.array([85.05493299, 82.85919403,
                               80.66345507, 78.46771611,
                               76.27197715, 74.07623819,
                               71.88049923, 69.68476027,
                               67.48902131, 65.29328234,
                               63.09754338, 60.90180442,
                               58.70606546, 56.5103265,
                               54.31458754, 52.11884858]),
                     np.array([49.94506701, 49.94506701,
                               49.94506701, 49.94506701,
                               49.94506701, 49.94506701,
                               49.94506701, 49.94506701,
                               49.94506701, 49.94506701,
                               49.94506701, 49.94506701,
                               49.94506701, 49.94506701,
                               49.94506701, 49.94506701])]

        afgh_lats = [np.array([46.52610743, 46.52610743,
                               46.52610743, 46.52610743,
                               46.52610743, 46.52610743,
                               46.52610743, 46.52610743,
                               46.52610743, 46.52610743,
                               46.52610743, 46.52610743,
                               46.52610743, 46.52610743,
                               46.52610743, 46.52610743]),
                     np.array([46.52610743, 44.99436458,
                               43.42055852, 41.804754,
                               40.14714935, 38.4480861,
                               36.70805834, 34.92772129,
                               33.10789917, 31.24959192,
                               29.35398073, 27.42243208,
                               25.45649997, 23.4579264,
                               21.4286396, 19.37075017]),
                     np.array([17.30750918, 17.30750918,
                               17.30750918, 17.30750918,
                               17.30750918, 17.30750918,
                               17.30750918, 17.30750918,
                               17.30750918, 17.30750918,
                               17.30750918, 17.30750918,
                               17.30750918, 17.30750918,
                               17.30750918, 17.30750918]),
                     np.array([17.30750918, 19.39146328,
                               21.44907771, 23.47806753,
                               25.47632393, 27.44192051,
                               29.37311717, 31.26836176,
                               33.12628971, 34.94572163,
                               36.72565938, 38.46528046,
                               40.16393131, 41.82111941,
                               43.43650469, 45.00989022])
                     ]

        # Correct values for these borders
        correct_values_afgh = [3053894.9120028536, 1620176.1036167517,
                               5187086.4642274799, 4155907.3124084808]

        maximum_extent_afgh = boundaries_to_extent(proj4_str,
                                                   None,
                                                   msg_extent,
                                                   afgh_lons, afgh_lats)

        for i in range(len(maximum_extent_afgh)):
            self.assertAlmostEqual(maximum_extent_afgh[i],
                                   correct_values_afgh[i], 2)

        # Correct values for combined boundaries
        correct_values_comb = [-2041009.079233268, 1620176.1036167517,
                               5187086.4642274799, 5387911.4915445326]

        maximum_extent_comb = boundaries_to_extent(proj4_str,
                                                   maximum_extent_euro4,
                                                   msg_extent,
                                                   afgh_lons, afgh_lats)
        for i in range(4):
            self.assertAlmostEqual(maximum_extent_comb[i],
                                   correct_values_comb[i], 2)

        # Borders where none of the corners are within the satellite view
        lons = [np.array([-170., 170., -170., 170])]
        lats = [np.array([89., 89., -89., -89])]

        # Correct values are the same as the full disc extent
        correct_values = [-5567248.07, -5570248.48, 5570248.48, 5567248.07]

        maximum_extent_full = boundaries_to_extent(proj4_str,
                                                   None,
                                                   msg_extent,
                                                   lons, lats)
        for i in range(4):
            self.assertAlmostEqual(maximum_extent_full[i],
                                   correct_values[i], 2)


class TestHelpers(unittest.TestCase):
    """Test the area helpers."""

    def test_lonlat_from_geos(self):
        """Get lonlats from geos."""
        geos_area = mock.MagicMock()
        lon_0 = 0
        h = 35785831.00
        geos_area.proj_dict = {'a': 6378169.00,
                               'b': 6356583.80,
                               'h': h,
                               'lon_0': lon_0}

        expected = np.array((lon_0, 0))

        import pyproj
        proj = pyproj.Proj(proj='geos', **geos_area.proj_dict)

        expected = proj(0, 0, inverse=True)

        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(0, 0, geos_area))

        expected = proj(0, 1000000, inverse=True)

        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(0, 1000000 / h,
                                                              geos_area))

        expected = proj(1000000, 0, inverse=True)

        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(1000000 / h, 0,
                                                              geos_area))

        expected = proj(2000000, -2000000, inverse=True)

        np.testing.assert_allclose(expected,
                                   hf._lonlat_from_geos_angle(2000000 / h,
                                                              -2000000 / h,
                                                              geos_area))

    def test_get_geostationary_bbox(self):
        """Get the geostationary bbox."""

        geos_area = mock.MagicMock()
        lon_0 = 0
        geos_area.proj_dict = {'a': 6378169.00,
                               'b': 6356583.80,
                               'h': 35785831.00,
                               'lon_0': lon_0}
        geos_area.area_extent = [-5500000., -5500000., 5500000., 5500000.]

        lon, lat = hf.get_geostationary_bounding_box(geos_area, 20)
        elon = np.array([-74.802824, -73.667708, -69.879687, -60.758081,
                         -32.224989, 32.224989, 60.758081, 69.879687,
                         73.667708, 74.802824, 74.802824, 73.667708,
                         69.879687, 60.758081, 32.224989, -32.224989,
                         -60.758081, -69.879687, -73.667708, -74.802824])

        elat = -np.array([-6.81982903e-15, -1.93889346e+01, -3.84764764e+01,
                          -5.67707359e+01, -7.18862588e+01, -7.18862588e+01,
                          -5.67707359e+01, -3.84764764e+01, -1.93889346e+01,
                          0.00000000e+00, 6.81982903e-15, 1.93889346e+01,
                          3.84764764e+01, 5.67707359e+01, 7.18862588e+01,
                          7.18862588e+01, 5.67707359e+01, 3.84764764e+01,
                          1.93889346e+01, -0.00000000e+00])

        np.testing.assert_allclose(lon, elon + lon_0)
        np.testing.assert_allclose(lat, elat)

    def test_get_geostationary_angle_extent(self):
        """Get max geostationary angles."""
        geos_area = mock.MagicMock()
        geos_area.proj_dict = {'a': 6378169.00,
                               'b': 6356583.80,
                               'h': 35785831.00}

        expected = (0.15185342867090912, 0.15133555510297725)

        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

        geos_area.proj_dict = {'a': 1000.0,
                               'b': 1000.0,
                               'h': np.sqrt(2) * 1000.0 - 1000.0}

        expected = (np.deg2rad(45), np.deg2rad(45))

        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

    def test_geostationary_mask(self):
        """Test geostationary mask"""
        # Compute mask of a very elliptical earth
        area = pyresample.geometry.AreaDefinition(
            'FLDK',
            'Full Disk',
            'geos',
            {'a': '6378169.0',
             'b': '3000000.0',
             'h': '35785831.0',
             'lon_0': '145.0',
             'proj': 'geos',
             'units': 'm'},
            101,
            101,
            (-6498000.088960204, -6498000.088960204,
             6502000.089024927, 6502000.089024927))

        mask = hf.get_geostationary_mask(area).astype(np.int).compute()

        # Check results along a couple of lines
        # a) Horizontal
        self.assertTrue(np.all(mask[50, :8] == 0))
        self.assertTrue(np.all(mask[50, 8:93] == 1))
        self.assertTrue(np.all(mask[50, 93:] == 0))

        # b) Vertical
        self.assertTrue(np.all(mask[:31, 50] == 0))
        self.assertTrue(np.all(mask[31:70, 50] == 1))
        self.assertTrue(np.all(mask[70:, 50] == 0))

        # c) Top left to bottom right
        self.assertTrue(np.all(mask[range(33), range(33)] == 0))
        self.assertTrue(np.all(mask[range(33, 68), range(33, 68)] == 1))
        self.assertTrue(np.all(mask[range(68, 101), range(68, 101)] == 0))

        # d) Bottom left to top right
        self.assertTrue(np.all(mask[range(101-1, 68-1, -1), range(33)] == 0))
        self.assertTrue(np.all(mask[range(68-1, 33-1, -1), range(33, 68)] == 1))
        self.assertTrue(np.all(mask[range(33-1, -1, -1), range(68, 101)] == 0))

    @mock.patch('satpy.readers.utils.AreaDefinition')
    def test_sub_area(self, adef):
        """Sub area slicing."""
        area = mock.MagicMock()
        area.pixel_size_x = 1.5
        area.pixel_size_y = 1.5
        area.pixel_upper_left = (0, 0)
        area.area_id = 'fakeid'
        area.name = 'fake name'
        area.proj_id = 'fakeproj'
        area.proj_dict = {'fake': 'dict'}

        hf.get_sub_area(area, slice(1, 4), slice(0, 3))
        adef.assert_called_once_with('fakeid', 'fake name', 'fakeproj',
                                     {'fake': 'dict'},
                                     3, 3,
                                     (0.75, -3.75, 5.25, 0.75))

    def test_np2str(self):
        """Test the np2str function."""
        # byte object
        npstring = np.string_('hej')
        self.assertEquals(hf.np2str(npstring), 'hej')

        # single element numpy array
        np_arr = np.array([npstring])
        self.assertEquals(hf.np2str(np_arr), 'hej')

        # scalar numpy array
        np_arr = np.array(npstring)
        self.assertEquals(hf.np2str(np_arr), 'hej')

        # multi-element array
        npstring = np.array([npstring, npstring])
        self.assertRaises(ValueError, hf.np2str, npstring)

        # non-array
        self.assertRaises(ValueError, hf.np2str, 5)


def suite():
    """The test suite for test_satin_helpers.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestHelpers))

    return mysuite
