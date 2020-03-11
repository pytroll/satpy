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
"""Testing of utils."""

import unittest
from unittest import mock
from numpy import sqrt
from satpy.utils import angle2xyz, lonlat2xyz, xyz2angle, xyz2lonlat, proj_units_to_meters, get_satpos


class TestUtils(unittest.TestCase):
    """Testing utils."""

    def test_lonlat2xyz(self):
        """Test the lonlat2xyz function."""
        x__, y__, z__ = lonlat2xyz(0, 0)
        self.assertAlmostEqual(x__, 1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(0, 90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = lonlat2xyz(180, 0)
        self.assertAlmostEqual(x__, -1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(-90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = lonlat2xyz(0, -90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, -1)

        x__, y__, z__ = lonlat2xyz(0, 45)
        self.assertAlmostEqual(x__, sqrt(2) / 2)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, sqrt(2) / 2)

        x__, y__, z__ = lonlat2xyz(0, 60)
        self.assertAlmostEqual(x__, sqrt(1) / 2)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, sqrt(3) / 2)

    def test_angle2xyz(self):
        """Test the lonlat2xyz function."""
        x__, y__, z__ = angle2xyz(0, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(0, 90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(180, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(-90, 0)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 1)

        x__, y__, z__ = angle2xyz(0, -90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(90, 90)
        self.assertAlmostEqual(x__, 1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(-90, 90)
        self.assertAlmostEqual(x__, -1)
        self.assertAlmostEqual(y__, 0)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(180, 90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(0, -90)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, -1)
        self.assertAlmostEqual(z__, 0)

        x__, y__, z__ = angle2xyz(0, 45)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, sqrt(2) / 2)
        self.assertAlmostEqual(z__, sqrt(2) / 2)

        x__, y__, z__ = angle2xyz(0, 60)
        self.assertAlmostEqual(x__, 0)
        self.assertAlmostEqual(y__, sqrt(3) / 2)
        self.assertAlmostEqual(z__, sqrt(1) / 2)

    def test_xyz2lonlat(self):
        """Test xyz2lonlat."""
        lon, lat = xyz2lonlat(1, 0, 0)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 0)

        lon, lat = xyz2lonlat(0, 1, 0)
        self.assertAlmostEqual(lon, 90)
        self.assertAlmostEqual(lat, 0)

        lon, lat = xyz2lonlat(0, 0, 1, asin=True)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 90)

        lon, lat = xyz2lonlat(0, 0, 1)
        self.assertAlmostEqual(lon, 0)
        self.assertAlmostEqual(lat, 90)

        lon, lat = xyz2lonlat(sqrt(2) / 2, sqrt(2) / 2, 0)
        self.assertAlmostEqual(lon, 45)
        self.assertAlmostEqual(lat, 0)

    def test_xyz2angle(self):
        """Test xyz2angle."""
        azi, zen = xyz2angle(1, 0, 0)
        self.assertAlmostEqual(azi, 90)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(0, 1, 0)
        self.assertAlmostEqual(azi, 0)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(0, 0, 1)
        self.assertAlmostEqual(azi, 0)
        self.assertAlmostEqual(zen, 0)

        azi, zen = xyz2angle(0, 0, 1, acos=True)
        self.assertAlmostEqual(azi, 0)
        self.assertAlmostEqual(zen, 0)

        azi, zen = xyz2angle(sqrt(2) / 2, sqrt(2) / 2, 0)
        self.assertAlmostEqual(azi, 45)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(-1, 0, 0)
        self.assertAlmostEqual(azi, -90)
        self.assertAlmostEqual(zen, 90)

        azi, zen = xyz2angle(0, -1, 0)
        self.assertAlmostEqual(azi, 180)
        self.assertAlmostEqual(zen, 90)

    def test_proj_units_to_meters(self):
        """Test proj units to meters conversion."""
        prj = '+asd=123123123123'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, prj)
        prj = '+a=6378.137'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000')
        prj = '+a=6378.137 +units=km'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000')
        prj = '+a=6378.137 +b=6378.137'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000 +b=6378137.000')
        prj = '+a=6378.137 +b=6378.137 +h=35785.863'
        res = proj_units_to_meters(prj)
        self.assertEqual(res, '+a=6378137.000 +b=6378137.000 +h=35785863.000')

    @mock.patch('satpy.utils.warnings.warn')
    def test_get_satpos(self, warn_mock):
        """Test getting the satellite position."""
        orb_params = {'nadir_longitude': 1,
                      'satellite_actual_longitude': 1.1,
                      'satellite_nominal_longitude': 1.2,
                      'projection_longitude': 1.3,
                      'nadir_latitude': 2,
                      'satellite_actual_latitude': 2.1,
                      'satellite_nominal_latitude': 2.2,
                      'projection_latitude': 2.3,
                      'satellite_actual_altitude': 3,
                      'satellite_nominal_altitude': 3.1,
                      'projection_altitude': 3.2}
        dataset = mock.MagicMock(attrs={'orbital_parameters': orb_params,
                                        'satellite_longitude': -1,
                                        'satellite_latitude': -2,
                                        'satellite_altitude': -3})

        # Nadir
        lon, lat, alt = get_satpos(dataset)
        self.assertTupleEqual((lon, lat, alt), (1, 2, 3))

        # Actual
        orb_params.pop('nadir_longitude')
        orb_params.pop('nadir_latitude')
        lon, lat, alt = get_satpos(dataset)
        self.assertTupleEqual((lon, lat, alt), (1.1, 2.1, 3))

        # Nominal
        orb_params.pop('satellite_actual_longitude')
        orb_params.pop('satellite_actual_latitude')
        orb_params.pop('satellite_actual_altitude')
        lon, lat, alt = get_satpos(dataset)
        self.assertTupleEqual((lon, lat, alt), (1.2, 2.2, 3.1))

        # Projection
        orb_params.pop('satellite_nominal_longitude')
        orb_params.pop('satellite_nominal_latitude')
        orb_params.pop('satellite_nominal_altitude')
        lon, lat, alt = get_satpos(dataset)
        self.assertTupleEqual((lon, lat, alt), (1.3, 2.3, 3.2))
        warn_mock.assert_called()

        # Legacy
        dataset.attrs.pop('orbital_parameters')
        lon, lat, alt = get_satpos(dataset)
        self.assertTupleEqual((lon, lat, alt), (-1, -2, -3))
