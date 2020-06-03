#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2014-2019 Satpy developers
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
from unittest import mock
import os
import numpy as np
import numpy.testing
import pyresample.geometry

from satpy.readers import utils as hf


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
        del geos_area.crs
        lon_0 = 0
        geos_area.proj_dict = {
            'proj': 'geos',
            'lon_0': lon_0,
            'a': 6378169.00,
            'b': 6356583.80,
            'h': 35785831.00,
            'units': 'm'}
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
        del geos_area.crs
        geos_area.proj_dict = {
            'proj': 'geos',
            'sweep': 'x',
            'lon_0': -89.5,
            'a': 6378169.00,
            'b': 6356583.80,
            'h': 35785831.00,
            'units': 'm'}

        expected = (0.15185342867090912, 0.15133555510297725)
        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

        geos_area.proj_dict['a'] = 1000.0
        geos_area.proj_dict['b'] = 1000.0
        geos_area.proj_dict['h'] = np.sqrt(2) * 1000.0 - 1000.0

        expected = (np.deg2rad(45), np.deg2rad(45))
        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

        geos_area.proj_dict = {
            'proj': 'geos',
            'sweep': 'x',
            'lon_0': -89.5,
            'ellps': 'GRS80',
            'h': 35785831.00,
            'units': 'm'}
        expected = (0.15185277703584374, 0.15133971368991794)

        np.testing.assert_allclose(expected,
                                   hf.get_geostationary_angle_extent(geos_area))

    def test_geostationary_mask(self):
        """Test geostationary mask."""
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
        self.assertEqual(hf.np2str(npstring), 'hej')

        # single element numpy array
        np_arr = np.array([npstring])
        self.assertEqual(hf.np2str(np_arr), 'hej')

        # scalar numpy array
        np_arr = np.array(npstring)
        self.assertEqual(hf.np2str(np_arr), 'hej')

        # multi-element array
        npstring = np.array([npstring, npstring])
        self.assertRaises(ValueError, hf.np2str, npstring)

        # non-array
        self.assertRaises(ValueError, hf.np2str, 5)

    def test_get_earth_radius(self):
        """Test earth radius computation."""
        a = 2.
        b = 1.

        def re(lat):
            """Compute ellipsoid radius at the given geodetic latitude.

            Reference: Capderou, M.: Handbook of Satellite Orbits, Equation (2.20).
            """
            lat = np.deg2rad(lat)
            e2 = 1 - b ** 2 / a ** 2
            n = a / np.sqrt(1 - e2*np.sin(lat)**2)
            return n * np.sqrt((1 - e2)**2 * np.sin(lat)**2 + np.cos(lat)**2)

        for lon in (0, 180, 270):
            self.assertEqual(hf.get_earth_radius(lon=lon, lat=0., a=a, b=b), a)
        for lat in (90, -90):
            self.assertEqual(hf.get_earth_radius(lon=0., lat=lat, a=a, b=b), b)
        self.assertTrue(np.isclose(hf.get_earth_radius(lon=123, lat=45., a=a, b=b), re(45.)))

    def test_reduce_mda(self):
        """Test metadata size reduction."""
        mda = {'a': 1,
               'b': np.array([1, 2, 3]),
               'c': np.array([1, 2, 3, 4]),
               'd': {'a': 1,
                     'b': np.array([1, 2, 3]),
                     'c': np.array([1, 2, 3, 4]),
                     'd': {'a': 1,
                           'b': np.array([1, 2, 3]),
                           'c': np.array([1, 2, 3, 4])}}}
        exp = {'a': 1,
               'b': np.array([1, 2, 3]),
               'd': {'a': 1,
                     'b': np.array([1, 2, 3]),
                     'd': {'a': 1,
                           'b': np.array([1, 2, 3])}}}
        numpy.testing.assert_equal(hf.reduce_mda(mda, max_size=3), exp)

        # Make sure, reduce_mda() doesn't modify the original dictionary
        self.assertIn('c', mda)
        self.assertIn('c', mda['d'])
        self.assertIn('c', mda['d']['d'])

    @mock.patch('satpy.readers.utils.bz2.BZ2File')
    @mock.patch('satpy.readers.utils.Popen')
    def test_unzip_file_pbzip2(self, mock_popen, mock_bz2):
        """Test the bz2 file unzipping techniques."""
        process_mock = mock.Mock()
        attrs = {'communicate.return_value': (b'output', b'error'),
                 'returncode': 0}
        process_mock.configure_mock(**attrs)
        mock_popen.return_value = process_mock

        bz2_mock = mock.MagicMock()
        bz2_mock.read.return_value = b'TEST'
        mock_bz2.return_value = bz2_mock

        filename = 'tester.DAT.bz2'
        whichstr = 'satpy.readers.utils.which'
        # no bz2 installed
        with mock.patch(whichstr) as whichmock:
            whichmock.return_value = None
            new_fname = hf.unzip_file(filename)
            self.assertTrue(bz2_mock.read.called)
            self.assertTrue(os.path.exists(new_fname))
            if os.path.exists(new_fname):
                os.remove(new_fname)
        # bz2 installed
        with mock.patch(whichstr) as whichmock:
            whichmock.return_value = '/usr/bin/pbzip2'
            new_fname = hf.unzip_file(filename)
            self.assertTrue(mock_popen.called)
            self.assertTrue(os.path.exists(new_fname))
            if os.path.exists(new_fname):
                os.remove(new_fname)

        filename = 'tester.DAT'
        new_fname = hf.unzip_file(filename)
        self.assertIsNone(new_fname)
