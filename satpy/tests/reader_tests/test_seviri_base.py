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
"""Test the MSG common (native and hrit format) functionionalities."""

import unittest

import numpy as np
import xarray as xr
import dask.array as da

from satpy.readers.seviri_base import dec10216, chebyshev, get_cds_time, \
    get_padding_area, pad_data_horizontally, pad_data_vertically
from satpy import CHUNK_SIZE


def chebyshev4(c, x, domain):
    """Evaluate 4th order Chebyshev polynomial."""
    start_x, end_x = domain
    t = (x - 0.5 * (end_x + start_x)) / (0.5 * (end_x - start_x))
    return c[0] + c[1]*t + c[2]*(2*t**2 - 1) + c[3]*(4*t**3 - 3*t) - 0.5*c[0]


class SeviriBaseTest(unittest.TestCase):
    """Test SEVIRI base."""

    def test_dec10216(self):
        """Test the dec10216 function."""
        res = dec10216(np.array([255, 255, 255, 255, 255], dtype=np.uint8))
        exp = (np.ones((4, )) * 1023).astype(np.uint16)
        np.testing.assert_equal(res, exp)
        res = dec10216(np.array([1, 1, 1, 1, 1], dtype=np.uint8))
        exp = np.array([4,  16,  64, 257], dtype=np.uint16)
        np.testing.assert_equal(res, exp)

    def test_chebyshev(self):
        """Test the chebyshev function."""
        coefs = [1, 2, 3, 4]
        time = 123
        domain = [120, 130]
        res = chebyshev(coefs=[1, 2, 3, 4], time=time, domain=domain)
        exp = chebyshev4(coefs, time, domain)
        np.testing.assert_allclose(res, exp)

    def test_get_cds_time(self):
        """Test the get_cds_time function."""
        # Scalar
        self.assertEqual(get_cds_time(days=21246, msecs=12*3600*1000),
                         np.datetime64('2016-03-03 12:00'))

        # Array
        days = np.array([21246, 21247, 21248])
        msecs = np.array([12*3600*1000, 13*3600*1000 + 1, 14*3600*1000 + 2])
        expected = np.array([np.datetime64('2016-03-03 12:00:00.000'),
                             np.datetime64('2016-03-04 13:00:00.001'),
                             np.datetime64('2016-03-05 14:00:00.002')])
        np.testing.assert_equal(get_cds_time(days=days, msecs=msecs), expected)

        days = 21246
        msecs = 12*3600*1000
        expected = np.datetime64('2016-03-03 12:00:00.000')
        np.testing.assert_equal(get_cds_time(days=days, msecs=msecs), expected)

    def test_pad_data_horizontally_bad_shape(self):
        """Test the error handling for the horizontal hrv padding."""
        data = xr.DataArray(data=np.zeros((1, 10)), dims=('y', 'x'))
        east_bound = 5
        west_bound = 10
        final_size = (1, 20)
        with self.assertRaises(IndexError):
            pad_data_horizontally(data, final_size, east_bound, west_bound)

    def test_pad_data_vertically_bad_shape(self):
        """Test the error handling for the vertical hrv padding."""
        data = xr.DataArray(data=np.zeros((10, 1)), dims=('y', 'x'))
        south_bound = 5
        north_bound = 10
        final_size = (20, 1)
        with self.assertRaises(IndexError):
            pad_data_vertically(data, final_size, south_bound, north_bound)

    @staticmethod
    def test_pad_data_horizontally():
        """Test the horizontal hrv padding."""
        data = xr.DataArray(data=np.zeros((1, 10)), dims=('y', 'x'))
        east_bound = 4
        west_bound = 13
        final_size = (1, 20)
        res = pad_data_horizontally(data, final_size, east_bound, west_bound)
        expected = np.array([[np.nan, np.nan, np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        np.testing.assert_equal(res, expected)

    @staticmethod
    def test_pad_data_vertically():
        """Test the vertical hrv padding."""
        data = xr.DataArray(data=np.zeros((10, 1)), dims=('y', 'x'))
        south_bound = 4
        north_bound = 13
        final_size = (20, 1)
        res = pad_data_vertically(data, final_size, south_bound, north_bound)
        expected = np.zeros(final_size)
        expected[:] = np.nan
        expected[south_bound-1:north_bound] = 0.
        np.testing.assert_equal(res, expected)

    @staticmethod
    def test_get_padding_area_float():
        """Test padding area generator for floats."""
        shape = (10, 10)
        dtype = np.float64
        res = get_padding_area(shape, dtype)
        expected = da.full(shape, np.nan, dtype=dtype, chunks=CHUNK_SIZE)
        np.testing.assert_array_equal(res, expected)

    @staticmethod
    def test_get_padding_area_int():
        """Test padding area generator for integers."""
        shape = (10, 10)
        dtype = np.int64
        res = get_padding_area(shape, dtype)
        expected = da.full(shape, 0, dtype=dtype, chunks=CHUNK_SIZE)
        np.testing.assert_array_equal(res, expected)
