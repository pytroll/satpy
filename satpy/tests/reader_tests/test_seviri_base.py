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
from datetime import datetime

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.seviri_base import (
    NoValidOrbitParams,
    OrbitPolynomial,
    OrbitPolynomialFinder,
    chebyshev,
    dec10216,
    get_cds_time,
    get_padding_area,
    get_satpos,
    pad_data_horizontally,
    pad_data_vertically,
)


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


ORBIT_POLYNOMIALS = {
    'StartTime': np.array([
        [
            datetime(2006, 1, 1, 6), datetime(2006, 1, 1, 12),
            datetime(2006, 1, 1, 18), datetime(1958, 1, 1, 0)]
    ]),
    'EndTime': np.array([
        [
            datetime(2006, 1, 1, 12), datetime(2006, 1, 1, 18),
            datetime(2006, 1, 2, 0), datetime(1958, 1, 1, 0)
        ]
    ]),
    'X': [np.zeros(8),
          [8.41607082e+04, 2.94319260e+00, 9.86748617e-01,
           -2.70135453e-01,
           -3.84364650e-02, 8.48718433e-03, 7.70548174e-04,
           -1.44262718e-04],
          np.zeros(8)],
    'Y': [np.zeros(8),
          [-5.21170255e+03, 5.12998948e+00, -1.33370453e+00,
           -3.09634144e-01,
           6.18232793e-02, 7.50505681e-03, -1.35131011e-03,
           -1.12054405e-04],
          np.zeros(8)],
    'Z': [np.zeros(8),
          [-6.51293855e+02, 1.45830459e+02, 5.61379400e+01,
           -3.90970565e+00,
           -7.38137565e-01, 3.06131644e-02, 3.82892428e-03,
           -1.12739309e-04],
          np.zeros(8)],
}
ORBIT_POLYNOMIALS_SYNTH = {
    # 12-31: Contiguous
    # 01-01: Small gap (12:00 - 13:00)
    # 01-02: Large gap (04:00 - 18:00)
    # 01-03: Overlap (10:00 - 13:00)
    'StartTime': np.array([
        [
            datetime(2005, 12, 31, 10), datetime(2005, 12, 31, 12),
            datetime(2006, 1, 1, 10), datetime(2006, 1, 1, 13),
            datetime(2006, 1, 2, 0), datetime(2006, 1, 2, 18),
            datetime(2006, 1, 3, 6), datetime(2006, 1, 3, 10),
        ]
    ]),
    'EndTime': np.array([
        [
            datetime(2005, 12, 31, 12), datetime(2005, 12, 31, 18),
            datetime(2006, 1, 1, 12), datetime(2006, 1, 1, 18),
            datetime(2006, 1, 2, 4), datetime(2006, 1, 2, 22),
            datetime(2006, 1, 3, 13), datetime(2006, 1, 3, 18),
        ]
    ]),
    'X': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    'Y': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1],
    'Z': [1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2],
}
ORBIT_POLYNOMIALS_INVALID = {
    'StartTime': np.array([
        [
            datetime(1958, 1, 1), datetime(1958, 1, 1)
        ]
    ]),
    'EndTime': np.array([
        [
            datetime(1958, 1, 1), datetime(1958, 1, 1)
        ]
    ]),
    'X': [1, 2],
    'Y': [3, 4],
    'Z': [5, 6],
}


class TestSatellitePosition:
    """Test locating the satellite."""

    @pytest.fixture
    def orbit_polynomial(self):
        """Get an orbit polynomial for testing."""
        return OrbitPolynomial(
            start_time=datetime(2006, 1, 1, 12),
            end_time=datetime(2006, 1, 1, 18),
            coefs=(
                np.array([8.41607082e+04, 2.94319260e+00, 9.86748617e-01,
                          -2.70135453e-01, -3.84364650e-02, 8.48718433e-03,
                          7.70548174e-04, -1.44262718e-04]),
                np.array([-5.21170255e+03, 5.12998948e+00, -1.33370453e+00,
                          -3.09634144e-01, 6.18232793e-02, 7.50505681e-03,
                          -1.35131011e-03, -1.12054405e-04]),
                np.array([-6.51293855e+02, 1.45830459e+02, 5.61379400e+01,
                          -3.90970565e+00, -7.38137565e-01, 3.06131644e-02,
                          3.82892428e-03, -1.12739309e-04])
            )
        )

    @pytest.fixture
    def time(self):
        """Get scan timestamp for testing."""
        return datetime(2006, 1, 1, 12, 15, 9, 304888)

    def test_eval_polynomial(self, orbit_polynomial, time):
        """Test getting the position in cartesian coordinates."""
        x, y, z = orbit_polynomial.evaluate(time)
        np.testing.assert_allclose(
            [x, y, z],
            [42078421.37095518, -2611352.744615312, -419828.9699940758]
        )

    def test_get_satpos(self, orbit_polynomial, time):
        """Test getting the position in geodetic coordinates."""
        lon, lat, alt = get_satpos(
            orbit_polynomial=orbit_polynomial,
            time=time,
            semi_major_axis=6378169.00,
            semi_minor_axis=6356583.80
        )
        np.testing.assert_allclose(
            [lon, lat, alt],
            [-3.55117540817073, -0.5711243456528018, 35783296.150123544]
        )


class TestOrbitPolynomialFinder:
    """Unit tests for orbit polynomial finder."""

    @pytest.mark.parametrize(
        ('orbit_polynomials', 'time', 'orbit_polynomial_exp'),
        [
            # Contiguous validity intervals (that's the norm)
            (
                ORBIT_POLYNOMIALS_SYNTH,
                datetime(2005, 12, 31, 12, 15),
                OrbitPolynomial(
                    coefs=(2.0, 2.1, 2.2),
                    start_time=np.datetime64('2005-12-31 12:00'),
                    end_time=np.datetime64('2005-12-31 18:00')
                )
            ),
            # No interval enclosing the given timestamp, but closest interval
            # not too far away
            (
                    ORBIT_POLYNOMIALS_SYNTH,
                    datetime(2006, 1, 1, 12, 15),
                    OrbitPolynomial(
                        coefs=(3.0, 3.1, 3.2),
                        start_time=np.datetime64('2006-01-01 10:00'),
                        end_time=np.datetime64('2006-01-01 12:00')
                    )
            ),
            # Overlapping intervals
            (
                    ORBIT_POLYNOMIALS_SYNTH,
                    datetime(2006, 1, 3, 12, 15),
                    OrbitPolynomial(
                        coefs=(8.0, 8.1, 8.2),
                        start_time=np.datetime64('2006-01-03 10:00'),
                        end_time=np.datetime64('2006-01-03 18:00')
                    )
            ),
        ]
    )
    def test_get_orbit_polynomial(self, orbit_polynomials, time,
                                  orbit_polynomial_exp):
        """Test getting the satellite locator."""
        finder = OrbitPolynomialFinder(orbit_polynomials)
        orbit_polynomial = finder.get_orbit_polynomial(time=time)
        assert orbit_polynomial == orbit_polynomial_exp

    @pytest.mark.parametrize(
        ('orbit_polynomials', 'time'),
        [
            # No interval enclosing the given timestamp and closest interval
            # too far away
            (ORBIT_POLYNOMIALS_SYNTH, datetime(2006, 1, 2, 12, 15)),
            # No valid polynomials at all
            (ORBIT_POLYNOMIALS_INVALID, datetime(2006, 1, 1, 12, 15))
        ]
    )
    def test_get_orbit_polynomial_exceptions(self, orbit_polynomials, time):
        """Test exceptions thrown while getting the satellite locator."""
        finder = OrbitPolynomialFinder(orbit_polynomials)
        with pytest.raises(NoValidOrbitParams):
            finder.get_orbit_polynomial(time=time)
