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

from datetime import datetime
import unittest

import numpy as np
import xarray as xr
import pytest

from satpy.readers.seviri_base import (
    dec10216, chebyshev, get_cds_time, SEVIRICalibrationHandler
)


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
        np.testing.assert_equal(res, exp)
        res = dec10216(np.array([1, 1, 1, 1, 1], dtype=np.uint8))
        exp = np.array([4,  16,  64, 257], dtype=np.uint16)
        np.testing.assert_equal(res, exp)

    def test_chebyshev(self):
        coefs = [1, 2, 3, 4]
        time = 123
        domain = [120, 130]
        res = chebyshev(coefs=[1, 2, 3, 4], time=time, domain=domain)
        exp = chebyshev4(coefs, time, domain)
        np.testing.assert_allclose(res, exp)

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
        np.testing.assert_equal(get_cds_time(days=days, msecs=msecs), expected)

        days = 21246
        msecs = 12*3600*1000
        expected = np.datetime64('2016-03-03 12:00:00.000')
        np.testing.assert_equal(get_cds_time(days=days, msecs=msecs), expected)


class TestCalibrationBase:
    """Base class for calibration tests."""

    platform_id = 324
    gains_nominal = np.arange(1, 13)
    offsets_nominal = np.arange(-1, -13, -1)
    gains_gsics = np.arange(0.1, 1.3, 0.1)
    offsets_gsics = np.arange(-0.1, -1.3, -0.1)
    radiance_types = 2 * np.ones(12)
    scan_time = datetime(2020, 1, 1)
    external_coefs = {
        'VIS006': {'gain': 10, 'offset': -10},
        'IR_108': {'gain': 20, 'offset': -20}
    }
    spectral_channel_ids = {'VIS006': 1, 'IR_108': 9}
    expected = {
        'VIS006': {
            'counts': {
                'NOMINAL': xr.DataArray(
                    [[0, 10],
                     [100, 255]],
                    dims=('y', 'x')
                )
            },
            'radiance': {
                'NOMINAL': xr.DataArray(
                    [[np.nan, 9],
                     [99, 254]],
                    dims=('y', 'x')
                ),
                'GSICS': xr.DataArray(
                    [[np.nan, 9],
                     [99, 254]],
                    dims=('y', 'x')
                ),
                'EXTERNAL': xr.DataArray(
                    [[np.nan, 90],
                     [990, 2540]],
                    dims=('y', 'x')
                )
            },
            'reflectance': {
                'NOMINAL': xr.DataArray(
                    [[np.nan, 40.47923],
                     [445.27155, 1142.414]],
                    dims=('y', 'x')
                ),
                'EXTERNAL': xr.DataArray(
                    [[np.nan, 404.7923],
                     [4452.7153, 11424.14]],
                    dims=('y', 'x')
                )
            }
        },
        'IR_108': {
            'counts': {
                'NOMINAL': xr.DataArray(
                    [[0, 10],
                     [100, 255]],
                    dims=('y', 'x')
                )
            },
            'radiance': {
                'NOMINAL': xr.DataArray(
                    [[np.nan, 81],
                     [891, 2286]],
                    dims=('y', 'x')
                ),
                'GSICS': xr.DataArray(
                    [[np.nan, 8.19],
                     [89.19, 228.69]],
                    dims=('y', 'x')
                ),
                'EXTERNAL': xr.DataArray(
                    [[np.nan, 180],
                     [1980, 5080]],
                    dims=('y', 'x')
                )
            },
            'brightness_temperature': {
                'NOMINAL': xr.DataArray(
                    [[np.nan, 279.82318],
                     [543.2585, 812.77167]],
                    dims=('y', 'x')
                ),
                'GSICS': xr.DataArray(
                    [[np.nan, 189.20985],
                     [285.53293, 356.06668]],
                    dims=('y', 'x')
                ),
                'EXTERNAL': xr.DataArray(
                    [[np.nan, 335.14236],
                     [758.6249, 1262.7567]],
                    dims=('y', 'x')
                ),
            }
        }
    }

    @pytest.fixture(name='counts')
    def counts(self):
        """Provide fake image counts."""
        return xr.DataArray(
            [[0, 10],
             [100, 255]],
            dims=('y', 'x')
        )

    def _get_expected(
            self, channel, calibration, calib_mode, use_ext_coefs
    ):
        if use_ext_coefs:
            return self.expected[channel][calibration]['EXTERNAL']
        return self.expected[channel][calibration][calib_mode]


class TestSeviriCalibrationHandler:
    """Unit tests for calibration handler."""

    def test_init(self):
        """Test initialization of the calibration handler."""
        with pytest.raises(ValueError):
            SEVIRICalibrationHandler(
                platform_id=None,
                channel_name=None,
                coefs=None,
                calib_mode='invalid',
                scan_time=None
            )

    @pytest.fixture(name='counts')
    def counts(self):
        """Provide fake counts."""
        return xr.DataArray(
            [[1, 2],
             [3, 4]],
            dims=('y', 'x')
        )

    @pytest.fixture(name='calib')
    def calib(self):
        """Provide a calibration handler."""
        return SEVIRICalibrationHandler(
            platform_id=324,
            channel_name='IR_108',
            coefs={
                'coefs': {
                    'NOMINAL': {
                        'gain': 10,
                        'offset': -1
                    },
                    'GSICS': {
                        'gain': 20,
                        'offset': -2
                    },
                    'EXTERNAL': {}
                },
                'radiance_type': 1
            },
            calib_mode='NOMINAL',
            scan_time=None
        )

    def test_calibrate_exceptions(self, counts, calib):
        """Test exception raised by the calibration handler."""
        with pytest.raises(ValueError):
            # Invalid calibration
            calib.calibrate(counts, 'invalid')
        with pytest.raises(NotImplementedError):
            # Invalid radiance type
            calib.coefs['radiance_type'] = 999
            calib.calibrate(counts, 'brightness_temperature')

    @pytest.mark.parametrize(
        ('calib_mode', 'ext_coefs', 'expected'),
        [
            ('NOMINAL', {}, (10, -1)),
            ('GSICS', {}, (20, -40)),
            ('GSICS', {'gain': 30, 'offset': -3}, (30, -3)),
            ('NOMINAL', {'gain': 30, 'offset': -3}, (30, -3))
        ]
    )
    def test_get_gain_offset(self, calib, calib_mode, ext_coefs, expected):
        """Test selection of gain and offset."""
        calib.calib_mode = calib_mode
        calib.coefs['coefs']['EXTERNAL'] = ext_coefs
        coefs = calib._get_gain_offset()
        assert coefs == expected
