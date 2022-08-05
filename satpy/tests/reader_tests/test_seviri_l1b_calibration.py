#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""Unittesting the native msg reader."""

import unittest
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from satpy.readers.seviri_base import SEVIRICalibrationAlgorithm, SEVIRICalibrationHandler

COUNTS_INPUT = xr.DataArray(
    np.array([[377.,  377.,  377.,  376.,  375.],
              [376.,  375.,  376.,  374.,  374.],
              [374.,  373.,  373.,  374.,  374.],
              [347.,  345.,  345.,  348.,  347.],
              [306.,  306.,  307.,  307.,  308.]], dtype=np.float32)
)

RADIANCES_OUTPUT = xr.DataArray(
    np.array([[66.84162903,  66.84162903,  66.84162903,  66.63659668,
               66.4315567],
              [66.63659668,  66.4315567,  66.63659668,  66.22652435,
               66.22652435],
              [66.22652435,  66.02148438,  66.02148438,  66.22652435,
               66.22652435],
              [60.69055939,  60.28048706,  60.28048706,  60.89559937,
               60.69055939],
              [52.28409576,  52.28409576,  52.48912811,  52.48912811,
               52.69416809]], dtype=np.float32)
)

GAIN = 0.20503567620766011
OFFSET = -10.456819486590666

CAL_TYPE1 = 1
CAL_TYPE2 = 2
CAL_TYPEBAD = -1
CHANNEL_NAME = 'IR_108'
PLATFORM_ID = 323  # Met-10

TBS_OUTPUT1 = xr.DataArray(
    np.array([[269.29684448,  269.29684448,  269.29684448,  269.13296509,
               268.96871948],
              [269.13296509,  268.96871948,  269.13296509,  268.80422974,
               268.80422974],
              [268.80422974,  268.63937378,  268.63937378,  268.80422974,
               268.80422974],
              [264.23751831,  263.88912964,  263.88912964,  264.41116333,
               264.23751831],
              [256.77682495,  256.77682495,  256.96743774,  256.96743774,
               257.15756226]], dtype=np.float32)
)

TBS_OUTPUT2 = xr.DataArray(
    np.array([[268.94519043,  268.94519043,  268.94519043,  268.77984619,
               268.61422729],
              [268.77984619,  268.61422729,  268.77984619,  268.44830322,
               268.44830322],
              [268.44830322,  268.28204346,  268.28204346,  268.44830322,
               268.44830322],
              [263.84396362,  263.49285889,  263.49285889,  264.01898193,
               263.84396362],
              [256.32858276,  256.32858276,  256.52044678,  256.52044678,
               256.71188354]], dtype=np.float32)
)

VIS008_SOLAR_IRRADIANCE = 73.1807

VIS008_RADIANCE = xr.DataArray(
    np.array([[0.62234485,  0.59405649,  0.59405649,  0.59405649,  0.59405649],
             [0.59405649,  0.62234485,  0.62234485,  0.59405649,  0.62234485],
             [0.76378691,  0.79207528,  0.79207528,  0.76378691,  0.79207528],
             [3.30974245,  3.33803129,  3.33803129,  3.25316572,  3.47947311],
             [7.52471399,  7.83588648,  8.2602129,  8.57138538,  8.99571133]],
             dtype=np.float32)
)

VIS008_REFLECTANCE = xr.DataArray(
    np.array([[2.739768, 2.615233, 2.615233, 2.615233, 2.615233],
              [2.615233, 2.739768, 2.739768, 2.615233, 2.739768],
              [3.362442, 3.486977, 3.486977, 3.362442, 3.486977],
              [14.570578, 14.695117, 14.695117, 14.321507, 15.317789],
              [33.126278, 34.49616, 36.364185, 37.73407, 39.60209]],
             dtype=np.float32)
)


class TestSEVIRICalibrationAlgorithm(unittest.TestCase):
    """Unit Tests for SEVIRI calibration algorithm."""

    def setUp(self):
        """Set up the SEVIRI Calibration algorithm for testing."""
        self.algo = SEVIRICalibrationAlgorithm(
            platform_id=PLATFORM_ID,
            scan_time=datetime(2020, 8, 15, 13, 0, 40)
        )

    def test_convert_to_radiance(self):
        """Test the conversion from counts to radiances."""
        result = self.algo.convert_to_radiance(COUNTS_INPUT, GAIN, OFFSET)
        xr.testing.assert_allclose(result, RADIANCES_OUTPUT)
        self.assertEqual(result.dtype, np.float32)

    def test_ir_calibrate(self):
        """Test conversion from radiance to brightness temperature."""
        result = self.algo.ir_calibrate(RADIANCES_OUTPUT,
                                        CHANNEL_NAME, CAL_TYPE1)
        xr.testing.assert_allclose(result, TBS_OUTPUT1, rtol=1E-5)
        self.assertEqual(result.dtype, np.float32)

        result = self.algo.ir_calibrate(RADIANCES_OUTPUT,
                                        CHANNEL_NAME, CAL_TYPE2)
        xr.testing.assert_allclose(result, TBS_OUTPUT2, rtol=1E-5)

        with self.assertRaises(NotImplementedError):
            self.algo.ir_calibrate(RADIANCES_OUTPUT, CHANNEL_NAME, CAL_TYPEBAD)

    def test_vis_calibrate(self):
        """Test conversion from radiance to reflectance."""
        result = self.algo.vis_calibrate(VIS008_RADIANCE,
                                         VIS008_SOLAR_IRRADIANCE)
        xr.testing.assert_allclose(result, VIS008_REFLECTANCE)
        self.assertTrue(result.sun_earth_distance_correction_applied)
        self.assertEqual(result.dtype, np.float32)


class TestSeviriCalibrationHandler:
    """Unit tests for SEVIRI calibration handler."""

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

    def _get_calibration_handler(self, calib_mode='NOMINAL', ext_coefs=None):
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
                    'EXTERNAL': ext_coefs or {}
                },
                'radiance_type': 1
            },
            calib_mode=calib_mode,
            scan_time=None
        )

    def test_calibrate_exceptions(self):
        """Test exceptions raised by the calibration handler."""
        calib = self._get_calibration_handler()
        with pytest.raises(ValueError):
            calib.calibrate(None, 'invalid')

    @pytest.mark.parametrize(
        ('calib_mode', 'ext_coefs', 'expected'),
        [
            ('NOMINAL', {}, (10, -1)),
            ('GSICS', {}, (20, -40)),
            ('GSICS', {'gain': 30, 'offset': -3}, (30, -3)),
            ('NOMINAL', {'gain': 30, 'offset': -3}, (30, -3))
        ]
    )
    def test_get_gain_offset(self, calib_mode, ext_coefs, expected):
        """Test selection of gain and offset."""
        calib = self._get_calibration_handler(calib_mode=calib_mode,
                                              ext_coefs=ext_coefs)
        coefs = calib.get_gain_offset()
        assert coefs == expected


class TestFileHandlerCalibrationBase:
    """Base class for file handler calibration tests."""

    platform_id = 324
    gains_nominal = np.arange(1, 13)
    offsets_nominal = np.arange(-1, -13, -1)
    # No GSICS coefficients for VIS channels -> set to zero
    gains_gsics = [0, 0, 0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 0]
    offsets_gsics = [0, 0, 0, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, 0]
    radiance_types = 2 * np.ones(12)
    scan_time = datetime(2020, 1, 1)
    external_coefs = {
        'VIS006': {'gain': 10, 'offset': -10},
        'IR_108': {'gain': 20, 'offset': -20},
        'HRV': {'gain': 5, 'offset': -5}
    }
    spectral_channel_ids = {'VIS006': 1, 'IR_108': 9, 'HRV': 12}
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
                    [[np.nan, 41.88985],
                     [460.7884, 1182.2247]],
                    dims=('y', 'x')
                ),
                'EXTERNAL': xr.DataArray(
                    [[np.nan, 418.89853],
                     [4607.8843, 11822.249]],
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
        },
        'HRV': {
            'counts': {
                'NOMINAL': xr.DataArray(
                    [[0, 10],
                     [100, 255]],
                    dims=('y', 'x')
                )
            },
            'radiance': {
                'NOMINAL': xr.DataArray(
                    [[np.nan, 108],
                     [1188, 3048]],
                    dims=('y', 'x')
                ),
                'GSICS': xr.DataArray(
                    [[np.nan, 108],
                     [1188, 3048]],
                    dims=('y', 'x')
                ),
                'EXTERNAL': xr.DataArray(
                    [[np.nan, 45],
                     [495, 1270]],
                    dims=('y', 'x')
                )
            },
            'reflectance': {
                'NOMINAL': xr.DataArray(
                    [[np.nan, 415.26767],
                     [4567.944, 11719.775]],
                    dims=('y', 'x')
                ),
                'EXTERNAL': xr.DataArray(
                    [[np.nan, 173.02817],
                     [1903.31, 4883.2397]],
                    dims=('y', 'x')
                )
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
