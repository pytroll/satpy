#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Tests for the goes imager nc reader (NOAA CLASS variant)."""

import datetime
import unittest
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.readers.goes_imager_nc import is_vis_channel
from satpy.tests.utils import make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - request


class GOESNCBaseFileHandlerTest(unittest.TestCase):
    """Testing the file handler."""

    longMessage = True

    @mock.patch('satpy.readers.goes_imager_nc.xr')
    @mock.patch.multiple('satpy.readers.goes_imager_nc.GOESNCBaseFileHandler',
                         _get_sector=mock.MagicMock())
    def setUp(self, xr_):
        """Set up the tests."""
        from satpy.readers.goes_imager_nc import CALIB_COEFS, GOESNCBaseFileHandler

        self.coefs = CALIB_COEFS['GOES-15']

        # Mock file access to return a fake dataset.
        self.time = datetime.datetime(2018, 8, 16, 16, 7)
        self.dummy3d = np.zeros((1, 2, 2))
        self.dummy2d = np.zeros((2, 2))
        self.band = 1
        self.nc = xr.Dataset(
            {'data': xr.DataArray(self.dummy3d, dims=('time', 'yc', 'xc')),
             'lon': xr.DataArray(data=self.dummy2d, dims=('yc', 'xc')),
             'lat': xr.DataArray(data=self.dummy2d, dims=('yc', 'xc')),
             'time': xr.DataArray(data=np.array([self.time],
                                                dtype='datetime64[ms]'),
                                  dims=('time',)),
             'bands': xr.DataArray(data=np.array([self.band]))},
            attrs={'Satellite Sensor': 'G-15'})
        xr_.open_dataset.return_value = self.nc

        # Instantiate reader using the mocked open_dataset() method. Also, make
        # the reader believe all abstract methods have been implemented.
        self.reader = GOESNCBaseFileHandler(filename='dummy', filename_info={},
                                            filetype_info={})

    def test_init(self):
        """Tests reader initialization."""
        self.assertEqual(self.reader.nlines, self.dummy2d.shape[0])
        self.assertEqual(self.reader.ncols, self.dummy2d.shape[1])
        self.assertEqual(self.reader.platform_name, 'GOES-15')
        self.assertEqual(self.reader.platform_shortname, 'goes15')
        self.assertEqual(self.reader.gvar_channel, self.band)
        self.assertIsInstance(self.reader.geo_data, xr.Dataset)

    def test_get_nadir_pixel(self):
        """Test identification of the nadir pixel."""
        from satpy.readers.goes_imager_nc import FULL_DISC

        earth_mask = np.array([[0, 0, 0, 0],
                               [0, 1, 0, 0],
                               [1, 1, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 0]])
        nadir_row, nadir_col = self.reader._get_nadir_pixel(
            earth_mask=earth_mask, sector=FULL_DISC)
        self.assertEqual((nadir_row, nadir_col), (2, 1),
                         msg='Incorrect nadir pixel')

    def test_viscounts2radiance(self):
        """Test conversion from VIS counts to radiance."""
        # Reference data is for detector #1
        slope = self.coefs['00_7']['slope'][0]
        offset = self.coefs['00_7']['offset'][0]
        counts = xr.DataArray([0, 100, 200, 500, 1000, 1023])
        rad_expected = xr.DataArray(
            [0., 41.54896, 100.06862,
             275.6276, 568.2259, 581.685422])
        rad = self.reader._viscounts2radiance(counts=counts, slope=slope,
                                              offset=offset)
        self.assertTrue(np.allclose(rad.data, rad_expected.data, atol=1E-6),
                        msg='Incorrect conversion from VIS counts to '
                            'radiance')

    def test_ircounts2radiance(self):
        """Test conversion from IR counts to radiance."""
        # Test counts
        counts = xr.DataArray([0, 100, 500, 1000, 1023])

        # Reference Radiance from NOAA lookup tables (same for detectors 1 and
        # 2, see [IR])
        rad_expected = {
            '03_9': np.array([0, 0.140, 1.899, 4.098, 4.199]),
            '06_5': np.array([0, 1.825, 12.124, 24.998, 25.590]),
            '10_7': np.array([0, 16.126, 92.630, 188.259, 192.658]),
            '13_3': np.array([0, 15.084, 87.421, 177.842, 182.001])
        }

        # The input counts are exact, but the accuracy of the output radiance is
        # limited to 3 digits
        atol = 1E-3

        for ch in sorted(rad_expected.keys()):
            coefs = self.coefs[ch]
            rad = self.reader._ircounts2radiance(
                counts=counts, scale=coefs['scale'], offset=coefs['offset'])
            self.assertTrue(np.allclose(rad.data, rad_expected[ch], atol=atol),
                            msg='Incorrect conversion from IR counts to '
                                'radiance in channel {}'.format(ch))

    def test_calibrate_vis(self):
        """Test VIS calibration."""
        rad = xr.DataArray([0, 1, 10, 100, 500])
        refl_expected = xr.DataArray([0., 0.188852, 1.88852, 18.8852, 94.426])
        refl = self.reader._calibrate_vis(radiance=rad,
                                          k=self.coefs['00_7']['k'])
        self.assertTrue(np.allclose(refl.data, refl_expected.data, atol=1E-6),
                        msg='Incorrect conversion from radiance to '
                            'reflectance')

    def test_calibrate_ir(self):
        """Test IR calibration."""
        # Test radiance values and corresponding BT from NOAA lookup tables
        # rev. H (see [IR]).
        rad = {
            '03_9': xr.DataArray([0, 0.1, 2, 3.997, 4.199]),
            '06_5': xr.DataArray([0, 0.821, 12.201, 25.590, 100]),
            '10_7': xr.DataArray([0, 11.727, 101.810, 189.407, 192.658]),
            '13_3': xr.DataArray([0, 22.679, 90.133, 182.001, 500])
        }
        bt_expected = {
            '03_9': np.array([[np.nan, 253.213, 319.451, 339.983, np.nan],
                              [np.nan, 253.213, 319.451, 339.983, np.nan]]),
            '06_5': np.array([[np.nan, 200.291, 267.860, 294.988, np.nan],
                              [np.nan, 200.308, 267.879, 295.008, np.nan]]),
            '10_7': np.array([[np.nan, 200.105, 294.437, 339.960, np.nan],
                              [np.nan, 200.097, 294.429, 339.953, np.nan]]),
            '13_3': np.array([[np.nan, 200.006, 267.517, 321.986, np.nan],
                              [np.nan, 200.014, 267.524, 321.990, np.nan]])
        }  # first row is for detector 1, second for detector 2.

        # The accuracy of the input radiance is limited to 3 digits so that
        # the results differ slightly.
        atol = {'03_9': 0.04, '06_5': 0.03, '10_7': 0.01, '13_3': 0.01}

        for ch in sorted(rad.keys()):
            coefs = self.coefs[ch]
            for det in [0, 1]:
                bt = self.reader._calibrate_ir(radiance=rad[ch],
                                               coefs={'a': coefs['a'][det],
                                                      'b': coefs['b'][det],
                                                      'n': coefs['n'][det],
                                                      'btmin': coefs['btmin'],
                                                      'btmax': coefs['btmax']})
                self.assertTrue(
                    np.allclose(bt.data, bt_expected[ch][det], equal_nan=True,
                                atol=atol[ch]),
                    msg='Incorrect conversion from radiance to brightness '
                        'temperature in channel {} detector {}'.format(ch, det))

    def test_start_time(self):
        """Test dataset start time stamp."""
        self.assertEqual(self.reader.start_time, self.time)

    def test_end_time(self):
        """Test dataset end time stamp."""
        from satpy.readers.goes_imager_nc import FULL_DISC, SCAN_DURATION, UNKNOWN_SECTOR
        expected = {
            UNKNOWN_SECTOR: self.time,
            FULL_DISC: self.time + SCAN_DURATION[FULL_DISC]
        }
        for sector, end_time in expected.items():
            self.reader.sector = sector
            self.assertEqual(self.reader.end_time, end_time)


class TestMetadata:
    """Testcase for dataset metadata."""

    @pytest.fixture(params=[1, 2])
    def channel_id(self, request):
        """Set channel ID."""
        return request.param

    @pytest.fixture(params=[True, False])
    def yaw_flip(self, request):
        """Set yaw-flip flag."""
        return request.param

    def _apply_yaw_flip(self, data_array, yaw_flip):
        if yaw_flip:
            data_array.data = np.flipud(data_array.data)
        return data_array

    @pytest.fixture
    def lons_lats(self, yaw_flip):
        """Get longitudes and latitudes."""
        lon = xr.DataArray(
            [[-1, 0, 1, 2],
             [-1, 0, 1, 2],
             [-1, 0, 1, 2]],
            dims=("yc", "xc")
        )
        lat = xr.DataArray(
            [[9999, 9999, 9999, 9999],
             [1, 1, 1, 1],
             [-1, -1, -1, -1]],
            dims=("yc", "xc")
        )
        self._apply_yaw_flip(lat, yaw_flip)
        return lon, lat

    @pytest.fixture
    def dataset(self, lons_lats, channel_id):
        """Create a fake dataset."""
        lon, lat = lons_lats
        data = xr.DataArray(
            [[[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]]],
            dims=("time", "yc", "xc")
        )
        time = xr.DataArray(
            [np.datetime64("2018-01-01 12:00:00")],
            dims="time"
        )
        bands = xr.DataArray([channel_id], dims="bands")
        return xr.Dataset(
            {
                'data': data,
                'lon': lon,
                'lat': lat,
                'time': time,
                'bands': bands,
            },
            attrs={'Satellite Sensor': 'G-15'}
        )

    @pytest.fixture
    def earth_mask(self, yaw_flip):
        """Get expected earth mask."""
        earth_mask = xr.DataArray(
            [[False, False, False, False],
             [True, True, True, True],
             [True, True, True, True]],
            dims=("yc", "xc"),
        )
        self._apply_yaw_flip(earth_mask, yaw_flip)
        return earth_mask

    @pytest.fixture
    def geometry(self, channel_id, yaw_flip):
        """Get expected geometry."""
        shapes = {
            1: {"width": 10847, "height": 10810},
            2: {"width": 2712, "height": 2702}
        }
        return {
            "nadir_row": 0 if yaw_flip else 1,
            "projection_longitude": -1 if yaw_flip else 1,
            "shape": shapes[channel_id]
        }

    @pytest.fixture
    def expected(self, geometry, earth_mask, yaw_flip):
        """Define expected metadata."""
        proj_dict = {
            'a': '6378169',
            'h': '35785831',
            'lon_0': '0',
            'no_defs': 'None',
            'proj': 'geos',
            'rf': '295.488065897001',
            'type': 'crs',
            'units': 'm',
            'x_0': '0',
            'y_0': '0'
        }
        area = AreaDefinition(
            area_id="goes_geos_uniform",
            proj_id="goes_geos_uniform",
            description="GOES-15 geostationary projection (uniform sampling)",
            projection=proj_dict,
            area_extent=(-5434201.1352, -5415668.5992, 5434201.1352, 5415668.5992),
            **geometry["shape"]
        )
        return {
            "area_def_uni": area,
            "earth_mask": earth_mask,
            "yaw_flip": yaw_flip,
            "lon0": 0,
            "lat0": geometry["projection_longitude"],
            "nadir_row": geometry["nadir_row"],
            "nadir_col": 1
        }

    @pytest.fixture
    def mocked_file_handler(self, dataset):
        """Mock file handler to load the given fake dataset."""
        from satpy.readers.goes_imager_nc import FULL_DISC, GOESNCFileHandler
        with mock.patch("satpy.readers.goes_imager_nc.xr") as xr_:
            xr_.open_dataset.return_value = dataset
            GOESNCFileHandler.vis_sectors[(3, 4)] = FULL_DISC
            GOESNCFileHandler.ir_sectors[(3, 4)] = FULL_DISC
            GOESNCFileHandler.yaw_flip_sampling_distance = 1
            return GOESNCFileHandler(
                filename='dummy',
                filename_info={},
                filetype_info={},
            )

    def test_metadata(self, mocked_file_handler, expected):
        """Test dataset metadata."""
        metadata = mocked_file_handler.meta
        self._assert_earth_mask_equal(metadata, expected)
        assert metadata == expected

    def _assert_earth_mask_equal(self, metadata, expected):
        earth_mask_tst = metadata.pop("earth_mask")
        earth_mask_ref = expected.pop("earth_mask")
        xr.testing.assert_allclose(earth_mask_tst, earth_mask_ref)


class GOESNCFileHandlerTest(unittest.TestCase):
    """Test the file handler."""

    longMessage = True

    @mock.patch('satpy.readers.goes_imager_nc.xr')
    def setUp(self, xr_):
        """Set up the tests."""
        from satpy.readers.goes_imager_nc import CALIB_COEFS, GOESNCFileHandler

        self.coefs = CALIB_COEFS['GOES-15']
        self.all_coefs = CALIB_COEFS
        self.channels = sorted(self.coefs.keys())
        self.ir_channels = sorted([ch for ch in self.channels
                                   if not is_vis_channel(ch)])
        self.vis_channels = sorted([ch for ch in self.channels
                                    if is_vis_channel(ch)])

        # Mock file access to return a fake dataset. Choose a medium count value
        # (100) to avoid elements being masked due to invalid
        # radiance/reflectance/BT
        nrows = ncols = 300
        self.counts = 100 * 32 * np.ones((1, nrows, ncols))  # emulate 10-bit
        self.lon = np.zeros((nrows, ncols))  # Dummy
        self.lat = np.repeat(np.linspace(-150, 150, nrows), ncols).reshape(
            nrows, ncols)  # Includes invalid values to be masked

        xr_.open_dataset.return_value = xr.Dataset(
            {'data': xr.DataArray(data=self.counts, dims=('time', 'yc', 'xc')),
             'lon': xr.DataArray(data=self.lon, dims=('yc', 'xc')),
             'lat': xr.DataArray(data=self.lat, dims=('yc', 'xc')),
             'time': xr.DataArray(data=np.array([0], dtype='datetime64[ms]'),
                                  dims=('time',)),
             'bands': xr.DataArray(data=np.array([1]))},
            attrs={'Satellite Sensor': 'G-15'})

        # Instantiate reader using the mocked open_dataset() method
        self.reader = GOESNCFileHandler(filename='dummy', filename_info={},
                                        filetype_info={})

    def test_get_dataset_coords(self):
        """Test whether coordinates returned by get_dataset() are correct."""
        lon = self.reader.get_dataset(key=make_dataid(name='longitude'),
                                      info={})
        lat = self.reader.get_dataset(key=make_dataid(name='latitude'),
                                      info={})
        # ... this only compares the valid (unmasked) elements
        self.assertTrue(np.all(lat.to_masked_array() == self.lat),
                        msg='get_dataset() returns invalid latitude')
        self.assertTrue(np.all(lon.to_masked_array() == self.lon),
                        msg='get_dataset() returns invalid longitude')

    def test_get_dataset_counts(self):
        """Test whether counts returned by get_dataset() are correct."""
        from satpy.readers.goes_imager_nc import ALTITUDE, UNKNOWN_SECTOR

        self.reader.meta.update({'lon0': -75.0,
                                 'lat0': 0.0,
                                 'sector': UNKNOWN_SECTOR,
                                 'nadir_row': 1,
                                 'nadir_col': 2,
                                 'area_def_uni': 'some_area'})
        attrs_exp = {'orbital_parameters': {'projection_longitude': -75.0,
                                            'projection_latitude': 0.0,
                                            'projection_altitude': ALTITUDE,
                                            'yaw_flip': True},
                     'platform_name': 'GOES-15',
                     'sensor': 'goes_imager',
                     'sector': UNKNOWN_SECTOR,
                     'nadir_row': 1,
                     'nadir_col': 2,
                     'area_def_uniform_sampling': 'some_area'}

        for ch in self.channels:
            counts = self.reader.get_dataset(
                key=make_dataid(name=ch, calibration='counts'), info={})
            # ... this only compares the valid (unmasked) elements
            self.assertTrue(np.all(self.counts/32. == counts.to_masked_array()),
                            msg='get_dataset() returns invalid counts for '
                                'channel {}'.format(ch))

            # Check attributes
            self.assertDictEqual(counts.attrs, attrs_exp)

    def test_get_dataset_masks(self):
        """Test whether data and coordinates are masked consistently."""
        # Requires that no element has been masked due to invalid
        # radiance/reflectance/BT (see setUp()).
        lon = self.reader.get_dataset(key=make_dataid(name='longitude'),
                                      info={})
        lon_mask = lon.to_masked_array().mask
        for ch in self.channels:
            for calib in ('counts', 'radiance', 'reflectance',
                          'brightness_temperature'):
                try:
                    data = self.reader.get_dataset(
                        key=make_dataid(name=ch, calibration=calib), info={})
                except ValueError:
                    continue
                data_mask = data.to_masked_array().mask
                self.assertTrue(np.all(data_mask == lon_mask),
                                msg='get_dataset() returns inconsistently '
                                    'masked {} in channel {}'.format(calib, ch))

    def test_get_dataset_invalid(self):
        """Test handling of invalid calibrations."""
        # VIS -> BT
        args = dict(key=make_dataid(name='00_7',
                                    calibration='brightness_temperature'),
                    info={})
        self.assertRaises(ValueError, self.reader.get_dataset, **args)

        # IR -> Reflectance
        args = dict(key=make_dataid(name='10_7',
                                    calibration='reflectance'),
                    info={})
        self.assertRaises(ValueError, self.reader.get_dataset, **args)

        # Unsupported calibration
        with pytest.raises(ValueError):
            args = dict(key=make_dataid(name='10_7',
                                        calibration='invalid'),
                        info={})

    def test_calibrate(self):
        """Test whether the correct calibration methods are called."""
        for ch in self.channels:
            if is_vis_channel(ch):
                calibs = {'radiance': '_viscounts2radiance',
                          'reflectance': '_calibrate_vis'}
            else:
                calibs = {'radiance': '_ircounts2radiance',
                          'brightness_temperature': '_calibrate_ir'}
            for calib, method in calibs.items():
                with mock.patch.object(self.reader, method) as target_func:
                    self.reader.calibrate(counts=self.reader.nc['data'],
                                          calibration=calib, channel=ch)
                    target_func.assert_called()

    def test_get_sector(self):
        """Test sector identification."""
        from satpy.readers.goes_imager_nc import (
            FULL_DISC,
            NORTH_HEMIS_EAST,
            NORTH_HEMIS_WEST,
            SOUTH_HEMIS_EAST,
            SOUTH_HEMIS_WEST,
            UNKNOWN_SECTOR,
        )
        shapes_vis = {
            (10800, 20754): FULL_DISC,
            (7286, 13900): NORTH_HEMIS_EAST,
            (2301, 13840): SOUTH_HEMIS_EAST,
            (5400, 13200): NORTH_HEMIS_WEST,
            (4300, 11090): SOUTH_HEMIS_WEST,
            (123, 456): UNKNOWN_SECTOR
        }
        shapes_ir = {
            (2700, 5200): FULL_DISC,
            (1850, 3450): NORTH_HEMIS_EAST,
            (600, 3500): SOUTH_HEMIS_EAST,
            (1310, 3300): NORTH_HEMIS_WEST,
            (1099, 2800): SOUTH_HEMIS_WEST,
            (123, 456): UNKNOWN_SECTOR
        }
        shapes = shapes_ir.copy()
        shapes.update(shapes_vis)
        for (nlines, ncols), sector_ref in shapes.items():
            if (nlines, ncols) in shapes_vis:
                channel = '00_7'
            else:
                channel = '10_7'
            sector = self.reader._get_sector(channel=channel, nlines=nlines,
                                             ncols=ncols)
            self.assertEqual(sector, sector_ref,
                             msg='Incorrect sector identification')


class TestChannelIdentification:
    """Test identification of channel type."""

    @pytest.mark.parametrize(
        "channel_name,expected",
        [
            ("00_7", True),
            ("10_7", False),
            (1, True),
            (2, False)
        ]
    )
    def test_is_vis_channel(self, channel_name, expected):
        """Test vis channel identification."""
        assert is_vis_channel(channel_name) == expected

    def test_invalid_channel(self):
        """Test handling of invalid channel type."""
        with pytest.raises(ValueError):
            is_vis_channel({"foo": "bar"})
