# -*- coding: utf-8 -*-

# Copyright (c) 2019 Satpy developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""The abi_l2_nc reader tests package."""

import unittest
from unittest import mock

import numpy as np
import xarray as xr


def _create_cmip_dataset():
    proj = xr.DataArray(
        [],
        attrs={
            'semi_major_axis': 1.,
            'semi_minor_axis': 1.,
            'perspective_point_height': 1.,
            'longitude_of_projection_origin': -90.,
            'sweep_angle_axis': u'x'
        }
    )
    x__ = xr.DataArray(
        [0, 1],
        attrs={'scale_factor': 2., 'add_offset': -1.},
        dims=('x',),
    )
    y__ = xr.DataArray(
        [0, 1],
        attrs={'scale_factor': -2., 'add_offset': 1.},
        dims=('y',),
    )

    ht_da = xr.DataArray(np.array([2, -1, -32768, 32767]).astype(np.int16).reshape((2, 2)),
                         dims=('y', 'x'),
                         attrs={'scale_factor': 0.3052037,
                                'add_offset': 0.,
                                '_FillValue': np.array(-1).astype(np.int16),
                                '_Unsigned': 'True',
                                'units': 'm'},)

    fake_dataset = xr.Dataset(
        data_vars={
            'goes_imager_projection': proj,
            'x': x__,
            'y': y__,
            'HT': ht_da,
            "nominal_satellite_subpoint_lat": np.array(0.0),
            "nominal_satellite_subpoint_lon": np.array(-89.5),
            "nominal_satellite_height": np.array(35786.02),
            "spatial_resolution": "10km at nadir",

        },
        attrs={
            "time_coverage_start": "2017-09-20T17:30:40.8Z",
            "time_coverage_end": "2017-09-20T17:41:17.5Z",
            "spatial_resolution": "2km at nadir",
        }
    )
    return fake_dataset


def _compare_subdict(actual_dict, exp_sub_dict):
    for key, value in exp_sub_dict.items():
        assert key in actual_dict
        assert actual_dict[key] == value


def _assert_orbital_parameters(orb_params):
    assert orb_params['satellite_nominal_longitude'] == -89.5
    assert orb_params['satellite_nominal_latitude'] == 0.0
    assert orb_params['satellite_nominal_altitude'] == 35786020.0


def _create_mcmip_dataset():
    fake_dataset = _create_cmip_dataset()
    fake_dataset = fake_dataset.copy(deep=True)
    fake_dataset['CMI_C14'] = fake_dataset['HT']
    del fake_dataset['HT']
    return fake_dataset


class Test_NC_ABI_L2_base(unittest.TestCase):
    """Test the NC_ABI_L2 reader."""

    def setUp(self):
        """Create fake data for the tests."""
        from satpy.readers.abi_l2_nc import NC_ABI_L2
        fake_cmip_dataset = _create_cmip_dataset()
        with mock.patch('satpy.readers.abi_base.xr') as xr_:
            xr_.open_dataset.return_value = fake_cmip_dataset
            self.reader = NC_ABI_L2(
                'filename',
                {
                    'platform_shortname': 'G16',
                    'scan_mode': 'M3',
                    'scene_abbr': 'M1',
                },
                {
                    'file_type': 'info',
                    'observation_type': 'ACHA',
                },
            )


class Test_NC_ABI_L2_get_dataset(Test_NC_ABI_L2_base):
    """Test get dataset function of the NC_ABI_L2 reader."""

    def test_get_dataset(self):
        """Test basic L2 load."""
        from satpy.tests.utils import make_dataid
        key = make_dataid(name='HT')
        res = self.reader.get_dataset(key, {'file_key': 'HT'})

        exp_data = np.array([[2 * 0.3052037, np.nan],
                             [32768 * 0.3052037, 32767 * 0.3052037]])

        exp_attrs = {'instrument_ID': None,
                     'modifiers': (),
                     'name': 'HT',
                     'observation_type': 'ACHA',
                     'orbital_slot': None,
                     'platform_name': 'GOES-16',
                     'platform_shortname': 'G16',
                     'production_site': None,
                     'scan_mode': 'M3',
                     'scene_abbr': 'M1',
                     'scene_id': None,
                     'sensor': 'abi',
                     'timeline_ID': None,
                     'units': 'm'}

        self.assertTrue(np.allclose(res.data, exp_data, equal_nan=True))
        _compare_subdict(res.attrs, exp_attrs)
        _assert_orbital_parameters(res.attrs['orbital_parameters'])


class TestMCMIPReading:
    """Test cases of the MCMIP file format."""

    @mock.patch('satpy.readers.abi_base.xr')
    def test_mcmip_get_dataset(self, xr_):
        """Test getting channel from MCMIP file."""
        from datetime import datetime

        from pyresample.geometry import AreaDefinition

        from satpy import Scene
        from satpy.dataset.dataid import WavelengthRange
        xr_.open_dataset.return_value = _create_mcmip_dataset()

        fn = "OR_ABI-L2-MCMIPF-M6_G16_s20192600241149_e20192600243534_c20192600245360.nc"
        scn = Scene(reader='abi_l2_nc', filenames=[fn])
        scn.load(['C14'])

        exp_data = np.array([[2 * 0.3052037, np.nan],
                             [32768 * 0.3052037, 32767 * 0.3052037]])

        exp_attrs = {'instrument_ID': None,
                     'modifiers': (),
                     'name': 'C14',
                     'observation_type': 'MCMIP',
                     'orbital_slot': None,
                     'reader': 'abi_l2_nc',
                     'platform_name': 'GOES-16',
                     'platform_shortname': 'G16',
                     'production_site': None,
                     'scan_mode': 'M6',
                     'scene_abbr': 'F',
                     'scene_id': None,
                     'sensor': 'abi',
                     'timeline_ID': None,
                     'start_time': datetime(2017, 9, 20, 17, 30, 40, 800000),
                     'end_time': datetime(2017, 9, 20, 17, 41, 17, 500000),
                     'calibration': 'brightness_temperature',
                     'ancillary_variables': [],
                     'wavelength': WavelengthRange(10.8, 11.2, 11.6, unit='µm'),
                     'units': 'm'}

        res = scn['C14']
        np.testing.assert_allclose(res.data, exp_data, equal_nan=True)
        assert isinstance(res.attrs['area'], AreaDefinition)
        _compare_subdict(res.attrs, exp_attrs)
        _assert_orbital_parameters(res.attrs['orbital_parameters'])


class Test_NC_ABI_L2_area_fixedgrid(Test_NC_ABI_L2_base):
    """Test the NC_ABI_L2 reader."""

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_fixedgrid(self, adef):
        """Test the area generation."""
        self.reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {'a': 1.0, 'b': 1.0, 'h': 1.0, 'lon_0': -90.0,
                                            'proj': 'geos', 'sweep': 'x', 'units': 'm'})
        self.assertEqual(call_args[4], self.reader.ncols)
        self.assertEqual(call_args[5], self.reader.nlines)
        np.testing.assert_allclose(call_args[6], (-2., -2.,  2.,  2.))


class Test_NC_ABI_L2_area_latlon(unittest.TestCase):
    """Test the NC_ABI_L2 reader."""

    @mock.patch('satpy.readers.abi_base.xr')
    def setUp(self, xr_):
        """Create fake data for the tests."""
        from satpy.readers.abi_l2_nc import NC_ABI_L2
        proj = xr.DataArray(
            [],
            attrs={'semi_major_axis': 1.,
                   'semi_minor_axis': 1.,
                   'inverse_flattening': 1.,
                   'longitude_of_prime_meridian': 0.0,
                   }
        )

        proj_ext = xr.DataArray(
            [],
            attrs={'geospatial_westbound_longitude': -85.0,
                   'geospatial_eastbound_longitude': -65.0,
                   'geospatial_northbound_latitude': 20.0,
                   'geospatial_southbound_latitude': -20.0,
                   'geospatial_lat_center': 0.0,
                   'geospatial_lon_center': -75.0,
                   })

        x__ = xr.DataArray(
            [0, 1],
            attrs={'scale_factor': 2., 'add_offset': -1.},
            dims=('lon',),
        )
        y__ = xr.DataArray(
            [0, 1],
            attrs={'scale_factor': -2., 'add_offset': 1.},
            dims=('lat',),
        )
        fake_dataset = xr.Dataset(
            data_vars={
                'goes_lat_lon_projection': proj,
                'geospatial_lat_lon_extent': proj_ext,
                'lon': x__,
                'lat': y__,
                'RSR': xr.DataArray(np.ones((2, 2)), dims=('lat', 'lon')),
            },
        )
        xr_.open_dataset.return_value = fake_dataset

        self.reader = NC_ABI_L2('filename',
                                {'platform_shortname': 'G16', 'observation_type': 'RSR',
                                 'scene_abbr': 'C', 'scan_mode': 'M3'},
                                {'filetype': 'info'})

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_latlon(self, adef):
        """Test the area generation."""
        self.reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {'proj': 'latlong', 'a': 1.0, 'b': 1.0, 'fi': 1.0, 'pm': 0.0,
                                            'lon_0': -75.0, 'lat_0': 0.0})
        self.assertEqual(call_args[4], self.reader.ncols)
        self.assertEqual(call_args[5], self.reader.nlines)
        np.testing.assert_allclose(call_args[6], (-85.0, -20.0, -65.0, 20))


class Test_NC_ABI_L2_area_AOD(unittest.TestCase):
    """Test the NC_ABI_L2 reader for the AOD product."""

    @mock.patch('satpy.readers.abi_base.xr')
    def setUp(self, xr_):
        """Create fake data for the tests."""
        from satpy.readers.abi_l2_nc import NC_ABI_L2
        proj = xr.DataArray(
            [],
            attrs={'semi_major_axis': 1.,
                   'semi_minor_axis': 1.,
                   'inverse_flattening': 1.,
                   'longitude_of_prime_meridian': 0.0,
                   }
        )

        proj_ext = xr.DataArray(
            [],
            attrs={'geospatial_westbound_longitude': -85.0,
                   'geospatial_eastbound_longitude': -65.0,
                   'geospatial_northbound_latitude': 20.0,
                   'geospatial_southbound_latitude': -20.0,
                   'geospatial_lat_center': 0.0,
                   'geospatial_lon_center': -75.0,
                   })

        x__ = xr.DataArray(
            [0, 1],
            attrs={'scale_factor': 2., 'add_offset': -1.},
            dims=('x',),
        )
        y__ = xr.DataArray(
            [0, 1],
            attrs={'scale_factor': -2., 'add_offset': 1.},
            dims=('y',),
        )
        fake_dataset = xr.Dataset(
            data_vars={
                'goes_lat_lon_projection': proj,
                'geospatial_lat_lon_extent': proj_ext,
                'x': x__,
                'y': y__,
                'RSR': xr.DataArray(np.ones((2, 2)), dims=('y', 'x')),
            },
        )
        xr_.open_dataset.return_value = fake_dataset

        self.reader = NC_ABI_L2('filename',
                                {'platform_shortname': 'G16', 'observation_type': 'RSR',
                                 'scene_abbr': 'C', 'scan_mode': 'M3'},
                                {'filetype': 'info'})

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_xy(self, adef):
        """Test the area generation."""
        self.reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {'proj': 'latlong', 'a': 1.0, 'b': 1.0, 'fi': 1.0, 'pm': 0.0,
                                            'lon_0': -75.0, 'lat_0': 0.0})
        self.assertEqual(call_args[4], self.reader.ncols)
        self.assertEqual(call_args[5], self.reader.nlines)
        np.testing.assert_allclose(call_args[6], (-85.0, -20.0, -65.0, 20))
