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
"""The glm_l2 reader tests package."""

import os
import unittest
from unittest import mock

import numpy as np
import xarray as xr


def setup_fake_dataset():
    """Create a fake dataset to avoid opening a file."""
    # flash_extent_density
    fed = (np.arange(10.).reshape((2, 5)) + 1.) * 50.
    fed = (fed + 1.) / 0.5
    fed = fed.astype(np.int16)
    fed = xr.DataArray(
        fed,
        dims=('y', 'x'),
        attrs={
            'scale_factor': 0.5,
            'add_offset': -1.,
            '_FillValue': 0,
            'units': 'Count per nominal    3136 microradian^2 pixel per 1.0 min',
            'grid_mapping': 'goes_imager_projection',
            'standard_name': 'flash_extent_density',
            'long_name': 'Flash extent density',
        }
    )
    dqf = xr.DataArray(
        fed.data.copy().astype(np.uint8),
        dims=('y', 'x'),
        attrs={
            '_FillValue': -1,
            'units': '1',
            'grid_mapping': 'goes_imager_projection',
            'standard_name': 'status_flag',
            'long_name': 'GLM data quality flags',
            'flag_meanings': "valid invalid",
        }
    )
    # create a variable that won't be configured to test available_datasets
    not_configured = xr.DataArray(
        fed.data.copy(),
        dims=('y', 'x'),
        attrs={
            'scale_factor': 0.5,
            'add_offset': -1.,
            '_FillValue': 0,
            'units': '1',
            'grid_mapping': 'goes_imager_projection',
            'standard_name': 'test',
            'long_name': 'Test',
        }
    )
    x__ = xr.DataArray(
        range(5),
        attrs={'scale_factor': 2., 'add_offset': -1.},
        dims=('x',),
    )
    y__ = xr.DataArray(
        range(2),
        attrs={'scale_factor': -2., 'add_offset': 1.},
        dims=('y',),
    )
    proj = xr.DataArray(
        [],
        attrs={
            'semi_major_axis': 1.,
            'semi_minor_axis': 1.,
            'perspective_point_height': 1.,
            'longitude_of_projection_origin': -90.,
            'latitude_of_projection_origin': 0.,
            'sweep_angle_axis': u'x'
        }
    )
    fake_dataset = xr.Dataset(
        data_vars={
            'flash_extent_density': fed,
            'not_configured': not_configured,
            'DQF': dqf,
            'x': x__,
            'y': y__,
            'goes_imager_projection': proj,
            "nominal_satellite_subpoint_lat": np.array(0.0),
            "nominal_satellite_subpoint_lon": np.array(-89.5),
            "nominal_satellite_height": np.array(35786.02)
        },
        attrs={
            "time_coverage_start": "2017-09-20T17:30:40Z",
            "time_coverage_end": "2017-09-20T17:41:17Z",
            "spatial_resolution": "2km at nadir",
        }
    )
    return fake_dataset


class TestGLML2FileHandler(unittest.TestCase):
    """Tests for the GLM L2 reader."""

    @mock.patch('satpy.readers.abi_base.xr')
    def setUp(self, xr_):
        """Create a fake file handler to test."""
        from satpy.readers.glm_l2 import NCGriddedGLML2
        fake_dataset = setup_fake_dataset()
        xr_.open_dataset.return_value = fake_dataset
        self.reader = NCGriddedGLML2('filename',
                                     {'platform_shortname': 'G16',
                                      'scene_abbr': 'C', 'scan_mode': 'M3'},
                                     {'filetype': 'glm_l2_imagery'})

    def test_basic_attributes(self):
        """Test getting basic file attributes."""
        from datetime import datetime
        self.assertEqual(self.reader.start_time,
                         datetime(2017, 9, 20, 17, 30, 40))
        self.assertEqual(self.reader.end_time,
                         datetime(2017, 9, 20, 17, 41, 17))

    def test_get_dataset(self):
        """Test the get_dataset method."""
        from satpy.tests.utils import make_dataid
        key = make_dataid(name='flash_extent_density')
        res = self.reader.get_dataset(key, {'info': 'info'})
        exp = {'instrument_ID': None,
               'modifiers': (),
               'name': 'flash_extent_density',
               'orbital_parameters': {'projection_altitude': 1.0,
                                      'projection_latitude': 0.0,
                                      'projection_longitude': -90.0,
                                      # 'satellite_nominal_altitude': 35786.02,
                                      'satellite_nominal_latitude': 0.0,
                                      'satellite_nominal_longitude': -89.5},
               'orbital_slot': None,
               'platform_name': 'GOES-16',
               'platform_shortname': 'G16',
               'production_site': None,
               'scan_mode': 'M3',
               'scene_abbr': 'C',
               'scene_id': None,
               "spatial_resolution": "2km at nadir",
               'sensor': 'glm',
               'timeline_ID': None,
               'grid_mapping': 'goes_imager_projection',
               'standard_name': 'flash_extent_density',
               'long_name': 'Flash extent density',
               'units': 'Count per nominal    3136 microradian^2 pixel per 1.0 min'}

        self.assertDictEqual(res.attrs, exp)

    def test_get_dataset_dqf(self):
        """Test the get_dataset method with special DQF var."""
        from satpy.tests.utils import make_dataid
        key = make_dataid(name='DQF')
        res = self.reader.get_dataset(key, {'info': 'info'})
        exp = {'instrument_ID': None,
               'modifiers': (),
               'name': 'DQF',
               'orbital_parameters': {'projection_altitude': 1.0,
                                      'projection_latitude': 0.0,
                                      'projection_longitude': -90.0,
                                      # 'satellite_nominal_altitude': 35786.02,
                                      'satellite_nominal_latitude': 0.0,
                                      'satellite_nominal_longitude': -89.5},
               'orbital_slot': None,
               'platform_name': 'GOES-16',
               'platform_shortname': 'G16',
               'production_site': None,
               'scan_mode': 'M3',
               'scene_abbr': 'C',
               'scene_id': None,
               "spatial_resolution": "2km at nadir",
               'sensor': 'glm',
               'timeline_ID': None,
               'grid_mapping': 'goes_imager_projection',
               'units': '1',
               '_FillValue': -1,
               'standard_name': 'status_flag',
               'long_name': 'GLM data quality flags',
               'flag_meanings': "valid invalid"}

        self.assertDictEqual(res.attrs, exp)
        self.assertTrue(np.issubdtype(res.dtype, np.integer))


class TestGLML2Reader(unittest.TestCase):
    """Test high-level reading functionality of GLM L2 reader."""

    yaml_file = "glm_l2.yaml"

    @mock.patch('satpy.readers.abi_base.xr')
    def setUp(self, xr_):
        """Create a fake reader to test."""
        from satpy._config import config_search_paths
        from satpy.readers import load_reader
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        fake_dataset = setup_fake_dataset()
        xr_.open_dataset.return_value = fake_dataset
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OR_GLM-L2-GLMC-M3_G16_s20192862159000_e20192862200000_c20192862200350.nc',
            'CSPP_CG_GLM-L2-GLMC-M3_G16_s20192862159000_e20192862200000_c20192862200350.nc',
        ])
        self.assertEqual(len(loadables), 2)
        r.create_filehandlers(loadables)
        self.reader = r

    def test_available_datasets(self):
        """Test that resolution is added to YAML configured variables."""
        # make sure we have some files
        self.assertTrue(self.reader.file_handlers)
        available_datasets = list(self.reader.available_dataset_ids)
        # flash_extent_density, DQF, and not_configured are available in our tests
        self.assertEqual(len(available_datasets), 3)
        for ds_id in available_datasets:
            self.assertEqual(ds_id['resolution'], 2000)
        # make sure not_configured was discovered
        names = [dataid['name'] for dataid in available_datasets]
        assert 'not_configured' in names
