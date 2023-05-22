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
"""Unit tests for MODIS L2 HDF reader."""

from __future__ import annotations

import dask
import dask.array as da
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from satpy import Scene, available_readers

from ..utils import CustomScheduler, make_dataid
from ._modis_fixtures import _shape_for_resolution

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - modis_l2_imapp_mask_byte1_file
# - modis_l2_imapp_mask_byte1_geo_files
# - modis_l2_imapp_snowmask_file
# - modis_l2_imapp_snowmask_geo_files
# - modis_l2_nasa_mod06_file
# - modis_l2_nasa_mod35_file
# - modis_l2_nasa_mod35_mod03_files


def _check_shared_metadata(data_arr, expect_area=False):
    assert data_arr.attrs["sensor"] == "modis"
    assert data_arr.attrs["platform_name"] == "EOS-Terra"
    assert "rows_per_scan" in data_arr.attrs
    assert isinstance(data_arr.attrs["rows_per_scan"], int)
    assert data_arr.attrs['reader'] == 'modis_l2'
    if expect_area:
        assert data_arr.attrs.get('area') is not None
    else:
        assert 'area' not in data_arr.attrs


class TestModisL2:
    """Test MODIS L2 reader."""

    def test_available_reader(self):
        """Test that MODIS L2 reader is available."""
        assert 'modis_l2' in available_readers()

    def test_scene_available_datasets(self, modis_l2_nasa_mod35_file):
        """Test that datasets are available."""
        scene = Scene(reader='modis_l2', filenames=modis_l2_nasa_mod35_file)
        available_datasets = scene.all_dataset_names()
        assert len(available_datasets) > 0
        assert 'cloud_mask' in available_datasets
        assert 'latitude' in available_datasets
        assert 'longitude' in available_datasets

    @pytest.mark.parametrize(
        ('input_files', 'has_5km', 'has_500', 'has_250', 'default_res'),
        [
            [lazy_fixture('modis_l2_nasa_mod35_file'),
             True, False, False, 1000],
        ]
    )
    def test_load_longitude_latitude(self, input_files, has_5km, has_500, has_250, default_res):
        """Test that longitude and latitude datasets are loaded correctly."""
        from .test_modis_l1b import _load_and_check_geolocation
        scene = Scene(reader='modis_l2', filenames=input_files)
        shape_5km = _shape_for_resolution(5000)
        shape_500m = _shape_for_resolution(500)
        shape_250m = _shape_for_resolution(250)
        default_shape = _shape_for_resolution(default_res)
        with dask.config.set(scheduler=CustomScheduler(max_computes=1 + has_5km + has_500 + has_250)):
            _load_and_check_geolocation(scene, "*", default_res, default_shape, True,
                                        check_callback=_check_shared_metadata)
            _load_and_check_geolocation(scene, 5000, 5000, shape_5km, has_5km,
                                        check_callback=_check_shared_metadata)
            _load_and_check_geolocation(scene, 500, 500, shape_500m, has_500,
                                        check_callback=_check_shared_metadata)
            _load_and_check_geolocation(scene, 250, 250, shape_250m, has_250,
                                        check_callback=_check_shared_metadata)

    def test_load_quality_assurance(self, modis_l2_nasa_mod35_file):
        """Test loading quality assurance."""
        scene = Scene(reader='modis_l2', filenames=modis_l2_nasa_mod35_file)
        dataset_name = 'quality_assurance'
        scene.load([dataset_name])
        quality_assurance_id = make_dataid(name=dataset_name, resolution=1000)
        assert quality_assurance_id in scene
        quality_assurance = scene[quality_assurance_id]
        assert quality_assurance.shape == _shape_for_resolution(1000)
        _check_shared_metadata(quality_assurance, expect_area=True)

    @pytest.mark.parametrize(
        ('input_files', 'loadables', 'request_resolution', 'exp_resolution', 'exp_area'),
        [
            [lazy_fixture('modis_l2_nasa_mod35_mod03_files'),
             ["cloud_mask"],
             1000, 1000, True],
            [lazy_fixture('modis_l2_imapp_mask_byte1_geo_files'),
             ["cloud_mask", "land_sea_mask", "snow_ice_mask"],
             None, 1000, True],
        ]
    )
    def test_load_category_dataset(self, input_files, loadables, request_resolution, exp_resolution, exp_area):
        """Test loading category products."""
        scene = Scene(reader='modis_l2', filenames=input_files)
        kwargs = {"resolution": request_resolution} if request_resolution is not None else {}
        scene.load(loadables, **kwargs)
        for ds_name in loadables:
            cat_id = make_dataid(name=ds_name, resolution=exp_resolution)
            assert cat_id in scene
            cat_data_arr = scene[cat_id]
            assert isinstance(cat_data_arr.data, da.Array)
            cat_data_arr = cat_data_arr.compute()
            assert cat_data_arr.shape == _shape_for_resolution(exp_resolution)
            assert cat_data_arr.values[0, 0] == 0.0
            assert cat_data_arr.attrs.get('resolution') == exp_resolution
            # mask variables should be integers
            assert np.issubdtype(cat_data_arr.dtype, np.integer)
            assert cat_data_arr.attrs.get('_FillValue') is not None
            _check_shared_metadata(cat_data_arr, expect_area=exp_area)

    @pytest.mark.parametrize(
        ('input_files', 'exp_area'),
        [
            [lazy_fixture('modis_l2_nasa_mod35_file'), False],
            [lazy_fixture('modis_l2_nasa_mod35_mod03_files'), True],
        ]
    )
    def test_load_250m_cloud_mask_dataset(self, input_files, exp_area):
        """Test loading 250m cloud mask."""
        scene = Scene(reader='modis_l2', filenames=input_files)
        dataset_name = 'cloud_mask'
        scene.load([dataset_name], resolution=250)
        cloud_mask_id = make_dataid(name=dataset_name, resolution=250)
        assert cloud_mask_id in scene
        cloud_mask = scene[cloud_mask_id]
        assert isinstance(cloud_mask.data, da.Array)
        cloud_mask = cloud_mask.compute()
        assert cloud_mask.shape == _shape_for_resolution(250)
        assert cloud_mask.values[0, 0] == 0.0
        # mask variables should be integers
        assert np.issubdtype(cloud_mask.dtype, np.integer)
        assert cloud_mask.attrs.get('_FillValue') is not None
        _check_shared_metadata(cloud_mask, expect_area=exp_area)

    @pytest.mark.parametrize(
        ('input_files', 'loadables', 'exp_resolution', 'exp_area', 'exp_value'),
        [
            [lazy_fixture('modis_l2_nasa_mod06_file'), ["surface_pressure"], 5000, True, 4.0],
            # snow mask is considered a category product, factor/offset ignored
            [lazy_fixture('modis_l2_imapp_snowmask_file'), ["snow_mask"], 1000, False, 1.0],
            [lazy_fixture('modis_l2_imapp_snowmask_geo_files'), ["snow_mask"], 1000, True, 1.0],
        ]
    )
    def test_load_l2_dataset(self, input_files, loadables, exp_resolution, exp_area, exp_value):
        """Load and check an L2 variable."""
        scene = Scene(reader='modis_l2', filenames=input_files)
        scene.load(loadables)
        for ds_name in loadables:
            assert ds_name in scene
            data_arr = scene[ds_name]
            assert isinstance(data_arr.data, da.Array)
            data_arr = data_arr.compute()
            assert data_arr.values[0, 0] == exp_value
            assert data_arr.shape == _shape_for_resolution(exp_resolution)
            assert data_arr.attrs.get('resolution') == exp_resolution
            _check_shared_metadata(data_arr, expect_area=exp_area)
