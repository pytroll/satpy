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
import pytest
from pytest_lazyfixture import lazy_fixture

from satpy import available_readers, Scene
from ._modis_fixtures import _shape_for_resolution
from ..utils import CustomScheduler, make_dataid


def _check_shared_metadata(data_arr):
    assert data_arr.attrs["sensor"] == "modis"
    assert data_arr.attrs["platform_name"] == "EOS-Terra"
    assert "rows_per_scan" in data_arr.attrs
    assert isinstance(data_arr.attrs["rows_per_scan"], int)
    assert data_arr.attrs['reader'] == 'modis_l2'


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
        _check_shared_metadata(quality_assurance)

    def test_load_1000m_cloud_mask_dataset(self, modis_l2_nasa_mod35_file):
        """Test loading 1000m cloud mask."""
        scene = Scene(reader='modis_l2', filenames=modis_l2_nasa_mod35_file)
        dataset_name = 'cloud_mask'
        scene.load([dataset_name], resolution=1000)
        cloud_mask_id = make_dataid(name=dataset_name, resolution=1000)
        assert cloud_mask_id in scene
        cloud_mask = scene[cloud_mask_id]
        assert cloud_mask.shape == _shape_for_resolution(1000)
        _check_shared_metadata(cloud_mask)

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
        assert cloud_mask.shape == _shape_for_resolution(250)
        _check_shared_metadata(cloud_mask)
        if exp_area:
            assert cloud_mask.attrs.get('area') is not None
        else:
            assert 'area' not in cloud_mask.attrs
