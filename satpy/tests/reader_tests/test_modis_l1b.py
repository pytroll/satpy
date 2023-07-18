#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Unit tests for MODIS L1b HDF reader."""

from __future__ import annotations

import dask
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from satpy import Scene, available_readers

from ..utils import CustomScheduler, make_dataid
from ._modis_fixtures import (
    AVAILABLE_1KM_PRODUCT_NAMES,
    AVAILABLE_HKM_PRODUCT_NAMES,
    AVAILABLE_QKM_PRODUCT_NAMES,
    _shape_for_resolution,
)

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - modis_l1b_imapp_1000m_file
# - modis_l1b_imapp_geo_file,
# - modis_l1b_nasa_1km_mod03_files,
# - modis_l1b_nasa_mod02hkm_file,
# - modis_l1b_nasa_mod02qkm_file,
# - modis_l1b_nasa_mod03_file,
# - modis_l1b_nasa_mod021km_file


def _check_shared_metadata(data_arr):
    assert data_arr.attrs["sensor"] == "modis"
    assert data_arr.attrs["platform_name"] == "EOS-Terra"
    assert "rows_per_scan" in data_arr.attrs
    assert isinstance(data_arr.attrs["rows_per_scan"], int)
    assert data_arr.attrs['reader'] == 'modis_l1b'


def _load_and_check_geolocation(scene, resolution, exp_res, exp_shape, has_res,
                                check_callback=_check_shared_metadata):
    scene.load(["longitude", "latitude"], resolution=resolution)
    lon_id = make_dataid(name="longitude", resolution=exp_res)
    lat_id = make_dataid(name="latitude", resolution=exp_res)
    if has_res:
        lon_arr = scene[lon_id]
        lat_arr = scene[lat_id]
        assert lon_arr.shape == exp_shape
        assert lat_arr.shape == exp_shape
        # compute lon/lat at the same time to avoid wasted computation
        lon_vals, lat_vals = dask.compute(lon_arr, lat_arr)
        np.testing.assert_array_less(lon_vals, 0)
        np.testing.assert_array_less(0, lat_vals)
        check_callback(lon_arr)
        check_callback(lat_arr)
    else:
        pytest.raises(KeyError, scene.__getitem__, lon_id)
        pytest.raises(KeyError, scene.__getitem__, lat_id)


class TestModisL1b:
    """Test MODIS L1b reader."""

    def test_available_reader(self):
        """Test that MODIS L1b reader is available."""
        assert 'modis_l1b' in available_readers()

    @pytest.mark.parametrize(
        ('input_files', 'expected_names', 'expected_data_res', 'expected_geo_res'),
        [
            [lazy_fixture('modis_l1b_nasa_mod021km_file'),
             AVAILABLE_1KM_PRODUCT_NAMES + AVAILABLE_HKM_PRODUCT_NAMES + AVAILABLE_QKM_PRODUCT_NAMES,
             [1000], [5000, 1000]],
            [lazy_fixture('modis_l1b_imapp_1000m_file'),
             AVAILABLE_1KM_PRODUCT_NAMES + AVAILABLE_HKM_PRODUCT_NAMES + AVAILABLE_QKM_PRODUCT_NAMES,
             [1000], [5000, 1000]],
            [lazy_fixture('modis_l1b_nasa_mod02hkm_file'),
             AVAILABLE_HKM_PRODUCT_NAMES + AVAILABLE_QKM_PRODUCT_NAMES, [500], [1000, 500, 250]],
            [lazy_fixture('modis_l1b_nasa_mod02qkm_file'),
             AVAILABLE_QKM_PRODUCT_NAMES, [250], [1000, 500, 250]],
        ]
    )
    def test_scene_available_datasets(self, input_files, expected_names, expected_data_res, expected_geo_res):
        """Test that datasets are available."""
        scene = Scene(reader='modis_l1b', filenames=input_files)
        available_datasets = scene.available_dataset_names()
        assert len(available_datasets) > 0
        assert 'longitude' in available_datasets
        assert 'latitude' in available_datasets
        for chan_name in expected_names:
            assert chan_name in available_datasets

        available_data_ids = scene.available_dataset_ids()
        available_datas = {x: [] for x in expected_data_res}
        available_geos = {x: [] for x in expected_geo_res}
        # Make sure that every resolution from the reader is what we expect
        for data_id in available_data_ids:
            res = data_id['resolution']
            if data_id['name'] in ['longitude', 'latitude']:
                assert res in expected_geo_res
                available_geos[res].append(data_id)
            else:
                assert res in expected_data_res
                available_datas[res].append(data_id)

        # Make sure that every resolution we expect has at least one dataset
        for exp_res, avail_id in available_datas.items():
            assert avail_id, f"Missing datasets for data resolution {exp_res}"
        for exp_res, avail_id in available_geos.items():
            assert avail_id, f"Missing geo datasets for geo resolution {exp_res}"

    @pytest.mark.parametrize(
        ('input_files', 'has_5km', 'has_500', 'has_250', 'default_res'),
        [
            [lazy_fixture('modis_l1b_nasa_mod021km_file'),
             True, False, False, 1000],
            [lazy_fixture('modis_l1b_imapp_1000m_file'),
             True, False, False, 1000],
            [lazy_fixture('modis_l1b_nasa_mod02hkm_file'),
             False, True, True, 250],
            [lazy_fixture('modis_l1b_nasa_mod02qkm_file'),
             False, True, True, 250],
            [lazy_fixture('modis_l1b_nasa_1km_mod03_files'),
             True, True, True, 250],
        ]
    )
    def test_load_longitude_latitude(self, input_files, has_5km, has_500, has_250, default_res):
        """Test that longitude and latitude datasets are loaded correctly."""
        scene = Scene(reader='modis_l1b', filenames=input_files)
        shape_5km = _shape_for_resolution(5000)
        shape_500m = _shape_for_resolution(500)
        shape_250m = _shape_for_resolution(250)
        default_shape = _shape_for_resolution(default_res)
        with dask.config.set(scheduler=CustomScheduler(max_computes=1 + has_5km + has_500 + has_250)):
            _load_and_check_geolocation(scene, "*", default_res, default_shape, True)
            _load_and_check_geolocation(scene, 5000, 5000, shape_5km, has_5km)
            _load_and_check_geolocation(scene, 500, 500, shape_500m, has_500)
            _load_and_check_geolocation(scene, 250, 250, shape_250m, has_250)

    def test_load_sat_zenith_angle(self, modis_l1b_nasa_mod021km_file):
        """Test loading satellite zenith angle band."""
        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file)
        dataset_name = 'satellite_zenith_angle'
        scene.load([dataset_name])
        dataset = scene[dataset_name]
        assert dataset.shape == _shape_for_resolution(1000)
        assert dataset.attrs['resolution'] == 1000
        _check_shared_metadata(dataset)

    def test_load_vis(self, modis_l1b_nasa_mod021km_file):
        """Test loading visible band."""
        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file)
        dataset_name = '1'
        scene.load([dataset_name])
        dataset = scene[dataset_name]
        assert dataset[0, 0] == 300.0
        assert dataset.shape == _shape_for_resolution(1000)
        assert dataset.attrs['resolution'] == 1000
        _check_shared_metadata(dataset)

    @pytest.mark.parametrize("mask_saturated", [False, True])
    def test_load_vis_saturation(self, mask_saturated, modis_l1b_nasa_mod021km_file):
        """Test loading visible band."""
        scene = Scene(reader='modis_l1b', filenames=modis_l1b_nasa_mod021km_file,
                      reader_kwargs={"mask_saturated": mask_saturated})
        dataset_name = '2'
        scene.load([dataset_name])
        dataset = scene[dataset_name]
        assert dataset.shape == _shape_for_resolution(1000)
        assert dataset.attrs['resolution'] == 1000
        _check_shared_metadata(dataset)

        # check saturation fill values
        data = dataset.values
        assert dataset[0, 0] == 300.0
        assert np.isnan(data[-1, -1])  # normal fill value
        if mask_saturated:
            assert np.isnan(data[-1, -2])  # saturation
            assert np.isnan(data[-1, -3])  # can't aggregate
        else:
            # test data factor/offset are 1/0
            # albedos are converted to %
            assert data[-1, -2] >= 32767 * 100.0  # saturation
            assert data[-1, -3] >= 32767 * 100.0  # can't aggregate
