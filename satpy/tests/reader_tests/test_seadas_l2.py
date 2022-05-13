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
"""Tests for the 'seadas_l2' reader."""

import numpy as np
import pytest
from pyhdf.SD import SD, SDC
from pyresample.geometry import SwathDefinition
from pytest_lazyfixture import lazy_fixture

from satpy import Scene, available_readers


@pytest.fixture(scope="module")
def seadas_l2_modis_chlor_a(tmp_path_factory):
    """Create MODIS SEADAS file."""
    filename = "a1.21322.1758.seadas.hdf"
    full_path = str(tmp_path_factory.mktemp("seadas_l2") / filename)
    return _create_seadas_chlor_a_file(full_path, "Aqua", "MODISA")


@pytest.fixture(scope="module")
def seadas_l2_viirs_npp_chlor_a(tmp_path_factory):
    """Create VIIRS NPP SEADAS file."""
    filename = "SEADAS_npp_d20211118_t1728125_e1739327.hdf"
    full_path = str(tmp_path_factory.mktemp("seadas") / filename)
    return _create_seadas_chlor_a_file(full_path, "NPP", "VIIRSN")


@pytest.fixture(scope="module")
def seadas_l2_viirs_j01_chlor_a(tmp_path_factory):
    """Create VIIRS JPSS-01 SEADAS file."""
    filename = "SEADAS_j01_d20211118_t1728125_e1739327.hdf"
    full_path = str(tmp_path_factory.mktemp("seadas") / filename)
    return _create_seadas_chlor_a_file(full_path, "JPSS-1", "VIIRSJ1")


def _create_seadas_chlor_a_file(full_path, mission, sensor):
    h = SD(full_path, SDC.WRITE | SDC.CREATE)
    setattr(h, "Sensor Name", sensor)
    h.Mission = mission
    setattr(h, "Start Time", "2021322175853191")
    setattr(h, "End Time", "2021322180551214")

    lon_info = {
        "type": SDC.FLOAT32,
        "data": np.zeros((5, 5), dtype=np.float32),
        "dim_labels": ["Number of Scan Lines", "Number of Pixel Control Points"],
        "attrs": {
            "long_name": "Longitude\x00",
            "standard_name": "longitude\x00",
            "units": "degrees_east\x00",
            "valid_range": (-180.0, 180.0),
        }
    }
    lat_info = {
        "type": SDC.FLOAT32,
        "data": np.zeros((5, 5), np.float32),
        "dim_labels": ["Number of Scan Lines", "Number of Pixel Control Points"],
        "attrs": {
            "long_name": "Latitude\x00",
            "standard_name": "latitude\x00",
            "units": "degrees_north\x00",
            "valid_range": (-90.0, 90.0),
        }
    }
    _add_variable_to_file(h, "longitude", lon_info)
    _add_variable_to_file(h, "latitude", lat_info)

    chlor_a_info = {
        "type": SDC.FLOAT32,
        "data": np.ones((5, 5), np.float32),
        "dim_labels": ["Number of Scan Lines", "Number of Pixel Control Points"],
        "attrs": {
            "long_name": "Chlorophyll Concentration, OCI Algorithm\x00",
            "units": "mg m^-3\x00",
            "standard_name": "mass_concentration_of_chlorophyll_in_sea_water\x00",
            "valid_range": (0.001, 100.0),
        }
    }
    _add_variable_to_file(h, "chlor_a", chlor_a_info)

    l2_flags = np.zeros((5, 5), dtype=np.int32)
    l2_flags[2, 2] = -1
    l2_flags_info = {
        "type": SDC.INT32,
        "data": l2_flags,
        "dim_labels": ["Number of Scan Lines", "Number of Pixel Control Points"],
        "attrs": {},
    }
    _add_variable_to_file(h, "l2_flags", l2_flags_info)
    return [full_path]


def _add_variable_to_file(h, var_name, var_info):
    v = h.create(var_name, var_info['type'], var_info['data'].shape)
    v[:] = var_info['data']
    for dim_count, dimension_name in enumerate(var_info['dim_labels']):
        v.dim(dim_count).setname(dimension_name)
    if var_info.get('fill_value'):
        v.setfillvalue(var_info['fill_value'])
    for attr_key, attr_val in var_info['attrs'].items():
        setattr(v, attr_key, attr_val)


class TestSEADAS:
    """Test the SEADAS L2 file reader."""

    def test_available_reader(self):
        """Test that SEADAS L2 reader is available."""
        assert 'seadas_l2' in available_readers()

    @pytest.mark.parametrize(
        "input_files",
        [
            lazy_fixture("seadas_l2_modis_chlor_a"),
            lazy_fixture("seadas_l2_viirs_npp_chlor_a"),
            lazy_fixture("seadas_l2_viirs_j01_chlor_a"),
        ])
    def test_scene_available_datasets(self, input_files):
        """Test that datasets are available."""
        scene = Scene(reader='seadas_l2', filenames=input_files)
        available_datasets = scene.all_dataset_names()
        assert len(available_datasets) > 0
        assert 'chlor_a' in available_datasets

    @pytest.mark.parametrize(
        ("input_files", "exp_plat", "exp_sensor", "exp_rps"),
        [
            (lazy_fixture("seadas_l2_modis_chlor_a"), "Aqua", {"modis"}, 10),
            (lazy_fixture("seadas_l2_viirs_npp_chlor_a"), "Suomi-NPP", {"viirs"}, 16),
            (lazy_fixture("seadas_l2_viirs_j01_chlor_a"), "NOAA-20", {"viirs"}, 16),
        ])
    @pytest.mark.parametrize("apply_quality_flags", [False, True])
    def test_load_chlor_a(self, input_files, exp_plat, exp_sensor, exp_rps, apply_quality_flags):
        """Test that we can load 'chlor_a'."""
        reader_kwargs = {"apply_quality_flags": apply_quality_flags}
        scene = Scene(reader='seadas_l2', filenames=input_files, reader_kwargs=reader_kwargs)
        scene.load(['chlor_a'])
        data_arr = scene['chlor_a']
        assert data_arr.attrs['platform_name'] == exp_plat
        assert data_arr.attrs['sensor'] == exp_sensor
        assert data_arr.attrs['units'] == 'mg m^-3'
        assert data_arr.dtype.type == np.float32
        assert isinstance(data_arr.attrs["area"], SwathDefinition)
        assert data_arr.attrs["rows_per_scan"] == exp_rps
        data = data_arr.data.compute()
        if apply_quality_flags:
            assert np.isnan(data[2, 2])
            assert np.count_nonzero(np.isnan(data)) == 1
        else:
            assert np.count_nonzero(np.isnan(data)) == 0
