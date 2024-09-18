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
from pytest_lazy_fixtures import lf as lazy_fixture

from satpy import Scene, available_readers
from satpy.tests.utils import CustomScheduler, make_dataid

from ._modis_fixtures import _shape_for_resolution, modis_l3_nasa_mcd12q1_file

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
    assert data_arr.attrs["reader"] == "mcd12q1"
    if expect_area:
        assert data_arr.attrs.get("area") is not None
    else:
        assert "area" not in data_arr.attrs


class TestModisL3MCD12Q1:
    """Test MODIS L3 MCD12Q1 reader."""

    def test_available_reader(self):
        """Test that MODIS L3 reader is available."""
        assert "mcd12q1" in available_readers()

    def test_scene_available_datasets(self, modis_l3_nasa_mcd12q1_file):
        """Test that datasets are available."""
        scene = Scene(reader="mcd12q1", filenames=modis_l3_nasa_mcd12q1_file)
        available_datasets = scene.all_dataset_names()
        assert len(available_datasets) > 0
        assert "LC_Type1" in available_datasets

    def test_load_l3_dataset(self, modis_l3_nasa_mcd12q1_file):
        """Load and check an L2 variable."""
        scene = Scene(reader="mcd12q1", filenames=modis_l3_nasa_mcd12q1_file)
        ds_name = 'LC_Type1'
        scene.load([ds_name])
        assert ds_name in scene
        data_arr = scene[ds_name]
        assert isinstance(data_arr.data, da.Array)
        assert data_arr.attrs.get("resolution") == 500

