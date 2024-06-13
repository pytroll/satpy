#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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
"""Unit tests for MODIS L3 HDF reader."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
from pyresample import geometry
from pytest_lazy_fixtures import lf as lazy_fixture

from satpy import Scene, available_readers


def _expected_area():
    proj_param = "EPSG:4326"

    return geometry.AreaDefinition("gridded_modis",
                                   "A gridded L3 MODIS area",
                                   "longlat",
                                   proj_param,
                                   7200,
                                   3600,
                                   (-180, -90, 180, 90))


class TestModisL3:
    """Test MODIS L3 reader."""

    def test_available_reader(self):
        """Test that MODIS L3 reader is available."""
        assert "modis_l3" in available_readers()

    @pytest.mark.parametrize(
        ("loadable", "filename"),
        [
            ("Coarse_Resolution_Surface_Reflectance_Band_2", lazy_fixture("modis_l3_nasa_mod09_file")),
            ("BRDF_Albedo_Parameter1_Band2", lazy_fixture("modis_l3_nasa_mod43_file")),
        ]
    )
    def test_scene_available_datasets(self, loadable, filename):
        """Test that datasets are available."""
        scene = Scene(reader="modis_l3", filenames=filename)
        available_datasets = scene.all_dataset_names()
        assert len(available_datasets) > 0
        assert loadable in available_datasets

        from satpy.readers.modis_l3 import ModisL3GriddedHDFFileHandler
        fh = ModisL3GriddedHDFFileHandler(filename[0], {}, {"file_type": "modis_l3_cmg_hdf"})
        configured_datasets = [[None, {"name": "none_ds", "file_type": "modis_l3_cmg_hdf"}],
                               [True, {"name": "true_ds", "file_type": "modis_l3_cmg_hdf"}],
                               [False, {"name": "false_ds", "file_type": "modis_l3_cmg_hdf"}],
                               [None, {"name": "other_ds", "file_type": "modis_l2_random"}]]
        for status, mda in fh.available_datasets(configured_datasets):
            if mda["name"] == "none_ds":
                assert mda["file_type"] == "modis_l3_cmg_hdf"
                assert status is False
            elif mda["name"] == "true_ds":
                assert mda["file_type"] == "modis_l3_cmg_hdf"
                assert status
            elif mda["name"] == "false_ds":
                assert mda["file_type"] == "modis_l3_cmg_hdf"
                assert status is False
            elif mda["name"] == "other_ds":
                assert mda["file_type"] == "modis_l2_random"
                assert status is None
            elif mda["name"] == loadable:
                assert mda["file_type"] == "modis_l3_cmg_hdf"
                assert status

    def test_load_l3_dataset(self, modis_l3_nasa_mod09_file):
        """Load and check an L2 variable."""
        scene = Scene(reader="modis_l3", filenames=modis_l3_nasa_mod09_file)

        ds_name = "Coarse_Resolution_Surface_Reflectance_Band_2"
        scene.load([ds_name])

        data_arr = scene[ds_name]
        assert isinstance(data_arr.data, da.Array)
        data_arr_comp = data_arr.compute()

        # Check types
        assert data_arr_comp.dtype == data_arr.dtype
        assert data_arr_comp.dtype == np.float32

        assert data_arr_comp.shape == (3600, 7200)
        assert data_arr_comp.attrs.get("resolution") == 0.05
        assert data_arr_comp.attrs.get("area") == _expected_area()
