"""Unit tests for MODIS L2 HDF reader."""

from __future__ import annotations

import dask.array as da

from satpy import Scene, available_readers

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - modis_l3_nasa_mcd12q1_file


class TestModisL3MCD12Q1:
    """Test MODIS L3 MCD12Q1 reader."""

    def test_available_reader(self):
        """Test that MODIS L3 reader is available."""
        assert "mcd12q1" in available_readers()

    def test_metadata(self, modis_l3_nasa_mcd12q1_file):
        """Test some basic metadata that should exist in the file."""
        scene = Scene(reader="mcd12q1", filenames=modis_l3_nasa_mcd12q1_file)
        ds_name = "LC_Type2"
        scene.load([ds_name])
        assert scene[ds_name].attrs["area"].description == "Tiled sinusoidal L3 MODIS area"
        assert scene[ds_name].attrs["sensor"] == "modis"

    def test_scene_available_datasets(self, modis_l3_nasa_mcd12q1_file):
        """Test that datasets are available."""
        scene = Scene(reader="mcd12q1", filenames=modis_l3_nasa_mcd12q1_file)
        available_datasets = scene.all_dataset_names()
        assert len(available_datasets) > 0
        assert "LC_Type1" in available_datasets

    def test_load_l3_dataset(self, modis_l3_nasa_mcd12q1_file):
        """Load and check an L2 variable."""
        scene = Scene(reader="mcd12q1", filenames=modis_l3_nasa_mcd12q1_file)
        ds_name = "LC_Type1"
        scene.load([ds_name])
        assert ds_name in scene
        data_arr = scene[ds_name]
        assert isinstance(data_arr.data, da.Array)
        assert data_arr.attrs.get("resolution") == 500
