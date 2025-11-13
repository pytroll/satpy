"""Tests for the H-SAF NC reader."""
import datetime as dt
import os
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition

from satpy._config import config_search_paths
from satpy.readers.core.loading import load_reader

# the readers file type
FILE_TYPE_H60 = "hsaf_h60_nc"
FILE_TYPE_H63 = "hsaf_h63_nc"
FILE_TYPE_H90 = "hsaf_h90_nc"

# parameters per file type
FILE_PARAMS = {
    FILE_TYPE_H60: {
        "fake_file": "h60_20251105_0000_fdk.nc",
        "real_file": "/".join(os.path.abspath(__file__).split("/")[0:-1]) + "/data/h60_20251111_1100_fdk.nc.gz",
        "yaml_file": "hsaf_nc.yaml",
    },
    FILE_TYPE_H63: {
        "fake_file": "h63_20251105_0000_fdk.nc",
        "real_file": "/".join(os.path.abspath(__file__).split("/")[0:-1]) + "/data/h63_20251111_1100_fdk.nc.gz",
        "yaml_file": "hsaf_nc.yaml",
    },
    FILE_TYPE_H90: {
        "fake_file": "h90_20251105_0000_fdk.nc",
        "real_file": "/".join(os.path.abspath(__file__).split("/")[0:-1]) + "/data/h90_20251111_1000_01_fdk.nc.gz",
        "yaml_file": "hsaf_nc.yaml",
    }
}

# constants for fake test data
DEFAULT_SHAPE = (5, 5)
rng = np.random.default_rng()
DEFAULT_RR = rng.random(DEFAULT_SHAPE)
DEFAULT_QIND = rng.integers(0, 100, size=DEFAULT_SHAPE)

def fake_hsaf_dataset(filename, **kwargs):
    """Mimic a HSAF NetCDF file content."""
    ds = xr.Dataset(
        {
            "rr": (("ny", "nx"), DEFAULT_RR),
            "acc_rr": (("ny", "nx"), DEFAULT_RR),
            "qind": (("ny", "nx"), DEFAULT_QIND),
        },
        coords={"ny": np.arange(DEFAULT_SHAPE[0]),
                "nx": np.arange(DEFAULT_SHAPE[1])},
        attrs={"satellite_identifier": "MSG1",
               "start_time": "2025-11-05T00:00:00",
               "sub_satellite_longitude": "0.0f"}
    )
    return ds

class TestHSAFNCReader:
    """Test HSAF H60 NetCDF reader."""

    def setup_method(self):
        """Load reader YAML and configs."""
        for file_type, params in FILE_PARAMS.items():
            # search for the yaml config
            params["reader_configs"] = config_search_paths(os.path.join("readers", params["yaml_file"]))
            # load the reader
            params["reader"] = load_reader(params["reader_configs"], name = file_type)

    @pytest.mark.parametrize(
        ("file_type", "expected_loadables"),
        [
            (FILE_PARAMS[FILE_TYPE_H60], 1),
            (FILE_PARAMS[FILE_TYPE_H63], 1),
            (FILE_PARAMS[FILE_TYPE_H90], 1),
        ],
    )
    def test_reader_creation(self, file_type, expected_loadables):
        """Test that the reader can create file handlers."""
        with mock.patch("satpy.readers.hsaf_nc.xr.open_dataset") as od:
            od.side_effect = fake_hsaf_dataset

            loadables = file_type["reader"].select_files_from_pathnames([file_type["fake_file"]])
            file_type["reader"].create_filehandlers(loadables)

            assert len(loadables) == expected_loadables
            assert file_type["reader"].file_handlers, "No file handlers created"

    @pytest.mark.parametrize(
        ("file_type", "loadable_ids", "unit", "resolution", "area_name"),
        [
            (FILE_PARAMS[FILE_TYPE_H60], ["rr", "qind"], "mm/h", 3000, "msg_seviri_fes_3km"),
            (FILE_PARAMS[FILE_TYPE_H63], ["rr", "qind"], "mm/h", 3000, "msg_seviri_iodc_3km"),
            (FILE_PARAMS[FILE_TYPE_H90], ["acc_rr", "qind"], "mm", 3000, "msg_seviri_iodc_3km"),
        ],
    )
    def test_load_datasets(self, file_type, loadable_ids, unit, resolution, area_name):
        """Test that datasets can be loaded correctly."""
        with mock.patch("satpy.readers.hsaf_nc.xr.open_dataset") as od:
            od.side_effect = fake_hsaf_dataset
            loadables = file_type["reader"].select_files_from_pathnames([file_type["fake_file"]])
            file_type["reader"].create_filehandlers(loadables)

            datasets = file_type["reader"].load(loadable_ids)
            dataset_names = {d["name"] for d in datasets.keys()}
            assert dataset_names == set(loadable_ids)

            # check array shapes and types
            assert datasets[loadable_ids[0]].shape == DEFAULT_SHAPE
            assert datasets[loadable_ids[1]].shape == DEFAULT_SHAPE
            assert np.issubdtype(datasets[loadable_ids[0]].dtype, np.floating)
            assert np.issubdtype(datasets[loadable_ids[1]].dtype, np.integer)

            data = datasets[loadable_ids[0]]
            assert data.attrs["spacecraft_name"] == "Meteosat-8"
            assert data.attrs["platform_name"] == "Meteosat-8"
            assert data.attrs["units"] == unit
            assert data.attrs["resolution"] == resolution
            assert data.attrs["start_time"] == dt.datetime(2025, 11, 5, 0, 0)
            assert data.attrs["end_time"] == dt.datetime(2025, 11, 5, 0, 15)
            assert data.attrs["area"].area_id == area_name
            assert data.dims == ("y", "x")


    @pytest.mark.parametrize(
        ("file_type", "loadable_ids"),
        [
            (FILE_PARAMS[FILE_TYPE_H60], ["rr", "qind"]),
            (FILE_PARAMS[FILE_TYPE_H63], ["rr", "qind"]),
            (FILE_PARAMS[FILE_TYPE_H90], ["acc_rr", "qind"]),
        ],
    )
    @pytest.mark.skipif(not os.path.exists(FILE_PARAMS[FILE_TYPE_H60]["real_file"]) or
                        not os.path.exists(FILE_PARAMS[FILE_TYPE_H63]["real_file"]),
                                           reason="Real HSAF file not present")
    def test_real_hsaf_file(self, file_type, loadable_ids):
        """Test the reader with a real HSAF NetCDF file."""
        # Select files
        loadables = file_type["reader"].select_files_from_pathnames([file_type["real_file"]])
        assert loadables, "No loadables found for the real file"

        # Create filehandlers
        file_type["reader"].create_filehandlers(loadables)
        assert file_type["reader"].file_handlers, "File handlers were not created"

        # Load all datasets defined in the YAML
        dataset_names = set(file_type["reader"].available_dataset_names)

        assert dataset_names, "No datasets found for the real file"
        assert dataset_names.issubset((loadable_ids)), f"Could not find {loadable_ids} in datasets"

    def test_get_area_def(self):
        """Test that the loaded dataset has a AreaDefinition and overwrite of lon_0 of the area works correctly."""
        with mock.patch("satpy.readers.hsaf_nc.xr.open_dataset") as od:
            od.side_effect = fake_hsaf_dataset
            file_type = FILE_PARAMS[FILE_TYPE_H63]
            loadables = file_type["reader"].select_files_from_pathnames([file_type["fake_file"]])
            file_type["reader"].create_filehandlers(loadables)

            datasets = file_type["reader"].load(["rr"])
            data = datasets["rr"]

            area = data.attrs["area"]
            assert isinstance(area, AreaDefinition)
            assert area.area_id == "msg_seviri_iodc_3km"
            assert area.proj_dict["lon_0"] == 0.0
