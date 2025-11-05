"""Tests for the H-SAF NC reader."""
import datetime as dt
import os
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from satpy._config import config_search_paths
from satpy.readers.core.loading import load_reader

# the readers file type
FILE_TYPE_H60 = "nc_hsaf_h60"
FILE_TYPE_H63 = "nc_hsaf_h63"

# parameters per file type
FILE_PARAMS = {
    FILE_TYPE_H60: {
        "fake_file": "h60_20251105_0000_fdk.nc",
        "real_file": "/".join(os.path.abspath(__file__).split('/')[0:-1]) + "/data/S-HSAF-h60_20251105_0915_fdk.nc.gz",
        "yaml_file": "hsaf_h60_nc.yaml",
    },
    FILE_TYPE_H63: {
        "fake_file": "h63_20251105_0000_fdk.nc",
        "real_file": "/".join(os.path.abspath(__file__).split('/')[0:-1]) + "/data/S-HSAF-h63_20251014_0915_fdk.nc.gz",
        "yaml_file": "hsaf_h63_nc.yaml",
    }
}

# constants for fake test data
DEFAULT_SHAPE = (5, 5)
DEFAULT_RR = np.random.rand(*DEFAULT_SHAPE)
DEFAULT_QIND = np.random.randint(0, 100, size=DEFAULT_SHAPE)

def fake_hsaf_dataset(filename, **kwargs):
    """Mimic a HSAF NetCDF file content."""
    ds = xr.Dataset(
        {
            "rr": (("y", "x"), DEFAULT_RR),
            "qind": (("y", "x"), DEFAULT_QIND),
        },
        coords={"y": np.arange(DEFAULT_SHAPE[0]),
                "x": np.arange(DEFAULT_SHAPE[1])},
        attrs={"satellite_identifier": "MSG",
               "start_time": "2025-11-05T00:00:00"}
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
            params["reader"] = load_reader(params["reader_configs"])

    @pytest.mark.parametrize(
        ("file_type", "expected_loadables"),
        [
            (FILE_PARAMS[FILE_TYPE_H60], 1),
            (FILE_PARAMS[FILE_TYPE_H63], 1),
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
        ("file_type", "loadable_ids"),
        [
            (FILE_PARAMS[FILE_TYPE_H60], ["rr", "qind"]),
            (FILE_PARAMS[FILE_TYPE_H63], ["rr", "qind"]),
        ],
    )
    def test_load_datasets(self, file_type, loadable_ids):
        """Test that datasets can be loaded correctly."""
        with mock.patch("satpy.readers.hsaf_nc.xr.open_dataset") as od:
            od.side_effect = fake_hsaf_dataset
            loadables = file_type["reader"].select_files_from_pathnames([file_type["fake_file"]])
            file_type["reader"].create_filehandlers(loadables)

            datasets = file_type["reader"].load(loadable_ids)
            dataset_names = {d['name'] for d in datasets.keys()}
            assert dataset_names == set(loadable_ids)

            # check array shapes and types
            assert datasets["rr"].shape == DEFAULT_SHAPE
            assert datasets["qind"].shape == DEFAULT_SHAPE
            assert np.issubdtype(datasets["rr"].dtype, np.floating)
            assert np.issubdtype(datasets["qind"].dtype, np.integer)

            # check dataset and attrs
            dataset_info = {"file_key": "rr", "units": "mm/h"}
            data = file_type["reader"].file_handlers[datasets["rr"].file_type][0].get_dataset({"name": "rr"}, dataset_info)

            assert data.attrs["spacecraft_name"] == "MSG"
            assert data.attrs["platform_name"] == "MSG"
            assert data.attrs["units"] == "mm/h"
            assert data.attrs["start_time"] == dt.datetime(2025, 11, 5, 0, 0)
            assert data.attrs["end_time"] == dt.datetime(2025, 11, 5, 0, 15)


    @pytest.mark.parametrize(
        ("file_type", "loadable_ids"),
        [
            (FILE_PARAMS[FILE_TYPE_H60], ["rr", "qind"]),
            (FILE_PARAMS[FILE_TYPE_H63], ["rr", "qind"]),
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
