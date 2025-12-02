"""Tests for the H-SAF NC reader."""
import datetime as dt
import gzip
import os
import shutil
from collections import namedtuple
from contextlib import suppress

import numpy as np
import pytest
import xarray as xr
from pyresample import AreaDefinition

from satpy import config
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
        "yaml_file": "hsaf_nc.yaml",
        "platform": "MSG3",
    },
    FILE_TYPE_H63: {
        "fake_file": "h63_20251105_0000_fdk.nc",
        "yaml_file": "hsaf_nc.yaml",
        "platform": "MSG2",
    },
    FILE_TYPE_H90: {
        "fake_file": "h90_20251105_0000_fdk.nc",
        "yaml_file": "hsaf_nc.yaml",
        "platform": "MSG2",
    }
}

# Avoid too many arguments for test_load_datasets
LoadDatasetsParams = namedtuple(
    "LoadDatasetsParams",
    ["file_type", "loadable_ids", "unit", "resolution", "area_name", "platform"]
)

# constants for fake test data
DEFAULT_SHAPE = (5, 5)
rng = np.random.default_rng()
DEFAULT_RR = rng.random(DEFAULT_SHAPE)
DEFAULT_QIND = rng.integers(0, 100, size=DEFAULT_SHAPE)

def fake_hsaf_dataset(filename, platform):
    """Mimic a HSAF NetCDF file content."""
    ds = xr.Dataset(
        {
            "rr": (("ny", "nx"), DEFAULT_RR),
            "acc_rr": (("ny", "nx"), DEFAULT_RR),
            "qind": (("ny", "nx"), DEFAULT_QIND),
        },
        coords={"ny": np.arange(DEFAULT_SHAPE[0]),
                "nx": np.arange(DEFAULT_SHAPE[1])},
        attrs={"satellite_identifier": platform,
               "start_time": "2025-11-05T00:00:00",
               "sub_satellite_longitude": "0.0f"}
    )
    filepath = os.path.join(config.get("tmp_dir"), filename)
    gzip_filepath = filepath + ".gz"
    ds.to_netcdf(path=filepath, format="NETCDF4")

    with open(filepath, "rb") as f_in, gzip.open(gzip_filepath, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return filepath

class TestHSAFNCReader:
    """Test HSAF H60 NetCDF reader."""

    def setup_method(self):
        """Load reader YAML and configs."""
        for file_type, params in FILE_PARAMS.items():
            # search for the yaml config
            params["reader_configs"] = config_search_paths(os.path.join("readers", params["yaml_file"]))
            # load the reader
            params["reader"] = load_reader(params["reader_configs"], name = file_type)
            params["filepath"] = fake_hsaf_dataset(params["fake_file"], params["platform"])


    def teardown_method(self):
        """Remove the previously created test file and reader."""
        for params in FILE_PARAMS.values():
            del params["reader"]

            with suppress(OSError):
                os.remove(params["filepath"])


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
        loadables = file_type["reader"].select_files_from_pathnames([file_type["filepath"]])
        file_handlers = file_type["reader"].create_filehandlers(loadables)

        assert len(file_handlers) == expected_loadables
        assert file_type["reader"].file_handlers, "No file handlers created"

    @pytest.mark.parametrize(
        "params",
        [
            LoadDatasetsParams(
                FILE_PARAMS[FILE_TYPE_H60], ["rr", "qind"], "mm/h", 3000, "msg_seviri_fes_3km", "Meteosat-10"
            ),
            LoadDatasetsParams(
                FILE_PARAMS[FILE_TYPE_H63], ["rr", "qind"], "mm/h", 3000, "msg_seviri_iodc_3km", "Meteosat-9"
            ),
            LoadDatasetsParams(
                FILE_PARAMS[FILE_TYPE_H90], ["acc_rr", "qind"], "mm", 3000, "msg_seviri_iodc_3km", "Meteosat-9"
            ),
        ],
    )
    def test_load_datasets(self, params):
        """Test that datasets can be loaded correctly."""
        loadables = params.file_type["reader"].select_files_from_pathnames([params.file_type["filepath"]])
        params.file_type["reader"].create_filehandlers(loadables)

        datasets = params.file_type["reader"].load(params.loadable_ids)
        dataset_names = {d["name"] for d in datasets.keys()}
        assert dataset_names == set(params.loadable_ids)

        # check array shapes and types
        assert datasets[params.loadable_ids[0]].shape == DEFAULT_SHAPE
        assert datasets[params.loadable_ids[1]].shape == DEFAULT_SHAPE
        assert np.issubdtype(datasets[params.loadable_ids[0]].dtype, np.floating)
        assert np.issubdtype(datasets[params.loadable_ids[1]].dtype, np.integer)

        data = datasets[params.loadable_ids[0]]
        assert data.attrs["platform_name"] == params.platform
        assert data.attrs["units"] == params.unit
        assert data.attrs["resolution"] == params.resolution
        assert data.attrs["start_time"] == dt.datetime(2025, 11, 5, 0, 0)
        assert data.attrs["end_time"] == dt.datetime(2025, 11, 5, 0, 15)
        assert data.attrs["area"].area_id == params.area_name
        assert data.dims == ("y", "x")


    def test_get_area_def(self):
        """Test that the loaded dataset has a AreaDefinition and overwrite of lon_0 of the area works correctly."""
        file_type = FILE_PARAMS[FILE_TYPE_H63]
        loadables = file_type["reader"].select_files_from_pathnames([file_type["filepath"]])
        file_type["reader"].create_filehandlers(loadables)

        datasets = file_type["reader"].load(["rr"])
        data = datasets["rr"]

        area = data.attrs["area"]
        assert isinstance(area, AreaDefinition)
        assert area.area_id == "msg_seviri_iodc_3km"
        assert area.proj_dict["lon_0"] == 0.0
