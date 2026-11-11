"""Module for testing the satpy.readers.omps_edr module."""

import copy
import datetime
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest
import xarray as xr

START_TIME1 = datetime.datetime(2025, 1, 1, 0, 0, 0)
SINGLE_GRAN_SHAPE = (30, 240)
# location of the negative values in the fake SO2 variables (avoids the "ErrorFlag" row)
NEGATIVE_ROW = 1
NUM_NEGATIVES = 5
# location of the eclipsed pixels (avoids the "ErrorFlag" and negative SO2 rows)
ECLIPSE_ROW = 2
NUM_ECLIPSED = 7
# the SO2 variables use the netCDF default float fill value, not the -9999.0 used elsewhere
SO2_FILL_VALUE = np.float32(-1.2676506e30)


def _fake_filename(
    start_time: datetime.datetime,
    end_time: datetime.datetime | None = None,
    prefix: str = "V8TOZ",
    version: str = "v4r3",
) -> str:
    stime_str = f"{start_time:%Y%m%d%H%M%S}0"
    if end_time is None:
        end_time = start_time + datetime.timedelta(seconds=90)
    etime_str = f"{end_time:%Y%m%d%H%M%S}0"
    ctime_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S0")
    return f"{prefix}-EDR_{version}_n21_s{stime_str}_e{etime_str}_c{ctime_str}.nc"


def _create_base_file(
    nc, start_time: datetime.datetime, platform: str = "NOAA21", include_eclipse: bool = True
) -> tuple:
    """Add the global attributes and variables common to all OMPS EDR products."""
    end_time = start_time + datetime.timedelta(seconds=90)
    rng = np.random.default_rng(12345)

    nc.platform = platform
    nc.platform_name = "NOAA21"
    nc.instrument = "OMPS"
    nc.instrument_name = "OMPS"
    nc.time_coverage_start = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    nc.time_coverage_end = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    nc.start_orbit_number = 43448
    nc.end_orbit_number = 43448

    shape = SINGLE_GRAN_SHAPE
    ntimes = nc.createDimension("nTimes", shape[0])
    nifov = nc.createDimension("nIFOV", shape[1])
    nwavelength = nc.createDimension("nWavelength", 12)
    dims = (ntimes, nifov)
    lon_var = nc.createVariable("Longitude", np.float32, dimensions=dims, fill_value=-9999.0)
    lon_var.standard_name = "longitude"
    lon_var.units = "degrees_east"
    lon_var.valid_range = (-360, 360)
    lon_var[:] = rng.random(shape).astype(np.float32) * 45.0
    lat_var = nc.createVariable("Latitude", np.float32, dimensions=dims, fill_value=-9999.0)
    lat_var.standard_name = "latitude"
    lat_var.units = "degrees_north"
    lat_var.valid_range = (-90, 90)
    lat_var[:] = rng.random(shape).astype(np.float32) * 45.0

    err_flag = nc.createVariable("ErrorFlag", np.int32, dimensions=dims, fill_value=-9999)
    err_flag.valid_range = (0, 10)
    err_flag.units = "1"
    err_data = np.zeros(shape, dtype=np.int32)
    err_data[0, :10] = np.arange(10, dtype=np.int32)
    err_flag[:] = err_data

    if include_eclipse:
        eclipse_flag = nc.createVariable("EclipseFlag", np.float32, dimensions=dims, fill_value=-9999.0)
        eclipse_flag.valid_range = (0, 10)
        eclipse_flag.units = "1"
        eclipse_data = np.zeros(shape, dtype=np.float32)
        eclipse_data[ECLIPSE_ROW, :NUM_ECLIPSED] = 1
        eclipse_flag[:] = eclipse_data

    # non-2D variables that the reader should ignore
    nvalue = nc.createVariable(
        "NvalueMeasured", np.float32, dimensions=(ntimes, nifov, nwavelength), fill_value=-9999.0
    )
    nvalue.valid_range = (0, 100)
    nvalue.units = "1"
    nvalue[:] = rng.random(shape + (12,)).astype(np.float32) * 100
    wavelengths = nc.createVariable("Wavelengths", np.float32, dimensions=(nwavelength,), fill_value=-9999.0)
    wavelengths.valid_range = (308, 380)
    wavelengths.units = "nm"
    wavelengths[:] = np.linspace(308, 380, 12).astype(np.float32)

    return dims, rng


def _create_ozone_variables(nc, dims, rng, o3_units: str = "0.01mm") -> None:
    """Add the ozone (V8TOZ) variables which are also included in the SO2 (V8TOS) files."""
    shape = SINGLE_GRAN_SHAPE

    amount_o3 = nc.createVariable("ColumnAmountO3", np.float32, dimensions=dims, fill_value=-9999.0)
    amount_o3.valid_range = (0, 1000)
    # the ozone files use "0.01mm" while the SO2 files use "DU" for the same variable
    amount_o3.units = o3_units
    amount_o3[:] = rng.random(shape).astype(np.float32) * 1000

    aerosol_idx = nc.createVariable("AerosolIndex", np.float32, dimensions=dims, fill_value=-9999.0)
    aerosol_idx.valid_range = (-100, 100)
    aerosol_idx.units = "1"
    aerosol_idx[:] = rng.random(shape).astype(np.float32) * 200 - 100

    refl331 = nc.createVariable("Reflectivity331", np.float32, dimensions=dims, fill_value=-9999.0)
    refl331.valid_range = (0, 100)
    refl331.units = "1"
    refl331[:] = rng.random(shape).astype(np.float32) * 100


def create_v8toz_file(tmp_path: Path, start_time: datetime.datetime, include_eclipse: bool = True) -> Path:
    """Create a fake total ozone file for testing."""
    from netCDF4 import Dataset

    end_time = start_time + datetime.timedelta(seconds=90)
    filename = tmp_path / _fake_filename(start_time, end_time=end_time, prefix="V8TOZ", version="v4r3")

    with Dataset(filename, "w") as nc:
        dims, rng = _create_base_file(nc, start_time, platform="NOAA21", include_eclipse=include_eclipse)
        _create_ozone_variables(nc, dims, rng)

    return filename


def create_v8tos_file(tmp_path: Path, start_time: datetime.datetime) -> Path:
    """Create a fake total SO2 file for testing."""
    from netCDF4 import Dataset

    end_time = start_time + datetime.timedelta(seconds=90)
    filename = tmp_path / _fake_filename(start_time, end_time=end_time, prefix="V8TOS", version="v4r5")
    shape = SINGLE_GRAN_SHAPE

    with Dataset(filename, "w") as nc:
        # the SO2 files use the dashed platform name and include the ozone product variables
        dims, rng = _create_base_file(nc, start_time, platform="NOAA-21")
        _create_ozone_variables(nc, dims, rng, o3_units="DU")

        for suffix in ("PBL", "TRL", "TRM", "STL"):
            so2_var = nc.createVariable(
                f"s_ColumnamountSO2_{suffix}", np.float32, dimensions=dims, fill_value=SO2_FILL_VALUE
            )
            # valid range includes negatives so they are only removed by "filter_negative_so2"
            so2_var.valid_range = (-300, 1000) if suffix == "PBL" else (-10, 2000)
            # SO2 variables use a non-standard "Unit" attribute instead of "units"
            so2_var.Unit = "Dobson"
            so2_var.without_data = -9999.0
            so2_data = rng.random(shape).astype(np.float32) * 100
            so2_data[NEGATIVE_ROW, :NUM_NEGATIVES] = -5.0
            so2_var[:] = so2_data

    return filename


def omps_reader_gen(file_paths: Iterable[Path], reader_kwargs: dict[str, Any] | None = None):
    """Create a reader instance with provided files loaded."""
    from satpy._config import config_search_paths
    from satpy.readers.core.loading import load_reader

    if reader_kwargs is None:
        reader_kwargs = {}

    reader_configs = config_search_paths("readers/omps_edr.yaml")
    reader = load_reader(reader_configs, **reader_kwargs)
    loadable_files = reader.select_files_from_pathnames(file_paths)
    reader.create_filehandlers(loadable_files, fh_kwargs=reader_kwargs)
    return reader


def test_available_datasets(tmp_path):
    """Test available datasets dynamically generated from file contents."""
    one_file = create_v8toz_file(tmp_path, START_TIME1)
    reader = omps_reader_gen([one_file])
    # make sure we have some files
    avail_datasets = list(data_id["name"] for data_id in reader.available_dataset_ids)
    assert "Reflectivity331" in avail_datasets
    assert "AerosolIndex" in avail_datasets
    assert "ColumnAmountO3" in avail_datasets
    # only 2D (nTimes, nIFOV) variables are provided
    assert "NvalueMeasured" not in avail_datasets
    assert "Wavelengths" not in avail_datasets


@pytest.mark.parametrize(
    ("file_func", "vars_to_load"),
    [
        (create_v8toz_file, ["Reflectivity331", "AerosolIndex", "ColumnAmountO3"]),
        (create_v8toz_file, ["Reflectivity331"]),
        (
            create_v8tos_file,
            ["s_ColumnamountSO2_PBL", "s_ColumnamountSO2_TRL", "Reflectivity331", "AerosolIndex", "ColumnAmountO3"],
        ),
    ],
)
@pytest.mark.parametrize(
    "error_flags_to_keep",
    [
        None,
        {},
        {"ColumnAmountO3": [0, 1, 2]},
        {"ColumnAmountO3": [0, 1, 2], "s_ColumnamountSO2_PBL": [0, 1, 5]},
        {"Reflectivity331": []},
        # applies to every variable (but never geolocation)
        defaultdict(lambda: [0, 1]),
    ],
    ids=["none", "empty", "ozone_only", "ozone_and_so2", "empty_values", "defaultdict"],
)
@pytest.mark.parametrize("filter_negative_so2", [False, True])
@pytest.mark.parametrize("filter_eclipse", [False, True])
def test_basic_load(tmp_path, file_func, vars_to_load, error_flags_to_keep, filter_negative_so2, filter_eclipse):
    """Test basic load from multiple files."""
    from satpy.tests.utils import assert_maximum_dask_computes

    # parametrized objects are shared between tests and defaultdict lookups insert keys
    # copy so every test starts with the original (possibly empty and falsy) mapping
    error_flags_to_keep = copy.copy(error_flags_to_keep)
    one_file = file_func(tmp_path, START_TIME1)
    two_file = file_func(tmp_path, START_TIME1 + datetime.timedelta(seconds=90))
    reader_kwargs = {
        "error_flags_to_keep": error_flags_to_keep,
        "filter_negative_so2": filter_negative_so2,
        "filter_eclipse": filter_eclipse,
    }
    reader = omps_reader_gen([one_file, two_file], reader_kwargs=reader_kwargs)
    with assert_maximum_dask_computes(0):
        loaded_dict = reader.load(vars_to_load)
    assert len(loaded_dict) == len(vars_to_load)

    for var_name in vars_to_load:
        _check_expected_array(
            loaded_dict[var_name],
            num_granules=2,
            error_flags_to_keep=error_flags_to_keep,
            filter_negative_so2=filter_negative_so2,
            filter_eclipse=filter_eclipse,
        )


def test_no_filtering_by_default(tmp_path):
    """Test that no flag filtering is done when no filtering kwargs are provided."""
    one_file = create_v8toz_file(tmp_path, START_TIME1)
    reader = omps_reader_gen([one_file])
    loaded_dict = reader.load(["ColumnAmountO3"])
    _check_expected_array(loaded_dict["ColumnAmountO3"], num_granules=1, filter_eclipse=False)


def test_missing_eclipse_flag(tmp_path):
    """Test that files without an "EclipseFlag" variable load when eclipse filtering is requested."""
    one_file = create_v8toz_file(tmp_path, START_TIME1, include_eclipse=False)
    reader = omps_reader_gen([one_file], reader_kwargs={"filter_eclipse": True})
    loaded_dict = reader.load(["ColumnAmountO3"])
    _check_expected_array(loaded_dict["ColumnAmountO3"], num_granules=1, filter_eclipse=False)


def _check_expected_array(
    data_arr: xr.DataArray,
    num_granules: int = 2,
    error_flags_to_keep: None | Mapping[str, Sequence[int]] = None,
    filter_negative_so2: bool = False,
    filter_eclipse: bool = False,
) -> None:
    from pyresample.geometry import SwathDefinition

    assert data_arr.dims == ("y", "x")
    assert data_arr.shape == (SINGLE_GRAN_SHAPE[0] * num_granules, SINGLE_GRAN_SHAPE[1])
    assert data_arr.dtype.type == np.float32
    # the non-standard "Unit" attribute of the SO2 variables is renamed to "units"
    assert "units" in data_arr.attrs
    assert "Unit" not in data_arr.attrs
    assert data_arr.attrs["platform_name"] == "NOAA-21"
    assert data_arr.attrs["sensor"] == "omps"

    data_np = data_arr.data.compute()
    assert data_np.dtype == data_arr.dtype
    var_name = data_arr.attrs["name"]
    is_so2 = var_name.startswith("s_ColumnamountSO2")
    expected_nans = _expected_nan_mask(
        num_granules,
        _error_flag_values_for_var(error_flags_to_keep, var_name),
        filter_negative_so2 and is_so2,
        filter_eclipse,
    )
    np.testing.assert_array_equal(np.isnan(data_np), expected_nans)

    area = data_arr.attrs["area"]
    assert isinstance(area, SwathDefinition)
    assert area.shape == (SINGLE_GRAN_SHAPE[0] * num_granules, SINGLE_GRAN_SHAPE[1])
    # geolocation is never filtered
    lons, lats = area.get_lonlats()
    assert not np.isnan(lons).any()
    assert not np.isnan(lats).any()

    if "valid_range" in data_arr.attrs:
        assert isinstance(data_arr.attrs["valid_range"], list)


def _error_flag_values_for_var(
    error_flags_to_keep: None | Mapping[str, Sequence[int]], var_name: str
) -> None | Sequence[int]:
    if error_flags_to_keep is None:
        return None
    try:
        return error_flags_to_keep[var_name]
    except KeyError:
        return None


def _expected_nan_mask(
    num_granules: int,
    error_flag_values: None | Iterable[int],
    expect_negative_filtering: bool,
    expect_eclipse_filtering: bool,
) -> np.ndarray:
    """Get the boolean mask of pixels expected to be NaN after all filtering."""
    gran_mask = np.zeros(SINGLE_GRAN_SHAPE, dtype=bool)
    if error_flag_values:
        for filt_val in range(10):
            gran_mask[0, filt_val] = filt_val not in error_flag_values
    if expect_negative_filtering:
        gran_mask[NEGATIVE_ROW, :NUM_NEGATIVES] = True
    if expect_eclipse_filtering:
        gran_mask[ECLIPSE_ROW, :NUM_ECLIPSED] = True
    return np.tile(gran_mask, (num_granules, 1))
