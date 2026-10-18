"""Tests for the zarr writer."""

import copy
import datetime as dt
import json
import os
import uuid
import warnings
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from dask.delayed import Delayed
from packaging.version import Version
from pyproj import CRS
from pyresample.geometry import AreaDefinition, StackedAreaDefinition, SwathDefinition
from zarr import open_group
from zarr.storage import MemoryStore

from satpy.dataset.dataid import DataQuery, WavelengthRange
from satpy.tests.utils import assert_maximum_dask_computes, make_dataid
from satpy.writers.core.compute import compute_writer_results
from satpy.writers.zarr import ZarrWriter

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - caplog
# - tmp_path

START_TIME = dt.datetime(2026, 1, 1, 12, 0, 0, 123456, tzinfo=dt.timezone.utc)
# Pixels are 1000 m wide and 500 m high, so swapping the x and y pixel sizes is caught
AREA = AreaDefinition("test", "test", "test", "EPSG:3395", 6, 4, (-1000.0, -500.0, 5000.0, 1500.0))
GEOS_AREA = AreaDefinition("geos", "geos", "geos", {"proj": "geos", "h": 35786023.0, "lon_0": -75.0, "sweep": "x"},
                           6, 4, (-3e6, -1e6, 3e6, 1e6))
# AREA rotated by 180 degrees, like SEVIRI data in their raw orientation
FLIPPED_AREA = AreaDefinition("flipped", "flipped", "flipped", "EPSG:3395", 6, 4, (5000.0, 1500.0, -1000.0, -500.0))
PROJ_UUID = "f17cb550-5864-4468-aeb7-f3180cfb622f"
SPATIAL_UUID = "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4"


def _data_array(name="test", *, ndim=2, area=AREA, start_time=START_TIME, chunks=2, **attrs):
    """Create a float32 dask DataArray like a reader or resampler would produce."""
    data = da.arange(24, dtype=np.float32).reshape(4, 6).rechunk(chunks)
    dims = ("y", "x")
    coords = {}
    if ndim == 3:
        data = da.stack([data, data + 24, data + 48]).rechunk((1, chunks, chunks))
        dims = ("bands", "y", "x")
        coords["bands"] = ["R", "G", "B"]
    if isinstance(area, AreaDefinition):
        x, y = area.get_proj_coords()
        coords.update(x=x[0], y=y[:, 0], crs=area.crs)
    if start_time is not None:
        attrs = {"start_time": start_time, "end_time": start_time + dt.timedelta(minutes=10), **attrs}
    return xr.DataArray(data, dims=dims, coords=coords, attrs={"name": name, "units": "K", "area": area, **attrs})


def _masked_data_array(**attrs):
    """Create float32 data whose first row of chunks is all NaN."""
    data_arr = _data_array(**attrs)
    return data_arr.copy(data=da.where(da.arange(4)[:, None] < 2, np.float32(np.nan), data_arr.data))


def _swath_definition(lon_offset=0.0):
    lons = xr.DataArray(da.linspace(lon_offset, lon_offset + 10.0, 24).reshape(4, 6).rechunk(2), dims=("y", "x"))
    lats = xr.DataArray(da.linspace(40.0, 50.0, 24).reshape(4, 6).rechunk(2), dims=("y", "x"))
    return SwathDefinition(lons, lats)


def _stacked_area_definition():
    """Create two areas shifted in x, which pyresample cannot merge into one grid, like shifted SEVIRI HRV windows."""
    upper = AreaDefinition("upper", "upper", "upper", "EPSG:3395", 6, 2, (-1000.0, 500.0, 5000.0, 1500.0))
    lower = AreaDefinition("lower", "lower", "lower", "EPSG:3395", 6, 2, (0.0, -500.0, 6000.0, 500.0))
    return StackedAreaDefinition(upper, lower)


def _messy_attrs():
    return {
        "sensor": {"abi"},
        "flag": True,
        "none_value": None,
        "_satpy_id": make_dataid(name="test"),
        "_last_resampler": object(),
        "orbital_parameters": {"satellite_nominal_longitude": -75.0, "satellite_actual_altitude": np.nan,
                               "nested": {"a": [1, 2]}},
        "prerequisites": [make_dataid(name="C01"), DataQuery(name="C02")],
        "array": np.array([1, 2, 3], dtype=np.int16),
        "ancillary_variables": [_data_array("quality")],
        "wavelength": WavelengthRange(10.1, 10.3, 10.5, "µm"),
        "np_scalar": np.float32(1.5),
        "valid_time": np.datetime64("2026-01-01T12:00"),
        "scan_times": np.array(["2026-01-01T12:00", "2026-01-01T12:05"], dtype="datetime64[ns]"),
        "limits": np.array([-np.inf, 0.0, np.inf], dtype=np.float32),
        "scan_duration": np.timedelta64(300_000_000_000, "ns"),
        "bytes_value": b"abc",
    }


def _strict_json(path):
    """Parse a JSON file, refusing the NaN and Infinity tokens that only Python's json accepts."""
    def _refuse(token):
        raise ValueError(f"{token} is not valid JSON")

    return json.loads(path.read_text(), parse_constant=_refuse)


def _hourly(*hours, **kwargs):
    """Create one DataArray per hour of 2026-01-01, to write as consecutive time steps."""
    return [_data_array(start_time=dt.datetime(2026, 1, 1, hour, tzinfo=dt.timezone.utc), **kwargs) for hour in hours]


def _save_steps(writer, store, data_arrays, **kwargs):
    """Write each DataArray to the store as a new time step."""
    for data_arr in data_arrays:
        writer.save_datasets([data_arr], filename=store, append_dim="time", **kwargs)


def _open(store, **kwargs):
    """Open a store with xarray's defaults, which warn for stores without consolidated metadata."""
    if open_group(store, mode="r").metadata.consolidated_metadata is None:
        kwargs.setdefault("consolidated", False)
    return xr.open_zarr(store, **kwargs)


def _assert_roundtrip(store, expected, **open_kwargs):
    ds = _open(store, **open_kwargs)
    for data_arr in expected:
        stored = ds[data_arr.attrs["name"]]
        assert stored.dims[-data_arr.ndim:] == data_arr.dims
        np.testing.assert_array_equal(stored.values.reshape(data_arr.shape), data_arr.values)
    return ds


def _times(*values):
    return [np.datetime64(value, "us") for value in values]


@pytest.fixture
def writer(tmp_path):
    """Create a zarr writer that writes to ``tmp_path``."""
    return ZarrWriter(base_dir=str(tmp_path))


@pytest.fixture
def icechunk_repo():
    """Create an in-memory Icechunk repository."""
    icechunk = pytest.importorskip("icechunk")
    return icechunk.Repository.create(icechunk.in_memory_storage())


def test_default_filename(writer, tmp_path):
    """Test that the default pattern names a single dataset's store after it."""
    data_arr = _data_array()
    writer.save_datasets([data_arr])
    _assert_roundtrip(tmp_path / "test_20260101_120000.zarr", [data_arr])


def test_scene_selects_writer_by_extension(tmp_path):
    """Test that Scene saving picks the zarr writer for filenames ending in .zarr."""
    from satpy import Scene

    scn = Scene()
    scn["C01"] = _data_array("C01")
    scn["C02"] = _data_array("C02")
    scn.save_datasets(filename=str(tmp_path / "{start_time:%Y%m%d}.zarr"))
    _assert_roundtrip(tmp_path / "20260101.zarr", [scn["C01"], scn["C02"]])
    scn.save_dataset("C01", filename=str(tmp_path / "{name}.zarr"))
    _assert_roundtrip(tmp_path / "C01.zarr", [scn["C01"]])


@pytest.mark.parametrize("ndim", [2, 3])
def test_roundtrip(writer, tmp_path, ndim):
    """Test that data, dims and coordinates round-trip and x/y get CF attributes."""
    data_arr = _data_array(ndim=ndim)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    ds = _assert_roundtrip(tmp_path / "out.zarr", [data_arr])
    assert "crs" not in ds.coords
    np.testing.assert_array_equal(ds["x"].values, data_arr["x"].values)
    assert ds["x"].attrs == {"standard_name": "projection_x_coordinate", "units": "m"}
    assert ds["y"].attrs == {"standard_name": "projection_y_coordinate", "units": "m"}
    assert open_group(tmp_path / "out.zarr", mode="r").metadata.zarr_format == 3


def test_attrs_sanitized(writer, tmp_path):
    """Test the conversion of Satpy attributes to JSON attributes without changing the input."""
    # The CF conversion cannot encode tuple keys, so they are not in _messy_attrs
    data_arr = _data_array(**_messy_attrs(), lookup={(1, 2): "a", 3: "b"})
    input_attrs = dict(data_arr.attrs)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    attrs = _strict_json(tmp_path / "out.zarr" / "test" / "zarr.json")["attributes"]
    for removed in ("area", "_satpy_id", "_last_resampler", "none_value"):
        assert removed not in attrs
    assert attrs["start_time"] == "2026-01-01T12:00:00.123456+00:00"
    assert attrs["sensor"] == ["abi"]
    assert attrs["flag"] is True
    assert attrs["orbital_parameters"] == {"satellite_nominal_longitude": -75.0, "satellite_actual_altitude": "NaN",
                                           "nested": {"a": [1, 2]}}
    assert attrs["prerequisites"] == [str(make_dataid(name="C01")), str(DataQuery(name="C02"))]
    assert attrs["array"] == [1, 2, 3]
    assert attrs["ancillary_variables"] == ["quality"]
    assert WavelengthRange.from_cf(attrs["wavelength"]) == WavelengthRange(10.1, 10.3, 10.5, "µm")
    assert attrs["np_scalar"] == 1.5
    assert attrs["valid_time"] == "2026-01-01T12:00"
    assert attrs["scan_times"] == ["2026-01-01T12:00:00.000000000", "2026-01-01T12:05:00.000000000"]
    assert attrs["limits"] == ["-Infinity", 0.0, "Infinity"]
    assert attrs["scan_duration"] == "0:05:00"
    assert attrs["lookup"] == {"(1, 2)": "a", "3": "b"}
    assert attrs["bytes_value"] == "abc"
    assert data_arr.attrs.keys() == input_attrs.keys()
    assert all(data_arr.attrs[key] is value for key, value in input_attrs.items())
    assert "crs" in data_arr.coords
    assert "standard_name" not in data_arr["x"].attrs


@pytest.mark.parametrize(
    ("name", "stored_name"),
    [
        pytest.param("dataarray_name", "dataarray_name", id="dataarray_name"),
        pytest.param(None, "data", id="unnamed"),
    ],
)
def test_variable_name_fallback(writer, tmp_path, caplog, name, stored_name):
    """Test the variable name of DataArrays without a 'name' attribute."""
    data_arr = _data_array().rename(name)
    del data_arr.attrs["name"]
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    assert list(_open(tmp_path / "out.zarr").data_vars) == [stored_name]
    assert ("writing it as 'data'" in caplog.text) == (stored_name == "data")


def test_no_datasets(writer):
    """Test that saving no datasets writes nothing."""
    assert writer.save_datasets([], compute=False) == []


def test_compute_false(writer, tmp_path):
    """Test that compute=False writes only metadata until the Delayed objects are computed."""
    data_arr = _data_array()
    delayeds = writer.save_datasets([data_arr], filename=tmp_path / "out.zarr", compute=False)
    assert len(delayeds) == 1
    assert isinstance(delayeds[0], Delayed)
    assert np.isnan(_open(tmp_path / "out.zarr")["test"].values).all()
    compute_writer_results([delayeds])
    _assert_roundtrip(tmp_path / "out.zarr", [data_arr])


def test_existing_store_raises_unless_mode_w(writer, tmp_path):
    """Test that an existing store is only overwritten with mode='w'."""
    writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr")
    with pytest.raises(FileExistsError):
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr")
    writer.save_datasets([_data_array("other")], filename=tmp_path / "out.zarr", mode="w")
    assert list(_open(tmp_path / "out.zarr").data_vars) == ["other"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_append_time(writer, tmp_path, zarr_format):
    """Test appending along a time dimension created from start_time, keeping the first write's attributes."""
    store = tmp_path / "out.zarr"
    first = _data_array()
    second = _data_array(start_time=dt.datetime(2026, 1, 1, 12, 10, 0, 654321, tzinfo=dt.timezone.utc), chunks=3)
    writer.save_datasets([first], filename=store, append_dim="time", zarr_format=zarr_format)
    writer.save_datasets([second], filename=store, append_dim="time", zarr_format=5 - zarr_format)
    ds = _open(store)
    assert list(ds.time.values) == _times("2026-01-01T12:00:00.123456", "2026-01-01T12:10:00.654321")
    assert ds["test"].dims == ("time", "y", "x")
    np.testing.assert_array_equal(ds["test"].values, np.stack([first.values, second.values]))
    assert ds["test"].attrs["start_time"] == first.attrs["start_time"].isoformat()
    metadata = open_group(store, mode="r").metadata
    assert metadata.zarr_format == zarr_format
    assert (metadata.consolidated_metadata is not None) == (zarr_format == 2)


@pytest.mark.parametrize("time_dims", [(), ("time",)])
def test_append_existing_time_coordinate(writer, tmp_path, time_dims):
    """Test appending data that already has a scalar or dimension time coordinate."""
    store = tmp_path / "out.zarr"
    data_arrays = [_data_array().assign_coords(time=np.datetime64(f"2026-01-01T{hour}:00", "ns")).expand_dims(time_dims)
                   for hour in (12, 13)]
    _save_steps(writer, store, data_arrays)
    assert list(_open(store).time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")


@pytest.mark.parametrize(
    ("start_time", "append_dim", "match"),
    [
        pytest.param(None, "time", "needs a 'start_time' attribute", id="missing_start_time"),
        pytest.param(START_TIME, "level", "only 'time' can be added", id="unknown_append_dim"),
    ],
)
def test_append_errors(writer, tmp_path, start_time, append_dim, match):
    """Test that appends the writer cannot set up raise."""
    with pytest.raises(ValueError, match=match):
        writer.save_datasets([_data_array(start_time=start_time)], filename=tmp_path / "out.zarr",
                             append_dim=append_dim)


def test_append_with_fill_value(writer, tmp_path):
    """Test appending integer data with a _FillValue attribute and a fill_value argument."""
    store = tmp_path / "out.zarr"
    data_arrays = [_data_array(start_time=dt.datetime(2026, 1, day), _FillValue=-1).astype(np.int16) for day in (1, 2)]
    _save_steps(writer, store, data_arrays, fill_value=-5)
    stored = _open(store)["test"]
    assert stored.sizes["time"] == 2
    assert stored.encoding["dtype"] == np.int16
    assert stored.encoding["_FillValue"] == -5
    assert open_group(store, mode="r")["test"].fill_value == -5


def test_append_with_encoding_argument(writer, tmp_path):
    """Test that the same encoding argument works for the first and every following append."""
    store = tmp_path / "out.zarr"
    encoding = {"test": {"dtype": "int16", "scale_factor": 0.25, "_FillValue": -1},
                "time": {"units": "seconds since 2000-01-01"}}
    input_encoding = copy.deepcopy(encoding)
    _save_steps(writer, store, _hourly(12, 13), encoding=encoding)
    assert encoding == input_encoding
    group = open_group(store, mode="r")
    assert group["test"].dtype == np.int16
    assert group["time"].attrs["units"] == "seconds since 2000-01-01"
    assert list(_open(store).time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_append_in_group(writer, tmp_path, zarr_format):
    """Test appending inside a group, which keeps the consolidated metadata in the root up to date."""
    store = tmp_path / "out.zarr"
    _save_steps(writer, store, _hourly(12, 13), zarr_format=zarr_format, xarray_kwargs={"group": "g"})
    assert list(_open(store, group="g").time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")


def test_mode_w_with_append_dim(writer, tmp_path):
    """Test that mode='w' with append_dim replaces the store with one time step, which later appends extend."""
    store = tmp_path / "out.zarr"
    _save_steps(writer, store, _hourly(12, 13))
    _save_steps(writer, store, _hourly(14), mode="w")
    _save_steps(writer, store, _hourly(15))
    assert list(_open(store).time.values) == _times("2026-01-01T14:00", "2026-01-01T15:00")


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    ("kwargs", "yaml_config", "zarr_format", "consolidated"),
    [
        pytest.param({}, {}, 3, False, id="default"),
        pytest.param({"consolidated": True}, {}, 3, True, id="format_3_consolidated"),
        pytest.param({"zarr_format": 2}, {}, 2, True, id="format_2"),
        pytest.param({"zarr_format": 2, "consolidated": False}, {}, 2, False, id="format_2_not_consolidated"),
        pytest.param({}, {"zarr_format": 2}, 2, True, id="yaml_config"),
        pytest.param({"zarr_format": 3}, {"zarr_format": 2}, 3, False, id="argument_over_yaml_config"),
    ],
)
def test_zarr_format(writer, tmp_path, kwargs, yaml_config, zarr_format, consolidated):
    """Test the zarr format and consolidated metadata of a new store, written without warnings."""
    with mock.patch.dict(writer.info, yaml_config):
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", **kwargs)
    group = open_group(tmp_path / "out.zarr", mode="r")
    assert group["test"].metadata.zarr_format == zarr_format
    assert (group.metadata.consolidated_metadata is not None) == consolidated


@pytest.mark.parametrize(
    ("variable_encoding", "encoding", "chunks", "dtype"),
    [
        pytest.param({}, None, (2, 6), np.float32, id="rechunked"),
        pytest.param({"dtype": "int16", "scale_factor": 0.25, "_FillValue": -1}, None, (2, 6), np.int16,
                     id="rechunked_keeps_encoding"),
        pytest.param({"chunks": (4, 3)}, None, (4, 3), np.float32, id="encoding_attribute_chunks"),
        pytest.param({}, {"test": {"chunks": (4, 3)}}, (4, 3), np.float32, id="encoding_kwarg_chunks"),
    ],
)
def test_irregular_chunks(writer, tmp_path, variable_encoding, encoding, chunks, dtype):
    """Test that irregular dask chunks are rechunked to their largest chunk size unless an encoding gives chunks."""
    data_arr = _data_array(area=_swath_definition(), chunks=((1, 2, 1), (6,)))
    data_arr.encoding = variable_encoding
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr", encoding=encoding)
    group = open_group(tmp_path / "out.zarr", mode="r")
    assert group["test"].chunks == chunks
    assert group["test"].dtype == dtype
    assert group["longitude"].chunks == (2, 6)
    _assert_roundtrip(tmp_path / "out.zarr", [data_arr])


def test_variable_encoding_used(writer, tmp_path):
    """Test that a DataArray's encoding is used, without the keys zarr does not understand."""
    data_arr = _data_array()
    data_arr.encoding = {"dtype": "int16", "scale_factor": 0.25, "add_offset": 0.0, "_FillValue": -1, "zlib": True,
                         "complevel": 4, "chunksizes": (4, 6), "source": "/in.nc", "original_shape": (4, 6)}
    input_encoding = copy.deepcopy(data_arr.encoding)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    assert data_arr.encoding == input_encoding
    array = open_group(tmp_path / "out.zarr", mode="r")["test"]
    assert array.dtype == np.int16
    assert array.attrs["scale_factor"] == 0.25
    assert array.chunks == (2, 2)
    _assert_roundtrip(tmp_path / "out.zarr", [data_arr])


def test_encoding_precedence(writer, tmp_path):
    """Test that fill_value overrides a DataArray's encoding and the encoding argument overrides both."""
    packed = _data_array("packed")
    packed.encoding = {"dtype": "int16", "scale_factor": 0.25, "_FillValue": -1}
    compressed = _data_array("compressed")
    compressed.encoding = {"compressors": [zarr.codecs.GzipCodec(level=5)], "dtype": "float64"}
    missing = _data_array("missing")
    missing.encoding = {"_FillValue": -5.0, "missing_value": -5.0}
    writer.save_datasets([packed, compressed, missing], filename=tmp_path / "out.zarr", fill_value=-9,
                         encoding={"compressed": {"dtype": "float32"}})
    group = open_group(tmp_path / "out.zarr", mode="r")
    assert group["packed"].dtype == np.int16
    assert group["packed"].attrs["_FillValue"] == -9
    assert [type(codec).__name__ for codec in group["compressed"].compressors] == ["GzipCodec"]
    assert group["compressed"].dtype == np.float32
    assert group["missing"].attrs["missing_value"] == -9


def test_variable_encoding_ignored_for_stored_variables(writer, tmp_path):
    """Test that appending uses the store's encoding instead of the appended DataArray's encoding."""
    store = tmp_path / "out.zarr"
    first, second = _hourly(12, 13)
    first.encoding = {"dtype": "int16", "scale_factor": 0.25, "_FillValue": -1}
    second.encoding = {"dtype": "float64"}
    _save_steps(writer, store, [first, second])
    assert open_group(store, mode="r")["test"].dtype == np.int16
    np.testing.assert_array_equal(_open(store)["test"].values, np.stack([first.values, second.values]))


@pytest.mark.parametrize(
    ("dtype", "variable_encoding", "fill_attr", "kwargs", "expected_fill", "stored_chunks"),
    [
        pytest.param(np.float32, {}, None, {}, np.nan, 3, id="float_nan"),
        pytest.param(np.float32, {}, -999.0, {}, np.nan, 3, id="float_fill_value_attribute"),
        pytest.param(np.float32, {}, np.nan, {}, np.nan, 3, id="float_nan_fill_value_attribute"),
        pytest.param(np.float32, {}, None, {"fill_value": -9}, -9, 3, id="fill_value_argument"),
        pytest.param(np.float32, {"dtype": "int16", "scale_factor": 0.25, "_FillValue": -1}, None, {}, -1, 3,
                     id="packed_encoding"),
        pytest.param(np.int16, {}, -1, {}, -1, 3, id="integer_fill_value_attribute"),
        pytest.param(np.int16, {}, -1, {"convert_to_cf": True}, -1, 3, id="integer_fill_value_attribute_cf"),
        pytest.param(np.float32, {"_FillValue": -9.0, "fill_value": 0.0}, None, {}, -9, 3,
                     id="variable_encoding_fill_value_dropped"),
        pytest.param(np.float32, {}, None, {"fill_value": -9, "encoding": {"test": {"fill_value": 0.0}}}, 0, 6,
                     id="encoding_argument"),
        pytest.param(np.float32, {"_FillValue": -9.0, "fill_value": 0.0}, None, {"fill_value": -9, "zarr_format": 2},
                     -9, 3, id="format_2"),
    ],
)
def test_zarr_fill_value(writer, tmp_path, dtype, variable_encoding, fill_attr, kwargs, expected_fill,
                         stored_chunks):
    """Test that new format 3 arrays get their _FillValue as zarr fill_value, so fill-only chunks are skipped."""
    data_arr = _masked_data_array()
    if fill_attr is not None:
        data_arr.attrs["_FillValue"] = fill_attr
    if np.issubdtype(dtype, np.integer):
        data_arr = data_arr.fillna(fill_attr).astype(dtype)
    data_arr.encoding = variable_encoding
    input_encoding = copy.deepcopy(variable_encoding)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr", **{"zarr_format": 3, **kwargs})
    assert data_arr.encoding == input_encoding
    array = open_group(tmp_path / "out.zarr", mode="r")["test"]
    np.testing.assert_equal(array.fill_value, expected_fill)
    assert array.nchunks_initialized == stored_chunks
    np.testing.assert_array_equal(_open(tmp_path / "out.zarr")["test"].values, _masked_data_array().values)


@pytest.mark.parametrize(
    ("time_encoding", "encoding", "units"),
    [
        pytest.param(None, None, "microseconds since 1970-01-01", id="default"),
        pytest.param({"units": "milliseconds since 2000-01-01", "dtype": "int64"}, None,
                     "milliseconds since 2000-01-01", id="encoding_attribute"),
        pytest.param(None, {"time": {"units": "seconds since 2000-01-01"}}, "seconds since 2000-01-01",
                     id="encoding_kwarg"),
    ],
)
def test_time_encoding(writer, tmp_path, time_encoding, encoding, units):
    """Test the units of the time coordinate."""
    data_arr = _data_array().assign_coords(time=np.datetime64("2026-01-01T12:00", "ns"))
    if time_encoding is not None:
        data_arr["time"].encoding = time_encoding
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr", encoding=encoding)
    assert open_group(tmp_path / "out.zarr", mode="r")["time"].attrs["units"] == units
    assert _open(tmp_path / "out.zarr").time.values == np.datetime64("2026-01-01T12:00")


def test_multiple_datasets_use_combined_metadata(writer, tmp_path):
    """Test that datasets share one store named and timed by their combined metadata."""
    first = _data_array("C01", platform_name="GOES-16")
    second = _data_array("C02", platform_name="GOES-16",
                         start_time=dt.datetime(2026, 1, 1, 11, 59, 59, 5, tzinfo=dt.timezone.utc))
    writer.save_datasets([first, second], filename=str(tmp_path / "{platform_name}_{start_time:%H%M%S%f}.zarr"),
                         append_dim="time")
    ds = _assert_roundtrip(tmp_path / "GOES-16_115959000005.zarr", [first, second])
    assert list(ds.time.values) == _times("2026-01-01T11:59:59.000005")


def test_split_variables(writer, tmp_path):
    """Test writing one store per dataset, each with the encoding argument entries of its dataset."""
    datasets = [_data_array("C01"), _data_array("C02")]
    writer.save_datasets(datasets, split_variables=True,
                         encoding={"C01": {"dtype": "float64"}, "C02": {"dtype": "int16", "_FillValue": -1}})
    for data_arr, dtype in zip(datasets, [np.float64, np.int16]):
        store = tmp_path / f"{data_arr.attrs['name']}_20260101_120000.zarr"
        ds = _assert_roundtrip(store, [data_arr])
        assert list(ds.data_vars) == [data_arr.attrs["name"]]
        assert open_group(store, mode="r")[data_arr.attrs["name"]].dtype == dtype


@pytest.mark.parametrize(
    ("names", "filename", "split_variables", "match"),
    [
        pytest.param(["C01", "C02"], None, False, "specify a 'filename' for multi-variable zarr stores",
                     id="no_common_name"),
        pytest.param(["C01"], "{platform_name}.zarr", False, "the dataset has no 'platform_name' attribute",
                     id="missing_attribute"),
        pytest.param(["C01", "C02"], "same.zarr", True, "would both be written to", id="same_filename"),
        pytest.param(["C01", "C01"], "same.zarr", False, "More than one dataset named 'C01'", id="duplicate_name"),
    ],
)
def test_filename_errors(writer, tmp_path, names, filename, split_variables, match):
    """Test errors for datasets that cannot be assigned to stores."""
    filename = None if filename is None else str(tmp_path / filename)
    with pytest.raises(ValueError, match=match):
        writer.save_datasets([_data_array(name) for name in names], filename=filename,
                             split_variables=split_variables)


@pytest.mark.parametrize(
    ("area", "other_area"),
    [
        pytest.param(AREA, AreaDefinition("other", "other", "other", "EPSG:3395", 6, 4, (0.0, 0.0, 6000.0, 2000.0)),
                     id="grid"),
        pytest.param(_swath_definition(), _swath_definition(lon_offset=1.0), id="swath"),
    ],
)
def test_different_coordinates_raise(writer, tmp_path, area, other_area):
    """Test that datasets on different grids or swaths cannot share a store."""
    with pytest.raises(ValueError, match="coordinates differ"):
        writer.save_datasets([_data_array("C01", area=area), _data_array("C02", area=other_area)],
                             filename=tmp_path / "out.zarr")


@pytest.mark.parametrize(
    ("second_start_time", "encoding", "match"),
    [
        pytest.param(None, None, "needs a 'start_time' attribute", id="missing_start_time"),
        pytest.param(START_TIME, {"C03": {"dtype": "float64"}}, r"entries for \['C03'\]", id="unknown_encoding_name"),
    ],
)
def test_invalid_store_writes_nothing(writer, tmp_path, second_start_time, encoding, match):
    """Test that no store is written when one is invalid, as to_zarr writes the metadata before computing."""
    datasets = [_data_array("C01"), _data_array("C02", start_time=second_start_time)]
    with pytest.raises(ValueError, match=match):
        writer.save_datasets(datasets, filename=str(tmp_path / "{name}.zarr"), split_variables=True,
                             append_dim="time", encoding=encoding, compute=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "second_kwargs",
    [
        pytest.param({}, id="default_mode"),
        pytest.param({"mode": "w"}, id="mode_w"),
        pytest.param({"mode": "a"}, id="mode_a"),
        pytest.param({"append_dim": "time"}, id="append"),
    ],
)
def test_groups_in_one_store(writer, tmp_path, zarr_format, second_kwargs):
    """Test writing datasets on different grids to groups of one store, all in the format of the store."""
    store = tmp_path / "out.zarr"
    fine = [_data_array("C01"), _data_array("C03")]
    coarse = _data_array("C13")
    writer.save_datasets(fine, filename=store, zarr_format=zarr_format, xarray_kwargs={"group": "1km"})
    writer.save_datasets([coarse], filename=store, zarr_format=5 - zarr_format, xarray_kwargs={"group": "2km"},
                         **second_kwargs)
    assert open_group(store, mode="r").metadata.zarr_format == zarr_format
    _assert_roundtrip(store, fine, group="1km")
    _assert_roundtrip(store, [coarse], group="2km")


@pytest.mark.parametrize(
    ("make_store", "split_variables"),
    [
        pytest.param(MemoryStore, False, id="one_store"),
        pytest.param(lambda: {"C01": MemoryStore(), "C02": MemoryStore()}, True, id="store_per_dataset"),
    ],
)
def test_store_objects(writer, make_store, split_variables):
    """Test writing to zarr stores instead of filenames."""
    datasets = [_data_array("C01"), _data_array("C02")]
    store = make_store()
    writer.save_datasets(datasets, store=store, split_variables=split_variables)
    for data_arr in datasets:
        _assert_roundtrip(store[data_arr.attrs["name"]] if split_variables else store, [data_arr])


@pytest.mark.parametrize(
    ("names", "make_kwargs", "match"),
    [
        pytest.param(["C01", "C02"], lambda: {"store": {"C01": MemoryStore(), "C02": MemoryStore()}},
                     "requires split_variables=True", id="dict_without_split"),
        pytest.param(["C01", "C02"], lambda: {"store": {"C01": MemoryStore()}, "split_variables": True},
                     "No store given for dataset 'C02'", id="missing_store"),
        pytest.param(["C01", "C01"], lambda: {"store": {"C01": MemoryStore()}, "split_variables": True},
                     "names are not unique", id="duplicate_names"),
        pytest.param(["C01", "C02"], lambda: {"store": MemoryStore(), "filename": "out.zarr"},
                     "cannot be used together", id="store_and_filename"),
        pytest.param(["C01", "C02"], lambda: {"store": MemoryStore(), "split_variables": True},
                     "Pass a dict of stores", id="single_store_split"),
        pytest.param(["C01", "C02"], lambda: {"store": MemoryStore(), "commit": "message"},
                     "requires an Icechunk Session", id="commit_plain_store"),
    ],
)
def test_store_errors(writer, names, make_kwargs, match):
    """Test invalid combinations of stores and keyword arguments."""
    with pytest.raises(ValueError, match=match):
        writer.save_datasets([_data_array(name) for name in names], **make_kwargs())


@pytest.mark.parametrize(
    ("filename", "forwarded"),
    [
        pytest.param("s3://bucket/out.zarr", True, id="remote"),
        pytest.param("memory://bucket/out.zarr", False, id="memory"),
        pytest.param("out.zarr", False, id="local"),
    ],
)
def test_storage_options_only_for_remote_urls(writer, filename, forwarded):
    """Test that storage_options are only passed to to_zarr for remote URLs."""
    with mock.patch.object(xr.Dataset, "to_zarr") as to_zarr:
        writer.save_datasets([_data_array()], filename=filename, storage_options={"anon": True}, compute=False)
    assert ("storage_options" in to_zarr.call_args.kwargs) == forwarded


def test_xarray_kwargs_override_writer_choices(writer, tmp_path):
    """Test that xarray_kwargs replace the to_zarr arguments chosen by the writer."""
    with mock.patch.object(xr.Dataset, "to_zarr") as to_zarr:
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", compute=False,
                             xarray_kwargs={"align_chunks": False, "safe_chunks": False})
    assert to_zarr.call_args.kwargs["align_chunks"] is False
    assert to_zarr.call_args.kwargs["safe_chunks"] is False


def test_xarray_kwargs_as_arguments(writer, tmp_path):
    """Test that to_zarr arguments in xarray_kwargs that save_datasets also takes are used as its arguments."""
    store = tmp_path / "out.zarr"
    xarray_kwargs = {"append_dim": "time", "zarr_format": 2, "consolidated": False, "group": "g",
                     "encoding": {"test": {"dtype": "int16"}}}
    for data_arr in _hourly(12, 13):
        writer.save_datasets([data_arr], filename=store, fill_value=-5, xarray_kwargs=xarray_kwargs)
    group = open_group(store, mode="r", path="g")
    assert group.metadata.zarr_format == 2
    assert group["test"].dtype == np.int16
    ds = _open(store, group="g")
    assert ds["test"].encoding["_FillValue"] == -5
    assert list(ds.time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        pytest.param("mode", "a", id="mode"),
        pytest.param("append_dim", "time", id="append_dim"),
        pytest.param("encoding", {}, id="encoding"),
        pytest.param("compute", False, id="compute"),
    ],
)
def test_arguments_also_in_xarray_kwargs_raise(writer, tmp_path, key, value):
    """Test that an argument passed both directly and in xarray_kwargs raises before anything is written."""
    with pytest.raises(ValueError, match=rf"\['{key}'\] given both as arguments and in 'xarray_kwargs'"):
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", xarray_kwargs={key: value},
                             **{key: value})
    assert not (tmp_path / "out.zarr").exists()


def test_memory_url(tmp_path):
    """Test writing to an fsspec URL without prepending base_dir or creating local directories."""
    import fsspec

    root = f"satpy-test-{uuid.uuid4()}"
    writer = ZarrWriter(base_dir=str(tmp_path), filename=f"memory://{root}/{{name}}.zarr")
    data_arr = _data_array()
    try:
        with mock.patch.object(os, "makedirs", wraps=os.makedirs) as makedirs:
            writer.save_datasets([data_arr], storage_options={})
        makedirs.assert_not_called()
        assert list(tmp_path.iterdir()) == []
        _assert_roundtrip(f"memory://{root}/test.zarr", [data_arr])
    finally:
        fsspec.filesystem("memory").rm(f"/{root}", recursive=True)


def test_unknown_kwargs_warn(writer, tmp_path):
    """Test that unknown keyword arguments warn while Satpy's image keyword arguments are ignored silently."""
    with pytest.warns(UserWarning, match=r"ignores these unknown keyword arguments: \['zar_format'\]"):
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", zar_format=2, overlay={},
                             decorate={})


def test_save_dataset_units_warn(writer, tmp_path):
    """Test that save_dataset warns that units are not converted."""
    with pytest.warns(UserWarning, match=r"unknown keyword arguments: \['units'\]"):
        writer.save_dataset(_data_array(), filename=tmp_path / "out.zarr", units="degC")
    assert _open(tmp_path / "out.zarr")["test"].attrs["units"] == "K"


@pytest.mark.parametrize("package", ["zarr", "xarray"])
def test_version_check(writer, tmp_path, package):
    """Test that too old zarr or xarray versions raise an ImportError."""
    def _version(name):
        return Version("1.0") if name == package else Version("9999.0")

    with mock.patch("satpy.writers.zarr._package_version", side_effect=_version), \
            pytest.raises(ImportError, match=f"requires {package}>="):
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr")


@pytest.mark.parametrize(
    "commit",
    [
        pytest.param(None, id="uncommitted"),
        pytest.param("Add data", id="committed"),
    ],
)
def test_icechunk_session(writer, icechunk_repo, commit):
    """Test writing to an Icechunk session, which is committed only with a commit message."""
    session = icechunk_repo.writable_session("main")
    initial_snapshot = icechunk_repo.lookup_branch("main")
    datasets = [_data_array("C01"), _data_array("C02")]
    writer.save_datasets(datasets, store=session, commit=commit)
    assert session.has_uncommitted_changes == (commit is None)
    assert (icechunk_repo.lookup_branch("main") != initial_snapshot) == (commit is not None)
    _assert_roundtrip(session.store, datasets)


def test_icechunk_append(writer, icechunk_repo):
    """Test appending to an Icechunk repository with a new session per write."""
    for data_arr in _hourly(12, 13):
        writer.save_datasets([data_arr], store=icechunk_repo.writable_session("main"), append_dim="time",
                             commit="Add a time step")
    ds = _open(icechunk_repo.readonly_session("main").store)
    assert list(ds.time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")


def test_icechunk_shared_session_commits_once(writer, icechunk_repo):
    """Test that one session holding several stores' datasets is committed once after all writes."""
    session = icechunk_repo.writable_session("main")
    datasets = [_data_array("C01"), _data_array("C02")]
    delayeds = writer.save_datasets(datasets, store={"C01": session, "C02": session}, split_variables=True,
                                    commit="Add data", mode="a", compute=False)
    assert len(delayeds) == 1
    compute_writer_results([delayeds])
    _assert_roundtrip(icechunk_repo.readonly_session("main").store, datasets)


def test_icechunk_zarr_format_2_raises(writer, icechunk_repo):
    """Test that zarr format 2 is refused for Icechunk."""
    with pytest.raises(ValueError, match="Icechunk only supports zarr format 3"):
        writer.save_datasets([_data_array()], store=icechunk_repo.writable_session("main"), zarr_format=2)


@pytest.mark.parametrize(
    ("area", "code", "bbox", "zarr_format"),
    [
        pytest.param(AREA, "EPSG:3395", [-1000.0, -500.0, 5000.0, 1500.0], 2, id="format_2"),
        pytest.param(AREA, "EPSG:3395", [-1000.0, -500.0, 5000.0, 1500.0], 3, id="format_3"),
        pytest.param(GEOS_AREA, None, [-3e6, -1e6, 3e6, 1e6], 3, id="no_authority_code"),
        pytest.param(FLIPPED_AREA, "EPSG:3395", [-1000.0, -500.0, 5000.0, 1500.0], 3, id="flipped"),
    ],
)
def test_geozarr_attrs(writer, tmp_path, area, code, bbox, zarr_format):
    """Test the GeoZarr proj and spatial attributes of gridded data and the root group."""
    writer.save_datasets([_data_array("C01", area=area), _data_array("rgb", ndim=3, area=area)],
                         filename=tmp_path / "out.zarr", zarr_format=zarr_format)
    group = open_group(tmp_path / "out.zarr", mode="r")
    cols, rows = np.meshgrid(np.arange(6) + 0.5, np.arange(4) + 0.5)
    x, y = area.get_proj_coords()
    for name in ("C01", "rgb"):
        attrs = group[name].attrs.asdict()
        assert [convention["uuid"] for convention in attrs["zarr_conventions"]] == [PROJ_UUID, SPATIAL_UUID]
        assert [convention["name"] for convention in attrs["zarr_conventions"]] == ["proj", "spatial"]
        assert attrs.get("proj:code") == code
        assert CRS(attrs["proj:wkt2"]) == area.crs
        assert CRS.from_json_dict(attrs["proj:projjson"]) == area.crs
        assert attrs["spatial:dimensions"] == ["y", "x"]
        a, b, c, d, e, f = attrs["spatial:transform"]
        np.testing.assert_allclose(a * cols + b * rows + c, x)
        np.testing.assert_allclose(d * cols + e * rows + f, y)
        assert attrs["spatial:bbox"] == bbox
        assert attrs["spatial:shape"] == [4, 6]
        assert attrs["spatial:registration"] == "pixel"
    root_attrs = group.attrs.asdict()
    assert [convention["uuid"] for convention in root_attrs["zarr_conventions"]] == [PROJ_UUID]
    assert root_attrs.get("proj:code") == code
    assert CRS(root_attrs["proj:wkt2"]) == area.crs


def test_geozarr_attrs_validate(writer, tmp_path):
    """Test the GeoZarr attributes with the validator of geozarr-toolkit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        geozarr_toolkit = pytest.importorskip("geozarr_toolkit")
    writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr")
    group = open_group(tmp_path / "out.zarr", mode="r")
    for attrs in (group["test"].attrs.asdict(), group.attrs.asdict()):
        errors = geozarr_toolkit.validate_attrs(attrs)
        assert not any(errors.values()), errors


@pytest.mark.parametrize(
    ("areas", "include_geozarr", "with_attrs"),
    [
        pytest.param(["swath"], True, [], id="swath"),
        pytest.param(["area", "swath"], True, ["C01"], id="area_and_swath"),
        pytest.param(["area"], False, [], id="disabled"),
    ],
)
def test_geozarr_attrs_skipped(writer, tmp_path, areas, include_geozarr, with_attrs):
    """Test that GeoZarr attributes are only added to gridded data and the root only if all are gridded."""
    area_defs = {"area": AREA, "swath": _swath_definition()}
    datasets = [_data_array(f"C0{num}", area=area_defs[area]) for num, area in enumerate(areas, start=1)]
    writer.save_datasets(datasets, filename=tmp_path / "out.zarr", include_geozarr=include_geozarr)
    group = open_group(tmp_path / "out.zarr", mode="r")
    assert [data_arr.attrs["name"] for data_arr in datasets if "proj:wkt2" in group[data_arr.attrs["name"]].attrs] \
        == with_attrs
    assert "zarr_conventions" not in group.attrs
    assert "proj:wkt2" not in group.attrs


@pytest.mark.parametrize(
    ("swath", "include_lonlats", "expected"),
    [(False, None, False), (False, True, True), (True, None, True), (True, False, False)],
)
def test_lonlats(writer, tmp_path, swath, include_lonlats, expected):
    """Test which data get longitude and latitude coordinates, without computing them early."""
    area = _swath_definition() if swath else AREA
    datasets = [_data_array("C01", area=area), _data_array("C02", area=area, chunks=3)]
    with assert_maximum_dask_computes(0):
        delayeds = writer.save_datasets(datasets, filename=tmp_path / "out.zarr", include_lonlats=include_lonlats,
                                        compute=False)
    compute_writer_results([delayeds])
    ds = _open(tmp_path / "out.zarr")
    assert ({"longitude", "latitude"} <= set(ds.coords)) == expected
    if expected:
        lons, lats = area.get_lonlats()
        np.testing.assert_allclose(ds["longitude"].values, lons)
        np.testing.assert_allclose(ds["latitude"].values, lats)
        assert ds["longitude"].attrs["standard_name"] == "longitude"


@pytest.mark.parametrize(
    ("include_lonlats", "warns", "has_lonlats"),
    [
        pytest.param(None, True, False, id="default"),
        pytest.param(True, False, True, id="include_lonlats"),
        pytest.param(False, False, False, id="no_lonlats"),
    ],
)
def test_stacked_area(writer, tmp_path, caplog, include_lonlats, warns, has_lonlats):
    """Test that data on a stack of shifted areas only get longitude/latitude, and a warning without them."""
    stacked = _stacked_area_definition()
    assert stacked.squeeze() is stacked
    writer.save_datasets([_data_array(area=stacked)], filename=tmp_path / "out.zarr", include_lonlats=include_lonlats)
    assert ("written without georeference" in caplog.text) == warns
    ds = _open(tmp_path / "out.zarr")
    assert ({"longitude", "latitude"} <= set(ds.coords)) == has_lonlats
    assert not any(key.startswith(("proj:", "spatial:")) for key in ds["test"].attrs)
    if has_lonlats:
        lons, lats = stacked.get_lonlats()
        np.testing.assert_allclose(ds["longitude"].values, lons)
        np.testing.assert_allclose(ds["latitude"].values, lats)


@pytest.mark.parametrize(
    "make_area",
    [
        pytest.param(_swath_definition, id="swath"),
        pytest.param(_stacked_area_definition, id="stacked"),
    ],
)
def test_existing_lonlats_kept(writer, tmp_path, caplog, make_area):
    """Test that existing longitude and latitude coordinates are not replaced and count as georeference."""
    data_arr = _data_array(area=make_area())
    data_arr = data_arr.assign_coords(longitude=data_arr * 0 + 1, latitude=data_arr * 0 + 2)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    ds = _open(tmp_path / "out.zarr")
    assert (ds["longitude"].values == 1).all()
    assert (ds["latitude"].values == 2).all()
    assert "without georeference" not in caplog.text


def test_lonlats_skipped_for_1d_data(writer, tmp_path):
    """Test that 1D swath data get no longitude and latitude coordinates."""
    writer.save_datasets([_data_array(area=_swath_definition()).isel(x=0)], filename=tmp_path / "out.zarr")
    assert "longitude" not in _open(tmp_path / "out.zarr").coords


def test_include_grid_mapping(writer, tmp_path):
    """Test the CF spatial_ref grid mapping coordinate, which is only added to gridded data."""
    writer.save_datasets([_data_array("C01"), _data_array("C02", area=_swath_definition())],
                         filename=tmp_path / "out.zarr", include_grid_mapping=True)
    group = open_group(tmp_path / "out.zarr", mode="r")
    assert group["C01"].attrs["grid_mapping"] == "spatial_ref"
    assert "grid_mapping" not in group["C02"].attrs
    assert CRS.from_cf(group["spatial_ref"].attrs.asdict()) == AREA.crs


def test_include_grid_mapping_rioxarray(writer, tmp_path):
    """Test that rioxarray reads the CRS and transform from the spatial_ref coordinate."""
    pytest.importorskip("rioxarray")
    writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", include_grid_mapping=True)
    data_arr = _open(tmp_path / "out.zarr")["test"]
    assert data_arr.rio.crs == AREA.crs
    assert tuple(data_arr.rio.transform())[:6] == (1000.0, 0.0, -1000.0, 0.0, -500.0, 1500.0)


def test_convert_to_cf(writer, tmp_path):
    """Test the CF conversion path."""
    writer.save_datasets([_data_array("C01", **_messy_attrs()), _data_array("C02")], filename=tmp_path / "out.zarr",
                         convert_to_cf=True, fill_value=-1, include_grid_mapping=True)
    group = open_group(tmp_path / "out.zarr", mode="r")
    assert group.attrs["Conventions"] == "CF-1.7"
    assert group["C01"].attrs["grid_mapping"] == "test"
    assert CRS.from_cf(group["test"].attrs.asdict()) == AREA.crs
    assert group["C01"].attrs["flag"] == "true"
    _strict_json(tmp_path / "out.zarr" / "C01" / "zarr.json")
    assert group["C01"].attrs["prerequisites"] == [str(make_dataid(name="C01")), str(DataQuery(name="C02"))]
    assert not any(key.startswith(("proj:", "spatial:")) for key in group["C01"].attrs)
    assert "spatial_ref" not in group
    assert {"longitude", "latitude"} <= set(_open(tmp_path / "out.zarr").coords)


def test_convert_to_cf_kwargs(writer, tmp_path):
    """Test that cf_kwargs are passed to the CF conversion."""
    writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", convert_to_cf=True,
                         cf_kwargs={"include_lonlats": False, "exclude_attrs": ["units"]})
    ds = _open(tmp_path / "out.zarr")
    assert "longitude" not in ds.coords
    assert "units" not in ds["test"].attrs


@pytest.mark.parametrize(
    ("cf_kwargs", "units"),
    [
        pytest.param({}, "microseconds since 1970-01-01", id="default"),
        pytest.param({"epoch": "milliseconds since 2000-01-01 00:00:00"}, "milliseconds since 2000-01-01", id="epoch"),
    ],
)
def test_convert_to_cf_time(writer, tmp_path, cf_kwargs, units):
    """Test the time and time_bnds encoding of the CF conversion and appending to its output."""
    store = tmp_path / "out.zarr"
    data_arrays = [_data_array(start_time=dt.datetime(2026, 1, 1, 12)).assign_coords(time=np.datetime64(time, "ns"))
                   for time in ("2026-01-01T12:00:00.250", "2026-01-01T12:10:00.500")]
    _save_steps(writer, store, data_arrays, convert_to_cf=True, cf_kwargs=cf_kwargs)
    group = open_group(store, mode="r")
    assert group["time"].attrs["units"] == units
    assert group["time_bnds"].attrs["units"] == units
    ds = _open(store)
    assert list(ds.time.values) == _times("2026-01-01T12:00:00.250", "2026-01-01T12:10:00.500")
    assert ds["time_bnds"].shape == (2, 2)


def test_convert_to_cf_append_time_from_start_time(writer, tmp_path):
    """Test that the CF conversion path also creates the time dimension from start_time, except for 0-d variables."""
    store = tmp_path / "out.zarr"
    _save_steps(writer, store, _hourly(12, 13, name="C01"), convert_to_cf=True, fill_value=-1)
    ds = _open(store)
    assert ds["C01"].dims == ("time", "y", "x")
    assert list(ds.time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")
    grid_mapping = ds["C01"].attrs["grid_mapping"]
    assert ds[grid_mapping].dims == ()
    assert "_FillValue" not in open_group(store, mode="r")[grid_mapping].attrs
