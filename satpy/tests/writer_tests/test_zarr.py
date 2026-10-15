"""Tests for the zarr writer."""

import copy
import datetime as dt
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
from pyresample.geometry import AreaDefinition, SwathDefinition
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
AREA = AreaDefinition("test", "test", "test", "EPSG:3395", 6, 4, (-1000.0, -500.0, 5000.0, 3500.0))
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


def _swath_definition():
    lons = xr.DataArray(da.linspace(0.0, 10.0, 24).reshape(4, 6).rechunk(2), dims=("y", "x"))
    lats = xr.DataArray(da.linspace(40.0, 50.0, 24).reshape(4, 6).rechunk(2), dims=("y", "x"))
    return SwathDefinition(lons, lats)


def _messy_attrs():
    return {
        "sensor": {"abi"},
        "flag": True,
        "none_value": None,
        "_satpy_id": make_dataid(name="test"),
        "_last_resampler": object(),
        "orbital_parameters": {"satellite_nominal_longitude": -75.0, "nested": {"a": [1, 2]}},
        "prerequisites": [make_dataid(name="C01"), DataQuery(name="C02")],
        "array": np.array([1, 2, 3], dtype=np.int16),
        "ancillary_variables": [_data_array("quality")],
        "wavelength": WavelengthRange(10.1, 10.3, 10.5, "µm"),
        "np_scalar": np.float32(1.5),
        "valid_time": np.datetime64("2026-01-01T12:00"),
        "bytes_value": b"abc",
    }


def _open(store, **kwargs):
    return xr.open_zarr(store, consolidated=False, **kwargs)


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
    assert zarr.open_group(tmp_path / "out.zarr", mode="r").metadata.zarr_format == 3


def test_attrs_sanitized(writer, tmp_path):
    """Test the conversion of Satpy attributes to JSON attributes without changing the input."""
    data_arr = _data_array(**_messy_attrs())
    input_attrs = dict(data_arr.attrs)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    attrs = zarr.open_group(tmp_path / "out.zarr", mode="r")["test"].attrs.asdict()
    for removed in ("area", "_satpy_id", "_last_resampler", "none_value"):
        assert removed not in attrs
    assert attrs["start_time"] == "2026-01-01T12:00:00.123456+00:00"
    assert attrs["sensor"] == ["abi"]
    assert attrs["flag"] is True
    assert attrs["orbital_parameters"] == {"satellite_nominal_longitude": -75.0, "nested": {"a": [1, 2]}}
    assert attrs["prerequisites"] == [str(make_dataid(name="C01")), str(DataQuery(name="C02"))]
    assert attrs["array"] == [1, 2, 3]
    assert attrs["ancillary_variables"] == ["quality"]
    assert WavelengthRange.from_cf(attrs["wavelength"]) == WavelengthRange(10.1, 10.3, 10.5, "µm")
    assert attrs["np_scalar"] == 1.5
    assert attrs["valid_time"] == "2026-01-01T12:00"
    assert attrs["bytes_value"] == "abc"
    assert data_arr.attrs.keys() == input_attrs.keys()
    assert "crs" in data_arr.coords
    assert "standard_name" not in data_arr["x"].attrs


@pytest.mark.parametrize(
    ("name", "stored_name"),
    [("dataarray_name", "dataarray_name"), (None, "data")],
    ids=["dataarray_name", "unnamed"],
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
    """Test appending along a time dimension created from start_time."""
    store = tmp_path / "out.zarr"
    first = _data_array()
    second = _data_array(start_time=dt.datetime(2026, 1, 1, 12, 10, 0, 654321, tzinfo=dt.timezone.utc), chunks=3)
    writer.save_datasets([first], filename=store, append_dim="time", zarr_format=zarr_format)
    writer.save_datasets([second], filename=store, append_dim="time", zarr_format=5 - zarr_format)
    ds = _open(store)
    assert list(ds.time.values) == _times("2026-01-01T12:00:00.123456", "2026-01-01T12:10:00.654321")
    assert ds["test"].dims == ("time", "y", "x")
    np.testing.assert_array_equal(ds["test"].values, np.stack([first.values, second.values]))
    metadata = zarr.open_group(store, mode="r").metadata
    assert metadata.zarr_format == zarr_format
    assert (metadata.consolidated_metadata is not None) == (zarr_format == 2)


@pytest.mark.parametrize("time_dims", [(), ("time",)])
def test_append_existing_time_coordinate(writer, tmp_path, time_dims):
    """Test appending data that already has a scalar or dimension time coordinate."""
    store = tmp_path / "out.zarr"
    for hour in (12, 13):
        data_arr = _data_array().assign_coords(time=np.datetime64(f"2026-01-01T{hour}:00", "ns"))
        if time_dims:
            data_arr = data_arr.expand_dims("time")
        writer.save_datasets([data_arr], filename=store, append_dim="time")
    assert list(_open(store).time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")


@pytest.mark.parametrize(
    ("start_time", "append_dim", "match"),
    [
        (None, "time", "needs a 'start_time' attribute"),
        (START_TIME, "level", "only 'time' can be added"),
    ],
    ids=["missing_start_time", "unknown_append_dim"],
)
def test_append_errors(writer, tmp_path, start_time, append_dim, match):
    """Test that appends the writer cannot set up raise."""
    with pytest.raises(ValueError, match=match):
        writer.save_datasets([_data_array(start_time=start_time)], filename=tmp_path / "out.zarr",
                             append_dim=append_dim)


def test_append_with_fill_value(writer, tmp_path):
    """Test appending integer data with a _FillValue attribute and a fill_value argument."""
    store = tmp_path / "out.zarr"
    for day in (1, 2):
        data_arr = _data_array(start_time=dt.datetime(2026, 1, day), _FillValue=-1).astype(np.int16)
        writer.save_datasets([data_arr], filename=store, append_dim="time", fill_value=-5)
    stored = _open(store)["test"]
    assert stored.sizes["time"] == 2
    assert stored.encoding["dtype"] == np.int16
    assert stored.encoding["_FillValue"] == -5


@pytest.mark.parametrize(
    ("zarr_format", "metadata_file", "consolidated"),
    [(2, ".zarray", True), (3, "zarr.json", False)],
)
def test_zarr_format(writer, tmp_path, zarr_format, metadata_file, consolidated):
    """Test the files written for each zarr format and their default consolidated metadata."""
    store = tmp_path / "out.zarr"
    writer.save_datasets([_data_array()], filename=store, zarr_format=zarr_format)
    assert (store / "test" / metadata_file).exists()
    assert (store / ".zmetadata").exists() == consolidated


def test_zarr_format_from_yaml_config(writer, tmp_path):
    """Test that the zarr_format of the writer configuration is the default."""
    with mock.patch.dict(writer.info, {"zarr_format": 2}):
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr")
    assert (tmp_path / "out.zarr" / "test" / ".zarray").exists()


@pytest.mark.parametrize(
    ("zarr_format", "consolidated"),
    [(3, None), (3, True), (2, None), (2, False)],
)
def test_no_warnings(writer, tmp_path, zarr_format, consolidated):
    """Test that writing emits no warnings, including zarr's consolidated metadata warning for format 3."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", zarr_format=zarr_format,
                             consolidated=consolidated)


def test_irregular_chunks_rechunked(writer, tmp_path):
    """Test that irregular dask chunks of variables and coordinates are rechunked to their largest chunk size."""
    data_arr = _data_array(area=_swath_definition())
    data_arr = data_arr.copy(data=data_arr.data.rechunk(((1, 2, 1), (6,))))
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    group = zarr.open_group(tmp_path / "out.zarr", mode="r")
    assert group["test"].chunks == (2, 6)
    assert group["longitude"].chunks == (2, 6)


@pytest.mark.parametrize("from_attribute", [False, True], ids=["encoding_kwarg", "encoding_attribute"])
def test_encoding_chunks_used(writer, tmp_path, from_attribute):
    """Test that chunks given in an encoding are used instead of regularizing the dask chunks."""
    data_arr = _data_array()
    data_arr = data_arr.copy(data=data_arr.data.rechunk(((1, 2, 1), (6,))))
    encoding = None
    if from_attribute:
        data_arr.encoding = {"chunks": (4, 3)}
    else:
        encoding = {"test": {"chunks": (4, 3)}}
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr", encoding=encoding)
    assert zarr.open_group(tmp_path / "out.zarr", mode="r")["test"].chunks == (4, 3)


def test_fill_value(writer, tmp_path):
    """Test that fill_value sets the _FillValue of the data variables and replaces a _FillValue attribute."""
    data_arr = _data_array(_FillValue=-1).astype(np.int16)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr", fill_value=-999)
    stored = _open(tmp_path / "out.zarr", mask_and_scale=False)["test"]
    assert stored.dtype == np.int16
    assert stored.attrs["_FillValue"] == -999


def test_variable_encoding_used(writer, tmp_path):
    """Test that a DataArray's encoding is used, without the keys zarr does not understand."""
    data_arr = _data_array()
    data_arr.encoding = {"dtype": "int16", "scale_factor": 0.25, "add_offset": 0.0, "_FillValue": -1, "zlib": True,
                         "complevel": 4, "chunksizes": (4, 6), "source": "/in.nc", "original_shape": (4, 6)}
    input_encoding = copy.deepcopy(data_arr.encoding)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    assert data_arr.encoding == input_encoding
    array = zarr.open_group(tmp_path / "out.zarr", mode="r")["test"]
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
    group = zarr.open_group(tmp_path / "out.zarr", mode="r")
    assert group["packed"].dtype == np.int16
    assert group["packed"].attrs["_FillValue"] == -9
    assert [type(codec).__name__ for codec in group["compressed"].compressors] == ["GzipCodec"]
    assert group["compressed"].dtype == np.float32
    assert group["missing"].attrs["missing_value"] == -9


def test_variable_encoding_ignored_for_stored_variables(writer, tmp_path):
    """Test that appending uses the store's encoding instead of the appended DataArray's encoding."""
    store = tmp_path / "out.zarr"
    first = _data_array()
    first.encoding = {"dtype": "int16", "scale_factor": 0.25, "_FillValue": -1}
    second = _data_array(start_time=dt.datetime(2026, 1, 1, 13, tzinfo=dt.timezone.utc))
    second.encoding = {"dtype": "float64"}
    writer.save_datasets([first], filename=store, append_dim="time")
    writer.save_datasets([second], filename=store, append_dim="time")
    assert zarr.open_group(store, mode="r")["test"].dtype == np.int16
    np.testing.assert_array_equal(_open(store)["test"].values, np.stack([first.values, second.values]))


@pytest.mark.parametrize(
    ("time_encoding", "encoding", "units"),
    [
        (None, None, "microseconds since 1970-01-01"),
        ({"units": "milliseconds since 2000-01-01", "dtype": "int64"}, None, "milliseconds since 2000-01-01"),
        (None, {"time": {"units": "seconds since 2000-01-01"}}, "seconds since 2000-01-01"),
    ],
    ids=["default", "encoding_attribute", "encoding_kwarg"],
)
def test_time_encoding(writer, tmp_path, time_encoding, encoding, units):
    """Test the units of the time coordinate."""
    data_arr = _data_array().assign_coords(time=np.datetime64("2026-01-01T12:00", "ns"))
    if time_encoding is not None:
        data_arr["time"].encoding = time_encoding
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr", encoding=encoding)
    assert zarr.open_group(tmp_path / "out.zarr", mode="r")["time"].attrs["units"] == units
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
    """Test writing one store per dataset."""
    datasets = [_data_array("C01"), _data_array("C02")]
    writer.save_datasets(datasets, split_variables=True)
    for data_arr in datasets:
        ds = _assert_roundtrip(tmp_path / f"{data_arr.attrs['name']}_20260101_120000.zarr", [data_arr])
        assert list(ds.data_vars) == [data_arr.attrs["name"]]


@pytest.mark.parametrize(
    ("names", "filename", "split_variables", "match"),
    [
        (["C01", "C02"], None, False, "specify a 'filename' for multi-variable zarr stores"),
        (["C01"], "{platform_name}.zarr", False, "the dataset has no 'platform_name' attribute"),
        (["C01", "C02"], "same.zarr", True, "would both be written to"),
        (["C01", "C01"], "same.zarr", False, "More than one dataset named 'C01'"),
    ],
    ids=["no_common_name", "missing_attribute", "same_filename", "duplicate_name"],
)
def test_filename_errors(writer, tmp_path, names, filename, split_variables, match):
    """Test errors for datasets that cannot be assigned to stores."""
    filename = None if filename is None else str(tmp_path / filename)
    with pytest.raises(ValueError, match=match):
        writer.save_datasets([_data_array(name) for name in names], filename=filename,
                             split_variables=split_variables)


def test_different_coordinates_raise(writer, tmp_path):
    """Test that datasets on different grids cannot share a store."""
    other_area = AreaDefinition("other", "other", "other", "EPSG:3395", 6, 4, (0.0, 0.0, 6000.0, 4000.0))
    with pytest.raises(ValueError, match="coordinates differ"):
        writer.save_datasets([_data_array("C01"), _data_array("C02", area=other_area)],
                             filename=tmp_path / "out.zarr")


def test_groups_in_one_store(writer, tmp_path):
    """Test writing datasets on different grids to groups of one store."""
    store = tmp_path / "out.zarr"
    fine = [_data_array("C01"), _data_array("C03")]
    coarse = _data_array("C13")
    writer.save_datasets(fine, filename=store, xarray_kwargs={"group": "1km"})
    writer.save_datasets([coarse], filename=store, mode="a", xarray_kwargs={"group": "2km"})
    _assert_roundtrip(store, fine, group="1km")
    _assert_roundtrip(store, [coarse], group="2km")


@pytest.mark.parametrize("split_variables", [False, True])
def test_store_objects(writer, split_variables):
    """Test writing to zarr stores instead of filenames."""
    datasets = [_data_array("C01"), _data_array("C02")]
    if split_variables:
        store = {"C01": MemoryStore(), "C02": MemoryStore()}
        writer.save_datasets(datasets, store=store, split_variables=True)
        for data_arr in datasets:
            _assert_roundtrip(store[data_arr.attrs["name"]], [data_arr])
    else:
        store = MemoryStore()
        writer.save_datasets(datasets, store=store)
        _assert_roundtrip(store, datasets)


@pytest.mark.parametrize(
    ("make_kwargs", "match"),
    [
        (lambda: {"store": {"C01": MemoryStore(), "C02": MemoryStore()}}, "requires split_variables=True"),
        (lambda: {"store": {"C01": MemoryStore()}, "split_variables": True}, "No store given for dataset 'C02'"),
        (lambda: {"store": MemoryStore(), "filename": "out.zarr"}, "cannot be used together"),
        (lambda: {"store": MemoryStore(), "split_variables": True}, "Pass a dict of stores"),
        (lambda: {"store": MemoryStore(), "commit": "message"}, "requires an Icechunk Session"),
    ],
    ids=["dict_without_split", "missing_store", "store_and_filename", "single_store_split", "commit_plain_store"],
)
def test_store_errors(writer, make_kwargs, match):
    """Test invalid combinations of stores and keyword arguments."""
    with pytest.raises(ValueError, match=match):
        writer.save_datasets([_data_array("C01"), _data_array("C02")], **make_kwargs())


def test_store_dict_with_duplicate_names_raises(writer):
    """Test that stores cannot be assigned by name to datasets with the same name."""
    with pytest.raises(ValueError, match="names are not unique"):
        writer.save_datasets([_data_array("C01"), _data_array("C01")], store={"C01": MemoryStore()},
                             split_variables=True)


@pytest.mark.parametrize(
    ("filename", "forwarded"),
    [("s3://bucket/out.zarr", True), ("memory://bucket/out.zarr", False), ("out.zarr", False)],
    ids=["remote", "memory", "local"],
)
def test_storage_options_only_for_remote_urls(writer, filename, forwarded):
    """Test that storage_options are only passed to to_zarr for remote URLs."""
    with mock.patch.object(xr.Dataset, "to_zarr") as to_zarr:
        writer.save_datasets([_data_array()], filename=filename, storage_options={"anon": True}, compute=False)
    assert ("storage_options" in to_zarr.call_args.kwargs) == forwarded


def test_xarray_kwargs_override_writer_choices(writer, tmp_path):
    """Test that xarray_kwargs replace the to_zarr arguments chosen by the writer."""
    writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr",
                         xarray_kwargs={"zarr_format": 2, "consolidated": False, "group": "g"})
    metadata = zarr.open_group(tmp_path / "out.zarr", mode="r", path="g").metadata
    assert metadata.zarr_format == 2
    assert metadata.consolidated_metadata is None


def test_memory_url(tmp_path):
    """Test writing to an fsspec URL without creating local directories."""
    import fsspec

    root = f"satpy-test-{uuid.uuid4()}"
    writer = ZarrWriter(filename=f"memory://{root}/{{name}}.zarr")
    data_arr = _data_array()
    try:
        with mock.patch.object(os, "makedirs", wraps=os.makedirs) as makedirs:
            writer.save_datasets([data_arr], storage_options={})
        makedirs.assert_not_called()
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


def test_icechunk_commit(writer, icechunk_repo):
    """Test writing to an Icechunk session and committing it."""
    datasets = [_data_array("C01"), _data_array("C02")]
    writer.save_datasets(datasets, store=icechunk_repo.writable_session("main"), commit="Add data")
    _assert_roundtrip(icechunk_repo.readonly_session("main").store, datasets)


def test_icechunk_append(writer, icechunk_repo):
    """Test appending to an Icechunk repository with a new session per write."""
    for hour in (12, 13):
        data_arr = _data_array(start_time=dt.datetime(2026, 1, 1, hour, tzinfo=dt.timezone.utc))
        writer.save_datasets([data_arr], store=icechunk_repo.writable_session("main"), append_dim="time",
                             commit=f"Add {hour}:00")
    ds = _open(icechunk_repo.readonly_session("main").store)
    assert list(ds.time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")


def test_icechunk_without_commit(writer, icechunk_repo):
    """Test that the session is left uncommitted without commit."""
    session = icechunk_repo.writable_session("main")
    writer.save_datasets([_data_array()], store=session)
    assert session.has_uncommitted_changes


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


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_geozarr_attrs(writer, tmp_path, zarr_format):
    """Test the GeoZarr proj and spatial attributes of gridded data and the root group."""
    writer.save_datasets([_data_array("C01"), _data_array("rgb", ndim=3)], filename=tmp_path / "out.zarr",
                         zarr_format=zarr_format)
    group = zarr.open_group(tmp_path / "out.zarr", mode="r")
    for name in ("C01", "rgb"):
        attrs = group[name].attrs.asdict()
        assert [convention["uuid"] for convention in attrs["zarr_conventions"]] == [PROJ_UUID, SPATIAL_UUID]
        assert [convention["name"] for convention in attrs["zarr_conventions"]] == ["proj", "spatial"]
        assert attrs["proj:code"] == "EPSG:3395"
        assert CRS(attrs["proj:wkt2"]) == AREA.crs
        assert CRS.from_json_dict(attrs["proj:projjson"]) == AREA.crs
        assert attrs["spatial:dimensions"] == ["y", "x"]
        assert attrs["spatial:transform"] == [1000.0, 0.0, -1000.0, 0.0, -1000.0, 3500.0]
        assert attrs["spatial:bbox"] == [-1000.0, -500.0, 5000.0, 3500.0]
        assert attrs["spatial:shape"] == [4, 6]
        assert attrs["spatial:registration"] == "pixel"
    root_attrs = group.attrs.asdict()
    assert [convention["uuid"] for convention in root_attrs["zarr_conventions"]] == [PROJ_UUID]
    assert root_attrs["proj:code"] == "EPSG:3395"


def test_geozarr_attrs_validate(writer, tmp_path):
    """Test the GeoZarr attributes with the validator of geozarr-toolkit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        geozarr_toolkit = pytest.importorskip("geozarr_toolkit")
    writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr")
    group = zarr.open_group(tmp_path / "out.zarr", mode="r")
    for attrs in (group["test"].attrs.asdict(), group.attrs.asdict()):
        errors = geozarr_toolkit.validate_attrs(attrs)
        assert not any(errors.values()), errors


def test_geozarr_without_authority_code(writer, tmp_path):
    """Test that proj:code is left out for a CRS without an authority code."""
    geos = AreaDefinition("geos", "geos", "geos", {"proj": "geos", "h": 35786023.0, "lon_0": -75.0, "sweep": "x"},
                          6, 4, (-3e6, -2e6, 3e6, 2e6))
    writer.save_datasets([_data_array(area=geos)], filename=tmp_path / "out.zarr")
    attrs = zarr.open_group(tmp_path / "out.zarr", mode="r")["test"].attrs.asdict()
    assert "proj:code" not in attrs
    assert CRS(attrs["proj:wkt2"]) == geos.crs


@pytest.mark.parametrize(
    ("areas", "include_geozarr", "with_attrs"),
    [
        (["swath"], True, []),
        (["area", "swath"], True, ["C01"]),
        (["area"], False, []),
    ],
    ids=["swath", "area_and_swath", "disabled"],
)
def test_geozarr_attrs_skipped(writer, tmp_path, areas, include_geozarr, with_attrs):
    """Test that GeoZarr attributes are only added to gridded data and the root only if all are gridded."""
    area_defs = {"area": AREA, "swath": _swath_definition()}
    datasets = [_data_array(f"C0{num}", area=area_defs[area]) for num, area in enumerate(areas, start=1)]
    writer.save_datasets(datasets, filename=tmp_path / "out.zarr", include_geozarr=include_geozarr)
    group = zarr.open_group(tmp_path / "out.zarr", mode="r")
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
    datasets = [_data_array("C01", area=area), _data_array("C02", area=area)]
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


def test_existing_lonlats_kept(writer, tmp_path):
    """Test that existing longitude and latitude coordinates are not replaced."""
    data_arr = _data_array(area=_swath_definition())
    data_arr = data_arr.assign_coords(longitude=data_arr * 0 + 1, latitude=data_arr * 0 + 2)
    writer.save_datasets([data_arr], filename=tmp_path / "out.zarr")
    ds = _open(tmp_path / "out.zarr")
    assert (ds["longitude"].values == 1).all()
    assert (ds["latitude"].values == 2).all()


def test_lonlats_skipped_for_1d_data(writer, tmp_path):
    """Test that 1D swath data get no longitude and latitude coordinates."""
    writer.save_datasets([_data_array(area=_swath_definition()).isel(x=0)], filename=tmp_path / "out.zarr")
    assert "longitude" not in _open(tmp_path / "out.zarr").coords


def test_include_grid_mapping(writer, tmp_path):
    """Test the CF spatial_ref grid mapping coordinate, which is only added to gridded data."""
    writer.save_datasets([_data_array("C01"), _data_array("C02", area=_swath_definition())],
                         filename=tmp_path / "out.zarr", include_grid_mapping=True)
    group = zarr.open_group(tmp_path / "out.zarr", mode="r")
    assert group["C01"].attrs["grid_mapping"] == "spatial_ref"
    assert "grid_mapping" not in group["C02"].attrs
    assert CRS.from_cf(group["spatial_ref"].attrs.asdict()) == AREA.crs


def test_include_grid_mapping_rioxarray(writer, tmp_path):
    """Test that rioxarray reads the CRS and transform from the spatial_ref coordinate."""
    pytest.importorskip("rioxarray")
    writer.save_datasets([_data_array()], filename=tmp_path / "out.zarr", include_grid_mapping=True)
    data_arr = _open(tmp_path / "out.zarr")["test"]
    assert data_arr.rio.crs == AREA.crs
    assert tuple(data_arr.rio.transform())[:6] == (1000.0, 0.0, -1000.0, 0.0, -1000.0, 3500.0)


def test_convert_to_cf(writer, tmp_path):
    """Test the CF conversion path."""
    writer.save_datasets([_data_array("C01", **_messy_attrs()), _data_array("C02")], filename=tmp_path / "out.zarr",
                         convert_to_cf=True, fill_value=-1, include_grid_mapping=True)
    group = zarr.open_group(tmp_path / "out.zarr", mode="r")
    assert group.attrs["Conventions"] == "CF-1.7"
    assert group["C01"].attrs["grid_mapping"] == "test"
    assert CRS.from_cf(group["test"].attrs.asdict()) == AREA.crs
    assert group["C01"].attrs["flag"] == "true"
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
    [({}, "microseconds since 1970-01-01"),
     ({"epoch": "milliseconds since 2000-01-01 00:00:00"}, "milliseconds since 2000-01-01")],
    ids=["default", "epoch"],
)
def test_convert_to_cf_time(writer, tmp_path, cf_kwargs, units):
    """Test the time and time_bnds encoding of the CF conversion and appending to its output."""
    store = tmp_path / "out.zarr"
    for time in ("2026-01-01T12:00:00.250", "2026-01-01T12:10:00.500"):
        data_arr = _data_array(start_time=dt.datetime(2026, 1, 1, 12)).assign_coords(time=np.datetime64(time, "ns"))
        writer.save_datasets([data_arr], filename=store, convert_to_cf=True, cf_kwargs=cf_kwargs, append_dim="time")
    group = zarr.open_group(store, mode="r")
    assert group["time"].attrs["units"] == units
    assert group["time_bnds"].attrs["units"] == units
    ds = _open(store)
    assert list(ds.time.values) == _times("2026-01-01T12:00:00.250", "2026-01-01T12:10:00.500")
    assert ds["time_bnds"].shape == (2, 2)


def test_convert_to_cf_append_time_from_start_time(writer, tmp_path):
    """Test that the CF conversion path also creates the time dimension from start_time."""
    store = tmp_path / "out.zarr"
    for hour in (12, 13):
        data_arr = _data_array(start_time=dt.datetime(2026, 1, 1, hour, tzinfo=dt.timezone.utc))
        writer.save_datasets([data_arr], filename=store, convert_to_cf=True, append_dim="time")
    ds = _open(store)
    assert ds["test"].dims == ("time", "y", "x")
    assert list(ds.time.values) == _times("2026-01-01T12:00", "2026-01-01T13:00")
