"""Tests for compatible netCDF/Zarr DataArray encodings."""
import copy
import datetime

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.cf.encoding import update_encoding


def _make_fake_ds(var_names=("foo", "bar"), chunks=(1, 2)):
    """Create a dask-backed dataset like the ones prepared by the CF writer."""
    data_vars = {
        var_name: (("y", "x"), da.arange(4 * i, 4 * (i + 1)).reshape(2, 2).rechunk(chunks))
        for i, var_name in enumerate(var_names)
    }
    lon = da.arange(7, 11).reshape(2, 2).rechunk(chunks)
    return xr.Dataset(data_vars, coords={"y": [1, 2], "x": [3, 4], "lon": (("y", "x"), lon)})


def _make_fake_ds_with_time(time_encoding=None):
    """Create a dataset with a time dimension."""
    ds = _make_fake_ds(chunks=2).expand_dims({"time": [datetime.datetime(2009, 7, 1, 12, 15)]})
    ds["time"].encoding = time_encoding or {}
    return ds


class TestUpdateEncoding:
    """Test update of dataset encodings."""

    @pytest.fixture
    def fake_ds(self):
        """Create fake data for testing."""
        return _make_fake_ds(chunks=-1)

    @pytest.fixture
    def fake_ds_digit(self):
        """Create fake data for testing."""
        return _make_fake_ds(var_names=("CHANNEL_1", "CHANNEL_2"))

    def test_dataset_name_digit(self, fake_ds_digit):
        """Test data with dataset name staring with a digit."""
        # Dataset with name staring with digit
        ds_digit = fake_ds_digit
        kwargs = {"encoding": {"1": {"dtype": "float32"},
                               "2": {"dtype": "float32"}},
                  "other": "kwargs"}
        enc, other_kwargs = update_encoding(ds_digit, kwargs, numeric_name_prefix="CHANNEL_")
        expected_dict = {
            "y": {"_FillValue": None},
            "x": {"_FillValue": None},
            "lon": {"chunksizes": (1, 2)},
            "CHANNEL_1": {"dtype": "float32", "chunksizes": (1, 2)},
            "CHANNEL_2": {"dtype": "float32", "chunksizes": (1, 2)}
        }
        assert enc == expected_dict
        assert other_kwargs == {"other": "kwargs"}

    def test_without_time(self, fake_ds):
        """Test data with no time dimension."""
        # Without time dimension
        ds = fake_ds.chunk(2)
        kwargs = {"encoding": {"bar": {"chunksizes": (1, 1)}},
                  "other": "kwargs"}
        enc, other_kwargs = update_encoding(ds, kwargs)
        expected_dict = {
            "y": {"_FillValue": None},
            "x": {"_FillValue": None},
            "lon": {"chunksizes": (2, 2)},
            "foo": {"chunksizes": (2, 2)},
            "bar": {"chunksizes": (1, 1)}
        }
        assert enc == expected_dict
        assert other_kwargs == {"other": "kwargs"}

        # Chunksize may not exceed shape
        ds = fake_ds.chunk(8)
        kwargs = {"encoding": {}, "other": "kwargs"}
        enc, other_kwargs = update_encoding(ds, kwargs)
        expected_dict = {
            "y": {"_FillValue": None},
            "x": {"_FillValue": None},
            "lon": {"chunksizes": (2, 2)},
            "foo": {"chunksizes": (2, 2)},
            "bar": {"chunksizes": (2, 2)}
        }
        assert enc == expected_dict

    def test_with_time(self, fake_ds):
        """Test data with a time dimension."""
        # With time dimension
        ds = fake_ds.chunk(8).expand_dims({"time": [datetime.datetime(2009, 7, 1, 12, 15)]})
        kwargs = {"encoding": {"bar": {"chunksizes": (1, 1, 1)}},
                  "other": "kwargs"}
        enc, other_kwargs = update_encoding(ds, kwargs)
        expected_dict = {
            "y": {"_FillValue": None},
            "x": {"_FillValue": None},
            "lon": {"chunksizes": (2, 2)},
            "foo": {"chunksizes": (1, 2, 2)},
            "bar": {"chunksizes": (1, 1, 1)},
            "time": {"_FillValue": None,
                     "calendar": "proleptic_gregorian",
                     "dtype": "float64",
                     "units": "days since 2009-07-01 12:15:00"},
            "time_bnds": {"_FillValue": None,
                          "dtype": "float64",
                          "calendar": "proleptic_gregorian",
                          "units": "days since 2009-07-01 12:15:00"}
        }
        assert enc == expected_dict
        # User-defined encoding may not be altered
        assert kwargs["encoding"] == {"bar": {"chunksizes": (1, 1, 1)}}


class TestVariableEncodingAttribute:
    """Test merging the variables' own ``encoding`` attribute into the encoding passed to xarray."""

    @pytest.mark.parametrize(
        ("own_encoding", "user_encoding", "expected"),
        [
            pytest.param(
                {"dtype": "int16", "scale_factor": 0.5, "_FillValue": -1, "zlib": True, "complevel": 4},
                {},
                {"chunksizes": (1, 2), "dtype": "int16", "scale_factor": 0.5, "_FillValue": -1, "zlib": True,
                 "complevel": 4},
                id="own_encoding_over_default_chunks",
            ),
            pytest.param(
                {"dtype": "int16", "scale_factor": 0.5, "_FillValue": -1, "chunksizes": (2, 1)},
                {"scale_factor": 0.25, "chunksizes": (2, 2)},
                {"chunksizes": (2, 2), "dtype": "int16", "scale_factor": 0.25, "_FillValue": -1},
                id="user_encoding_over_own_encoding_per_key",
            ),
            pytest.param(
                {"zlib": True, "complevel": 4, "shuffle": True},
                {"compression": "zlib"},
                {"chunksizes": (1, 2), "shuffle": True, "compression": "zlib"},
                id="compression_from_one_source",
            ),
            pytest.param(
                {"chunksizes": (2, 1)},
                {"contiguous": True},
                {"contiguous": True},
                id="chunking_from_one_source",
            ),
        ],
    )
    def test_precedence(self, own_encoding, user_encoding, expected):
        """Test precedence of default, own and user-defined encodings."""
        ds = _make_fake_ds()
        ds["foo"].encoding = own_encoding
        enc, _ = update_encoding(ds, {"encoding": {"foo": user_encoding}})
        assert enc["foo"] == expected

    @pytest.mark.parametrize(
        ("original_shape", "chunksizes", "exp_chunksizes"),
        [
            ((2, 2), (2, 1), (2, 1)),
            ((4, 2), (2, 1), (1, 2)),
            ((2, 2), (4, 4), (1, 2)),
            ((2, 2), None, (1, 2)),
        ],
    )
    def test_encoding_read_from_file(self, original_shape, chunksizes, exp_chunksizes):
        """Test that only keys that xarray accepts for writing are taken from encoding read from a file.

        Chunk sizes are only used if they fit the data as they are now.
        """
        ds = _make_fake_ds()
        ds["foo"].encoding = {
            "dtype": np.dtype("int16"), "zlib": True, "szip": False, "zstd": False, "bzip2": False, "blosc": False,
            "shuffle": True, "complevel": 4, "fletcher32": False, "contiguous": chunksizes is None,
            "chunksizes": chunksizes, "preferred_chunks": {"y": 2, "x": 1}, "source": "/path/to/file.nc",
            "original_shape": original_shape, "_FillValue": np.int16(-1), "coordinates": "lon",
        }
        enc, _ = update_encoding(ds, {})
        assert enc["foo"] == {
            "dtype": np.dtype("int16"), "zlib": True, "shuffle": True, "complevel": 4, "fletcher32": False,
            "chunksizes": exp_chunksizes, "_FillValue": np.int16(-1),
        }

    def test_encoding_also_in_attrs(self):
        """Test that encoding keys which are also attributes are ignored, as xarray would refuse to write them."""
        ds = _make_fake_ds()
        ds["foo"].attrs["_FillValue"] = 255
        ds["foo"].encoding = {"_FillValue": 254, "dtype": "uint8"}
        enc, _ = update_encoding(ds, {})
        assert enc["foo"] == {"chunksizes": (1, 2), "dtype": "uint8"}

    @pytest.mark.parametrize(
        ("user_encoding", "exp_fill_value"),
        [
            ({}, None),
            ({"x": {"_FillValue": -1}}, -1),
        ],
    )
    def test_dimension_coordinate_fill_value(self, user_encoding, exp_fill_value):
        """Test that dimension coordinates get no fill value unless the user asks for one."""
        ds = _make_fake_ds()
        ds["x"].encoding = {"_FillValue": np.nan, "dtype": "float32"}
        enc, _ = update_encoding(ds, {"encoding": user_encoding})
        assert enc["x"] == {"_FillValue": exp_fill_value, "dtype": "float32"}

    @pytest.mark.parametrize(
        ("time_encoding", "user_encoding", "exp_units"),
        [
            pytest.param({}, {}, "days since 2009-07-01 12:15:00", id="default"),
            pytest.param({"units": "seconds since 1970-01-01"}, {}, "seconds since 1970-01-01", id="epoch"),
            pytest.param({"units": "seconds since 1970-01-01"}, {"time": {"units": "hours since 2009-07-01"}},
                         "hours since 2009-07-01", id="user"),
        ],
    )
    def test_time_units(self, time_encoding, user_encoding, exp_units):
        """Test that time coordinates and bounds share the units with the highest precedence."""
        ds = _make_fake_ds_with_time(time_encoding)
        enc, _ = update_encoding(ds, {"encoding": user_encoding})
        assert enc["time"] == {"units": exp_units, "calendar": "proleptic_gregorian", "dtype": "float64",
                               "_FillValue": None}
        assert enc["time_bnds"] == {"units": exp_units, "calendar": "proleptic_gregorian", "dtype": "float64",
                                    "_FillValue": None}

    def test_inputs_not_modified(self):
        """Test that neither the dataset's encodings nor the user-defined encoding are modified."""
        ds = _make_fake_ds_with_time({"units": "seconds since 1970-01-01"})
        ds["foo"].encoding = {"dtype": "int16", "source": "/path/to/file.nc"}
        own_encodings = {var_name: copy.deepcopy(var.encoding) for var_name, var in ds.variables.items()}
        kwargs = {"encoding": {"foo": {"zlib": True}, "bar": {"dtype": "int16"}, "time": {"calendar": "standard"}}}
        orig_kwargs = copy.deepcopy(kwargs)
        update_encoding(ds, kwargs)
        assert kwargs == orig_kwargs
        assert {var_name: var.encoding for var_name, var in ds.variables.items()} == own_encodings
