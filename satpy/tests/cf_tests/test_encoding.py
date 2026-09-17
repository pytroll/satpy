"""Tests for compatible netCDF/Zarr DataArray encodings."""
import datetime

import pytest
import xarray as xr


class TestUpdateEncoding:
    """Test update of dataset encodings."""

    @pytest.fixture
    def fake_ds(self):
        """Create fake data for testing."""
        ds = xr.Dataset({"foo": (("y", "x"), [[1, 2], [3, 4]]),
                         "bar": (("y", "x"), [[3, 4], [5, 6]])},
                        coords={"y": [1, 2],
                                "x": [3, 4],
                                "lon": (("y", "x"), [[7, 8], [9, 10]])})
        return ds

    @pytest.fixture
    def fake_ds_digit(self):
        """Create fake data for testing."""
        ds_digit = xr.Dataset({"CHANNEL_1": (("y", "x"), [[1, 2], [3, 4]]),
                               "CHANNEL_2": (("y", "x"), [[3, 4], [5, 6]])},
                              coords={"y": [1, 2],
                                      "x": [3, 4],
                                      "lon": (("y", "x"), [[7, 8], [9, 10]])})
        return ds_digit

    def test_dataset_name_digit(self, fake_ds_digit):
        """Test data with dataset name staring with a digit."""
        from satpy.cf.encoding import update_encoding

        # Dataset with name staring with digit
        ds_digit = fake_ds_digit
        kwargs = {"encoding": {"1": {"dtype": "float32"},
                               "2": {"dtype": "float32"}},
                  "other": "kwargs"}
        enc, other_kwargs = update_encoding(ds_digit, kwargs, numeric_name_prefix="CHANNEL_")
        expected_dict = {
            "y": {"_FillValue": None},
            "x": {"_FillValue": None},
            "CHANNEL_1": {"dtype": "float32"},
            "CHANNEL_2": {"dtype": "float32"}
        }
        assert enc == expected_dict
        assert other_kwargs == {"other": "kwargs"}

    def test_without_time(self, fake_ds):
        """Test data with no time dimension."""
        from satpy.cf.encoding import update_encoding

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
        from satpy.cf.encoding import update_encoding

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
