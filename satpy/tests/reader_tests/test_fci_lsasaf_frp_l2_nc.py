"""Tests for the FCI LSASAF FRP NetCDF Level-2 Reader."""

from datetime import datetime, timedelta

import dask.array as da
import netCDF4
import numpy as np
import pytest
import xarray as xr

from satpy.readers.fci_lsasaf_frp_l2_nc import VAR_MAP, FRPFileHandler


@pytest.fixture
def filename_info():
    """Return Satpy filename information."""
    return {
        "platform_name": "MTG",
        "start_time": datetime(2026, 7, 31, 12, 0, 0),
        "facility_or_tool": "LSASAF-LISBON",
        "coverage": "FD",
        "disposition_mode": "C",
    }


@pytest.fixture
def filetype_info():
    """Return file type information."""
    return {}


@pytest.fixture
def sample_nc_file(tmp_path):
    """Create a minimal NetCDF test file with ListProduct group."""
    filename = tmp_path / (
        "W_PT-LSASAF-LISBON,SATELLITE,LSA-509_MTG_MTFRPPIXEL_MTG-FD_"
        "C_LPMG_20260731120000.nc"
    )

    with netCDF4.Dataset(filename, mode="w", format="NETCDF4") as root:
        # Global/root attributes
        root.platform = "MTI1"
        root.sensor = "FCI"
        root.product_frequency = "10-min"

        # The actual data are stored in the NetCDF group
        group = root.createGroup("ListProduct")

        group.createDimension("index", 3)

        frp = group.createVariable(
            "FRP",
            "f4",
            ("index",),
            zlib=True,
            chunksizes=(3,),
            fill_value=-999.0,
        )
        frp[:] = [10.5, 20.0, 30.0]
        frp.units = "MW"

        fire_confidence = group.createVariable(
            "FIRE_CONFIDENCE",
            "f4",
            ("index",),
            zlib=True,
            chunksizes=(3,),
            fill_value=-999.0,
        )
        fire_confidence[:] = [80.0, 90.0, 70.0]
        fire_confidence.units = "%"

        latitude = group.createVariable(
            "LATITUDE",
            "f4",
            ("index",),
            zlib=True,
            chunksizes=(3,),
            fill_value=-999.0,
        )
        latitude[:] = [50.1, 50.2, 50.3]

        longitude = group.createVariable(
            "LONGITUDE",
            "f4",
            ("index",),
            zlib=True,
            chunksizes=(3,),
            fill_value=-999.0,
        )
        longitude[:] = [8.6, 8.7, 8.8]

        abs_line = group.createVariable(
            "ABS_LINE",
            "i4",
            ("index",),
            zlib=True,
            chunksizes=(3,),
        )
        abs_line[:] = [0, 2, 4]

        abs_samp = group.createVariable(
            "ABS_SAMP",
            "i4",
            ("index",),
            zlib=True,
            chunksizes=(3,),
        )
        abs_samp[:] = [0, 3, 5]

    return filename


@pytest.fixture
def reader(sample_nc_file, filename_info, filetype_info):
    """Return initialized NetCDF reader."""
    return FRPFileHandler(
        str(sample_nc_file),
        filename_info,
        filetype_info,
    )


def test_start_time(reader, filename_info):
    """Test start time."""
    assert reader.start_time == filename_info["start_time"]


def test_end_time_is_ten_minutes_after_start(reader, filename_info):
    """Test default end time derived from product frequency."""
    expected = filename_info["start_time"] + timedelta(minutes=10)
    assert reader.end_time == expected


def test_root_attributes_are_available(reader):
    """Test that root attributes are available through global_attrs."""
    assert reader.global_attrs["platform"] == "MTI1"
    assert reader.global_attrs["sensor"] == "FCI"
    assert reader.global_attrs["product_frequency"] == "10-min"


def test_satellite_name(reader):
    """Test satellite/platform name translation."""
    assert reader.satellite_name == "Meteosat-12"


def test_sensor_name(reader):
    """Test sensor name."""
    assert reader.sensor_name == "fci"


@pytest.mark.parametrize(
    "key",
    [
        "frp",
        "fire_confidence",
        "latitude",
        "longitude",
        "abs_line",
        "abs_samp",
    ],
)
def test_contains_known_dataset(reader, key):
    """Test availability of known datasets."""
    assert key in reader


def test_contains_unknown_dataset(reader):
    """Test unavailable dataset."""
    assert "not_a_dataset" not in reader


def test_getitem_returns_expected_variable(reader):
    """Test source variable lookup."""
    data = reader["frp"]

    assert isinstance(data, xr.DataArray)
    assert data.name == "FRP"
    assert data.dims == ("index",)


def test_getitem_uses_var_map(reader):
    """Test mapping of logical names to NetCDF names."""
    assert reader["frp"].name == VAR_MAP["frp"]
    assert reader["latitude"].name == VAR_MAP["latitude"]
    assert reader["abs_line"].name == VAR_MAP["abs_line"]


def test_variables_are_dask_backed(reader):
    """Test that data loaded from the group are Dask-backed."""
    for key in ("frp", "latitude", "longitude", "abs_line", "abs_samp"):
        data = reader[key]

        assert isinstance(data.data, da.Array)
        assert data.dims == ("index",)
        assert data.shape == (3,)


def test_get_dataset_latitude(reader):
    """Test one-dimensional latitude dataset."""
    dsid = {"name": "latitude"}
    dsinfo = {
        "units": "degrees_north",
        "standard_name": "active_fire_pixel_centre_latitude",
    }

    data = reader.get_dataset(dsid, dsinfo)

    assert isinstance(data, xr.DataArray)
    assert data.dims == ("y",)
    assert data.shape == (3,)

    np.testing.assert_allclose(
        data.compute().values,
        [50.1, 50.2, 50.3],
    )

    assert data.attrs["units"] == "degrees_north"
    assert (
        data.attrs["standard_name"]
        == "active_fire_pixel_centre_latitude"
    )
    assert data.attrs["satellite_name"] == "Meteosat-12"
    assert data.attrs["platform_name"] == "MTG"
    assert data.attrs["sensor"] == "fci"
    assert data.attrs["start_time"] == reader.start_time
    assert data.attrs["end_time"] == reader.end_time


def test_get_dataset_frp(reader, monkeypatch):
    """Test FRP dataset creation without allocating full-disk grid."""
    def fake_get_array_on_fci_grid(data_array):
        return data_array

    monkeypatch.setattr(
        reader,
        "get_array_on_fci_grid",
        fake_get_array_on_fci_grid,
    )

    dsid = {"name": "frp"}
    dsinfo = {
        "units": "MW",
        "standard_name": "fire_radiative_power",
        "resolution": 1000,
    }

    data = reader.get_dataset(dsid, dsinfo)

    assert isinstance(data, xr.DataArray)
    assert data.dims == ("y",)
    assert data.shape == (3,)

    np.testing.assert_allclose(
        data.compute().values,
        [10.5, 20.0, 30.0],
    )

    assert data.attrs["units"] == "MW"
    assert data.attrs["standard_name"] == "fire_radiative_power"
    assert data.attrs["resolution"] == 1000
    assert data.attrs["satellite_name"] == "Meteosat-12"
    assert data.attrs["platform_name"] == "MTG"
    assert data.attrs["sensor"] == "fci"


def test_get_array_on_fci_grid(reader, monkeypatch):
    """Test placing sparse values on a small raster grid."""
    monkeypatch.setattr(
        "satpy.readers.fci_lsasaf_frp_l2_nc.FRP_GRID_SHAPE",
        (5, 6),
    )

    values = xr.DataArray(
        np.array([10.5, 20.0, 30.0], dtype=np.float32),
        dims=("index",),
        attrs={
            "satellite_name": "Meteosat-12",
            "platform_name": "MTG",
            "sensor": "fci",
        },
    )

    gridded = reader.get_array_on_fci_grid(values)

    assert isinstance(gridded, xr.DataArray)
    assert gridded.dims == ("y", "x")
    assert gridded.shape == (5, 6)

    result = gridded.compute().values

    assert np.isclose(result[0, 0], 10.5)
    assert np.isclose(result[2, 3], 20.0)
    assert np.isclose(result[4, 5], 30.0)

    assert np.isnan(result[0, 1])

    assert gridded.attrs["satellite_name"] == "Meteosat-12"
    assert gridded.attrs["platform_name"] == "MTG"
    assert gridded.attrs["sensor"] == "fci"


def test_get_area_def(reader, monkeypatch):
    """Test area definition lookup."""
    called = {}

    def fake_get_area_def(name):
        called["name"] = name
        return "dummy_area"

    monkeypatch.setattr(
        "satpy.readers.fci_lsasaf_frp_l2_nc.get_area_def",
        fake_get_area_def,
    )

    area = reader.get_area_def({"name": "frp"})

    assert called["name"] == "mtg_fci_fdss_1km"
    assert area == "dummy_area"
