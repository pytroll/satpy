"""Tests for the LSA SAF MTG FCI FRP Level-2 reader."""

from datetime import datetime, timedelta

import dask.array as da
import netCDF4
import numpy as np
import pytest
import xarray as xr

from satpy.readers.fci_lsasaf_frp_l2 import (
    NC_VAR_MAP,
    CSVFileHandler,
    NCFileHandler,
)


@pytest.fixture
def filename_info():
    """Return Satpy filename information."""
    return {
        "platform_name": "MTG",
        "start_time": datetime(2026, 7, 31, 12, 0),
        "facility_or_tool": "LSASAF-LISBON",
        "coverage": "FD",
        "disposition_mode": "C",
    }


@pytest.fixture
def filetype_info():
    """Return file type information."""
    return {}


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create a minimal CSV FRP product."""
    filename = tmp_path / (
        "LSASAF-LISBON-509_MTG_MTFRPPIXEL-ListProduct_"
        "MTG-FD_202607311200.csv"
    )

    filename.write_text(
        "FRP,LATITUDE,LONGITUDE,ABS_LINE,ABS_SAMP\n"
        "10.5,50.1,8.6,0,0\n"
        "20.0,50.2,8.7,2,3\n"
        "30.0,50.3,8.8,4,5\n"
    )

    return filename


@pytest.fixture
def sample_nc_file(tmp_path):
    """Create a minimal NetCDF FRP product with ListProduct group."""
    filename = tmp_path / (
        "W_PT-LSASAF-LISBON,SATELLITE,LSA-509_MTG_MTFRPPIXEL_"
        "MTG-FD_C_LPMG_20260731120000.nc"
    )

    with netCDF4.Dataset(filename, mode="w", format="NETCDF4") as root:
        root.platform = "MTI1"
        root.sensor = "FCI"
        root.product_frequency = "10-min"

        group = root.createGroup("ListProduct")
        group.createDimension("index", 3)

        variables = {
            "FRP": ("f4", [10.5, 20.0, 30.0]),
            "FIRE_CONFIDENCE": ("f4", [80.0, 90.0, 70.0]),
            "LATITUDE": ("f4", [50.1, 50.2, 50.3]),
            "LONGITUDE": ("f4", [8.6, 8.7, 8.8]),
            "ABS_LINE": ("i4", [0, 2, 4]),
            "ABS_SAMP": ("i4", [0, 3, 5]),
        }

        for name, (dtype, values) in variables.items():
            var = group.createVariable(
                name,
                dtype,
                ("index",),
                zlib=True,
                chunksizes=(3,),
            )
            var[:] = values

        group["FRP"].units = "MW"
        group["FIRE_CONFIDENCE"].units = "%"

    return filename


@pytest.fixture(params=["csv", "nc"])
def reader(request, sample_csv_file, sample_nc_file, filename_info, filetype_info):
    """Create the requested FRP file handler."""
    if request.param == "csv":
        return CSVFileHandler(
            str(sample_csv_file),
            filename_info,
            filetype_info,
        )

    return NCFileHandler(
        str(sample_nc_file),
        filename_info,
        filetype_info,
    )


@pytest.fixture
def reader_kind(request):
    """Return the parameterized reader kind."""
    return request.param


def test_start_time(reader, filename_info):
    """Test observation start time."""
    assert reader.start_time == filename_info["start_time"]


def test_end_time(reader, filename_info):
    """Test observation end time."""
    expected = filename_info["start_time"] + timedelta(minutes=10)
    assert reader.end_time == expected


@pytest.mark.parametrize(
    "name",
    [
        "frp",
        "latitude",
        "longitude",
        "abs_line",
        "abs_samp",
    ],
)
def test_contains_common_datasets(reader, name):
    """Test availability of datasets provided by both formats."""
    assert name in reader


def test_does_not_contain_unknown_dataset(reader):
    """Test unavailable datasets."""
    assert "not_a_dataset" not in reader


def test_getitem_returns_mapped_source(reader):
    """Test source variable or column lookup."""
    data = reader["frp"]

    assert data is not None
    assert data.name == "FRP"


def test_get_dataset_latitude(reader):
    """Test creation of the latitude dataset."""
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
    """Test FRP dataset metadata without allocating a full disk."""
    monkeypatch.setattr(
        reader,
        "get_array_on_fci_grid",
        lambda data: data,
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


def test_get_array_on_fci_grid(reader, monkeypatch):
    """Test placement of sparse detections on a small grid."""
    monkeypatch.setattr(
        "satpy.readers.fci_lsasaf_frp_l2.FRP_GRID_SHAPE",
        (5, 6),
    )

    values = xr.DataArray(
        np.array([10.5, 20.0, 30.0], dtype=np.float32),
        dims=("y",),
        attrs={"test_attribute": "preserved"},
    )

    gridded = reader.get_array_on_fci_grid(values)
    result = gridded.compute().values

    assert isinstance(gridded, xr.DataArray)
    assert gridded.dims == ("y", "x")
    assert gridded.shape == (5, 6)

    assert result[0, 0] == pytest.approx(10.5)
    assert result[2, 3] == pytest.approx(20.0)
    assert result[4, 5] == pytest.approx(30.0)
    assert np.isnan(result[0, 1])

    assert gridded.attrs["test_attribute"] == "preserved"


def test_get_area_def(reader, monkeypatch):
    """Test retrieval of the native FCI area definition."""
    called = {}

    def fake_get_area_def(area_id):
        called["area_id"] = area_id
        return "dummy_area"

    monkeypatch.setattr(
        "satpy.readers.fci_lsasaf_frp_l2.get_area_def",
        fake_get_area_def,
    )

    assert reader.get_area_def({"name": "frp"}) == "dummy_area"
    assert called["area_id"] == "mtg_fci_fdss_1km"


def test_nc_reader_contains_fire_confidence(
    sample_nc_file,
    filename_info,
    filetype_info,
):
    """Test that FIRE_CONFIDENCE is exclusive to the NetCDF reader."""
    reader = NCFileHandler(
        str(sample_nc_file),
        filename_info,
        filetype_info,
    )

    assert "fire_confidence" in reader
    assert reader["fire_confidence"].name == NC_VAR_MAP["fire_confidence"]


def test_csv_reader_does_not_contain_fire_confidence(
    sample_csv_file,
    filename_info,
    filetype_info,
):
    """Test that CSV products have no FIRE_CONFIDENCE field."""
    reader = CSVFileHandler(
        str(sample_csv_file),
        filename_info,
        filetype_info,
    )

    assert "fire_confidence" not in reader


def test_nc_data_is_dask_backed(sample_nc_file, filename_info, filetype_info):
    """Test lazy data access for NetCDF variables."""
    reader = NCFileHandler(
        str(sample_nc_file),
        filename_info,
        filetype_info,
    )

    data = reader["frp"]

    assert isinstance(data.data, da.Array)
    assert data.dims == ("index",)
    assert data.shape == (3,)
