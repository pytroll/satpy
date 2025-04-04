"""Tests for the CAMEL L3 netCDF reader."""

import datetime as dt

import numpy as np
import pytest
import xarray as xr

from satpy.readers.camel_l3_nc import CAMELL3NCFileHandler
from satpy.tests.utils import make_dataid

rng = np.random.default_rng()
ndvi_data = rng.integers(0, 1000, (3600, 7200), dtype=np.int16)
emis_data = rng.integers(0, 1000, (3600, 7200, 5), dtype=np.int16)

lon_data = np.arange(-180, 180, 0.05)
lat_data = np.arange(-90, 90, 0.05)

start_time = dt.datetime(2023, 8, 1, 0, 0, 0)
end_time = dt.datetime(2023, 9, 1, 0, 0, 0)

fill_val = -999
scale_val = 0.001

dimensions = {"longitude": 7200, "latitude": 3600, "spectra": 13}

exp_ext = (-180.0, -90.0, 180.0, 90.0)

global_attrs = {"time_coverage_start": start_time.strftime("%Y-%m-%d %H:%M:%SZ"),
                "time_coverage_end": end_time.strftime("%Y-%m-%d %H:%M:%SZ"),
                "geospatial_lon_resolution": "0.05 degree grid ",
                "geospatial_lat_resolution": "0.05 degree grid ",
                }

bad_attrs1 = global_attrs.copy()
bad_attrs1["geospatial_lon_resolution"] = "0.1 degree grid "
bad_attrs2 = global_attrs.copy()
bad_attrs2["geospatial_lat_resolution"] = "0.1 degree grid "


def _make_ds(the_attrs, tmp_factory):
    """Make a dataset for use in tests."""
    fname = f'{tmp_factory.mktemp("data")}/CAM5K30EM_emis_202308_V003.nc'
    ds = xr.Dataset({"aster_ndvi": (["Rows", "Columns"], ndvi_data),
                     "camel_emis": (["latitude", "longitude", "spectra"], emis_data)},
                    coords={"latitude": (["Rows"], lat_data),
                            "longitude": (["Columns"], lon_data)},
                    attrs=the_attrs)
    ds.to_netcdf(fname)
    return fname


def camel_l3_filehandler(fname):
    """Instantiate a Filehandler."""
    fileinfo = {"start_period": "202308",
                "version": "003"}
    filetype = None
    fh = CAMELL3NCFileHandler(fname, fileinfo, filetype)
    return fh


@pytest.fixture(scope="session")
def camel_filename(tmp_path_factory):
    """Create a fake camel l3 file."""
    return _make_ds(global_attrs, tmp_path_factory)


@pytest.fixture(scope="session")
def camel_filename_bad1(tmp_path_factory):
    """Create a fake camel l3 file."""
    return _make_ds(bad_attrs1, tmp_path_factory)


@pytest.fixture(scope="session")
def camel_filename_bad2(tmp_path_factory):
    """Create a fake camel l3 file."""
    return _make_ds(bad_attrs2, tmp_path_factory)


def test_startend(camel_filename):
    """Test start and end times are set correctly."""
    fh = camel_l3_filehandler(camel_filename)
    assert fh.start_time == start_time
    assert fh.end_time == end_time


def test_camel_l3_area_def(camel_filename, caplog):
    """Test reader handles area definition correctly."""
    ps = "+proj=longlat +datum=WGS84 +no_defs +type=crs"

    # Check case where input data is correct size.
    fh = camel_l3_filehandler(camel_filename)
    ndvi_id = make_dataid(name="aster_ndvi")
    area_def = fh.get_area_def(ndvi_id)
    assert area_def.width == dimensions["longitude"]
    assert area_def.height == dimensions["latitude"]
    assert np.allclose(area_def.area_extent, exp_ext)

    assert area_def.proj4_string == ps


def test_bad_longitude(camel_filename_bad1):
    """Check case where longitude grid is not correct."""
    with pytest.raises(ValueError, match="Only 0.05 degree grid data is supported."):
        camel_l3_filehandler(camel_filename_bad1)


def test_bad_latitude(camel_filename_bad2):
    """Check case where latitude grid is not correct."""
    with pytest.raises(ValueError, match="Only 0.05 degree grid data is supported."):
        camel_l3_filehandler(camel_filename_bad2)


def test_load_ndvi_data(camel_filename):
    """Test that data is loaded successfully."""
    fh = camel_l3_filehandler(camel_filename)
    ndvi_id = make_dataid(name="aster_ndvi")
    ndvi = fh.get_dataset(ndvi_id, {"file_key": "aster_ndvi"})
    assert np.allclose(ndvi.data, ndvi_data)


def test_load_emis_data(camel_filename):
    """Test that data is loaded successfully."""
    fh = camel_l3_filehandler(camel_filename)
    emis_id = make_dataid(name="camel_emis")

    # This is correct data
    emis = fh.get_dataset(emis_id, {"file_key": "camel_emis", "band_id": 2})
    assert np.allclose(emis.data, emis_data[:, :, 2])

    # This will fail as we are requesting a band too high data
    with pytest.raises(ValueError, match="Band id requested is larger than dataset."):
        fh.get_dataset(emis_id, {"file_key": "camel_emis", "band_id": 12})
