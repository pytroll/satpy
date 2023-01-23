"""Tests for the Insat3D reader."""
import os
from datetime import datetime

import dask.array as da
import h5netcdf
import numpy as np
import pytest

from satpy import Scene
from satpy.readers.insat3d_img_l1b_h5 import (
    CHANNELS_BY_RESOLUTION,
    LUT_SUFFIXES,
    Insat3DIMGL1BH5FileHandler,
    get_lonlat_suffix,
    open_dataset,
    open_datatree,
)
from satpy.tests.utils import make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path_factory

# real shape is 1, 11220, 11264
shape_1km = (1, 1122, 1126)
shape_4km = (1, 2816, 2805)
shape_8km = (1, 1408, 1402)
rad_units = "mW.cm-2.sr-1.micron-1"
alb_units = "%"
temp_units = "K"
chunks_1km = (1, 46, 1126)
values_1km = np.random.randint(0, 1000, shape_1km, dtype=np.uint16)
values_1km[0, 0, 0] = 0
values_4km = np.random.randint(0, 1000, shape_4km, dtype=np.uint16)
values_8km = np.random.randint(0, 1000, shape_8km, dtype=np.uint16)

values_by_resolution = {1000: values_1km,
                        4000: values_4km,
                        8000: values_8km}

lut_values_2 = np.arange(0, 1024 * 2, 2)
lut_values_3 = np.arange(0, 1024 * 3, 3)

dimensions = {"GeoX": shape_4km[2],
              "GeoY": shape_4km[1],
              "GeoX1": shape_8km[2],
              "GeoY1": shape_8km[1],
              "GeoX2": shape_1km[2],
              "GeoY2": shape_1km[1],
              "time": 1,
              "GreyCount": 1024,
              }
dimensions_by_resolution = {1000: ("GeoY2", "GeoX2"),
                            4000: ("GeoY", "GeoX"),
                            8000: ("GeoY1", "GeoX1")}

channel_names = {"vis": "Visible",
                 "mir": "Middle Infrared",
                 "swir": "Shortwave Infrared",
                 "tir1": "Thermal Infrared1",
                 "tir2": "Thermal Infrared2",
                 "wv": "Water Vapor"}

calibrated_names = {"": "Count",
                    "RADIANCE": "Radiance",
                    "ALBEDO": "Albedo",
                    "TEMP": "Brightness Temperature"}

calibrated_units = {"": "1",
                    "RADIANCE": "mW.cm-2.sr-1.micron-1",
                    "ALBEDO": "%",
                    "TEMP": "K"}

start_time = datetime(2009, 6, 9, 9, 0)
end_time = datetime(2009, 6, 9, 9, 30)

time_pattern = "%d-%b-%YT%H:%M:%S"

global_attrs = {"Observed_Altitude(km)": 35778.490219,
                "Field_of_View(degrees)": 17.973925,
                "Acquisition_Start_Time": start_time.strftime(time_pattern),
                "Acquisition_End_Time": end_time.strftime(time_pattern),
                "Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude": [0.0, 82.0],
                "Nominal_Altitude(km)": 36000.0,
                }


@pytest.fixture(scope="session")
def insat_filename(tmp_path_factory):
    """Create a fake insat 3d l1b file."""
    filename = tmp_path_factory.mktemp("data") / "3DIMG_25OCT2022_0400_L1B_STD_V01R00.h5"
    with h5netcdf.File(filename, mode="w") as h5f:
        h5f.dimensions = dimensions
        h5f.attrs.update(global_attrs)
        for resolution, channels in CHANNELS_BY_RESOLUTION.items():
            _create_channels(channels, h5f, resolution)
            _create_lonlats(h5f, resolution)

    return filename


def mask_array(array):
    """Mask an array with nan instead of 0."""
    return np.where(array == 0, np.nan, array)


def _create_channels(channels, h5f, resolution):
    for channel in channels:
        var_name = "IMG_" + channel.upper()

        var = h5f.create_variable(var_name, ("time",) + dimensions_by_resolution[resolution], np.uint16,
                                  chunks=chunks_1km)
        var[:] = values_by_resolution[resolution]
        var.attrs["_FillValue"] = 0
        for suffix, lut_values in zip(LUT_SUFFIXES[channel], (lut_values_2, lut_values_3)):
            lut_name = "_".join((var_name, suffix))
            var = h5f.create_variable(lut_name, ("GreyCount",), float)
            var[:] = lut_values
            var.attrs["units"] = bytes(calibrated_units[suffix], "ascii")
            var.attrs["long_name"] = " ".join((channel_names[channel], calibrated_names[suffix]))


def _create_lonlats(h5f, resolution):
    lonlat_suffix = get_lonlat_suffix(resolution)
    for var_name in ["Longitude" + lonlat_suffix, "Latitude" + lonlat_suffix]:
        var = h5f.create_variable(var_name, dimensions_by_resolution[resolution], np.uint16,
                                  chunks=chunks_1km[1:])
        var[:] = values_by_resolution[resolution]
        var.attrs["scale_factor"] = 0.01
        var.attrs["add_offset"] = 0.0


def test_insat3d_backend_has_1km_channels(insat_filename):
    """Test the insat3d backend."""
    res = open_dataset(insat_filename, resolution=1000)
    assert res["IMG_VIS"].shape == shape_1km
    assert res["IMG_SWIR"].shape == shape_1km


@pytest.mark.parametrize("resolution,name,shape,expected_values,expected_name,expected_units",
                         [(1000, "IMG_VIS_RADIANCE", shape_1km, mask_array(values_1km * 2),
                           "Visible Radiance", rad_units),
                          (1000, "IMG_VIS_ALBEDO", shape_1km, mask_array(values_1km * 3),
                           "Visible Albedo", alb_units),
                          (4000, "IMG_MIR_RADIANCE", shape_4km, mask_array(values_4km * 2),
                           "Middle Infrared Radiance", rad_units),
                          (4000, "IMG_MIR_TEMP", shape_4km, mask_array(values_4km * 3),
                           "Middle Infrared Brightness Temperature", temp_units),
                          (4000, "IMG_TIR1_RADIANCE", shape_4km, mask_array(values_4km * 2),
                           "Thermal Infrared1 Radiance", rad_units),
                          (4000, "IMG_TIR2_RADIANCE", shape_4km, mask_array(values_4km * 2),
                           "Thermal Infrared2 Radiance", rad_units),
                          (8000, "IMG_WV_RADIANCE", shape_8km, mask_array(values_8km * 2),
                           "Water Vapor Radiance", rad_units),
                          ])
def test_insat3d_has_calibrated_arrays(insat_filename,
                                       resolution, name, shape, expected_values, expected_name, expected_units):
    """Check that calibration happens as expected."""
    res = open_dataset(insat_filename, resolution=resolution)
    assert res[name].shape == shape
    np.testing.assert_allclose(res[name], expected_values)
    assert res[name].attrs["units"] == expected_units
    assert res[name].attrs["long_name"] == expected_name


def test_insat3d_has_dask_arrays(insat_filename):
    """Test that the backend uses dask."""
    res = open_dataset(insat_filename, resolution=1000)
    assert isinstance(res["IMG_VIS_RADIANCE"].data, da.Array)
    assert res["IMG_VIS"].chunks is not None


def test_insat3d_only_has_3_resolutions(insat_filename):
    """Test that we only accept 1000, 4000, 8000."""
    with pytest.raises(ValueError):
        _ = open_dataset(insat_filename, resolution=1024)


@pytest.mark.parametrize("resolution", [1000, 4000, 8000, ])
def test_insat3d_returns_lonlat(insat_filename, resolution):
    """Test that lons and lats are loaded."""
    res = open_dataset(insat_filename, resolution=resolution)
    expected = values_by_resolution[resolution].squeeze() / 100.0
    assert isinstance(res["Latitude"].data, da.Array)
    np.testing.assert_allclose(res["Latitude"], expected)
    assert isinstance(res["Longitude"].data, da.Array)
    np.testing.assert_allclose(res["Longitude"], expected)


@pytest.mark.parametrize("resolution", [1000, 4000, 8000, ])
def test_insat3d_has_global_attributes(insat_filename, resolution):
    """Test that the backend supports global attributes."""
    res = open_dataset(insat_filename, resolution=resolution)
    assert res.attrs.keys() >= global_attrs.keys()


@pytest.mark.parametrize("resolution", [1000, 4000, 8000, ])
def test_insat3d_opens_datatree(insat_filename, resolution):
    """Test that a datatree is produced."""
    res = open_datatree(insat_filename)
    assert str(resolution) in res.keys()


def test_insat3d_datatree_has_global_attributes(insat_filename):
    """Test that the backend supports global attributes in the datatree."""
    res = open_datatree(insat_filename)
    assert res.attrs.keys() >= global_attrs.keys()


@pytest.mark.parametrize("calibration,expected_values",
                         [("counts", values_1km),
                          ("radiance", mask_array(values_1km * 2)),
                          ("reflectance", mask_array(values_1km * 3))])
def test_filehandler_returns_data_array(insat_filehandler, calibration, expected_values):
    """Test that the filehandler can get dataarrays."""
    fh = insat_filehandler
    ds_info = None

    ds_id = make_dataid(name="VIS", resolution=1000, calibration=calibration)
    darr = fh.get_dataset(ds_id, ds_info)
    np.testing.assert_allclose(darr, expected_values.squeeze())
    assert darr.dims == ("y", "x")


def test_filehandler_returns_masked_data_in_space(insat_filehandler):
    """Test that the filehandler masks space pixels."""
    fh = insat_filehandler
    ds_info = None

    ds_id = make_dataid(name="VIS", resolution=1000, calibration='reflectance')
    darr = fh.get_dataset(ds_id, ds_info)
    assert np.isnan(darr[0, 0])


def test_insat3d_has_orbital_parameters(insat_filehandler):
    """Test that the filehandler returns data with orbital parameter attributes."""
    fh = insat_filehandler
    ds_info = None

    ds_id = make_dataid(name="VIS", resolution=1000, calibration='reflectance')
    darr = fh.get_dataset(ds_id, ds_info)

    assert "orbital_parameters" in darr.attrs
    assert "satellite_nominal_longitude" in darr.attrs["orbital_parameters"]
    assert "satellite_nominal_latitude" in darr.attrs["orbital_parameters"]
    assert "satellite_nominal_altitude" in darr.attrs["orbital_parameters"]
    assert "satellite_actual_altitude" in darr.attrs["orbital_parameters"]
    assert "platform_name" in darr.attrs
    assert "sensor" in darr.attrs


def test_filehandler_returns_coords(insat_filehandler):
    """Test that lon and lat can be loaded."""
    fh = insat_filehandler
    ds_info = None

    lon_id = make_dataid(name="longitude", resolution=1000)
    darr = fh.get_dataset(lon_id, ds_info)
    np.testing.assert_allclose(darr, values_1km.squeeze() / 100)


@pytest.fixture(scope="session")
def insat_filehandler(insat_filename):
    """Instantiate a Filehandler."""
    fileinfo = {}
    filetype = None
    fh = Insat3DIMGL1BH5FileHandler(insat_filename, fileinfo, filetype)
    return fh


def test_filehandler_returns_area(insat_filehandler):
    """Test that filehandle returns an area."""
    fh = insat_filehandler

    ds_id = make_dataid(name="MIR", resolution=4000, calibration="brightness_temperature")
    area_def = fh.get_area_def(ds_id)
    lons, lats = area_def.get_lonlats(chunks=1000)


def test_filehandler_has_start_and_end_time(insat_filehandler):
    """Test that the filehandler handles start and end time."""
    fh = insat_filehandler

    assert fh.start_time == start_time
    assert fh.end_time == end_time


def test_satpy_load_array(insat_filename):
    """Test that satpy can load the VIS array."""
    scn = Scene(filenames=[os.fspath(insat_filename)], reader="insat3d_img_l1b_h5")
    scn.load(["VIS"])
    expected = mask_array(values_1km * 3).squeeze()
    np.testing.assert_allclose(scn["VIS"], expected)


def test_satpy_load_two_arrays(insat_filename):
    """Test that satpy can load the VIS array."""
    scn = Scene(filenames=[os.fspath(insat_filename)], reader="insat3d_img_l1b_h5")
    scn.load(["TIR1", "WV"])
    expected = mask_array(values_4km * 3).squeeze()
    np.testing.assert_allclose(scn["TIR1"], expected)
