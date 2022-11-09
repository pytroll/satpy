"""Tests for the Insat3D reader."""
import dask.array as da
import h5netcdf
import numpy as np
import pytest
from xarray import open_dataset

from satpy.readers.insat3d_img_l1b_h5 import CHANNELS_BY_RESOLUTION, LUT_SUFFIXES, I3DBackend

# real shape is 1, 11220, 11264
shape_1km = (1, 1122, 1126)
shape_4km = (1, 2816, 2805)
shape_8km = (1, 1408, 1402)
rad_units = "mW.cm-2.sr-1.micron-1"
alb_units = "%"
temp_units = "K"
chunks_1km = (1, 46, 1126)
values_1km = np.random.randint(0, 1000, shape_1km, dtype=np.uint16)
values_4km = np.random.randint(0, 1000, shape_4km, dtype=np.uint16)
values_8km = np.random.randint(0, 1000, shape_8km, dtype=np.uint16)

values_by_resolution = {1000: values_1km,
                        4000: values_4km,
                        8000: values_8km}

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


@pytest.fixture(scope="session")
def insat_filename(tmp_path_factory):
    """Create a fake insat 3d l1b file."""
    lut_values_2 = np.arange(0, 1024 * 2, 2)
    lut_values_3 = np.arange(0, 1024 * 3, 3)

    filename = tmp_path_factory.mktemp("data") / "3DIMG_25OCT2022_0400_L1B_STD_V01R00.h5"
    with h5netcdf.File(filename, mode="w") as h5f:
        h5f.dimensions = dimensions
        for resolution, channels in CHANNELS_BY_RESOLUTION.items():
            for channel in channels:
                var_name = "IMG_" + channel.upper()

                var = h5f.create_variable(var_name, ("time", ) + dimensions_by_resolution[resolution], np.uint16,
                                          chunks=chunks_1km)
                var[:] = values_by_resolution[resolution]
                for suffix, lut_values in zip(LUT_SUFFIXES[channel], (lut_values_2, lut_values_3)):
                    lut_name = "_".join((var_name, suffix))
                    var = h5f.create_variable(lut_name, ("GreyCount",), float)
                    var[:] = lut_values
                    var.attrs["units"] = bytes(calibrated_units[suffix], "ascii")
                    var.attrs["long_name"] = " ".join((channel_names[channel], calibrated_names[suffix]))

            if resolution == 1000:
                var_name = "Longitude_VIS"
                var = h5f.create_variable(var_name, dimensions_by_resolution[resolution], np.uint16,
                                          chunks=chunks_1km[1:])
                var[:] = values_by_resolution[resolution]
                var.attrs["scale_factor"] = 0.01
                var.attrs["add_offset"] = 0.0
                var_name = "Latitude_VIS"
                var = h5f.create_variable(var_name, dimensions_by_resolution[resolution], np.uint16,
                                          chunks=chunks_1km[1:])
                var[:] = values_by_resolution[resolution]
                var.attrs["scale_factor"] = 0.01
                var.attrs["add_offset"] = 0.0
            elif resolution == 4000:
                var_name = "Longitude"
                var = h5f.create_variable(var_name, dimensions_by_resolution[resolution], np.uint16,
                                          chunks=chunks_1km[1:])
                var[:] = values_by_resolution[resolution]
                var.attrs["scale_factor"] = 0.01
                var.attrs["add_offset"] = 0.0
                var_name = "Latitude"
                var = h5f.create_variable(var_name, dimensions_by_resolution[resolution], np.uint16,
                                          chunks=chunks_1km[1:])
                var[:] = values_by_resolution[resolution]
                var.attrs["scale_factor"] = 0.01
                var.attrs["add_offset"] = 0.0
            elif resolution == 8000:
                var_name = "Longitude_WV"
                var = h5f.create_variable(var_name, dimensions_by_resolution[resolution], np.uint16,
                                          chunks=chunks_1km[1:])
                var[:] = values_by_resolution[resolution]
                var.attrs["scale_factor"] = 0.01
                var.attrs["add_offset"] = 0.0
                var_name = "Latitude_WV"
                var = h5f.create_variable(var_name, dimensions_by_resolution[resolution], np.uint16,
                                          chunks=chunks_1km[1:])
                var[:] = values_by_resolution[resolution]
                var.attrs["scale_factor"] = 0.01
                var.attrs["add_offset"] = 0.0
    return filename


def test_insat3d_backend_has_1km_channels(insat_filename):
    """Test the insat3d backend."""
    res = open_dataset(insat_filename, engine=I3DBackend, chunks={}, resolution=1000)
    assert res["IMG_VIS"].shape == shape_1km
    assert res["IMG_SWIR"].shape == shape_1km


@pytest.mark.parametrize("resolution,name,shape,expected_values,expected_name,expected_units",
                         [(1000, "IMG_VIS_RADIANCE", shape_1km, values_1km * 2, "Visible Radiance", rad_units),
                          (1000, "IMG_VIS_ALBEDO", shape_1km, values_1km * 3, "Visible Albedo", alb_units),
                          (4000, "IMG_MIR_RADIANCE", shape_4km, values_4km * 2, "Middle Infrared Radiance", rad_units),
                          (4000, "IMG_MIR_TEMP", shape_4km, values_4km * 3, "Middle Infrared Brightness Temperature",
                           temp_units),
                          (4000, "IMG_TIR1_RADIANCE", shape_4km, values_4km * 2, "Thermal Infrared1 Radiance",
                           rad_units),
                          (4000, "IMG_TIR2_RADIANCE", shape_4km, values_4km * 2, "Thermal Infrared2 Radiance",
                           rad_units),
                          (8000, "IMG_WV_RADIANCE", shape_8km, values_8km * 2, "Water Vapor Radiance", rad_units),
                          ])
def test_insat3d_has_calibrated_arrays(insat_filename,
                                       resolution, name, shape, expected_values, expected_name, expected_units):
    """Check that calibration happens as expected."""
    res = open_dataset(insat_filename, engine=I3DBackend, chunks={}, resolution=resolution)
    assert res[name].shape == shape
    np.testing.assert_allclose(res[name], expected_values)
    assert res[name].attrs["units"] == expected_units
    assert res[name].attrs["long_name"] == expected_name


def test_insat3d_has_dask_arrays(insat_filename):
    """Test that the backend uses dask."""
    res = open_dataset(insat_filename, engine=I3DBackend, chunks={}, resolution=1000)
    assert isinstance(res["IMG_VIS_RADIANCE"].data, da.Array)
    assert res["IMG_VIS"].chunks == da.core.normalize_chunks(chunks_1km, shape_1km)


def test_insat3d_only_has_3_resolutions(insat_filename):
    """Test that we only accept 1000, 4000, 8000."""
    with pytest.raises(ValueError):
        _ = open_dataset(insat_filename, engine=I3DBackend, chunks={}, resolution=1024)


def test_insat3d_returns_lonlat(insat_filename):
    """Test that lons and lats are loaded."""
    res = open_dataset(insat_filename, engine=I3DBackend, chunks={}, resolution=1000)
    assert isinstance(res["Latitude"].data, da.Array)
    np.testing.assert_allclose(res["Latitude"], values_1km.squeeze() / 100.0)
    assert isinstance(res["Longitude"].data, da.Array)
    np.testing.assert_allclose(res["Longitude"], values_1km.squeeze() / 100.0)
