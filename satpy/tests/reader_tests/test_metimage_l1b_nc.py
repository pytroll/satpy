"""The metimage_l1b_nc reader tests package.

This version tests the readers for METimage.

"""


import datetime
import os
import unittest
import uuid

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset

from satpy.readers.core.metimage import (
    MEAN_EARTH_RADIUS,
    ROWS_PER_SCAN,
    SCAN_ALT_TIE_POINTS,
    TIE_POINTS_FACTOR,
)
from satpy.readers.metimage_l1b_nc import METimageL1BNCFileHandler

TEST_FILE = "test_file_vii_l1b_nc.nc"

NUM_SCANS = 25
NUM_TIE_POINTS_ACT = 10
NUM_LINES = NUM_SCANS * ROWS_PER_SCAN
NUM_PIXELS = (NUM_TIE_POINTS_ACT - 1) * TIE_POINTS_FACTOR
NUM_TIE_POINTS_ALT = NUM_SCANS * SCAN_ALT_TIE_POINTS


def _create_l1b_file(path, with_tie_points=True):
    """Write a small METimage L1B file with realistic dimensions and on-disk chunking."""
    with Dataset(path, "w") as nc:
        nc.sensing_start_time_utc = "20170920173040.888"
        nc.sensing_end_time_utc = "20170920174117.555"
        nc.spacecraft = "SGA1"
        nc.instrument = "VII"

        data = nc.createGroup("data")
        data.createDimension("num_chan_solar", 11)
        data.createDimension("num_chan_thermal", 9)
        data.createDimension("num_pixels", NUM_PIXELS)
        data.createDimension("num_lines", NUM_LINES)

        calibration = data.createGroup("calibration_data")
        for name, dim, size in (("bt_conversion_a", "num_chan_thermal", 9),
                                ("bt_conversion_b", "num_chan_thermal", 9),
                                ("channel_cw_thermal", "num_chan_thermal", 9),
                                ("band_averaged_solar_irradiance", "num_chan_solar", 11)):
            var = calibration.createVariable(name, np.float32, dimensions=(dim,))
            var[:] = np.arange(1, size + 1)

        quality = nc.createGroup("quality")
        quality.createDimension("gap_items", 2)
        for name in ("duration_of_product", "duration_of_data_present",
                     "duration_of_data_missing", "duration_of_data_degraded"):
            quality.createVariable(name, np.double, dimensions=())[:] = 1.0
        for name in ("gap_start_time_utc", "gap_end_time_utc"):
            quality.createVariable(name, np.double, dimensions=("gap_items",))[:] = [0.0, 0.0]

        measurement = data.createGroup("measurement_data")
        # The real files store one full row of pixels per on-disk chunk.
        radiance = measurement.createVariable("vii_668", np.float32,
                                              dimensions=("num_lines", "num_pixels"),
                                              chunksizes=(1, NUM_PIXELS))
        radiance[:] = np.arange(NUM_LINES * NUM_PIXELS).reshape(NUM_LINES, NUM_PIXELS)
        delta_lat = measurement.createVariable("delta_lat", np.float32,
                                               dimensions=("num_lines", "num_pixels"),
                                               chunksizes=(1, NUM_PIXELS))
        delta_lat[:] = 1.0

        if not with_tie_points:
            return

        measurement.createDimension("num_tie_points_act", NUM_TIE_POINTS_ACT)
        measurement.createDimension("num_tie_points_alt", NUM_TIE_POINTS_ALT)
        tie_shape = (NUM_TIE_POINTS_ALT, NUM_TIE_POINTS_ACT)
        tie_dims = ("num_tie_points_alt", "num_tie_points_act")
        lon = measurement.createVariable("longitude", np.float32, dimensions=tie_dims,
                                         chunksizes=(1, NUM_TIE_POINTS_ACT))
        lon[:] = np.linspace(-10.0, 10.0, np.prod(tie_shape)).reshape(tie_shape)
        lat = measurement.createVariable("latitude", np.float32, dimensions=tie_dims,
                                         chunksizes=(1, NUM_TIE_POINTS_ACT))
        lat[:] = np.linspace(40.0, 60.0, np.prod(tie_shape)).reshape(tie_shape)
        # Not used by the reader (yet), but part of the real product.
        sza = measurement.createVariable("solar_zenith", np.float32, dimensions=tie_dims,
                                         chunksizes=(1, NUM_TIE_POINTS_ACT))
        sza[:] = 25.0


def _make_handler(path):
    return METimageL1BNCFileHandler(
        filename=str(path),
        filename_info={
            "creation_time": datetime.datetime(2017, 9, 22, 22, 40, 10),
            "sensing_start_time": datetime.datetime(2017, 9, 20, 12, 30, 30),
            "sensing_end_time": datetime.datetime(2017, 9, 20, 18, 30, 50),
        },
        filetype_info={"cached_longitude": "data/measurement_data/longitude",
                       "cached_latitude": "data/measurement_data/latitude"},
    )


class TestMETimageL1bNCFileHandler(unittest.TestCase):
    """Test the METimageL1BNCFileHandler reader."""

    def setUp(self):
        """Set up the test."""
        # Easiest way to test the reader is to create a test netCDF file on the fly
        # uses a UUID to avoid permission conflicts during execution of tests in parallel
        self.test_file_name = TEST_FILE + str(uuid.uuid1()) + ".nc"
        _create_l1b_file(self.test_file_name)

        self.reader = _make_handler(self.test_file_name)

    def tearDown(self):
        """Remove the previously created test file."""
        # Catch Windows PermissionError for removing the created test file.
        try:
            os.remove(self.test_file_name)
        except OSError:
            pass

    def test_calibration_functions(self):
        """Test the calibration functions."""
        radiance = np.array([[1.0, 2.0, 5.0], [7.0, 10.0, 20.0]])

        cw = 13.0
        a = 3.0
        b = 100.0
        bt = self.reader._calibrate_bt(radiance, cw, a, b)
        expected_bt = np.array([[675.04993213, 753.10301462, 894.93149648],
                                [963.20401882, 1048.95086402, 1270.95546218]])
        assert np.allclose(bt, expected_bt)

        isi = 2.0
        refl = self.reader._calibrate_refl(radiance, isi)
        expected_refl = np.array([[157.07963268, 314.15926536, 785.3981634],
                                  [1099.55742876, 1570.79632679, 3141.59265359]])
        assert np.allclose(refl, expected_refl)

    def test_functions(self):
        """Test the functions."""
        # Checks that the _perform_orthorectification function is correctly executed
        variable = xr.DataArray(
            dims=("num_lines", "num_pixels"),
            name="test_name",
            attrs={
                "key_1": "value_1",
                "key_2": "value_2"
            },
            data=da.from_array(np.ones((NUM_LINES, NUM_PIXELS)))
        )

        orthorect_variable = self.reader._perform_orthorectification(variable, "data/measurement_data/delta_lat")
        expected_values = (np.degrees(np.ones((NUM_LINES, NUM_PIXELS)) / MEAN_EARTH_RADIUS)
                           + np.ones((NUM_LINES, NUM_PIXELS)))
        assert np.allclose(orthorect_variable.values, expected_values)

        # Checks that the _perform_calibration function is correctly executed in all cases
        # radiance calibration: return value is simply a copy of the variable
        return_variable = self.reader._perform_calibration(variable, {"calibration": "radiance"})
        assert np.all(return_variable == variable)

        # invalid calibration: raises a ValueError
        with pytest.raises(ValueError, match="Unknown calibration invalid for dataset test"):
            self.reader._perform_calibration(variable,
                                             {"calibration": "invalid", "name": "test"})

        # brightness_temperature calibration: checks that the return value is correct
        calibrated_variable = self.reader._perform_calibration(variable,
                                                               {"calibration": "brightness_temperature",
                                                                "chan_thermal_index": 3})
        expected_values = np.full((NUM_LINES, NUM_PIXELS), 1237.52069907)
        assert np.allclose(calibrated_variable.values, expected_values)

        # reflectance calibration: checks that the return value is correct
        calibrated_variable = self.reader._perform_calibration(variable,
                                                               {"calibration": "reflectance",
                                                                "wavelength": [0.658, 0.668, 0.678],
                                                                "chan_solar_index": 2})
        expected_values = np.full((NUM_LINES, NUM_PIXELS), 104.71975512)
        assert np.allclose(calibrated_variable.values, expected_values)


# NOTE:
# The tests below use the following fixture, defined in this module:
#   metimage_l1b_file - path to a small but structurally realistic L1B netCDF file


@pytest.fixture
def metimage_l1b_file(tmp_path):
    """Create a small METimage L1B file."""
    path = tmp_path / "metimage_l1b.nc"
    _create_l1b_file(path)
    return path


# Row chunks the reader is expected to produce for the test file (600 rows of 72 float32 pixels,
# 24 rows per scan) at a few dask "array.chunk-size" settings. The test file is small, so small
# chunk sizes are needed to get more than one chunk out of it. # The 24KiB case is included
# because its tie point chunking does not line up with whole scans after interpolation.
CHUNK_CASES = [
    ("16KiB", (48,) * 12 + (24,)),
    ("24KiB", (72,) * 8 + (24,)),
    ("64KiB", (216, 216, 168)),
]

# Dataset information as the YAML reader builds it from metimage_l1b_nc.yaml. These cover the
# three ways a dataset reaches the user: read straight from the file, served from the cached
# interpolated geolocation, and interpolated from tie points on the fly.
DATASET_INFOS = [
    {"name": "vii_668", "file_key": "data/measurement_data/vii_668",
     "calibration": "radiance", "chan_solar_index": 2},
    {"name": "lon_pixels", "file_key": "cached_longitude", "standard_name": "longitude"},
    {"name": "lat_pixels", "file_key": "cached_latitude", "standard_name": "latitude"},
    {"name": "solar_zenith_angle", "file_key": "data/measurement_data/solar_zenith",
     "standard_name": "solar_zenith_angle", "interpolate": True},
]


@pytest.mark.parametrize(("chunk_size", "expected_row_chunks"), CHUNK_CASES)
@pytest.mark.parametrize("dataset_info", DATASET_INFOS, ids=lambda info: info["name"])
def test_datasets_are_chunked_by_whole_scans(metimage_l1b_file, dataset_info,
                                             chunk_size, expected_row_chunks):
    """Test that every dataset keeps whole rows of pixels and is chunked in whole scans."""
    with dask.config.set({"array.chunk-size": chunk_size}):
        handler = _make_handler(metimage_l1b_file)
        variable = handler.get_dataset(None, dataset_info)

    assert variable.chunks == (expected_row_chunks, (NUM_PIXELS,))


def test_file_without_tie_points_is_still_chunked(tmp_path):
    """Test that a file without tie point dimensions still chunks its pixel variables."""
    path = tmp_path / "metimage_l1b_no_tie_points.nc"
    _create_l1b_file(path, with_tie_points=False)

    with dask.config.set({"array.chunk-size": "64KiB"}):
        handler = _make_handler(path)
        variable = handler.get_dataset(None, DATASET_INFOS[0])

    assert handler.longitude is None
    assert handler.latitude is None
    assert variable.chunks == ((216, 216, 168), (NUM_PIXELS,))
