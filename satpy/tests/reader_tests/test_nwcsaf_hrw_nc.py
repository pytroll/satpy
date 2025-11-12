# Copyright (c) 2025- Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Unittests for NWC SAF GEO HRW reader."""

import datetime as dt

import h5py
import numpy as np
import pytest

from satpy.readers.nwcsaf_hrw_nc import POST_V2025_DATASETS, PRE_V2025_SEVIRI_WIND_CHANNELS
from satpy.tests.utils import RANDOM_GEN

# This is the actual dtype of the trajectory items. We do not support it, so won't add this
# complexity to the test file creation. It's here anyway if someone wants to add it.
TRAJECTORY_DTYPE_ACTUAL = np.dtype([
    ("latitude", "<f4"),
    ("longitude", "<f4"),
    ("latitude_increment", "<f4"),
    ("longitude_increment", "<f4"),
    ("air_temperature", "<f4"),
    ("air_pressure", "<f4"),
    ("air_pressure_error", "<f4"),
    ("air_pressure_correction", "<f4"),
    ("wind_speed", "<f4"),
    ("wind_from_direction", "<f4"),
    ("quality_index_with_forecast", "i1"),
    ("quality_index_without_forecast", "i1"),
    ("quality_index_iwwg_value", "i1"),
    ("tracer_correlation_method", "i1"),
    ("tracer_type", "i1"),
    ("height_assignment_method", "i1"),
    ("orographic_index", "i1"),
    ("cloud_type", "i1"),
    ("correlation", "i1")
])

WIND_DTYPES = [
    ("wind_idx", np.dtype("uint32")),
    ("previous_wind_idx", np.dtype("uint32")),
    ("number_of_winds", np.dtype("uint8")),
    ("correlation_test", np.dtype("uint8")),
    ("quality_test", np.dtype("uint16")),
    ("segment_x", np.dtype("uint32")),
    ("segment_y", np.dtype("uint32")),
    ("segment_x_pix", np.dtype("uint32")),
    ("segment_y_pix", np.dtype("uint32")),
    ("latitude", np.dtype("<f4")),
    ("longitude", np.dtype("<f4")),
    ("latitude_increment", np.dtype("<f4")),
    ("longitude_increment", np.dtype("<f4")),
    ("air_temperature", np.dtype("<f4")),
    ("air_pressure", np.dtype("<f4")),
    ("air_pressure_error", np.dtype("<f4")),
    ("air_pressure_correction", np.dtype("<f4")),
    ("air_pressure_nwp_at_best_fit_level", np.dtype("<f4")),
    ("wind_speed", np.dtype("<f4")),
    ("wind_from_direction", np.dtype("<f4")),
    ("wind_speed_nwp_at_amv_level", np.dtype("<f4")),
    ("wind_from_direction_nwp_at_amv_level", np.dtype("<f4")),
    ("wind_speed_nwp_at_best_fit_level", np.dtype("<f4")),
    ("wind_from_direction_nwp_at_best_fit_level", np.dtype("<f4")),
    ("wind_speed_difference_nwp_at_amv_level", np.dtype("<f4")),
    ("wind_from_direction_difference_nwp_at_amv_level", np.dtype("<f4")),
    ("wind_speed_difference_nwp_at_best_fit_level", np.dtype("<f4")),
    ("wind_from_direction_difference_nwp_at_best_fit_level", np.dtype("<f4")),
    ("quality_index_with_forecast", np.dtype("uint8")),
    ("quality_index_without_forecast", np.dtype("uint8")),
    ("quality_index_iwwg_value", np.dtype("uint8")),
    ("tracer_correlation_method", np.dtype("uint8")),
    ("tracer_type", np.dtype("uint8")),
    ("height_assignment_method", np.dtype("uint8")),
    ("orographic_index", np.dtype("uint8")),
    ("cloud_type", np.dtype("uint8")),
    ("correlation", np.dtype("uint8")),
]

# Global attributes accessed by the file handler
NOMINAL_PRODUCT_TIME = np.bytes_("2025-02-06T13:00:00Z")
NOMINAL_PRODUCT_TIME_V2025 = np.bytes_("2025-09-23T13:00:00Z")
SPATIAL_RESOLUTION = np.array([3.], dtype=np.float32)
SATELLITE_IDENTIFIER = np.bytes_("MSG3")
SATELLITE_IDENTIFIER_V2025 = np.bytes_("MTI1")
ALGORITHM_VERSION_V2021 = np.bytes_("6.2.2")
ALGORITHM_VERSION_V2025 = np.bytes_("7.0")
SAMPLING_INTERVAL_V2025 = np.bytes_("10.00 minutes")

# Dataset attributes accessed by the file handler
TIME_PERIOD = np.array([15.], dtype=np.float32)

NUM_OBS = 123

FILENAME_INFO = {"platform_id": "MSG3", "region_id": "MSG-N-BS", "start_time": dt.datetime(2025, 2, 6, 13, 0)}
FILENAME_INFO_V2025 = {"platform_id": "MTI1", "region_id": "MSG-N-BS", "start_time": dt.datetime(2025, 9, 23, 13, 0)}
FILETYPE_INFO = {"file_type": "nc_nwcsaf_geo_hrw"}


@pytest.fixture(scope="module")
def hrw_file(tmp_path_factory):
    """Create a HRW data file."""
    fname = tmp_path_factory.mktemp("data") / "S_NWC_HRW_MSG3_MSG-N-BS_20250206T130000Z.nc"
    with h5py.File(fname, "w") as h5f:
        h5f.attrs["nominal_product_time"] = NOMINAL_PRODUCT_TIME
        h5f.attrs["spatial_resolution"] = SPATIAL_RESOLUTION
        h5f.attrs["satellite_identifier"] = SATELLITE_IDENTIFIER
        h5f.attrs["product_algorithm_version"] = ALGORITHM_VERSION_V2021
        for ch in PRE_V2025_SEVIRI_WIND_CHANNELS:
            _create_channel_dataset(h5f, ch)
    return fname


def _create_channel_dataset(h5f, name):
    dset = h5f.create_dataset(name, shape=(NUM_OBS,), dtype=np.dtype(WIND_DTYPES))
    dset.attrs["time_period"] = TIME_PERIOD
    data = _create_data()
    for i in range(NUM_OBS):
        dset[i] = data


def _create_data():
    data = []
    for key, dtype in WIND_DTYPES:
        val = np.array(255 * RANDOM_GEN.random(), dtype=dtype).item()
        data.append(val)

    return tuple(data)


@pytest.fixture(scope="module")
def hrw_v2025_file(tmp_path_factory):
    """Create a HRW data file in v2025 format."""
    fname = tmp_path_factory.mktemp("data") / "S_NWC_HRW_MTI1_MSG-N-BS_20250923T130000Z.nc"
    with h5py.File(fname, "w") as h5f:
        h5f.attrs["nominal_product_time"] = NOMINAL_PRODUCT_TIME_V2025
        h5f.attrs["spatial_resolution"] = SPATIAL_RESOLUTION
        h5f.attrs["satellite_identifier"] = SATELLITE_IDENTIFIER_V2025
        h5f.attrs["product_algorithm_version"] = ALGORITHM_VERSION_V2025
        h5f.attrs["sampling_interval"] = SAMPLING_INTERVAL_V2025
        # Just add one, they are pretty much the same:
        dset = h5f.create_dataset("wind_speed", shape=(NUM_OBS,), dtype=np.double)
        dset.attrs["standard_name"] = np.bytes_("wind_speed")
        dset.attrs["units"] = np.bytes_("m s-1")
        dset.attrs["valid_range"] = np.array([0., 409.4])
        dset.attrs["_FillValue"] = np.double(409.5)
        # Make the first value invalid
        data = np.array(409.4 * RANDOM_GEN.random((NUM_OBS,)), dtype=np.double)
        data[0] = 409.5
        dset[:] = data
        # And coordinates, those are needed
        lons = h5f.create_dataset("lon", shape=(NUM_OBS,), dtype=np.double)
        lons.attrs["standard_name"] = "longitude"
        lons.attrs["units"] = "degrees_east"
        lons.attrs["valid_range"] = np.array([-90., 90.], dtype=np.double)
        lons.attrs["_FillValue"] = np.double(245.)
        lons[:] = 180 * RANDOM_GEN.random((NUM_OBS,), dtype=np.double) - 90
        lats = h5f.create_dataset("lat", shape=(NUM_OBS,), dtype=np.double)
        lats.attrs["standard_name"] = "latitude"
        lats.attrs["units"] = "degrees_north"
        lats.attrs["valid_range"] = np.array([-180., 179.99999], dtype=np.double)
        lats.attrs["_FillValue"] = np.double(491.)
        lats[:] = 360 * RANDOM_GEN.random((NUM_OBS,), dtype=np.double) - 180.00001
    return fname



def test_hrw_handler_init(hrw_file):
    """Test the filehandler initialization works."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_file, FILENAME_INFO, FILETYPE_INFO)

    assert fh.filename_info == FILENAME_INFO
    assert fh.filetype_info == FILETYPE_INFO
    assert fh.platform_name is not None
    assert fh.sensor == "seviri"


def test_available_datasets(hrw_file):
    """Test that dynamic creation of available datasets works."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_file, FILENAME_INFO, FILETYPE_INFO)

    avail = fh.available_datasets()

    assert len(list(avail)) == len(PRE_V2025_SEVIRI_WIND_CHANNELS) * len(WIND_DTYPES)


def test_available_merged_datasets(hrw_file):
    """Test that dynamic creation of available datasets works when the channels are merge together."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_file, FILENAME_INFO, FILETYPE_INFO, merge_channels=True)

    avail = fh.available_datasets()

    assert len(list(avail)) == len(WIND_DTYPES)


def test_get_dataset(hrw_file):
    """Test reading a dataset."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_file, FILENAME_INFO, FILETYPE_INFO)

    data = fh.get_dataset({"name": "wind_vis06_wind_speed"}, {"the_answer": 42})

    assert data.attrs["the_answer"] == 42
    _check_common_attrs(data)


def test_get_merged_dataset(hrw_file):
    """Test reading a merged dataset."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_file, FILENAME_INFO, FILETYPE_INFO, merge_channels=True)

    data = fh.get_dataset({"name": "wind_speed"}, {})

    assert data.size == len(PRE_V2025_SEVIRI_WIND_CHANNELS) * NUM_OBS


def _check_common_attrs(data):
    assert "wind_vis06_longitude" in data.coords
    assert "wind_vis06_latitude" in data.coords
    assert data.dtype == np.float32
    assert data.values.dtype == np.float32
    assert "units" in data.attrs


def test_hrw_via_scene(hrw_file):
    """Test reading HRW datasets via Scene."""
    from satpy import Scene

    scn = Scene(reader="nwcsaf-geo", filenames=[hrw_file])
    scn.load(["wind_vis06_wind_from_direction"])
    data = scn["wind_vis06_wind_from_direction"]

    assert scn.start_time == dt.datetime(2025, 2, 6, 13, 0)
    assert scn.end_time == dt.datetime(2025, 2, 6, 13, 15)
    _check_scene_dataset_attrs(data)
    _check_common_attrs(data)


def _check_scene_dataset_attrs(data):
    from pyresample.geometry import SwathDefinition

    assert data.attrs["file_type"] == "nc_nwcsaf_geo_hrw"
    assert data.attrs["resolution"] == 3000.0
    assert data.attrs["name"] == "wind_vis06_wind_from_direction"
    assert data.attrs["coordinates"] == ("wind_vis06_longitude", "wind_vis06_latitude")
    assert data.attrs["start_time"] == dt.datetime(2025, 2, 6, 13, 0)
    assert data.attrs["end_time"] == dt.datetime(2025, 2, 6, 13, 15)
    assert data.attrs["reader"] == "nwcsaf-geo"
    assert isinstance(data.attrs["area"], SwathDefinition)
    assert data.attrs["area"].shape == (NUM_OBS,)


def test_merged_hrw_via_scene(hrw_file):
    """Test reading HRW merged datasets via Scene."""
    from satpy import Scene

    scn = Scene(reader="nwcsaf-geo", filenames=[hrw_file], reader_kwargs={"merge_channels": True})
    scn.load(["wind_from_direction"])
    data = scn["wind_from_direction"]

    assert data.shape == (len(PRE_V2025_SEVIRI_WIND_CHANNELS) * NUM_OBS,)


def test_hrw_handler_init_v2025(hrw_v2025_file):
    """Test the filehandler initialization works."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_v2025_file, FILENAME_INFO_V2025, FILETYPE_INFO)

    assert fh.filename_info == FILENAME_INFO_V2025
    assert fh.filetype_info == FILETYPE_INFO
    assert fh.platform_name is not None
    assert fh.sensor == "fci"


def test_available_datasets_v2025(hrw_v2025_file):
    """Test that dynamic creation of available datasets works."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_v2025_file, FILENAME_INFO, FILETYPE_INFO)

    avail = fh.available_datasets()
    avail_dsets = set(d[1]["name"] for d in avail)
    assert avail_dsets == set(POST_V2025_DATASETS)


def test_get_dataset_v2025(hrw_v2025_file):
    """Test reading a dataset from a file in v2025 format."""
    from satpy.readers.nwcsaf_hrw_nc import NWCSAFGEOHRWFileHandler

    fh = NWCSAFGEOHRWFileHandler(hrw_v2025_file, FILENAME_INFO, FILETYPE_INFO)

    data = fh.get_dataset({"name": "wind_speed"}, {"the_answer": 42})
    assert "units" in data.attrs
    assert data.shape == (NUM_OBS,)


def test_hrw_via_scene_v2025(hrw_v2025_file):
    """Test reading HRW v2025 datasets via Scene."""
    from satpy import Scene

    scn = Scene(reader="nwcsaf-geo", filenames=[hrw_v2025_file])
    scn.load(["wind_speed"])
    data = scn["wind_speed"]

    assert data.shape == (NUM_OBS,)
    assert "units" in data.attrs
    assert "standard_name" in data.attrs
    assert np.isnan(data[0].compute())


@pytest.mark.parametrize("dset", ["longitude", "latitude"])
def test_lon_and_lat_via_scene_v2025(hrw_v2025_file, dset):
    """Test reading longitude and latitude datasets v2025 datasets via Scene."""
    from satpy import Scene

    scn = Scene(reader="nwcsaf-geo", filenames=[hrw_v2025_file])
    scn.load([dset])
    data = scn[dset]

    assert data.shape == (NUM_OBS,)
    assert "units" in data.attrs
    assert "standard_name" in data.attrs
