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

from satpy.readers.nwcsaf_hrw_nc import WIND_CHANNELS

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
NOMINAL_PRODUCT_TIME = "2025-02-06T13:00:00Z"
SPATIAL_RESOLUTION = np.array([3.], dtype=np.float32)
SATELLITE_IDENTIFIER = np.bytes_("MSG3")
# Dataset attributes accessed by the file handler
TIME_PERIOD = np.array([15.], dtype=np.float32)

NUM_OBS = 123

FILENAME_INFO = {"platform_id": "MSG3", "region_id": "MSG-N-BS", "start_time": dt.datetime(2025, 2, 6, 13, 0)}
FILETYPE_INFO = {"file_type": "nc_nwcsaf_geo_hrw"}


@pytest.fixture(scope="module")
def hrw_file(tmp_path_factory):
    """Create a HRW data file."""
    fname = tmp_path_factory.mktemp("data") / "S_NWC_HRW_MSG3_MSG-N-BS_20250206T130000Z.nc"
    with h5py.File(fname, "w") as h5f:
        h5f.attrs["nominal_product_time"] = NOMINAL_PRODUCT_TIME
        h5f.attrs["spatial_resolution"] = SPATIAL_RESOLUTION
        h5f.attrs["satellite_identifier"] = SATELLITE_IDENTIFIER
        for ch in WIND_CHANNELS:
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
        val = np.array(255 * np.random.random(), dtype=dtype).item()  # noqa
        data.append(val)

    return tuple(data)


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

    assert len(list(avail)) == len(WIND_CHANNELS) * len(WIND_DTYPES)


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

    assert data.size == len(WIND_CHANNELS) * NUM_OBS


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

    assert data.shape == (len(WIND_CHANNELS) * NUM_OBS,)
