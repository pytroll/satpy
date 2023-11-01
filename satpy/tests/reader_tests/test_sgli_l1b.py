"""Tests for the SGLI L1B backend."""
from datetime import datetime, timedelta

import dask
import dask.array as da
import h5py
import numpy as np
import pytest
from dask.array.core import normalize_chunks
from xarray import DataArray, Dataset, open_dataset

from satpy.readers.sgli_l1b import HDF5SGLI

START_TIME = datetime.now()
END_TIME = START_TIME + timedelta(minutes=5)
FULL_KM_ARRAY = np.arange(1955 * 1250, dtype=np.uint16).reshape((1955, 1250))
MASK = 16383
LON_LAT_ARRAY = np.arange(197 * 126, dtype=np.uint16).reshape((197, 126))


def test_open_dataset():
    """Test open_dataset function."""
    from satpy.readers.sgli_l1b import SGLIBackend
    filename = "/home/a001673/data/satellite/gcom-c/GC1SG1_202002231142M25511_1BSG_VNRDL_1008.h5"
    res = open_dataset(filename, engine=SGLIBackend, chunks={})
    assert isinstance(res, Dataset)
    data_array = res["Lt_VN01"]
    assert isinstance(data_array, DataArray)
    assert isinstance(data_array.data, da.Array)
    assert data_array.chunks == normalize_chunks((116, 157), data_array.shape)


@pytest.fixture(scope="session")
def sgli_file(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "test_file.h5"
    with h5py.File(filename, "w") as h5f:
        global_attributes = h5f.create_group("Global_attributes")
        global_attributes.attrs["Scene_start_time"] = np.array([START_TIME.strftime("%Y%m%d %H:%M:%S.%f")[:-3]],
                                                               dtype="|S21")
        global_attributes.attrs["Scene_end_time"] = np.array([END_TIME.strftime("%Y%m%d %H:%M:%S.%f")[:-3]],
                                                             dtype="|S21")

        image_data = h5f.create_group("Image_data")
        image_data.attrs["Number_of_lines"] = 1955
        image_data.attrs["Number_of_pixels"] = 1250
        vn01 = image_data.create_dataset("Lt_VN01", data=FULL_KM_ARRAY, chunks=(116, 157))
        vn01.attrs["Slope_reflectance"] = np.array([5e-05], dtype=np.float32)
        vn01.attrs["Offset_reflectance"] = np.array([-0.05], dtype=np.float32)
        vn01.attrs["Slope"] = np.array([0.02], dtype=np.float32)
        vn01.attrs["Offset"] = np.array([-25], dtype=np.float32)
        vn01.attrs["Mask"] = np.array([16383], dtype=np.uint16)
        vn01.attrs["Bit00(LSB)-13"] = np.array([b"Digital Number\n16383 : Missing value\n16382 : Saturation value"],
                                               dtype="|S61")

        geometry_data = h5f.create_group("Geometry_data")
        longitude = geometry_data.create_dataset("Longitude", data=LON_LAT_ARRAY, chunks=(47, 63))
        longitude.attrs["Resampling_interval"] = 10
        latitude = geometry_data.create_dataset("Latitude", data=LON_LAT_ARRAY, chunks=(47, 63))
        latitude.attrs["Resampling_interval"] = 10
    return filename

def test_start_time(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    microseconds = START_TIME.microsecond % 1000
    assert handler.start_time == START_TIME - timedelta(microseconds=microseconds)


def test_end_time(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    microseconds = END_TIME.microsecond % 1000
    assert handler.end_time == END_TIME - timedelta(microseconds=microseconds)

def test_get_dataset_counts(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="counts")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01"})
    assert np.allclose(res, FULL_KM_ARRAY & MASK)
    assert res.dtype == np.uint16
    assert res.attrs["platform_name"] == "GCOM-C1"

def test_get_dataset_reflectances(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="reflectance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01"})
    assert np.allclose(res[0, :] / 100, FULL_KM_ARRAY[0, :] * 5e-5 - 0.05)
    assert res.dtype == np.float32

def test_get_dataset_radiance(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="radiance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01"})
    assert np.allclose(res[0, :], FULL_KM_ARRAY[0, :] * np.float32(0.02) - 25)
    assert res.dtype == np.float32

def test_channel_is_masked(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="counts")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01"})
    assert res.max() == MASK

def test_missing_values_are_masked(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="radiance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01"})
    assert np.isnan(res).sum() == 149

def test_channel_is_chunked(sgli_file):
    with dask.config.set({"array.chunk-size": "1MiB"}):
        handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
        did = dict(name="VN1", resolution=1000, polarization=None, calibration="counts")
        res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01"})
        assert res.chunks[0][0] > 116

def test_loading_lon_lat(sgli_file):
    handler = HDF5SGLI(sgli_file, {"resolution": "L"}, {})
    did = dict(name="longitude_v", resolution=1000, polarization=None)
    res = handler.get_dataset(did, {"file_key": "Geometry_data/Longitude"})
    assert res.shape == (1955, 1250)
    assert res.chunks is not None
    assert res.dtype == np.float32
