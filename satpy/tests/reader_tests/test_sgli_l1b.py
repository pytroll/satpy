"""Tests for the SGLI L1B backend."""

import datetime as dt
import sys

import dask
import h5py
import numpy as np
import pytest

from satpy.readers.sgli_l1b import HDF5SGLI
from satpy.tests.utils import RANDOM_GEN

START_TIME = dt.datetime.now()
END_TIME = START_TIME + dt.timedelta(minutes=5)
FULL_KM_ARRAY = np.arange(1955 * 1250, dtype=np.uint16).reshape((1955, 1250))
MASK = 16383
LON_LAT_ARRAY = np.arange(197 * 126, dtype=np.float32).reshape((197, 126))
AZI_ARRAY = RANDOM_GEN.integers(-180 * 100, 180 * 100, size=(197, 126), dtype=np.int16)
ZEN_ARRAY = RANDOM_GEN.integers(0, 180 * 100, size=(197, 126), dtype=np.int16)


@pytest.fixture(scope="module")
def sgli_vn_file(tmp_path_factory):
    """Create a stub VN file."""
    filename = tmp_path_factory.mktemp("data") / "test_vn_file.h5"
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

        add_downsampled_geometry_data(h5f)

    return filename

@pytest.fixture(scope="module")
def sgli_ir_file(tmp_path_factory):
    """Create a stub IR file."""
    filename = tmp_path_factory.mktemp("data") / "test_ir_file.h5"
    with h5py.File(filename, "w") as h5f:
        global_attributes = h5f.create_group("Global_attributes")
        global_attributes.attrs["Scene_start_time"] = np.array([START_TIME.strftime("%Y%m%d %H:%M:%S.%f")[:-3]],
                                                               dtype="|S21")
        global_attributes.attrs["Scene_end_time"] = np.array([END_TIME.strftime("%Y%m%d %H:%M:%S.%f")[:-3]],
                                                             dtype="|S21")

        image_data = h5f.create_group("Image_data")
        image_data.attrs["Number_of_lines"] = 1854
        image_data.attrs["Number_of_pixels"] = 1250

        sw01 = image_data.create_dataset("Lt_SW01", data=FULL_KM_ARRAY, chunks=(116, 157))
        sw01.attrs["Slope_reflectance"] = np.array([5e-05], dtype=np.float32)
        sw01.attrs["Offset_reflectance"] = np.array([0.0], dtype=np.float32)
        sw01.attrs["Slope"] = np.array([0.02], dtype=np.float32)
        sw01.attrs["Offset"] = np.array([-25], dtype=np.float32)
        sw01.attrs["Mask"] = np.array([16383], dtype=np.uint16)
        sw01.attrs["Bit00(LSB)-13"] = np.array([b"Digital Number\n16383 : Missing value\n16382 : Saturation value"],
                                               dtype="|S61")


        ti01 = image_data.create_dataset("Lt_TI01", data=FULL_KM_ARRAY, chunks=(116, 157))
        ti01.attrs["Slope"] = np.array([0.0012], dtype=np.float32)
        ti01.attrs["Offset"] = np.array([-1.65], dtype=np.float32)
        ti01.attrs["Mask"] = np.array([16383], dtype=np.uint16)
        ti01.attrs["Bit00(LSB)-13"] = np.array([b"Digital Number\n16383 : Missing value\n16382 : Saturation value"],
                                               dtype="|S61")
        ti01.attrs["Center_wavelength"] = np.array([12000], dtype=np.float32)

        add_downsampled_geometry_data(h5f)

    return filename

@pytest.fixture(scope="module")
def sgli_pol_file(tmp_path_factory):
    """Create a POL stub file."""
    filename = tmp_path_factory.mktemp("data") / "test_pol_file.h5"
    with h5py.File(filename, "w") as h5f:
        global_attributes = h5f.create_group("Global_attributes")
        global_attributes.attrs["Scene_start_time"] = np.array([START_TIME.strftime("%Y%m%d %H:%M:%S.%f")[:-3]],
                                                               dtype="|S21")
        global_attributes.attrs["Scene_end_time"] = np.array([END_TIME.strftime("%Y%m%d %H:%M:%S.%f")[:-3]],
                                                             dtype="|S21")

        image_data = h5f.create_group("Image_data")
        image_data.attrs["Number_of_lines"] = 1854
        image_data.attrs["Number_of_pixels"] = 1250

        p1_0 = image_data.create_dataset("Lt_P1_0", data=FULL_KM_ARRAY, chunks=(116, 157))
        p1_0.attrs["Slope_reflectance"] = np.array([5e-05], dtype=np.float32)
        p1_0.attrs["Offset_reflectance"] = np.array([0.0], dtype=np.float32)
        p1_0.attrs["Slope"] = np.array([0.02], dtype=np.float32)
        p1_0.attrs["Offset"] = np.array([-25], dtype=np.float32)
        p1_0.attrs["Mask"] = np.array([16383], dtype=np.uint16)
        p1_0.attrs["Bit00(LSB)-13"] = np.array([b"Digital Number\n16383 : Missing value\n16382 : Saturation value"],
                                               dtype="|S61")


        p1_m60 = image_data.create_dataset("Lt_P1_m60", data=FULL_KM_ARRAY, chunks=(116, 157))
        p1_m60.attrs["Slope_reflectance"] = np.array([5e-05], dtype=np.float32)
        p1_m60.attrs["Offset_reflectance"] = np.array([-60.0], dtype=np.float32)
        p1_m60.attrs["Slope"] = np.array([0.0012], dtype=np.float32)
        p1_m60.attrs["Offset"] = np.array([-1.65], dtype=np.float32)
        p1_m60.attrs["Mask"] = np.array([16383], dtype=np.uint16)
        p1_m60.attrs["Bit00(LSB)-13"] = np.array([b"Digital Number\n16383 : Missing value\n16382 : Saturation value"],
                                               dtype="|S61")

        p1_60 = image_data.create_dataset("Lt_P1_60", data=FULL_KM_ARRAY, chunks=(116, 157))
        p1_60.attrs["Slope_reflectance"] = np.array([5e-05], dtype=np.float32)
        p1_60.attrs["Offset_reflectance"] = np.array([60.0], dtype=np.float32)
        p1_60.attrs["Slope"] = np.array([0.0012], dtype=np.float32)
        p1_60.attrs["Offset"] = np.array([-1.65], dtype=np.float32)
        p1_60.attrs["Mask"] = np.array([16383], dtype=np.uint16)
        p1_60.attrs["Bit00(LSB)-13"] = np.array([b"Digital Number\n16383 : Missing value\n16382 : Saturation value"],
                                               dtype="|S61")

        geometry_data = h5f.create_group("Geometry_data")
        longitude = geometry_data.create_dataset("Longitude", data=FULL_KM_ARRAY.astype(np.float32), chunks=(47, 63))
        longitude.attrs["Resampling_interval"] = 1
        latitude = geometry_data.create_dataset("Latitude", data=FULL_KM_ARRAY.astype(np.float32), chunks=(47, 63))
        latitude.attrs["Resampling_interval"] = 1

        return filename

def add_downsampled_geometry_data(h5f):
    """Add downsampled geometry data to an h5py file instance."""
    geometry_data = h5f.create_group("Geometry_data")
    longitude = geometry_data.create_dataset("Longitude", data=LON_LAT_ARRAY, chunks=(47, 63))
    longitude.attrs["Resampling_interval"] = 10
    latitude = geometry_data.create_dataset("Latitude", data=LON_LAT_ARRAY, chunks=(47, 63))
    latitude.attrs["Resampling_interval"] = 10

    angles_slope = np.array([0.01], dtype=np.float32)
    angles_offset = np.array([0], dtype=np.float32)

    azimuth = geometry_data.create_dataset("Sensor_azimuth", data=AZI_ARRAY, chunks=(47, 63))
    azimuth.attrs["Resampling_interval"] = 10
    azimuth.attrs["Slope"] = angles_slope
    azimuth.attrs["Offset"] = angles_offset
    zenith = geometry_data.create_dataset("Sensor_zenith", data=ZEN_ARRAY, chunks=(47, 63))
    zenith.attrs["Resampling_interval"] = 10
    zenith.attrs["Slope"] = angles_slope
    zenith.attrs["Offset"] = angles_offset

    sazimuth = geometry_data.create_dataset("Solar_azimuth", data=AZI_ARRAY, chunks=(47, 63))
    sazimuth.attrs["Resampling_interval"] = 10
    sazimuth.attrs["Slope"] = angles_slope
    sazimuth.attrs["Offset"] = angles_offset
    szenith = geometry_data.create_dataset("Solar_zenith", data=ZEN_ARRAY, chunks=(47, 63))
    szenith.attrs["Resampling_interval"] = 10
    szenith.attrs["Slope"] = angles_slope
    szenith.attrs["Offset"] = angles_offset


def test_start_time(sgli_vn_file):
    """Test that the start time is extracted."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    microseconds = START_TIME.microsecond % 1000
    assert handler.start_time == START_TIME - dt.timedelta(microseconds=microseconds)


def test_end_time(sgli_vn_file):
    """Test that the end time is extracted."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    microseconds = END_TIME.microsecond % 1000
    assert handler.end_time == END_TIME - dt.timedelta(microseconds=microseconds)

def test_get_dataset_counts(sgli_vn_file):
    """Test that counts can be extracted from a file."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="counts")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01", "units": "",
                                    "standard_name": ""})
    assert np.allclose(res, FULL_KM_ARRAY & MASK)
    assert res.dtype == np.uint16
    assert res.attrs["platform_name"] == "GCOM-C1"
    assert res.attrs["sensor"] == "sgli"

def test_get_dataset_for_unknown_channel(sgli_vn_file):
    """Test that counts can be extracted from a file."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="VIN", resolution=1000, polarization=None, calibration="counts")
    with pytest.raises(KeyError):
        handler.get_dataset(did, {"file_key": "Image_data/Lt_VIN01", "units": "",
                                        "standard_name": ""})

def test_get_vn_dataset_reflectances(sgli_vn_file):
    """Test that the vn datasets can be calibrated to reflectances."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="reflectance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01", "units": "%",
                                    "standard_name": ""})
    assert np.allclose(res[0, :] / 100, FULL_KM_ARRAY[0, :] * 5e-5 - 0.05)
    assert res.dtype == np.float32
    assert res.dims == ("y", "x")
    assert res.attrs["units"] == "%"

def test_get_vn_dataset_radiance(sgli_vn_file):
    """Test that datasets can be calibrated to radiance."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="radiance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01", "units": "W m-2 um-1 sr-1",
                                    "standard_name": "toa_outgoing_radiance_per_unit_wavelength"})
    assert np.allclose(res[0, :], FULL_KM_ARRAY[0, :] * np.float32(0.02) - 25)
    assert res.dtype == np.float32
    assert res.attrs["units"] == "W m-2 um-1 sr-1"
    assert res.attrs["standard_name"] == "toa_outgoing_radiance_per_unit_wavelength"

def test_channel_is_masked(sgli_vn_file):
    """Test that channels are masked for no-data."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="counts")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01", "units": "",
                                    "standard_name": ""})
    assert res.max() == MASK

def test_missing_values_are_masked(sgli_vn_file):
    """Check that missing values are masked."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="VN1", resolution=1000, polarization=None, calibration="radiance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01", "units": "",
                                    "standard_name": ""})
    assert np.isnan(res).sum() == 149

def test_channel_is_chunked(sgli_vn_file):
    """Test that the channel data is chunked."""
    with dask.config.set({"array.chunk-size": "1MiB"}):
        handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
        did = dict(name="VN1", resolution=1000, polarization=None, calibration="counts")
        res = handler.get_dataset(did, {"file_key": "Image_data/Lt_VN01", "units": "",
                                    "standard_name": ""})
        assert res.chunks[0][0] > 116

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Python 3.10 or higher needed for geotiepoints")
def test_loading_lon_lat(sgli_vn_file):
    """Test that loading lons and lats works."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="longitude_v", resolution=1000, polarization=None)
    res = handler.get_dataset(did, {"file_key": "Geometry_data/Longitude", "units": "",
                                    "standard_name": ""})
    assert res.shape == (1955, 1250)
    assert res.chunks is not None
    assert res.dtype == np.float32
    assert res.dims == ("y", "x")

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Python 3.10 or higher needed for geotiepoints")
def test_loading_sensor_angles(sgli_vn_file):
    """Test loading the satellite angles."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="satellite_zenith_angle", resolution=1000, polarization=None)
    res = handler.get_dataset(did, {"file_key": "Geometry_data/Sensor_zenith", "units": "",
                                    "standard_name": ""})
    assert res.shape == (1955, 1250)
    assert res.chunks is not None
    assert res.dtype == np.float32
    assert res.min() >= 0

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Python 3.10 or higher needed for geotiepoints")
def test_loading_solar_angles(sgli_vn_file):
    """Test loading sun angles."""
    handler = HDF5SGLI(sgli_vn_file, {"resolution": "L"}, {})
    did = dict(name="solar_azimuth_angle", resolution=1000, polarization=None)
    res = handler.get_dataset(did, {"file_key": "Geometry_data/Sensor_zenith", "units": "",
                                    "standard_name": ""})
    assert res.shape == (1955, 1250)
    assert res.chunks is not None
    assert res.dtype == np.float32
    assert res.max() <= 180

def test_get_sw_dataset_reflectances(sgli_ir_file):
    """Test getting SW dataset reflectances."""
    handler = HDF5SGLI(sgli_ir_file, {"resolution": "L"}, {})
    did = dict(name="SW1", resolution=1000, polarization=None, calibration="reflectance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_SW01", "units": "",
                                    "standard_name": ""})
    assert np.allclose(res[0, :] / 100, FULL_KM_ARRAY[0, :] * 5e-5)
    assert res.dtype == np.float32

def test_get_ti_dataset_radiance(sgli_ir_file):
    """Test getting thermal IR radiances."""
    handler = HDF5SGLI(sgli_ir_file, {"resolution": "L"}, {})
    did = dict(name="TI1", resolution=1000, polarization=None, calibration="radiance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_TI01", "units": "",
                                    "standard_name": ""})
    assert np.allclose(res[0, :], FULL_KM_ARRAY[0, :] * np.float32(0.0012) - 1.65)
    assert res.dtype == np.float32

def test_get_ti_dataset_bt(sgli_ir_file):
    """Test getting brightness temperatures for IR channels."""
    handler = HDF5SGLI(sgli_ir_file, {"resolution": "L"}, {})
    did = dict(name="TI1", resolution=1000, polarization=None, calibration="brightness_temperature")
    with pytest.raises(NotImplementedError):
        _ = handler.get_dataset(did, {"file_key": "Image_data/Lt_TI01", "units": "K",
                                    "standard_name": "toa_brightness_temperature"})

@pytest.mark.skipif(sys.version_info < (3, 10), reason="Python 3.10 or higher needed for geotiepoints")
def test_get_ti_lon_lats(sgli_ir_file):
    """Test getting the lons and lats for IR channels."""
    handler = HDF5SGLI(sgli_ir_file, {"resolution": "L"}, {})
    did = dict(name="longitude_ir", resolution=1000, polarization=None)
    res = handler.get_dataset(did, {"file_key": "Geometry_data/Longitude", "units": "",
                                    "standard_name": ""})
    assert res.shape == (1854, 1250)
    assert res.chunks is not None
    assert res.dtype == np.float32

@pytest.mark.parametrize("polarization", [0, -60, 60])
def test_get_polarized_dataset_reflectance(sgli_pol_file, polarization):
    """Test getting polarized reflectances."""
    handler = HDF5SGLI(sgli_pol_file, {"resolution": "L"}, {})
    did = dict(name="P1", resolution=1000, polarization=polarization, calibration="reflectance")
    res = handler.get_dataset(did, {"file_key": "Image_data/Lt_P1_{polarization}", "units": "%",
                                    "standard_name": "toa_bidirectional_reflectance"})
    assert res.dtype == np.float32
    expected = (FULL_KM_ARRAY[0, :] * np.float32(5e-5) + np.float32(polarization)) * 100
    np.testing.assert_allclose(res[0, :], expected)
    assert res.attrs["units"] == "%"
    assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"

def test_get_polarized_longitudes(sgli_pol_file):
    """Test getting polarized reflectances."""
    handler = HDF5SGLI(sgli_pol_file, {"resolution": "L"}, {})
    did = dict(name="longitude", resolution=1000, polarization=0)
    res = handler.get_dataset(did, {"file_key": "Geometry_data/Longitude", "units": "",
                                    "standard_name": ""})
    assert res.dtype == np.float32
    expected = FULL_KM_ARRAY.astype(np.float32)
    np.testing.assert_allclose(res, expected)
