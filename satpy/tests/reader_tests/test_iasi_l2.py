#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Unit tests for IASI L2 reader."""

import datetime as dt
import math
import os

import numpy as np
import pytest
import xarray as xr

from satpy.readers.iasi_l2 import IASIL2HDF5

SCAN_WIDTH = 120
NUM_LEVELS = 138
NUM_SCANLINES = 10
FNAME = "W_XX-EUMETSAT-kan,iasi,metopb+kan_C_EUMS_20170920103559_IASI_PW3_02_M01_20170920102217Z_20170920102912Z.hdf"
# Structure for the test data, to be written to HDF5 file
TEST_DATA = {
    "INFO": {
        "OmC": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                "attrs": {"long_name": "Cloud signal. Predicted average window channel 'Obs minus Calc",
                          "units": "K"}},
        "FLG_AMSUBAD": {"data": np.zeros((NUM_SCANLINES, 30), dtype=np.uint8),
                        "attrs": {}},
        "FLG_IASIBAD": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.uint8),
                        "attrs": {}},
        "FLG_MHSBAD": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.uint8),
                        "attrs": {}},
        # Not implemented in the reader
        "mdist": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                  "attrs": {}},
    },
    "L1C": {
        "Latitude": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                     "attrs": {"units": "degrees_north"}},
        "Longitude": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      "attrs": {"units": "degrees_north"}},
        "SatAzimuth": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                       "attrs": {"units": "degrees"}},
        "SatZenith": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      "attrs": {"units": "degrees"}},
        "SensingTime_day": {"data": 6472 * np.ones(NUM_SCANLINES, dtype=np.uint16),
                            "attrs": {}},
        "SensingTime_msec": {"data": np.arange(37337532, 37338532, 100, dtype=np.uint32),
                             "attrs": {}},
        "SunAzimuth": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                       "attrs": {"units": "degrees"}},
        "SunZenith": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      "attrs": {"units": "degrees"}},
    },
    "Maps": {
        "Height": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                   "attrs": {"units": "m"}},
        "HeightStd": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      "attrs": {"units": "m"}},
    },
    "PWLR": {
        "E": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH, 10), dtype=np.float32),
              "attrs": {"emissivity_wavenumbers": np.array([699.3, 826.4,
                                                            925.9, 1075.2,
                                                            1204.8, 1315.7,
                                                            1724.1, 2000.0,
                                                            2325.5, 2702.7],
                                                           dtype=np.float32)}},
        "O": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              "attrs": {"long_name": "Ozone mixing ratio vertical profile",
                        "units": "kg/kg"}},
        "OC": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {}},
        "P": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              "attrs": {"long_name": "Atmospheric pressures at which the vertical profiles are given. "
                                     "Last value is the surface pressure",
                        "units": "hpa"}},
        "QE": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {}},
        "QO": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {}},
        "QP": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {}},
        "QT": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {}},
        "QTs": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                "attrs": {}},
        "QW": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {}},
        "T": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              "attrs": {"long_name": "Temperature vertical profile", "units": "K"}},
        "Ts": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {"long_name": "Surface skin temperature", "units": "K"}},
        "W": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              "attrs": {"long_name": "Water vapour mixing ratio vertical profile", "units": "kg/kg"}},
        "WC": {"data": np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               "attrs": {"long_name": "Water vapour total columnar amount", "units": "mm"}},
    }
}


FNAME_INFO = {"start_time": dt.datetime(2017, 9, 20, 10, 22, 17),
              "end_time": dt.datetime(2017, 9, 20, 10, 29, 12),
              "processing_time": dt.datetime(2017, 9, 20, 10, 35, 59),
              "processing_location": "kan",
              "long_platform_id": "metopb",
              "instrument": "iasi",
              "platform_id": "M01"}

FTYPE_INFO = {"file_reader": IASIL2HDF5,
              "file_patterns": ["{fname}.hdf"],
              "file_type": "iasi_l2_hdf5"}


@pytest.fixture(scope="module")
def test_data(tmp_path_factory):
    """Save the test to the indicated directory."""
    import h5py
    test_file = os.path.join(tmp_path_factory.mktemp("data"), FNAME)
    with h5py.File(test_file, "w") as fid:
        # Create groups
        for grp in TEST_DATA:
            fid.create_group(grp)
            # Write datasets
            for dset in TEST_DATA[grp]:
                fid[grp][dset] = TEST_DATA[grp][dset]["data"]
                # Write dataset attributes
                for attr in TEST_DATA[grp][dset]["attrs"]:
                    fid[grp][dset].attrs[attr] = \
                        TEST_DATA[grp][dset]["attrs"][attr]
    return test_file


@pytest.fixture
def iasi_filehandler(test_data):
    """Create a filehandler."""
    return IASIL2HDF5(test_data, FNAME_INFO, FTYPE_INFO)


def test_scene(test_data):
    """Test scene creation."""
    from satpy import Scene
    scn = Scene(reader="iasi_l2", filenames=[test_data])
    assert scn.start_time is not None
    assert scn.end_time is not None
    assert scn.sensor_names
    assert "iasi" in scn.sensor_names


def test_scene_load_available_datasets(test_data):
    """Test that all datasets are available."""
    from satpy import Scene
    scn = Scene(reader="iasi_l2", filenames=[test_data])
    scn.load(scn.available_dataset_names())


def test_scene_load_pressure(test_data):
    """Test loading pressure data."""
    from satpy import Scene
    scn = Scene(reader="iasi_l2", filenames=[test_data])
    scn.load(["pressure"])
    pres = scn["pressure"].compute()
    check_pressure(pres, scn.attrs)


def test_scene_load_emissivity(test_data):
    """Test loading emissivity data."""
    from satpy import Scene
    scn = Scene(reader="iasi_l2", filenames=[test_data])
    scn.load(["emissivity"])
    emis = scn["emissivity"].compute()
    check_emissivity(emis)


def test_scene_load_sensing_times(test_data):
    """Test loading sensing times."""
    from satpy import Scene
    scn = Scene(reader="iasi_l2", filenames=[test_data])
    scn.load(["sensing_time"])
    times = scn["sensing_time"].compute()
    check_sensing_times(times)


def test_init(test_data, iasi_filehandler):
    """Test reader initialization."""
    assert iasi_filehandler.filename == test_data
    assert iasi_filehandler.finfo == FNAME_INFO
    assert iasi_filehandler.lons is None
    assert iasi_filehandler.lats is None
    assert iasi_filehandler.mda["platform_name"] == "Metop-B"
    assert iasi_filehandler.mda["sensor"] == "iasi"


def test_time_properties(iasi_filehandler):
    """Test time properties."""
    import datetime as dt
    assert isinstance(iasi_filehandler.start_time, dt.datetime)
    assert isinstance(iasi_filehandler.end_time, dt.datetime)


def test_get_dataset(iasi_filehandler):
    """Test get_dataset() for different datasets."""
    from satpy.tests.utils import make_dataid
    info = {"eggs": "spam"}
    key = make_dataid(name="pressure")
    data = iasi_filehandler.get_dataset(key, info).compute()
    check_pressure(data)
    assert "eggs" in data.attrs
    assert data.attrs["eggs"] == "spam"
    key = make_dataid(name="emissivity")
    data = iasi_filehandler.get_dataset(key, info).compute()
    check_emissivity(data)
    key = make_dataid(name="sensing_time")
    data = iasi_filehandler.get_dataset(key, info).compute()
    assert data.shape == (NUM_SCANLINES, SCAN_WIDTH)


def check_pressure(pres, attrs=None):
    """Test reading pressure dataset.

    Helper function.
    """
    assert np.all(pres == 0.0)
    assert pres.x.size == SCAN_WIDTH
    assert pres.y.size == NUM_SCANLINES
    assert pres.level.size == NUM_LEVELS
    if attrs:
        assert pres.attrs["start_time"] == attrs["start_time"]
        assert pres.attrs["end_time"] == attrs["end_time"]
    assert "long_name" in pres.attrs
    assert "units" in pres.attrs


def check_emissivity(emis):
    """Test reading emissivity dataset.

    Helper function.
    """
    assert np.all(emis == 0.0)
    assert emis.x.size == SCAN_WIDTH
    assert emis.y.size == NUM_SCANLINES
    assert "emissivity_wavenumbers" in emis.attrs


def check_sensing_times(times):
    """Test reading sensing times.

    Helper function.
    """
    # Times should be equal in blocks of four, but not beyond, so
    # there should be SCAN_WIDTH/4 different values
    for i in range(int(SCAN_WIDTH / 4)):
        assert np.unique(times[0, i * 4:i * 4 + 4]).size == 1
    assert np.unique(times[0, :]).size == SCAN_WIDTH / 4


@pytest.mark.parametrize(("dset", "dtype", "units"), [
    ("amsu_instrument_flags", np.uint8, None),
    ("iasi_instrument_flags", np.uint8, None),
    ("mhs_instrument_flags", np.uint8, None),
    ("observation_minus_calculation", np.float32, "K"),
    ("surface_elevation", np.float32, "m"),
    ("surface_elevation_std", np.float32, "m")
    ])
def test_get_info_and_maps(iasi_filehandler, dset, dtype, units):
    """Test datasets in INFO and Maps groups are read."""
    from satpy.tests.utils import make_dataid
    info = {"eggs": "spam"}
    key = make_dataid(name=dset)
    data = iasi_filehandler.get_dataset(key, info).compute()
    assert data.shape == (NUM_SCANLINES, SCAN_WIDTH)
    assert data.dtype == dtype
    if units:
        assert data.attrs["units"] == units
    assert data.attrs["platform_name"] == "Metop-B"


def test_read_dataset(test_data):
    """Test read_dataset() function."""
    import h5py

    from satpy.readers.iasi_l2 import read_dataset
    from satpy.tests.utils import make_dataid
    with h5py.File(test_data, "r") as fid:
        key = make_dataid(name="pressure")
        data = read_dataset(fid, key).compute()
        check_pressure(data)
        key = make_dataid(name="emissivity")
        data = read_dataset(fid, key).compute()
        check_emissivity(data)
        # This dataset doesn't have any attributes
        key = make_dataid(name="ozone_total_column")
        data = read_dataset(fid, key).compute()
        assert len(data.attrs) == 0


def test_read_geo(test_data):
    """Test read_geo() function."""
    import h5py

    from satpy.readers.iasi_l2 import read_geo
    from satpy.tests.utils import make_dataid
    with h5py.File(test_data, "r") as fid:
        key = make_dataid(name="sensing_time")
        data = read_geo(fid, key).compute()
        assert data.shape == (NUM_SCANLINES, SCAN_WIDTH)
        key = make_dataid(name="latitude")
        data = read_geo(fid, key).compute()
        assert data.shape == (NUM_SCANLINES, SCAN_WIDTH)


def test_form_datetimes():
    """Test _form_datetimes() function."""
    from satpy.readers.iasi_l2 import _form_datetimes
    days = TEST_DATA["L1C"]["SensingTime_day"]["data"]
    msecs = TEST_DATA["L1C"]["SensingTime_msec"]["data"]
    times = _form_datetimes(days, msecs)
    check_sensing_times(times)


@pytest.fixture
def fake_iasi_l2_cdr_nc_dataset():
    """Create minimally fake IASI L2 CDR NC dataset."""
    shp = (3, 4, 5)
    fv = -999
    dims = ("scan_lines", "pixels", "vertical_levels")
    coords2 = "latitude longitude"
    coords3 = "latitude longitude pressure_levels"
    lons = xr.DataArray(
            np.array([[0, 0, 0, 0], [1, 1, 1, 1],
                      [2, 2, 2, 2]], dtype="float32"),
            dims=dims[:2],
            attrs={"coordinates": coords2,
                   "standard_name": "longitude"})
    lats = xr.DataArray(
            np.array([[3, 3, 3, 3], [2, 2, 2, 2],
                      [1, 1, 1, 1]], dtype="float32"),
            dims=dims[:2],
            attrs={"coordinates": coords2,
                   "standard_name": "latitude"})
    pres = xr.DataArray(
            np.linspace(0, 1050, math.prod(shp), dtype="float32").reshape(shp),
            dims=dims,
            attrs={"coordinates": coords3})

    temps = np.linspace(100, 400, math.prod(shp), dtype="float32").reshape(shp)
    temps[0, 0, 0] = fv
    temp = xr.DataArray(
            temps, dims=dims,
            attrs={"coordinates": coords3, "_FillValue": fv, "units": "K"})

    iasibad = xr.DataArray(
            np.zeros(shp[:2], dtype="uint8"),
            dims=dims[:2],
            attrs={"coordinates": coords2,
                   "standard_name": "flag_information_IASI_L1c"})
    iasibad[0, 0] = 1

    cf = xr.DataArray(
            np.zeros(shp[:2], dtype="uint8"),
            dims=dims[:2],
            attrs={"coordinates": coords2,
                   "standard_name": "cloud_area_fraction",
                   "_FillValue": 255,
                   "valid_min": 0,
                   "valid_max": 100})

    return xr.Dataset(
            {"T": temp, "FLG_IASIBAD": iasibad, "CloudFraction": cf},
            coords={
                "longitude": lons,
                "latitude": lats,
                "pressure_levels": pres})


@pytest.fixture
def fake_iasi_l2_cdr_nc_file(fake_iasi_l2_cdr_nc_dataset, tmp_path):
    """Write a NetCDF file with minimal fake IASI L2 CDR NC data."""
    fn = ("W_XX-EUMETSAT-Darmstadt,HYPERSPECT+SOUNDING,METOPA+PW3+"
          "IASI_C_EUMP_19210624090000Z_19210623090100Z_eps_r_l2_0101.nc")
    of = tmp_path / fn
    fake_iasi_l2_cdr_nc_dataset.to_netcdf(of)
    return os.fspath(of)


def test_iasi_l2_cdr_nc(fake_iasi_l2_cdr_nc_file):
    """Test the IASI L2 CDR NC reader."""
    from satpy import Scene
    sc = Scene(filenames=[fake_iasi_l2_cdr_nc_file], reader=["iasi_l2_cdr_nc"])
    sc.load(["T", "FLG_IASIBAD", "CloudFraction"])
    assert sc["T"].dims == ("y", "x", "vertical_levels")
    assert sc["T"].shape == (3, 4, 5)
    assert sc["T"].attrs["area"].shape == (3, 4)
    (lons, lats) = sc["T"].attrs["area"].get_lonlats()
    np.testing.assert_array_equal(
            lons,
            np.array([[0, 0, 0, 0], [1, 1, 1, 1],
                      [2, 2, 2, 2]]))
    assert np.isnan(sc["T"][0, 0, 0])
    assert sc["FLG_IASIBAD"][0, 0] == 1
    assert sc["CloudFraction"].dtype == np.dtype("uint8")
    assert sc["T"].attrs["units"] == "K"
