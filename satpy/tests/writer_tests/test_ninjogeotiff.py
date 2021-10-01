#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021- Satpy developers
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
"""Tests for writing GeoTIFF files with NinJoTIFF tags."""

import datetime
import math
import unittest.mock

import dask.array as da
import pytest
import xarray as xr


@pytest.fixture(scope="module")
def test_area_small_eqc_sphere():
    """Create 100x200 test equirectangular area centered on (40, -30), spherical geoid."""
    from pyresample import create_area_def
    shp = (100, 200)
    test_area = create_area_def(
        "test-area",
        {"proj": "eqc", "lat_ts": 0, "lat_0": 0, "lon_0": 0,
         "x_0": 0, "y_0": 0, "ellps": "sphere", "units": "m",
         "no_defs": None, "type": "crs"},
        units="m",
        shape=shp,
        resolution=1000,
        center=(40, -30))
    return test_area


@pytest.fixture(scope="module")
def test_area_large_eqc_wgs84():
    """Create 1000x2000 test equirectangular area centered on (50, 90), wgs84."""
    from pyresample import create_area_def
    shp = (1000, 2000)
    test_area = create_area_def(
        "test-area",
        {"proj": "eqc", "lat_ts": 0, "lat_0": 0, "lon_0": 0,
         "x_0": 0, "y_0": 0, "ellps": "wgs84", "units": "m",
         "no_defs": None, "type": "crs"},
        units="m",
        shape=shp,
        resolution=1000,
        center=(50, 90))
    return test_area


@pytest.fixture(scope="module")
def test_area_small_stereographic_wgs84():
    """Create a 200x100 test stereographic area centered on the north pole, wgs84."""
    from pyresample import create_area_def
    shp = (200, 100)
    test_area = create_area_def(
        "test-area",
        {"proj": "stere", "lat_0": 75.0, "lon_0": 0.0, "lat_ts": 60.0,
            "ellps": "WGS84", "units": "m", "type": "crs"},
        units="m",
        shape=shp,
        resolution=1000,
        center=(90, 0))
    return test_area


@pytest.fixture(scope="module")
def test_dataset_small_mid_atlantic_L(test_area_small_eqc_sphere):
    """Get a small testdataset in mode L, over Atlantic."""
    arr = xr.DataArray(
        da.zeros(test_area_small_eqc_sphere.shape, chunks=50),
        dims=("y", "x", "bands"),
        coords={"bands": ["L"]},
        attrs={
            "name": "test-small-mid-atlantic",
            "start_time": datetime.datetime(1985, 8, 13, 15, 0),
            "area": test_area_small_eqc_sphere})
    return arr


@pytest.fixture(scope="module")
def test_dataset_large_asia_RGB(test_area_large_eqc_wgs84):
    """Get a large-ish test dataset in mode RGB, over Asia."""
    arr = xr.DataArray(
        da.zeros(test_area_large_eqc_wgs84.shape + (3,), chunks=50),
        dims=("y", "x", "bands"),
        coords={"bands": ["R", "G", "B"]},
        attrs={
            "name": "test-large-asia",
            "start_time": datetime.datetime(2015, 10, 21, 22, 25, 0),
            "area": test_area_large_eqc_wgs84,
            "mode": "RGB"})
    return arr


@pytest.fixture(scope="module")
def test_dataset_small_arctic_P(test_area_small_stereographic_wgs84):
    """Get a small-ish test dataset in mode P, over Arctic."""
    arr = xr.DataArray(
        da.zeros(test_area_small_stereographic_wgs84.shape + (1,), chunks=50),
        dims=("y", "x", "bands"),
        coords={"bands": ["P"]},
        attrs={
            "name": "test-small-arctic",
            "start_time": datetime.datetime(2027, 8, 2, 10, 20),
            "area": test_area_small_stereographic_wgs84,
            "mode": "P"})
    return arr


@pytest.fixture(scope="module")
def fake_datasets(test_dataset_small_mid_atlantic_L, test_dataset_large_asia_RGB, test_dataset_small_arctic_P):
    """Create fake datasets for testing writing routines."""
    return [test_dataset_small_mid_atlantic_L, test_dataset_large_asia_RGB,
            test_dataset_small_arctic_P]


@pytest.fixture(scope="module")
def ntg1(test_dataset_small_mid_atlantic_L):
    """Create instance of NinJoTagGenerator class."""
    from satpy.writers.ninjogeotiff import NinJoTagGenerator
    return NinJoTagGenerator(
            test_dataset_small_mid_atlantic_L,
            {"ChannelID": 900015,
             "DataSource": "FIXME",
             "DataType": "GPRN",
             "PhysicUnit": "C",
             "SatelliteNameID": 6400014})


@pytest.fixture(scope="module")
def ntg2(test_dataset_large_asia_RGB):
    """Create instance of NinJoTagGenerator class."""
    from satpy.writers.ninjogeotiff import NinJoTagGenerator
    return NinJoTagGenerator(
            test_dataset_large_asia_RGB,
            {"ChannelID": 1000015,
             "DataSource": "FIXME",
             "DataType": "GORN",
             "SatelliteNameID": 6400014})


@pytest.fixture(scope="module")
def ntg3(test_dataset_small_arctic_P):
    """Create instance of NinJoTagGenerator class."""
    from satpy.writers.ninjogeotiff import NinJoTagGenerator
    return NinJoTagGenerator(
            test_dataset_small_arctic_P,
            {"ChannelID": 800012,
             "DataSource": "FIXME",
             "DataType": "PPRN",
             "SatelliteNameID": 6500014})


exp_tags = {"AxisIntercept": -88,
            "CentralMeridian": 0.0,
            "ChannelID": 900015,
            "ColorDepth": 24,
            "CreationDateID": 1632820093,
            "DataSource": "FIXME",
            "DataType": "GPRN",
            "DateID": 1623581777,
            "EarthRadiusLarge": 6378137.0,
            "EarthRadiusSmall": 6356752.5,
            "FileName": "papapath.tif",
            "Gradient": 0.5,
            "HeaderVersion": 2,
            "IsAtmosphereCorrected": 0,
            "IsBlackLineCorrection": 0,
            "IsCalibrated": 1,
            "IsNormalized": 0,
            "Magic": "NINJO",
            "MaxGrayValue": 255,
            "MeridianEast": 45.0,
            "MeridianWest": -135.0,
            "MinGrayValue": 0,
            "PhysicUnit": "C",
            "PhysicValue": "unknown",
            "Projection": "NPOL",
            "ReferenceLatitude1": 60.0,
            "ReferenceLatitude2": 0.0,
            "SatelliteNameID": 6400014,
            "TransparentPixel": 0,
            "XMaximum": 200,
            "XMinimum": 1,
            "YMaximum": 100,
            "YMinimum": 1}


def test_ninjogeotiff(fake_datasets):
    """Test that it writes a GeoTIFF with the appropriate NinJo-tags."""
    from satpy.writers.ninjogeotiff import NinJoGeoTIFFWriter
    w = NinJoGeoTIFFWriter()
    with unittest.mock.patch("satpy.writers.geotiff.GeoTIFFWriter.save_dataset") as swggs:
        w.save_dataset(
                fake_datasets[0],
                ninjo_tags=dict(
                    PhysicUnit="C",
                    SatelliteNameID=6400014,
                    ChannelID=900015,
                    DataType="GPRN",
                    DataSource="FIXME"))
        swggs.assert_called_with(
                fake_datasets[0],
                tags={f"ninjo_{k:s}": v for (k, v) in exp_tags.items()})


def test_calc_tags(fake_datasets):
    """Test calculating all tags from dataset."""
    from satpy.writers.ninjogeotiff import calc_tags_from_dataset
    ds = fake_datasets[0]
    tags = calc_tags_from_dataset(
            ds,
            {"ChannelID": 900015,
             "DataSource": "FIXME",
             "DataType": "GPRN",
             "PhysicUnit": "C",
             "SatelliteNameID": 6400014})
    assert tags == exp_tags


def test_calc_single_tag_by_name(ntg1):
    """Test calculating single tag from dataset."""
    assert ntg1.get_tag("Magic") == "NINJO"
    assert ntg1.get_tag("DataType") == "GPRN"
    assert ntg1.get_tag("IsCalibrated") == 1
    with pytest.raises(ValueError):
        ntg1.get_tag("invalid")


def test_get_axis_intercept(ntg1):
    """Test calculating the axis intercept."""
    intercept = ntg1.get_axis_intercept()
    assert isinstance(intercept, float)
    assert math.isclose(intercept, -88.0)


def test_get_central_meridian(ntg1):
    """Test calculating the central meridian."""
    cm = ntg1.get_central_meridian()
    assert isinstance(cm, float)
    assert math.isclose(cm, 0.0)


def test_get_color_depth(ntg1):
    """Test extracting the color depth."""
    cd = ntg1.get_color_depth()
    assert isinstance(cd, int)
    assert cd == 24


def test_get_creation_date_id(ntg1):
    """Test getting the creation date ID."""
    cdid = ntg1.get_creation_date_id()
    assert isinstance(cdid, int)
    assert cdid == 1632820093


def test_get_date_id(ntg1):
    """Test getting the date ID."""
    did = ntg1.get_date_id()
    assert isinstance(did, int)
    assert did == 1623581777


def test_get_earth_radius_large(ntg1):
    """Test getting the Earth semi-major axis."""
    erl = ntg1.get_earth_radius_large()
    assert isinstance(erl, float)
    assert math.isclose(erl, 6378137.0)


def test_get_earth_radius_small(ntg1):
    """Test getting the Earth semi-minor axis."""
    ers = ntg1.get_earth_radius_small()
    assert isinstance(ers, float)
    assert math.isclose(ers, 6356752.5)


def test_get_filename(ntg1):
    """Test getting the filename."""
    assert ntg1.get_filename() == "papapath.tif"


def test_get_gradient(ntg1):
    """Test getting the gradient."""
    grad = ntg1.get_gradient()
    assert isinstance(grad, float)
    assert math.isclose(grad, 0.5)


def test_get_atmosphere_corrected(ntg1):
    """Test whether the atmosphere is corrected."""
    corr = ntg1.get_atmosphere_corrected()
    assert isinstance(corr, int)  # on purpose not a boolean
    assert corr == 0


def test_get_black_line_corrected(ntg1):
    """Test whether black line correction applied."""
    blc = ntg1.get_black_line_corrected()
    assert isinstance(blc, int)  # on purpose not a boolean
    assert blc == 0


def test_is_calibrated(ntg1):
    """Test whether calibrated."""
    calib = ntg1.get_is_calibrated()
    assert isinstance(calib, int)
    assert calib == 1


def test_is_normalized(ntg1):
    """Test whether normalized."""
    is_norm = ntg1.get_is_normalized()
    assert isinstance(is_norm, int)
    assert is_norm == 0


def test_get_max_gray(ntg1):
    """Test getting max gray value."""
    mg = ntg1.get_max_gray_value()
    assert isinstance(mg, int)
    assert mg == 255


def test_get_meridian_east(ntg1):
    """Test getting east meridian."""
    me = ntg1.get_meridian_east()
    assert isinstance(me, float)
    assert math.isclose(me, 45.0)


def test_get_meridian_west(ntg1):
    """Test getting west meridian."""
    mw = ntg1.get_meridian_west()
    assert isinstance(mw, float)
    assert math.isclose(mw, -135.0)


def test_get_min_gray_value(ntg1):
    """Test getting min gray value."""
    mg = ntg1.get_min_gray_value()
    assert isinstance(mg, int)
    assert mg == 0


def test_get_projection(ntg1):
    """Test getting projection string."""
    assert ntg1.get_projection() == "NPOL"


def test_get_ref_lat_1(ntg1):
    """Test getting reference latitude 1."""
    rl1 = ntg1.get_ref_lat_1()
    assert isinstance(rl1, float)
    assert math.isclose(rl1, 60.0)


def test_get_ref_lat_2(ntg1):
    """Test getting reference latitude 2."""
    rl2 = ntg1.get_ref_lat_2()
    assert isinstance(rl2, float)
    assert math.isclose(rl2, 0.0)


def test_get_transparent_pixel(ntg1):
    """Test getting fill value."""
    tp = ntg1.get_transparent_pixel()
    assert isinstance(tp, int)
    assert tp == 0


def test_get_xmax(ntg1):
    """Test getting maximum x."""
    xmax = ntg1.get_xmaximum()
    assert isinstance(xmax, int)
    assert xmax == 200


def test_get_ymax(ntg1):
    """Test getting maximum y."""
    ymax = ntg1.get_ymaximum()
    assert isinstance(ymax, int)
    assert ymax == 100
