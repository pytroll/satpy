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
import os

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from pyresample import create_area_def
from satpy.writers import get_enhanced_image


def _get_fake_da(lo, hi, shp, dtype="f4"):
    """Generate dask array with synthetic data.

    This is more or less a 2d linspace: it'll return a 2-d dask array of shape
    shape, lowest value is lo, highest value is hi.
    """
    return da.arange(lo, hi, (hi-lo)/math.prod(shp), chunks=50, dtype=dtype).reshape(shp)


@pytest.fixture(scope="module")
def test_area_tiny_eqc_sphere():
    """Create 10x00 test equirectangular area centered on (40, -30), spherical geoid."""
    shp = (10, 20)
    test_area = create_area_def(
        "test-area-eqc-sphere",
        {"proj": "eqc", "lat_ts": 0., "lat_0": 0., "lon_0": 0.,
         "x_0": 0., "y_0": 0., "ellps": "sphere", "units": "m",
         "no_defs": None, "type": "crs"},
        units="m",
        shape=shp,
        resolution=1000,
        center=(-3330000.0, 4440000.0))
    return test_area


@pytest.fixture(scope="module")
def test_area_small_eqc_wgs84():
    """Create 50x100 test equirectangular area centered on (50, 90), wgs84."""
    shp = (50, 100)
    test_area = create_area_def(
            "test-area-eqc-wgs84",
            {"proj": "eqc", "lat_0": 2.5, "lon_0": 1., "ellps": "WGS84"},
            units="m",
            shape=shp,
            resolution=1000,
            center=(10000000.0, 6000000.0))
    return test_area


@pytest.fixture(scope="module")
def test_area_tiny_stereographic_wgs84():
    """Create a 20x10 test stereographic area centered on the north pole, wgs84."""
    shp = (20, 10)
    test_area = create_area_def(
        "test-area-north-stereo",
        {"proj": "stere", "lat_0": 75.0, "lon_0": 2.0, "lat_ts": 60.0,
            "ellps": "WGS84", "units": "m", "type": "crs"},
        units="m",
        shape=shp,
        resolution=1000,
        center=(0.0, 1500000.0))
    return test_area


@pytest.fixture(scope="module")
def test_image_small_mid_atlantic_L(test_area_tiny_eqc_sphere):
    """Get a small test image in mode L, over Atlantic."""
    arr = xr.DataArray(
        _get_fake_da(-80, 40, test_area_tiny_eqc_sphere.shape + (1,)),
        dims=("y", "x", "bands"),
        attrs={
            "name": "test-small-mid-atlantic",
            "start_time": datetime.datetime(1985, 8, 13, 15, 0),
            "area": test_area_tiny_eqc_sphere})
    return get_enhanced_image(arr)


@pytest.fixture(scope="module")
def test_image_large_asia_RGB(test_area_small_eqc_wgs84):
    """Get a large-ish test image in mode RGB, over Asia."""
    arr = xr.DataArray(
        _get_fake_da(0, 255, test_area_small_eqc_wgs84.shape + (3,), "uint8"),
        dims=("y", "x", "bands"),
        coords={"bands": ["R", "G", "B"]},
        attrs={
            "name": "test-large-asia",
            "start_time": datetime.datetime(2015, 10, 21, 22, 25, 0),
            "area": test_area_small_eqc_wgs84,
            "mode": "RGB"})
    return get_enhanced_image(arr)


@pytest.fixture(scope="module")
def test_image_small_arctic_P(test_area_tiny_stereographic_wgs84):
    """Get a small-ish test image in mode P, over Arctic."""
    arr = xr.DataArray(
        _get_fake_da(0, 10, test_area_tiny_stereographic_wgs84.shape + (1,), "uint8"),
        dims=("y", "x", "bands"),
        coords={"bands": ["P"]},
        attrs={
            "name": "test-small-arctic",
            "start_time": datetime.datetime(2027, 8, 2, 10, 20),
            "area": test_area_tiny_stereographic_wgs84,
            "mode": "P"})
    return get_enhanced_image(arr)


@pytest.fixture(scope="module")
def fake_images(test_image_small_mid_atlantic_L, test_image_large_asia_RGB,
                test_image_small_arctic_P):
    """Create fake datasets for testing writing routines."""
    return [test_image_small_mid_atlantic_L, test_image_large_asia_RGB,
            test_image_small_arctic_P]


@pytest.fixture(scope="module")
def ntg1(test_image_small_mid_atlantic_L):
    """Create instance of NinJoTagGenerator class."""
    from satpy.writers.ninjogeotiff import NinJoTagGenerator
    return NinJoTagGenerator(
            test_image_small_mid_atlantic_L,
            255,
            "quinoa.tif",
            {"ChannelID": 900015,
             "DataType": "GORN",
             "PhysicUnit": "C",
             "PhysicValue": "Temperature",
             "SatelliteNameID": 6400014,
             "DataSource": "dowsing rod"})


@pytest.fixture(scope="module")
def ntg2(test_image_large_asia_RGB):
    """Create instance of NinJoTagGenerator class."""
    from satpy.writers.ninjogeotiff import NinJoTagGenerator
    return NinJoTagGenerator(
            test_image_large_asia_RGB,
            0,
            "seitan.tif",
            {"ChannelID": 1000015,
             "DataType": "GORN",
             "PhysicUnit": "N/A",
             "PhysicValue": "N/A",
             "SatelliteNameID": 6400014})


@pytest.fixture(scope="module")
def ntg3(test_image_small_arctic_P):
    """Create instance of NinJoTagGenerator class."""
    from satpy.writers.ninjogeotiff import NinJoTagGenerator
    return NinJoTagGenerator(
            test_image_small_arctic_P,
            12,
            "spelt.tif",
            {"ChannelID": 800012,
             "DataType": "PPRN",
             "PhysicUnit": "N/A",
             "PhysicValue": "N/A",
             "SatelliteNameID": 6500014,
             "OverFlightTime": 42})


@pytest.fixture
def patch_datetime_now(monkeypatch):
    """Get a fake datetime.datetime.now()."""
    # Source: https://stackoverflow.com/a/20503374/974555, CC-BY-SA 4.0

    class mydatetime(datetime.datetime):
        """Drop-in replacement for datetime.datetime."""

        @classmethod
        def now(cls):
            """Drop-in replacement for datetime.datetime.now."""
            return datetime.datetime(2033, 5, 18, 5, 33, 20)

    monkeypatch.setattr(datetime, 'datetime', mydatetime)


def test_write_and_read_file(fake_images, tmp_path):
    """Test that it writes a GeoTIFF with the appropriate NinJo-tags."""
    import rasterio
    from satpy.writers.ninjogeotiff import NinJoGeoTIFFWriter
    fn = os.fspath(tmp_path / "test.tif")
    ngtw = NinJoGeoTIFFWriter()
    ngtw.save_dataset(
        fake_images[0].data,
        filename=fn,
        fill_value=0,
        PhysicUnit="C",
        PhysicValue="Temperature",
        SatelliteNameID=6400014,
        ChannelID=900015,
        DataType="GORN",
        DataSource="dowsing rod")
    src = rasterio.open(fn)
    tgs = src.tags()
    assert tgs["ninjo_FileName"] == fn
    assert tgs["ninjo_DataSource"] == "dowsing rod"


def test_get_all_tags(ntg1, ntg3):
    """Test getting all tags from dataset."""
    # test that passed, dynamic, and mandatory tags are all included, and
    # nothing more
    t1 = ntg1.get_all_tags()
    assert set(t1.keys()) == (
            ntg1.fixed_tags.keys() |
            ntg1.passed_tags |
            ntg1.dynamic_tags.keys() |
            {"DataSource"})
    # test that when extra tag is passed this is also included
    t3 = ntg3.get_all_tags()
    assert t3.keys() == (
            ntg3.fixed_tags.keys() |
            ntg3.passed_tags |
            ntg3.dynamic_tags.keys() |
            {"OverFlightTime"})
    assert t3["OverFlightTime"] == 42


def test_calc_single_tag_by_name(ntg1, ntg2, ntg3):
    """Test calculating single tag from dataset."""
    assert ntg1.get_tag("Magic") == "NINJO"
    assert ntg1.get_tag("DataType") == "GORN"
    assert ntg2.get_tag("DataType") == "GORN"
    assert ntg3.get_tag("DataType") == "PPRN"
    assert ntg1.get_tag("DataSource") == "dowsing rod"
    with pytest.raises(ValueError):
        ntg1.get_tag("invalid")
    with pytest.raises(ValueError):
        ntg1.get_tag("OriginalHeader")


def test_get_central_meridian(ntg1, ntg2, ntg3):
    """Test calculating the central meridian."""
    cm = ntg1.get_central_meridian()
    assert isinstance(cm, float)
    np.testing.assert_allclose(cm, 0.0)
    np.testing.assert_allclose(ntg2.get_central_meridian(), 1.0)
    np.testing.assert_allclose(ntg3.get_central_meridian(), 2.0)


def test_get_color_depth(ntg1, ntg2, ntg3):
    """Test extracting the color depth."""
    cd = ntg1.get_color_depth()
    assert isinstance(cd, int)
    assert cd == 8  # mode L
    assert ntg2.get_color_depth() == 24  # mode RGB
    assert ntg3.get_color_depth() == 8  # mode P


def test_get_creation_date_id(ntg1, ntg2, ntg3, patch_datetime_now):
    """Test getting the creation date ID.

    This is the time at which the file was created.

    This test believes it is run at 2033-5-18 05:33:20Z.
    """
    cdid = ntg1.get_creation_date_id()
    assert isinstance(cdid, int)
    assert cdid == 2000000000
    assert ntg2.get_creation_date_id() == 2000000000
    assert ntg3.get_creation_date_id() == 2000000000


def test_get_date_id(ntg1, ntg2, ntg3):
    """Test getting the date ID."""
    did = ntg1.get_date_id()
    assert isinstance(did, int)
    assert did == 492786000
    assert ntg2.get_date_id() == 1445459100
    assert ntg3.get_date_id() == 1817194800


def test_get_earth_radius_large(ntg1, ntg2, ntg3):
    """Test getting the Earth semi-major axis."""
    erl = ntg1.get_earth_radius_large()
    assert isinstance(erl, float)
    np.testing.assert_allclose(erl, 6370997.0)
    np.testing.assert_allclose(ntg2.get_earth_radius_large(), 6378137.0)
    np.testing.assert_allclose(ntg3.get_earth_radius_large(), 6378137.0)


def test_get_earth_radius_small(ntg1, ntg2, ntg3):
    """Test getting the Earth semi-minor axis."""
    ers = ntg1.get_earth_radius_small()
    assert isinstance(ers, float)
    np.testing.assert_allclose(ers, 6370997.0)
    np.testing.assert_allclose(ntg2.get_earth_radius_small(), 6356752.314245179)
    np.testing.assert_allclose(ntg3.get_earth_radius_small(), 6356752.314245179)


def test_get_filename(ntg1, ntg2, ntg3):
    """Test getting the filename."""
    assert ntg1.get_filename() == "quinoa.tif"
    assert ntg2.get_filename() == "seitan.tif"
    assert ntg3.get_filename() == "spelt.tif"


def test_get_min_gray_value_L(ntg1):
    """Test getting min gray value for mode L."""
    mg = ntg1.get_min_gray_value()
    assert isinstance(mg.compute().item(), int)
    assert mg.compute() == 0


def test_get_min_gray_value_RGB(ntg2):
    """Test getting min gray value for RGB.

    Note that min/max gray value is mandatory in NinJo even for RGBs?
    """
    assert ntg2.get_min_gray_value().compute().item() == 1  # fill value 0


def test_get_min_gray_value_P(ntg3):
    """Test getting min gray value for mode P."""
    assert ntg3.get_min_gray_value().compute().item() == 0


def test_get_max_gray_value_L(ntg1):
    """Test getting max gray value for mode L."""
    mg = ntg1.get_max_gray_value().compute().item()
    assert isinstance(mg, int)
    assert mg == 254  # fill value is 255


def test_get_max_gray_value_RGB(ntg2):
    """Test max gray value for RGB."""
    assert ntg2.get_max_gray_value() == 255


@pytest.mark.xfail(reason="Needs GeoTIFF P fixes, see GH#1844")
def test_get_max_gray_value_P(ntg3):
    """Test getting max gray value for mode P."""
    assert ntg3.get_max_gray_value().compute().item() == 10


@pytest.mark.xfail(reason="not easy, not needed, not implemented")
def test_get_meridian_east(ntg1, ntg2, ntg3):
    """Test getting east meridian."""
    np.testing.assert_allclose(ntg1.get_meridian_east(), -29.048101549452294)
    np.testing.assert_allclose(ntg2.get_meridian_east(), 180.0)
    np.testing.assert_allclose(ntg3.get_meridian_east(), 99.81468125314737)


@pytest.mark.xfail(reason="not easy, not needed, not implemented")
def test_get_meridian_west(ntg1, ntg2, ntg3):
    """Test getting west meridian."""
    np.testing.assert_allclose(ntg1.get_meridian_west(), -30.846745608241903)
    np.testing.assert_allclose(ntg2.get_meridian_east(), -180.0)
    np.testing.assert_allclose(ntg3.get_meridian_west(), 81.84837557075694)


def test_get_projection(ntg1, ntg2, ntg3):
    """Test getting projection string."""
    assert ntg1.get_projection() == "PLAT"
    assert ntg2.get_projection() == "PLAT"
    assert ntg3.get_projection() == "NPOL"


def test_get_ref_lat_1(ntg1, ntg2, ntg3):
    """Test getting reference latitude 1."""
    rl1 = ntg1.get_ref_lat_1()
    assert isinstance(rl1, float)
    np.testing.assert_allclose(rl1, 0.0)
    np.testing.assert_allclose(ntg2.get_ref_lat_1(), 2.5)
    np.testing.assert_allclose(ntg3.get_ref_lat_1(), 75)


@pytest.mark.xfail(reason="Not implemented, what is this?")
def test_get_ref_lat_2(ntg1, ntg2, ntg3):
    """Test getting reference latitude 2."""
    rl2 = ntg1.get_ref_lat_2()
    assert isinstance(rl2, float)
    np.testing.assert_allclose(rl2, 0.0)
    np.testing.assert_allclose(ntg2.get_ref_lat_2(), 0.0)
    np.testing.assert_allclose(ntg2.get_ref_lat_3(), 0.0)


def test_get_transparent_pixel(ntg1, ntg2, ntg3):
    """Test getting fill value."""
    tp = ntg1.get_transparent_pixel()
    assert isinstance(tp, int)
    assert tp == 255
    assert ntg2.get_transparent_pixel() == 0  # when not set ??
    assert ntg3.get_transparent_pixel() == 12


def test_get_xmax(ntg1, ntg2, ntg3):
    """Test getting maximum x."""
    xmax = ntg1.get_xmaximum()
    assert isinstance(xmax, int)
    assert xmax == 20
    assert ntg2.get_xmaximum() == 100
    assert ntg3.get_xmaximum() == 10


def test_get_ymax(ntg1, ntg2, ntg3):
    """Test getting maximum y."""
    ymax = ntg1.get_ymaximum()
    assert isinstance(ymax, int)
    assert ymax == 10
    assert ntg2.get_ymaximum() == 50
    assert ntg3.get_ymaximum() == 20
