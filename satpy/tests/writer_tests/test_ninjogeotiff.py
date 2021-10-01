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
def fake_datasets():
    """Create fake datasets for testing writing routines."""
    from pyresample import create_area_def
    shp = (100, 200)
    fake_area = create_area_def(
        "test-area",
        {"proj": "eqc", "lat_ts": 0, "lat_0": 0, "lon_0": 0,
         "x_0": 0, "y_0": 0, "ellps": "sphere", "units": "m",
         "no_defs": None, "type": "crs"},
        units="m",
        shape=shp,
        resolution=1000,
        center=(0, 0))
    return [xr.DataArray(
        da.zeros(shp, chunks=50),
        dims=("y", "x"),
        attrs={"name": "test",
               "start_time": datetime.datetime(1985, 8, 13, 15, 0),
               "area": fake_area})]


@pytest.fixture(scope="module")
def ntg(fake_datasets):
    """Create instance of NinJoTagGenerator class."""
    from satpy.writers.ninjogeotiff import NinJoTagGenerator
    return NinJoTagGenerator(
            fake_datasets[0],
            {"ChannelID": 900015,
             "DataSource": "FIXME",
             "DataType": "GPRN",
             "PhysicUnit": "C",
             "SatelliteNameID": 6400014})


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


def test_calc_single_tag_by_name(ntg):
    """Test calculating single tag from dataset."""
    assert ntg.get_tag("Magic") == "NINJO"
    assert ntg.get_tag("DataType") == "GPRN"
    assert ntg.get_tag("IsCalibrated") == 1
    with pytest.raises(ValueError):
        ntg.get_tag("invalid")


def test_get_axis_intercept(ntg):
    """Test calculating the axis intercept."""
    intercept = ntg.get_axis_intercept()
    assert isinstance(intercept, float)
    assert math.isclose(intercept, -88.0)


def test_get_central_meridian(ntg):
    """Test calculating the central meridian."""
    cm = ntg.get_central_meridian()
    assert isinstance(cm, float)
    assert math.isclose(cm, 0.0)


def test_get_color_depth(ntg):
    """Test extracting the color depth."""
    cd = ntg.get_color_depth()
    assert isinstance(cd, int)
    assert cd == 24


def test_get_creation_date_id(ntg):
    """Test getting the creation date ID."""
    cdid = ntg.get_creation_date_id()
    assert isinstance(cdid, int)
    assert cdid == 1632820093


def test_get_date_id(ntg):
    """Test getting the date ID."""
    did = ntg.get_date_id()
    assert isinstance(did, int)
    assert did == 1623581777


def test_get_earth_radius_large(ntg):
    """Test getting the Earth semi-major axis."""
    erl = ntg.get_earth_radius_large()
    assert isinstance(erl, float)
    assert math.isclose(erl, 6378137.0)


def test_get_earth_radius_small(ntg):
    """Test getting the Earth semi-minor axis."""
    ers = ntg.get_earth_radius_small()
    assert isinstance(ers, float)
    assert math.isclose(ers, 6356752.5)


def test_get_filename(ntg):
    """Test getting the filename."""
    assert ntg.get_filename() == "papapath.tif"


def test_get_gradient(ntg):
    """Test getting the gradient."""
    grad = ntg.get_gradient()
    assert isinstance(grad, float)
    assert math.isclose(grad, 0.5)


def test_get_atmosphere_corrected(ntg):
    """Test whether the atmosphere is corrected."""
    corr = ntg.get_atmosphere_corrected()
    assert isinstance(corr, int)  # on purpose not a boolean
    assert corr == 0


def test_get_black_line_corrected(ntg):
    """Test whether black line correction applied."""
    blc = ntg.get_black_line_corrected()
    assert isinstance(blc, int)  # on purpose not a boolean
    assert blc == 0


def test_is_calibrated(ntg):
    """Test whether calibrated."""
    calib = ntg.get_is_calibrated()
    assert isinstance(calib, int)
    assert calib == 1


def test_is_normalized(ntg):
    """Test whether normalized."""
    is_norm = ntg.get_is_normalized()
    assert isinstance(is_norm, int)
    assert is_norm == 0


def test_get_max_gray(ntg):
    """Test getting max gray value."""
    mg = ntg.get_max_gray_value()
    assert isinstance(mg, int)
    assert mg == 255


def test_get_meridian_east(ntg):
    """Test getting east meridian."""
    me = ntg.get_meridian_east()
    assert isinstance(me, float)
    assert math.isclose(me, 45.0)


def test_get_meridian_west(ntg):
    """Test getting west meridian."""
    mw = ntg.get_meridian_west()
    assert isinstance(mw, float)
    assert math.isclose(mw, -135.0)


def test_get_min_gray_value(ntg):
    """Test getting min gray value."""
    mg = ntg.get_min_gray_value()
    assert isinstance(mg, int)
    assert mg == 0


def test_get_projection(ntg):
    """Test getting projection string."""
    assert ntg.get_projection() == "NPOL"


def test_get_ref_lat_1(ntg):
    """Test getting reference latitude 1."""
    rl1 = ntg.get_ref_lat_1()
    assert isinstance(rl1, float)
    assert math.isclose(rl1, 60.0)


def test_get_ref_lat_2(ntg):
    """Test getting reference latitude 2."""
    rl2 = ntg.get_ref_lat_2()
    assert isinstance(rl2, float)
    assert math.isclose(rl2, 0.0)


def test_get_transparent_pixel(ntg):
    """Test getting fill value."""
    tp = ntg.get_transparent_pixel()
    assert isinstance(tp, int)
    assert tp == 0


def test_get_xmax(ntg):
    """Test getting maximum x."""
    xmax = ntg.get_xmaximum()
    assert isinstance(xmax, int)
    assert xmax == 200


def test_get_ymax(ntg):
    """Test getting maximum y."""
    ymax = ntg.get_ymaximum()
    assert isinstance(ymax, int)
    assert ymax == 100
