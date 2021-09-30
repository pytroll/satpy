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
import unittest.mock

import dask.array as da
import pytest
import xarray as xr


@pytest.fixture
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


@pytest.fixture
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
