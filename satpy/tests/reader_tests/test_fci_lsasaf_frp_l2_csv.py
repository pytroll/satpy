#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Module for testing the satpy.readers.fci_lsasaf_frp_l2_csv module."""


from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from satpy.readers.fci_lsasaf_frp_l2_csv import COLUMN_MAP, FRPFileHandler

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


@pytest.fixture
def filename_info():
    """Return filename_info."""
    return {
        "satellite_name": "MTG",
        "start_time": datetime(2026, 7, 31, 12, 0),
        "facility_or_tool": "LSA",
        "coverage": "FD",
    }


@pytest.fixture
def filetype_info():
    """Return filetype_info."""
    return {}


@pytest.fixture
def sample_csv_file(tmp_path):
    """Create simple csv testdata, stored in temporary place."""
    csv_content = """FRP,LATITUDE,LONGITUDE,ABS_LINE,ABS_SAMP
10.5,50.1,8.6,1,1
20.0,50.2,8.7,2,3
30.0,50.3,8.8,4,5
"""
    fn = tmp_path / "LSA-509_MTG_MTFRPPIXEL-ListProduct_MTG-FD_202607311200.csv"
    fn.write_text(csv_content)
    return fn


@pytest.fixture
def reader(sample_csv_file, filename_info, filetype_info):
    """Test reader behavior."""
    return FRPFileHandler(str(sample_csv_file), filename_info, filetype_info)


def test_start_time(reader, filename_info):
    """Test parsing of start_time."""
    assert reader.start_time == filename_info["start_time"]


def test_end_time_is_10_minutes_after_start(reader, filename_info):
    """Test calculation of end_time."""
    assert reader.end_time == filename_info["start_time"] + timedelta(minutes=10)


@pytest.mark.parametrize("key", ["frp", "latitude", "longitude", "abs_line", "abs_samp"])
def test_contains_known_columns(reader, key):
    """Test column return."""
    assert key in reader


def test_contains_unknown_column(reader):
    """Test return of not existing column."""
    assert "not_a_dataset" not in reader


def test_getitem_returns_expected_column(reader):
    """Test column frp to be returned correct."""
    series = reader["frp"]
    assert series is not None


def test_getitem_uses_column_map(reader):
    """Test whether mapping via column_map works."""
    frp_series = reader["frp"]
    lat_series = reader["latitude"]

    assert frp_series.name == COLUMN_MAP["frp"]
    assert lat_series.name == COLUMN_MAP["latitude"]


def test_get_dataset_latitude_returns_dataarray(reader):
    """Test return of latitude dataarray."""
    dsid = {"name": "latitude"}
    dsinfo = {
        "units": "degrees_north",
        "standard_name": "active_fire_pixel_centre_latitude",
    }

    data = reader.get_dataset(dsid, dsinfo)

    assert isinstance(data, xr.DataArray)
    assert data.dims == ("y",)
    assert data.attrs["units"] == "degrees_north"
    assert data.attrs["standard_name"] == "active_fire_pixel_centre_latitude"
    assert data.attrs["satellite_name"] == "MTG"
    assert data.attrs["platform_name"] == "Meteosat-12"
    assert data.attrs["sensor"] == "fci"
    assert data.attrs["start_time"] == reader.start_time
    assert data.attrs["end_time"] == reader.end_time


def test_get_dataset_frp_adds_lat_lon_coords(monkeypatch, reader):
    """Test latitude an longitude on frp dataset."""
    def fake_get_array_on_fci_grid(data):
        return data
    #monkeypatched (5x6) instead of huge 11136x11136 px native fci grid
    monkeypatch.setattr(reader, "get_array_on_fci_grid", fake_get_array_on_fci_grid)

    dsid = {"name": "frp"}
    dsinfo = {
        "units": "MW",
        "standard_name": "fire_radiative_power",
        "resolution": 1000,
    }

    data = reader.get_dataset(dsid, dsinfo)

    assert isinstance(data, xr.DataArray)
    assert data.dims == ("y",)
    assert "longitude" in data.coords
    assert "latitude" in data.coords
    assert data.attrs["units"] == "MW"
    assert data.attrs["standard_name"] == "fire_radiative_power"
    assert data.attrs["resolution"] == 1000
    assert data.attrs["satellite_name"] == "MTG"
    assert data.attrs["platform_name"] == "Meteosat-12"
    assert data.attrs["sensor"] == "fci"


def test_get_array_on_fci_grid(monkeypatch, reader):
    """Test mapping on a 2D grid, here for a small testgrid."""
    monkeypatch.setattr(
        "satpy.readers.fci_lsasaf_frp_l2_csv.FRP_GRID_SHAPE",
        (5, 6),
    )

    data = xr.DataArray(
        np.array([10.5, 20.0, 30.0], dtype=np.float32),
        dims=("y",),
        attrs={
            "satellite_name": "MTG",
            "platform_name": "Meteosat-12",
            "sensor": "fci",
        },
    )

    gridded = reader.get_array_on_fci_grid(data)

    assert isinstance(gridded, xr.DataArray)
    assert gridded.dims == ("y", "x")
    assert gridded.shape == (5, 6)

    assert np.isclose(gridded.values[0, 0], 10.5)
    assert np.isclose(gridded.values[1, 2], 20.0)
    assert np.isclose(gridded.values[3, 4], 30.0)

    assert np.isnan(gridded.values[0, 1])
    assert gridded.attrs["satellite_name"] == "MTG"
    assert gridded.attrs["platform_name"] == "Meteosat-12"


def test_get_area_def(monkeypatch, reader):
    """Test area definition."""
    called = {}

    def fake_get_area_def(name):
        called["name"] = name
        return "dummy_area"

    monkeypatch.setattr(
        "satpy.readers.fci_lsasaf_frp_l2_csv.get_area_def",
        fake_get_area_def,
    )

    area = reader.get_area_def({"name": "frp"})

    assert called["name"] == "mtg_fci_fdss_1km"
    assert area == "dummy_area"
