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
    return [xr.DataArray(
        da.zeros((100, 200), chunks=50),
        dims=("y", "x"),
        attrs={"name": "test",
               "start_time": datetime.datetime(1985, 8, 13, 15, 0)})]


def test_ninjogeotiff(fake_datasets):
    """Test that it writes a GeoTIFF with the appropriate NinJo-tags."""
    from satpy.writers.ninjogeotiff import NinJoGeoTIFFWriter
    w = NinJoGeoTIFFWriter()
    with unittest.mock.patch("satpy.writers.geotiff.GeoTIFFWriter.save_datasets") as swggs:
        w.save_datasets(
                fake_datasets,
                ninjo_tags=dict(
                    physic_unit="C",
                    sat_id="6400014",
                    chan_id="900015",
                    data_cat="GORN",
                    data_source="EUMETCAST",
                    ch_min_measurement_unit="-87.5",
                    ch_max_measurement_unit="40"))
        swggs.assert_called_with(
                fake_datasets,
                tags={"ninjo_physic_unit": "C",
                      "ninjo_sat_id": "6400014",
                      "ninjo_chan_id": "900015",
                      "ninjo_data_cat": "GORN",
                      "ninjo_data_source": "EUMETCAST",
                      "ninjo_ch_min_measurement_unit": "-87.5",
                      "ninjo_ch_max_measurement_unit": "40",
                      "ninjo_TransparentPixel": "0"})
