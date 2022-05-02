#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Tests for the 'cmsaf-claas2_l2_nc' reader."""

import datetime
import os

import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.tests.utils import make_dataid


@pytest.fixture
def fake_dataset():
    """Create a CLAAS-like test dataset."""
    cph = xr.DataArray(
        [[[0, 1], [2, 0]]],
        dims=("time", "y", "x")
    )
    ctt = xr.DataArray(
        [[280, 290], [300, 310]],
        dims=("y", "x")
    )
    time_bounds = xr.DataArray(
        [[12436.91666667, 12436.92534722]],
        dims=("time", "bndsize")
    )
    attrs = {
        "CMSAF_proj4_params": "+a=6378169.0 +h=35785831.0 "
                              "+b=6356583.8 +lon_0=0 +proj=geos",
        "CMSAF_area_extent": np.array(
            [-5456233.41938636, -5453233.01608472,
             5453233.01608472, 5456233.41938636]),
        "time_coverage_start": "1985-08-13T13:15:00Z",
        "time_coverage_end": "2085-08-13T13:15:00Z",
    }
    return xr.Dataset(
        {
            "cph": cph,
            "ctt": ctt,
            "time_bnds": time_bounds
        },
        attrs=attrs
    )


@pytest.fixture
def encoding():
    """Dataset encoding."""
    return {
        "ctt": {"scale_factor": np.float32(0.01)},
    }


@pytest.fixture
def fake_file(fake_dataset, encoding, tmp_path):
    """Write a fake dataset to file."""
    filename = tmp_path / "CPPin20140101001500305SVMSG01MD.nc"
    fake_dataset.to_netcdf(filename, encoding=encoding)
    yield filename


@pytest.fixture
def fake_files(fake_dataset, encoding, tmp_path):
    """Write the same fake dataset into two different files."""
    filenames = [
        tmp_path / "CPPin20140101001500305SVMSG01MD.nc",
        tmp_path / "CPPin20140101003000305SVMSG01MD.nc",
    ]
    for filename in filenames:
        fake_dataset.to_netcdf(filename, encoding=encoding)
    yield filenames


@pytest.fixture
def reader():
    """Return reader for CMSAF CLAAS-2."""
    from satpy._config import config_search_paths
    from satpy.readers import load_reader

    reader_configs = config_search_paths(
        os.path.join("readers", "cmsaf-claas2_l2_nc.yaml"))
    reader = load_reader(reader_configs)
    return reader


def test_file_pattern(reader):
    """Test file pattern matching."""
    filenames = [
            "CTXin20040120091500305SVMSG01MD.nc",
            "CTXin20040120093000305SVMSG01MD.nc",
            "CTXin20040120094500305SVMSG01MD.nc",
            "abcde52034294023489248MVSSG03DD.nc"]

    files = reader.select_files_from_pathnames(filenames)
    # only 3 out of 4 above should match
    assert len(files) == 3


class TestCLAAS2MultiFile:
    """Test reading multiple CLAAS-2 files."""

    @pytest.fixture
    def multi_file_reader(self, reader, fake_files):
        """Create a multi-file reader."""
        loadables = reader.select_files_from_pathnames(fake_files)
        reader.create_filehandlers(loadables)
        return reader

    @pytest.fixture
    def multi_file_dataset(self, multi_file_reader):
        """Load datasets from multiple files."""
        ds_ids = [make_dataid(name=name) for name in ["cph", "ctt"]]
        datasets = multi_file_reader.load(ds_ids)
        return datasets

    def test_combine_timestamps(self, multi_file_reader):
        """Test combination of timestamps."""
        assert multi_file_reader.start_time == datetime.datetime(1985, 8, 13, 13, 15)
        assert multi_file_reader.end_time == datetime.datetime(2085, 8, 13, 13, 15)

    @pytest.mark.parametrize(
        "ds_name,expected",
        [
            ("cph", [[0, 1], [2, 0], [0, 1], [2, 0]]),
            ("ctt", [[280, 290], [300, 310], [280, 290], [300, 310]]),
        ]
    )
    def test_combine_datasets(self, multi_file_dataset, ds_name, expected):
        """Test combination of datasets."""
        np.testing.assert_array_almost_equal(
            multi_file_dataset[ds_name].data, expected
        )

    def test_number_of_datasets(self, multi_file_dataset):
        """Test number of datasets."""
        assert 2 == len(multi_file_dataset)


class TestCLAAS2SingleFile:
    """Test reading a single CLAAS2 file."""

    @pytest.fixture
    def file_handler(self, fake_file):
        """Return a CLAAS-2 file handler."""
        from satpy.readers.cmsaf_claas2 import CLAAS2
        return CLAAS2(fake_file, {}, {})

    def test_get_area_def(self, file_handler):
        """Test area definition."""
        MAJOR_AXIS_OF_EARTH_ELLIPSOID = 6378169.0
        MINOR_AXIS_OF_EARTH_ELLIPSOID = 6356583.8
        SATELLITE_ALTITUDE = 35785831.0
        PROJECTION_LONGITUDE = 0.0
        PROJ_DICT = {
            "a": MAJOR_AXIS_OF_EARTH_ELLIPSOID,
            "b": MINOR_AXIS_OF_EARTH_ELLIPSOID,
            "h": SATELLITE_ALTITUDE,
            "lon_0": PROJECTION_LONGITUDE,
            "proj": "geos",
            "units": "m",
        }
        area_exp = AreaDefinition(
            area_id="some_area_name",
            description="on-the-fly area",
            proj_id="geos",
            projection=PROJ_DICT,
            area_extent=[-5456233.41938636, -5453233.01608472,
                         5453233.01608472, 5456233.41938636],
            width=2,
            height=2,
        )
        area = file_handler.get_area_def(make_dataid(name="foo"))
        assert area == area_exp

    @pytest.mark.parametrize(
        "ds_name,expected",
        [
            ("ctt", xr.DataArray([[280, 290], [300, 310]], dims=('y', 'x'))),
            ("cph", xr.DataArray([[0, 1], [2, 0]], dims=('y', 'x'))),
        ]
    )
    def test_get_dataset(self, file_handler, ds_name, expected):
        """Test dataset loading."""
        dsid = make_dataid(name=ds_name)
        ds = file_handler.get_dataset(dsid, {})
        xr.testing.assert_allclose(ds, expected)

    def test_start_time(self, file_handler):
        """Test start time property."""
        assert file_handler.start_time == datetime.datetime(1985, 8, 13, 13, 15)

    def test_end_time(self, file_handler):
        """Test end time property."""
        assert file_handler.end_time == datetime.datetime(2085, 8, 13, 13, 15)
