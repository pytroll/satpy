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

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - request


@pytest.fixture(
    params=[datetime.datetime(2017, 12, 5), datetime.datetime(2017, 12, 6)]
)
def start_time(request):
    """Get start time of the dataset."""
    return request.param


@pytest.fixture
def start_time_str(start_time):
    """Get string representation of the start time."""
    return start_time.strftime("%Y-%m-%dT%H:%M:%SZ")


@pytest.fixture()
def fake_dataset(start_time_str):
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
        "time_coverage_start": start_time_str,
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

    def test_combine_timestamps(self, multi_file_reader, start_time):
        """Test combination of timestamps."""
        assert multi_file_reader.start_time == start_time
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

    @pytest.fixture
    def area_extent_exp(self, start_time):
        """Get expected area extent."""
        if start_time < datetime.datetime(2017, 12, 6):
            return (-5454733.160460291, -5454733.160460292, 5454733.160460292, 5454733.160460291)
        return (-5456233.362099582, -5453232.958821001, 5453232.958821001, 5456233.362099582)

    @pytest.fixture
    def area_exp(self, area_extent_exp):
        """Get expected area definition."""
        proj_dict = {
            "a": 6378169.0,
            "b": 6356583.8,
            "h": 35785831.0,
            "lon_0": 0.0,
            "proj": "geos",
            "units": "m",
        }
        return AreaDefinition(
            area_id="msg_seviri_fes_3km",
            description="MSG SEVIRI Full Earth Scanning service area definition with 3 km resolution",
            proj_id="geos",
            projection=proj_dict,
            area_extent=area_extent_exp,
            width=3636,
            height=3636,
        )

    def test_get_area_def(self, file_handler, area_exp):
        """Test area definition."""
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

    def test_start_time(self, file_handler, start_time):
        """Test start time property."""
        assert file_handler.start_time == start_time

    def test_end_time(self, file_handler):
        """Test end time property."""
        assert file_handler.end_time == datetime.datetime(2085, 8, 13, 13, 15)
