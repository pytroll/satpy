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

import os
import datetime
import numpy as np
import xarray as xr
import numpy.testing
import pytest
from unittest import mock
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    """Class for faking the NetCDF4 Filehandler."""

    _nrows = 30
    _ncols = 40

    def __init__(self, *args, auto_maskandscale, **kwargs):
        # make sure that CLAAS2 reader asks NetCDF4FileHandler for having
        # auto_maskandscale enabled
        assert auto_maskandscale
        super().__init__(*args, **kwargs)

    def _get_global_attributes(self):
        data = {}
        attrs = {
                "CMSAF_proj4_params": "+a=6378169.0 +h=35785831.0 "
                                      "+b=6356583.8 +lon_0=0 +proj=geos",
                "CMSAF_area_extent": np.array(
                    [-5456233.41938636, -5453233.01608472,
                     5453233.01608472, 5456233.41938636]),
                "time_coverage_start": "1985-08-13T13:15:00Z",
                "time_coverage_end": "2085-08-13T13:15:00Z",
                }
        for (k, v) in attrs.items():
            data["/attr/" + k] = v
        return data

    def _get_data(self):
        data = {
                "cph": xr.DataArray(
                    np.arange(self._nrows*self._ncols, dtype="i4").reshape(
                        (1, self._nrows, self._ncols))/100,
                    dims=("time", "y", "x")),
                "ctt": xr.DataArray(
                    np.arange(self._nrows*self._ncols, 0, -1,
                              dtype="i4").reshape(
                                  (self._nrows, self._ncols))/100,
                    dims=("y", "x")),
                "time_bnds": xr.DataArray(
                    [[12436.91666667, 12436.92534722]],
                    dims=("time", "time_bnds"))}
        for k in set(data.keys()):
            data[f"{k:s}/dimensions"] = data[k].dims
            data[f"{k:s}/attr/fruit"] = "apple"
            data[f"{k:s}/attr/scale_factor"] = np.float32(0.01)
        return data

    def _get_dimensions(self):
        data = {
                "/dimension/x": self._nrows,
                "/dimension/y": self._ncols,
                "/dimension/time": 1,
                "/dimension/time_bnds": 2,
                }
        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        """Get the content of the test data."""
        # mock global attributes
        # - root groups global
        # - other groups global
        # mock data variables
        # mock dimensions
        #
        # ... but only what satpy is using ...

        D = {}
        D.update(self._get_data())
        D.update(self._get_dimensions())
        D.update(self._get_global_attributes())
        return D


@pytest.fixture
def reader():
    """Return reader for CMSAF CLAAS-2."""
    from satpy.config import config_search_paths
    from satpy.readers import load_reader

    reader_configs = config_search_paths(
        os.path.join("readers", "cmsaf-claas2_l2_nc.yaml"))
    reader = load_reader(reader_configs)
    return reader


@pytest.fixture(autouse=True, scope="class")
def fake_handler():
    """Wrap NetCDF4 FileHandler with our own fake handler."""
    # implementation strongly inspired by test_viirs_l1b.py
    from satpy.readers.cmsaf_claas2 import CLAAS2
    p = mock.patch.object(
            CLAAS2,
            "__bases__",
            (FakeNetCDF4FileHandler2,))
    with p:
        p.is_local = True
        yield p


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


def test_load(reader):
    """Test loading."""
    from satpy import DatasetID

    # testing two filenames to test correctly combined
    filenames = [
        "CTXin20040120091500305SVMSG01MD.nc",
        "CTXin20040120093000305SVMSG01MD.nc"]

    loadables = reader.select_files_from_pathnames(filenames)
    reader.create_filehandlers(loadables)
    res = reader.load(
            [DatasetID(name=name) for name in ["cph", "ctt"]])
    assert 2 == len(res)
    assert reader.start_time == datetime.datetime(1985, 8, 13, 13, 15)
    assert reader.end_time == datetime.datetime(2085, 8, 13, 13, 15)
    np.testing.assert_array_almost_equal(
            res["cph"].data,
            np.tile(np.arange(0.0, 12.0, 0.01).reshape((30, 40)), [2, 1]))
    np.testing.assert_array_almost_equal(
            res["ctt"].data,
            np.tile(np.arange(12.0, 0.0, -0.01).reshape((30, 40)), [2, 1]))
