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
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import make_dataid


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    """Class for faking the NetCDF4 Filehandler."""

    _nrows = 30
    _ncols = 40

    def __init__(self, *args, auto_maskandscale, **kwargs):
        """Init the file handler."""
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


class TestCLAAS2:
    @pytest.fixture
    def reader(self):
        """Return reader for CMSAF CLAAS-2."""
        from satpy._config import config_search_paths
        from satpy.readers import load_reader

        reader_configs = config_search_paths(
            os.path.join("readers", "cmsaf-claas2_l2_nc.yaml"))
        reader = load_reader(reader_configs)
        return reader

    @pytest.fixture(autouse=True, scope="class")
    def fake_handler(self):
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

    def test_file_pattern(self, reader):
        """Test file pattern matching."""
        filenames = [
                "CTXin20040120091500305SVMSG01MD.nc",
                "CTXin20040120093000305SVMSG01MD.nc",
                "CTXin20040120094500305SVMSG01MD.nc",
                "abcde52034294023489248MVSSG03DD.nc"]

        files = reader.select_files_from_pathnames(filenames)
        # only 3 out of 4 above should match
        assert len(files) == 3

    def test_load(self, reader):
        """Test loading."""

        # testing two filenames to test correctly combined
        filenames = [
            "CTXin20040120091500305SVMSG01MD.nc",
            "CTXin20040120093000305SVMSG01MD.nc"]

        loadables = reader.select_files_from_pathnames(filenames)
        reader.create_filehandlers(loadables)
        res = reader.load(
                [make_dataid(name=name) for name in ["cph", "ctt"]])
        assert 2 == len(res)
        assert reader.start_time == datetime.datetime(1985, 8, 13, 13, 15)
        assert reader.end_time == datetime.datetime(2085, 8, 13, 13, 15)
        np.testing.assert_array_almost_equal(
                res["cph"].data,
                np.tile(np.arange(0.0, 12.0, 0.01).reshape((30, 40)), [2, 1]))
        np.testing.assert_array_almost_equal(
                res["ctt"].data,
                np.tile(np.arange(12.0, 0.0, -0.01).reshape((30, 40)), [2, 1]))


class TestCLAAS2New:
    """Test CLAAS2 file handler by writing and reading test file."""

    @pytest.fixture
    def fake_dataset(self):
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
    def encoding(self):
        return {
            "ctt": {"scale_factor": np.float32(0.01)},
        }

    @pytest.fixture
    def fake_file(self, fake_dataset, encoding):
        filename = "CPPin20140101001500305SVMSG01MD.nc"
        fake_dataset.to_netcdf(filename, encoding=encoding)
        yield filename
        os.unlink(filename)

    @pytest.fixture
    def file_handler(self, fake_file):
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
