#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Module for testing the satpy.readers.clavrx module."""

import os
import unittest
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.tests.reader_tests.test_hdf4_utils import FakeHDF4FileHandler

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)

class FakeHDF4FileHandlerGeo(FakeHDF4FileHandler):
    """Swap-in HDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        file_content = {
            "/attr/platform": "HIM8",
            "/attr/sensor": "AHI",
            # this is a Level 2 file that came from a L1B file
            "/attr/L1B": "clavrx_H08_20180806_1800",
        }

        file_content["longitude"] = xr.DataArray(
            DEFAULT_LON_DATA,
            dims=("y", "x"),
            attrs={
                "_FillValue": np.nan,
                "scale_factor": 1.,
                "add_offset": 0.,
                "standard_name": "longitude",
            })
        file_content["longitude/shape"] = DEFAULT_FILE_SHAPE

        file_content["latitude"] = xr.DataArray(
            DEFAULT_LAT_DATA,
            dims=("y", "x"),
            attrs={
                "_FillValue": np.nan,
                "scale_factor": 1.,
                "add_offset": 0.,
                "standard_name": "latitude",
            })
        file_content["latitude/shape"] = DEFAULT_FILE_SHAPE

        file_content["refl_1_38um_nom"] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.float32),
            dims=("y", "x"),
            attrs={
                "SCALED": 1,
                "add_offset": 59.0,
                "scale_factor": 0.0018616290763020515,
                "units": "%",
                "_FillValue": -32768,
                "valid_range": [-32767, 32767],
                "actual_range": [-2., 120.],
                "actual_missing": -999.0
            })
        file_content["refl_1_38um_nom/shape"] = DEFAULT_FILE_SHAPE

        # data with fill values
        file_content["variable2"] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.float32),
            dims=("y", "x"),
            attrs={
                "_FillValue": -1,
                "scale_factor": 1.,
                "add_offset": 0.,
                "units": "1",
            })
        file_content["variable2/shape"] = DEFAULT_FILE_SHAPE
        file_content["variable2"] = file_content["variable2"].where(
            file_content["variable2"] % 2 != 0)

        # category
        file_content["variable3"] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.byte),
            dims=("y", "x"),
            attrs={
                "SCALED": 0,
                "_FillValue": -128,
                "flag_meanings": "clear water supercooled mixed ice unknown",
                "flag_values": [0, 1, 2, 3, 4, 5],
                "units": "1",
            })
        file_content["variable3/shape"] = DEFAULT_FILE_SHAPE

        return file_content


class TestCLAVRXReaderGeo(unittest.TestCase):
    """Test CLAVR-X Reader with Geo files."""

    yaml_file = "clavrx.yaml"

    def setUp(self):
        """Wrap HDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.clavrx import CLAVRXHDF4FileHandler
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(CLAVRXHDF4FileHandler, "__bases__", (FakeHDF4FileHandlerGeo,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            "clavrx_H08_20180806_1800.level2.hdf",
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_no_nav_donor(self):
        """Test exception raised when no donor file is available."""
        import xarray as xr

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        fake_fn = "clavrx_H08_20180806_1800.level2.hdf"
        with mock.patch("satpy.readers.clavrx.SDS", xr.DataArray):
            loadables = r.select_files_from_pathnames([fake_fn])
            r.create_filehandlers(loadables)
            l1b_base = fake_fn.split(".")[0]
            msg = f"Missing navigation donor {l1b_base}"
            with pytest.raises(IOError, match=msg):
                r.load(["refl_1_38um_nom", "variable2", "variable3"])

    def test_load_all_old_donor(self):
        """Test loading all test datasets with old donor."""
        import xarray as xr

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch("satpy.readers.clavrx.SDS", xr.DataArray):
            loadables = r.select_files_from_pathnames([
                "clavrx_H08_20180806_1800.level2.hdf",
            ])
            r.create_filehandlers(loadables)
        with mock.patch("satpy.readers.clavrx.glob") as g, mock.patch("satpy.readers.clavrx.netCDF4.Dataset") as d:
            g.return_value = ["fake_donor.nc"]
            x = np.linspace(-0.1518, 0.1518, 300)
            y = np.linspace(0.1518, -0.1518, 10)
            proj = mock.Mock(
                semi_major_axis=6378.137,
                semi_minor_axis=6356.7523142,
                perspective_point_height=35791,
                longitude_of_projection_origin=140.7,
                sweep_angle_axis="y",
            )
            d.return_value = fake_donor = mock.MagicMock(
                variables={"Projection": proj, "x": x, "y": y},
            )
            fake_donor.__getitem__.side_effect = lambda key: fake_donor.variables[key]
            datasets = r.load(["refl_1_38um_nom", "variable2", "variable3"])
        assert len(datasets) == 3
        for v in datasets.values():
            assert "calibration" not in v.attrs
            assert v.attrs["units"] in ["1", "%"]
            assert isinstance(v.attrs["area"], AreaDefinition)
            if v.attrs.get("flag_values"):
                assert "_FillValue" in v.attrs
            else:
                assert "_FillValue" not in v.attrs
            if v.attrs["name"] == "refl_1_38um_nom":
                assert "valid_range" in v.attrs
                assert isinstance(v.attrs["valid_range"], list)
            else:
                assert "valid_range" not in v.attrs
            if "flag_values" in v.attrs:
                assert np.issubdtype(v.dtype, np.integer)
                assert v.attrs.get("flag_meanings") is not None

    def test_load_all_new_donor(self):
        """Test loading all test datasets with new donor."""
        import xarray as xr

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch("satpy.readers.clavrx.SDS", xr.DataArray):
            loadables = r.select_files_from_pathnames([
                "clavrx_H08_20180806_1800.level2.hdf",
            ])
            r.create_filehandlers(loadables)
        with mock.patch("satpy.readers.clavrx.glob") as g, mock.patch("satpy.readers.clavrx.netCDF4.Dataset") as d:
            g.return_value = ["fake_donor.nc"]
            x = np.linspace(-0.1518, 0.1518, 300)
            y = np.linspace(0.1518, -0.1518, 10)
            proj = mock.Mock(
                semi_major_axis=6378137,
                semi_minor_axis=6356752.3142,
                perspective_point_height=35791000,
                longitude_of_projection_origin=140.7,
                sweep_angle_axis="y",
            )
            d.return_value = fake_donor = mock.MagicMock(
                variables={"goes_imager_projection": proj, "x": x, "y": y},
            )
            fake_donor.__getitem__.side_effect = lambda key: fake_donor.variables[key]
            datasets = r.load(["refl_1_38um_nom", "variable2", "variable3"])
        assert len(datasets) == 3
        for v in datasets.values():
            assert "calibration" not in v.attrs
            assert v.attrs["units"] in ["1", "%"]
            assert isinstance(v.attrs["area"], AreaDefinition)
            assert v.attrs["area"].is_geostationary is True
            assert v.attrs["platform_name"] == "himawari8"
            assert v.attrs["sensor"] == "ahi"
        assert datasets["variable3"].attrs.get("flag_meanings") is not None
