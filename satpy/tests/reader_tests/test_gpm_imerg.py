#!/usr/bin/env python
# Copyright (c) 2020 Satpy developers
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

"""Unittests for GPM IMERG reader."""

import datetime as dt
import os
import unittest
from unittest import mock

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler

DEFAULT_FILE_SHAPE = (3600, 1800)
DEFAULT_LAT_DATA = np.linspace(-89.95, 89.95,
                               DEFAULT_FILE_SHAPE[1]).astype(np.float32)
DEFAULT_LON_DATA = np.linspace(-179.95, 179.95,
                               DEFAULT_FILE_SHAPE[0]).astype(np.float32)


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def _get_geo_data(self, num_rows, num_cols):
        geo = {
            "Grid/lon":
                xr.DataArray(DEFAULT_LON_DATA,
                             attrs={"units": "degrees_east", },
                             dims=("lon")),
            "Grid/lat":
                xr.DataArray(DEFAULT_LAT_DATA,
                             attrs={"units": "degrees_north", },
                             dims=("lat")),
        }
        return geo

    def _get_precip_data(self, num_rows, num_cols):
        selection = {
            "Grid/IRprecipitation":
            xr.DataArray(
                da.ones((1, num_cols, num_rows), chunks=1024,
                        dtype=np.float32),
                attrs={
                    "_FillValue": -9999.9,
                    "units": "mm/hr",
                    "Units": "mm/hr",
                    "badval": h5py.h5r.Reference(),
                    "badvals": np.array([[h5py.h5r.Reference()]])
                },
                dims=("time", "lon", "lat")),
        }
        return selection

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        num_rows = 1800
        num_cols = 3600

        test_content = {}
        data = {}
        data = self._get_geo_data(num_rows, num_cols)
        test_content.update(data)
        data = self._get_precip_data(num_rows, num_cols)
        test_content.update(data)

        return test_content


class TestHdf5IMERG(unittest.TestCase):
    """Test the GPM IMERG reader."""

    yaml_file = "gpm_imerg.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.gpm_imerg import Hdf5IMERG
        self.reader_configs = config_search_paths(os.path.join("readers", self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(Hdf5IMERG, "__bases__", (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_load_data(self):
        """Test loading data."""
        from pyproj import CRS

        from satpy.readers.core.loading import load_reader

        # Filename to test, needed for start and end times
        filenames = [
            "3B-HHR.MS.MRG.3IMERG.20200131-S233000-E235959.1410.V06B.HDF5", ]

        # Expected projection in area def
        pdict = {"proj": "longlat",
                 "datum": "WGS84",
                 "no_defs": None,
                 "type": "crs"}

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert 1 == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers
        res = reader.load(["IRprecipitation"])
        assert 1 == len(res)
        assert res["IRprecipitation"].start_time == dt.datetime(2020, 1, 31, 23, 30, 0)
        assert res["IRprecipitation"].end_time == dt.datetime(2020, 1, 31, 23, 59, 59)
        assert res["IRprecipitation"].resolution == 0.1
        assert res["IRprecipitation"].area.width == 3600
        assert res["IRprecipitation"].area.height == 1800
        assert res["IRprecipitation"].area.crs == CRS(pdict)
        np.testing.assert_almost_equal(res["IRprecipitation"].area.area_extent,
                                       (-179.95, -89.95, 179.95, 89.95), 5)
