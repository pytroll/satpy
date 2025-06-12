# Copyright (c) 2025 Satpy developers
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
"""Tests for the writer base module."""
from __future__ import annotations

import datetime as dt
import os
import shutil

import pytest
import xarray as xr
from dask import array as da


class TestBaseWriter:
    """Test the base writer class."""

    def setup_method(self):
        """Set up tests."""
        import tempfile

        from pyresample.geometry import AreaDefinition

        from satpy.scene import Scene

        adef = AreaDefinition(
            "test", "test", "test", "EPSG:4326",
            100, 200, (-180., -90., 180., 90.),
        )
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=("y", "x"),
            attrs={
                "name": "test",
                "start_time": dt.datetime(2018, 1, 1, 0, 0, 0),
                "sensor": "fake_sensor",
                "area": adef,
            }
        )
        ds2 = ds1.copy()
        ds2.attrs["sensor"] = {"fake_sensor1", "fake_sensor2"}
        self.scn = Scene()
        self.scn["test"] = ds1
        self.scn["test2"] = ds2

        # Temp dir
        self.base_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory created for a test."""
        try:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_save_dataset_static_filename(self):
        """Test saving a dataset with a static filename specified."""
        self.scn.save_datasets(base_dir=self.base_dir, filename="geotiff.tif")
        assert os.path.isfile(os.path.join(self.base_dir, "geotiff.tif"))

    @pytest.mark.parametrize(
        ("fmt_fn", "exp_fns"),
        [
            ("geotiff_{name}_{start_time:%Y%m%d_%H%M%S}.tif",
             ["geotiff_test_20180101_000000.tif", "geotiff_test2_20180101_000000.tif"]),
            ("geotiff_{name}_{sensor}.tif",
             ["geotiff_test_fake_sensor.tif", "geotiff_test2_fake_sensor1-fake_sensor2.tif"]),
        ]
    )
    def test_save_dataset_dynamic_filename(self, fmt_fn, exp_fns):
        """Test saving a dataset with a format filename specified."""
        self.scn.save_datasets(base_dir=self.base_dir, filename=fmt_fn)
        for exp_fn in exp_fns:
            exp_path = os.path.join(self.base_dir, exp_fn)
            assert os.path.isfile(exp_path)

    def test_save_dataset_dynamic_filename_with_dir(self):
        """Test saving a dataset with a format filename that includes a directory."""
        fmt_fn = os.path.join("{start_time:%Y%m%d}", "geotiff_{name}_{start_time:%Y%m%d_%H%M%S}.tif")
        exp_fn = os.path.join("20180101", "geotiff_test_20180101_000000.tif")
        self.scn.save_datasets(base_dir=self.base_dir, filename=fmt_fn)
        assert os.path.isfile(os.path.join(self.base_dir, exp_fn))

        # change the filename pattern but keep the same directory
        fmt_fn2 = os.path.join("{start_time:%Y%m%d}", "geotiff_{name}_{start_time:%Y%m%d_%H}.tif")
        exp_fn2 = os.path.join("20180101", "geotiff_test_20180101_00.tif")
        self.scn.save_datasets(base_dir=self.base_dir, filename=fmt_fn2)
        assert os.path.isfile(os.path.join(self.base_dir, exp_fn2))
        # the original file should still exist
        assert os.path.isfile(os.path.join(self.base_dir, exp_fn))
