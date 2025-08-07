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
"""Tests for the writer compute module."""
from __future__ import annotations

import datetime as dt
import os
import shutil

import numpy as np
import xarray as xr
from dask import array as da


def test_group_results_by_output_file(tmp_path):
    """Test grouping results by output file.

    Add a test for grouping the results from save_datasets(..., compute=False)
    by output file.  This is useful if for some reason we want to treat each
    output file as a seperate computation (that can still be computed together
    later).
    """
    from pyresample import create_area_def

    from satpy.tests.utils import make_fake_scene
    from satpy.writers.core.compute import group_results_by_output_file
    x = 10
    fake_area = create_area_def("sargasso", 4326, resolution=1, width=x, height=x, center=(0, 0))
    fake_scene = make_fake_scene(
        {"dragon_top_height": (dat := xr.DataArray(
            dims=("y", "x"),
            data=da.arange(x*x).reshape((x, x)))),
         "penguin_bottom_height": dat,
         "kraken_depth": dat},
        daskify=True,
        area=fake_area,
        common_attrs={"start_time": dt.datetime(2022, 11, 16, 13, 27)})
    # NB: even if compute=False, ``save_datasets`` creates (empty) files
    (sources, targets) = fake_scene.save_datasets(
            filename=os.fspath(tmp_path / "test-{name}.tif"),
            writer="ninjogeotiff",
            compress="NONE",
            fill_value=0,
            compute=False,
            ChannelID="x",
            DataType="x",
            PhysicUnit="K",
            PhysicValue="Temperature",
            SatelliteNameID="x")

    grouped = group_results_by_output_file(sources, targets)

    assert len(grouped) == 3
    assert len({x.rfile.path for x in grouped[0][1]}) == 1
    for x in grouped:
        assert len(x[0]) == len(x[1])
    assert sources[:5] == grouped[0][0]
    assert targets[:5] == grouped[0][1]
    assert sources[10:] == grouped[2][0]
    assert targets[10:] == grouped[2][1]


class TestComputeWriterResults:
    """Test compute_writer_results()."""

    def setup_method(self):
        """Create temporary directory to save files to and a mock scene."""
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
            attrs={"name": "test",
                   "start_time": dt.datetime(2018, 1, 1, 0, 0, 0),
                   "area": adef}
        )
        self.scn = Scene()
        self.scn["test"] = ds1

        # Temp dir
        self.base_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory created for a test."""
        try:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_empty(self):
        """Test empty result list."""
        from satpy.writers.core.compute import compute_writer_results
        compute_writer_results([])

    def test_simple_image(self):
        """Test writing to PNG file."""
        from satpy.writers.core.compute import compute_writer_results
        fname = os.path.join(self.base_dir, "simple_image.png")
        res = self.scn.save_datasets(filename=fname,
                                     datasets=["test"],
                                     writer="simple_image",
                                     compute=False)
        compute_writer_results([res])
        assert os.path.isfile(fname)

    def test_geotiff(self):
        """Test writing to mitiff file."""
        from satpy.writers.core.compute import compute_writer_results
        fname = os.path.join(self.base_dir, "geotiff.tif")
        res = self.scn.save_datasets(filename=fname,
                                     datasets=["test"],
                                     writer="geotiff", compute=False)
        compute_writer_results([res])
        assert os.path.isfile(fname)

    def test_multiple_geotiff(self):
        """Test writing to mitiff file."""
        from satpy.writers.core.compute import compute_writer_results
        fname1 = os.path.join(self.base_dir, "geotiff1.tif")
        res1 = self.scn.save_datasets(filename=fname1,
                                      datasets=["test"],
                                      writer="geotiff", compute=False)
        fname2 = os.path.join(self.base_dir, "geotiff2.tif")
        res2 = self.scn.save_datasets(filename=fname2,
                                      datasets=["test"],
                                      writer="geotiff", compute=False)
        compute_writer_results([res1, res2])
        assert os.path.isfile(fname1)
        assert os.path.isfile(fname2)

    def test_multiple_simple(self):
        """Test writing to geotiff files."""
        from satpy.writers.core.compute import compute_writer_results
        fname1 = os.path.join(self.base_dir, "simple_image1.png")
        res1 = self.scn.save_datasets(filename=fname1,
                                      datasets=["test"],
                                      writer="simple_image", compute=False)
        fname2 = os.path.join(self.base_dir, "simple_image2.png")
        res2 = self.scn.save_datasets(filename=fname2,
                                      datasets=["test"],
                                      writer="simple_image", compute=False)
        compute_writer_results([res1, res2])
        assert os.path.isfile(fname1)
        assert os.path.isfile(fname2)

    def test_mixed(self):
        """Test writing to multiple mixed-type files."""
        from satpy.writers.core.compute import compute_writer_results
        fname1 = os.path.join(self.base_dir, "simple_image3.png")
        res1 = self.scn.save_datasets(filename=fname1,
                                      datasets=["test"],
                                      writer="simple_image", compute=False)
        fname2 = os.path.join(self.base_dir, "geotiff3.tif")
        res2 = self.scn.save_datasets(filename=fname2,
                                      datasets=["test"],
                                      writer="geotiff", compute=False)
        res3 = []
        compute_writer_results([res1, res2, res3])
        assert os.path.isfile(fname1)
        assert os.path.isfile(fname2)

    def test_source_only(self):
        """Test writers who only return dask arrays with no targets.

        With newer versions of dask it is recommended to not pass Arrays to
        Delayed functions as the tasks aren't properly/completely optimized.
        The alternative is to reduce the Array into a single array-like operation
        with map_blocks, blockwise, or some other reduction function.
        """
        from satpy.writers.core.compute import compute_writer_results

        compute_count = 0

        def reduced_map_blocks(_: np.ndarray) -> str:
            nonlocal compute_count
            compute_count += 1
            return "abc"

        arr1 = da.zeros((5, 5))
        arr2 = da.ones((5, 5))
        res1 = da.map_blocks(
            reduced_map_blocks,
            arr1.rechunk(arr1.shape),
            dtype=str,
            meta=np.ndarray((), dtype=str),
            chunks=(1, 1),
        )
        res2 = da.map_blocks(
            reduced_map_blocks,
            arr2.rechunk(arr2.shape),
            dtype=str,
            meta=np.ndarray((), dtype=str),
            chunks=(1, 1),
        )

        compute_writer_results([res1, res2])
        assert compute_count == 2
