#!/usr/bin/python
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
"""Unittests for generic image reader."""

import os
import unittest

import dask.array as da
import numpy as np
import xarray as xr

from satpy.tests.utils import make_dataid


class TestGenericImage(unittest.TestCase):
    """Test generic image reader."""

    def setUp(self):
        """Create temporary images to test on."""
        import tempfile
        from datetime import datetime

        from pyresample.geometry import AreaDefinition

        from satpy.scene import Scene

        self.date = datetime(2018, 1, 1)

        # Create area definition
        pcs_id = "ETRS89 / LAEA Europe"
        proj4_dict = "EPSG:3035"
        self.x_size = 100
        self.y_size = 100
        area_extent = (2426378.0132, 1528101.2618, 6293974.6215, 5446513.5222)
        self.area_def = AreaDefinition("geotiff_area", pcs_id, pcs_id,
                                       proj4_dict, self.x_size, self.y_size,
                                       area_extent)

        # Create datasets for L, LA, RGB and RGBA mode images
        r__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50)).astype(np.uint8)
        g__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50)).astype(np.uint8)
        b__ = da.random.randint(0, 256, size=(self.y_size, self.x_size),
                                chunks=(50, 50)).astype(np.uint8)
        a__ = 255 * np.ones((self.y_size, self.x_size), dtype=np.uint8)
        a__[:10, :10] = 0
        a__ = da.from_array(a__, chunks=(50, 50))

        r_nan__ = np.random.uniform(0., 1., size=(self.y_size, self.x_size))
        r_nan__[:10, :10] = np.nan
        r_nan__ = da.from_array(r_nan__, chunks=(50, 50))

        ds_l = xr.DataArray(da.stack([r__]), dims=("bands", "y", "x"),
                            attrs={"name": "test_l",
                                   "start_time": self.date})
        ds_l["bands"] = ["L"]
        ds_la = xr.DataArray(da.stack([r__, a__]), dims=("bands", "y", "x"),
                             attrs={"name": "test_la",
                                    "start_time": self.date})
        ds_la["bands"] = ["L", "A"]
        ds_rgb = xr.DataArray(da.stack([r__, g__, b__]),
                              dims=("bands", "y", "x"),
                              attrs={"name": "test_rgb",
                                     "start_time": self.date})
        ds_rgb["bands"] = ["R", "G", "B"]
        ds_rgba = xr.DataArray(da.stack([r__, g__, b__, a__]),
                               dims=("bands", "y", "x"),
                               attrs={"name": "test_rgba",
                                      "start_time": self.date})
        ds_rgba["bands"] = ["R", "G", "B", "A"]

        ds_l_nan = xr.DataArray(da.stack([r_nan__]),
                                dims=("bands", "y", "x"),
                                attrs={"name": "test_l_nan",
                                       "start_time": self.date})
        ds_l_nan["bands"] = ["L"]

        # Temp dir for the saved images
        self.base_dir = tempfile.mkdtemp()

        # Put the datasets to Scene for easy saving
        scn = Scene()
        scn["l"] = ds_l
        scn["l"].attrs["area"] = self.area_def
        scn["la"] = ds_la
        scn["la"].attrs["area"] = self.area_def
        scn["rgb"] = ds_rgb
        scn["rgb"].attrs["area"] = self.area_def
        scn["rgba"] = ds_rgba
        scn["rgba"].attrs["area"] = self.area_def
        scn["l_nan"] = ds_l_nan
        scn["l_nan"].attrs["area"] = self.area_def

        # Save the images.  Two images in PNG and two in GeoTIFF
        scn.save_dataset("l", os.path.join(self.base_dir, "test_l.png"), writer="simple_image")
        scn.save_dataset("la", os.path.join(self.base_dir, "20180101_0000_test_la.png"), writer="simple_image")
        scn.save_dataset("rgb", os.path.join(self.base_dir, "20180101_0000_test_rgb.tif"), writer="geotiff")
        scn.save_dataset("rgba", os.path.join(self.base_dir, "test_rgba.tif"), writer="geotiff")
        scn.save_dataset("l_nan", os.path.join(self.base_dir, "test_l_nan_fillvalue.tif"),
                         writer="geotiff", fill_value=0)
        scn.save_dataset("l_nan", os.path.join(self.base_dir, "test_l_nan_nofillvalue.tif"),
                         writer="geotiff")

        self.scn = scn

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_png_scene(self):
        """Test reading PNG images via satpy.Scene()."""
        from satpy import Scene

        fname = os.path.join(self.base_dir, "test_l.png")
        scn = Scene(reader="generic_image", filenames=[fname])
        scn.load(["image"])
        assert scn["image"].shape == (1, self.y_size, self.x_size)
        assert scn.sensor_names == {"images"}
        assert scn.start_time is None
        assert scn.end_time is None
        assert "area" not in scn["image"].attrs

        fname = os.path.join(self.base_dir, "20180101_0000_test_la.png")
        scn = Scene(reader="generic_image", filenames=[fname])
        scn.load(["image"])
        data = da.compute(scn["image"].data)
        assert scn["image"].shape == (1, self.y_size, self.x_size)
        assert scn.sensor_names == {"images"}
        assert scn.start_time == self.date
        assert scn.end_time == self.date
        assert "area" not in scn["image"].attrs
        assert np.sum(np.isnan(data)) == 100

    def test_geotiff_scene(self):
        """Test reading TIFF images via satpy.Scene()."""
        from satpy import Scene

        fname = os.path.join(self.base_dir, "20180101_0000_test_rgb.tif")
        scn = Scene(reader="generic_image", filenames=[fname])
        scn.load(["image"])
        assert scn["image"].shape == (3, self.y_size, self.x_size)
        assert scn.sensor_names == {"images"}
        assert scn.start_time == self.date
        assert scn.end_time == self.date
        assert scn["image"].area == self.area_def

        fname = os.path.join(self.base_dir, "test_rgba.tif")
        scn = Scene(reader="generic_image", filenames=[fname])
        scn.load(["image"])
        assert scn["image"].shape == (3, self.y_size, self.x_size)
        assert scn.sensor_names == {"images"}
        assert scn.start_time is None
        assert scn.end_time is None
        assert scn["image"].area == self.area_def

    def test_geotiff_scene_nan(self):
        """Test reading TIFF images originally containing NaN values via satpy.Scene()."""
        from satpy import Scene

        fname = os.path.join(self.base_dir, "test_l_nan_fillvalue.tif")
        scn = Scene(reader="generic_image", filenames=[fname])
        scn.load(["image"])
        assert scn["image"].shape == (1, self.y_size, self.x_size)
        assert np.sum(scn["image"].data[0][:10, :10].compute()) == 0

        fname = os.path.join(self.base_dir, "test_l_nan_nofillvalue.tif")
        scn = Scene(reader="generic_image", filenames=[fname])
        scn.load(["image"])
        assert scn["image"].shape == (1, self.y_size, self.x_size)
        assert np.all(np.isnan(scn["image"].data[0][:10, :10].compute()))

    def test_GenericImageFileHandler(self):
        """Test direct use of the reader."""
        from satpy.readers.generic_image import GenericImageFileHandler

        fname = os.path.join(self.base_dir, "test_rgba.tif")
        fname_info = {"start_time": self.date}
        ftype_info = {}
        reader = GenericImageFileHandler(fname, fname_info, ftype_info)

        foo = make_dataid(name="image")
        assert reader.file_content
        assert reader.finfo["filename"] == fname
        assert reader.finfo["start_time"] == self.date
        assert reader.finfo["end_time"] == self.date
        assert reader.area == self.area_def
        assert reader.get_area_def(None) == self.area_def
        assert reader.start_time == self.date
        assert reader.end_time == self.date

        dataset = reader.get_dataset(foo, {})
        assert isinstance(dataset, xr.DataArray)
        assert "spatial_ref" in dataset.coords
        assert np.all(np.isnan(dataset.data[:, :10, :10].compute()))

    def test_GenericImageFileHandler_masking_only_integer(self):
        """Test direct use of the reader."""
        from satpy.readers.generic_image import GenericImageFileHandler

        class FakeGenericImageFileHandler(GenericImageFileHandler):

            def __init__(self, filename, filename_info, filetype_info, file_content, **kwargs):
                """Get fake file content from 'get_test_content'."""
                super(GenericImageFileHandler, self).__init__(filename, filename_info, filetype_info)
                self.file_content = file_content
                self.dataset_name = None
                self.file_content.update(kwargs)

        data = self.scn["rgba"]

        # do nothing if not integer
        float_data = data / 255.
        reader = FakeGenericImageFileHandler("dummy", {}, {}, {"image": float_data})
        assert reader.get_dataset(make_dataid(name="image"), {}) is float_data

        # masking if integer
        data = data.astype(np.uint32)
        assert data.bands.size == 4
        reader = FakeGenericImageFileHandler("dummy", {}, {}, {"image": data})
        ret_data = reader.get_dataset(make_dataid(name="image"), {})
        assert ret_data.bands.size == 3

    def test_GenericImageFileHandler_datasetid(self):
        """Test direct use of the reader."""
        from satpy.readers.generic_image import GenericImageFileHandler

        fname = os.path.join(self.base_dir, "test_rgba.tif")
        fname_info = {"start_time": self.date}
        ftype_info = {}
        reader = GenericImageFileHandler(fname, fname_info, ftype_info)

        foo = make_dataid(name="image-custom")
        assert reader.file_content
        dataset = reader.get_dataset(foo, {})
        assert isinstance(dataset, xr.DataArray)

    def test_GenericImageFileHandler_nodata(self):
        """Test nodata handling with direct use of the reader."""
        from satpy.readers.generic_image import GenericImageFileHandler

        fname = os.path.join(self.base_dir, "test_l_nan_fillvalue.tif")
        fname_info = {"start_time": self.date}
        ftype_info = {}
        reader = GenericImageFileHandler(fname, fname_info, ftype_info)

        foo = make_dataid(name="image-custom")
        assert reader.file_content
        info = {"nodata_handling": "nan_mask"}
        dataset = reader.get_dataset(foo, info)
        assert isinstance(dataset, xr.DataArray)
        assert np.all(np.isnan(dataset.data[0][:10, :10].compute()))
        assert np.isnan(dataset.attrs["_FillValue"])

        info = {"nodata_handling": "fill_value"}
        dataset = reader.get_dataset(foo, info)
        assert isinstance(dataset, xr.DataArray)
        assert np.sum(dataset.data[0][:10, :10].compute()) == 0
        assert dataset.attrs["_FillValue"] == 0

        # default same as 'nodata_handling': 'fill_value'
        dataset = reader.get_dataset(foo, {})
        assert isinstance(dataset, xr.DataArray)
        assert np.sum(dataset.data[0][:10, :10].compute()) == 0
        assert dataset.attrs["_FillValue"] == 0
