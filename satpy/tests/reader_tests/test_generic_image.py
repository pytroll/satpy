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

import datetime as dt

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition
from rasterio.errors import NotGeoreferencedWarning

from satpy import Scene
from satpy.readers.generic_image import GenericImageFileHandler
from satpy.tests.utils import RANDOM_GEN, make_dataid

DATA_DATE = dt.datetime(2018, 1, 1)

X_SIZE = 100
Y_SIZE = 100
AREA_DEFINITION = AreaDefinition("geotiff_area", "ETRS89 / LAEA Europe", "ETRS89 / LAEA Europe",
                                 "EPSG:3035", X_SIZE, Y_SIZE,
                                 (2426378.0132, 1528101.2618, 6293974.6215, 5446513.5222))


@pytest.fixture
def random_image_channel():
    """Create random data."""
    return da.random.randint(0, 256, size=(Y_SIZE, X_SIZE), chunks=(50, 50)).astype(np.uint8)


random_image_channel_l = random_image_channel
random_image_channel_r = random_image_channel
random_image_channel_g = random_image_channel
random_image_channel_b = random_image_channel


@pytest.fixture
def alpha_channel():
    """Create alpha channel with fully transparent and opaque areas."""
    a__ = 255 * np.ones((Y_SIZE, X_SIZE), dtype=np.uint8)
    a__[:10, :10] = 0
    return da.from_array(a__, chunks=(50, 50))


@pytest.fixture
def random_image_channel_with_nans():
    """Create random data and replace a portion of it with NaN values."""
    arr = RANDOM_GEN.uniform(0., 1., size=(Y_SIZE, X_SIZE))
    arr[:10, :10] = np.nan
    return da.from_array(arr, chunks=(50, 50))


@pytest.fixture
def test_image_l(tmp_path, random_image_channel_l):
    """Create a test image with mode L."""
    dset = xr.DataArray(da.stack([random_image_channel_l]), dims=("bands", "y", "x"),
                        attrs={"name": "test_l", "start_time": DATA_DATE})
    dset["bands"] = ["L"]
    fname = tmp_path / "test_l.png"
    _save_image(dset, fname, "simple_image")

    return fname


@pytest.fixture
def test_image_l_nan(tmp_path, random_image_channel_with_nans):
    """Create a test image with mode L where data has NaN values."""
    dset = xr.DataArray(da.stack([random_image_channel_with_nans]), dims=("bands", "y", "x"),
                        attrs={"name": "test_l_nan", "start_time": DATA_DATE})
    dset["bands"] = ["L"]
    fname = tmp_path / "test_l_nan_nofillvalue.tif"
    _save_image(dset, fname, "geotiff")

    return fname


@pytest.fixture
def test_image_l_nan_fill_value(tmp_path, random_image_channel_with_nans):
    """Create a test image with mode L where data has NaN values and fill value is set."""
    dset = xr.DataArray(da.stack([random_image_channel_with_nans]), dims=("bands", "y", "x"),
                        attrs={"name": "test_l_nan", "start_time": DATA_DATE})
    dset["bands"] = ["L"]
    fname = tmp_path / "test_l_nan_fillvalue.tif"
    _save_image(dset, fname, "geotiff", fill_value=0)

    return fname


@pytest.fixture
def test_image_la(tmp_path, random_image_channel_l, alpha_channel):
    """Create a test image with mode LA."""
    dset = xr.DataArray(da.stack([random_image_channel_l, alpha_channel]),
                        dims=("bands", "y", "x"),
                        attrs={"name": "test_la", "start_time": DATA_DATE})
    dset["bands"] = ["L", "A"]
    fname = tmp_path / "20180101_0000_test_la.png"
    _save_image(dset, fname, "simple_image")

    return fname


@pytest.fixture
def test_image_rgb(tmp_path, random_image_channel_r, random_image_channel_g, random_image_channel_b):
    """Create a test image with mode RGB."""
    dset = xr.DataArray(da.stack([random_image_channel_r, random_image_channel_g, random_image_channel_b]),
                              dims=("bands", "y", "x"),
                              attrs={"name": "test_rgb",
                                     "start_time": DATA_DATE})
    dset["bands"] = ["R", "G", "B"]
    fname = tmp_path / "20180101_0000_test_rgb.tif"
    _save_image(dset, fname, "geotiff")

    return fname


@pytest.fixture
def rgba_dset(random_image_channel_r, random_image_channel_g, random_image_channel_b, alpha_channel):
    """Create an RGB dataset."""
    dset = xr.DataArray(
    da.stack([random_image_channel_r, random_image_channel_g, random_image_channel_b, alpha_channel]),
    dims=("bands", "y", "x"),
    attrs={"name": "test_rgba",
            "start_time": DATA_DATE})
    dset["bands"] = ["R", "G", "B", "A"]
    return dset


@pytest.fixture
def test_image_rgba(tmp_path, rgba_dset):
    """Create a test image with mode RGBA."""
    fname = tmp_path / "test_rgba.tif"
    _save_image(rgba_dset, fname, "geotiff")

    return fname


def _save_image(dset, fname, writer, fill_value=None):
    scn = Scene()
    scn["data"] = dset
    scn["data"].attrs["area"] = AREA_DEFINITION
    scn.save_dataset("data", str(fname), writer=writer, fill_value=fill_value)


def test_png_scene_l_mode(test_image_l):
    """Test reading a PNG image with L mode via satpy.Scene()."""
    with pytest.warns(NotGeoreferencedWarning, match=r"Dataset has no geotransform"):
        scn = Scene(reader="generic_image", filenames=[test_image_l])
    scn.load(["image"])
    _assert_image_common(scn, 1, None, None, np.float32)
    assert "area" not in scn["image"].attrs


def _assert_image_common(scn, channels, start_time, end_time, dtype):
    assert scn["image"].shape == (channels, Y_SIZE, X_SIZE)
    assert scn.sensor_names == {"images"}
    try:
        assert scn.start_time is start_time
        assert scn.end_time is end_time
    except AssertionError:
        assert scn.start_time == start_time
        assert scn.end_time == end_time
    assert scn["image"].dtype == dtype


def test_png_scene_la_mode(test_image_la):
    """Test reading a PNG image with LA mode via satpy.Scene()."""
    with pytest.warns(NotGeoreferencedWarning, match=r"Dataset has no geotransform"):
        scn = Scene(reader="generic_image", filenames=[test_image_la])
    scn.load(["image"])
    data = da.compute(scn["image"].data)
    assert np.sum(np.isnan(data)) == 100
    assert "area" not in scn["image"].attrs
    _assert_image_common(scn, 1, DATA_DATE, DATA_DATE, np.float32)


def test_geotiff_scene_rgb(test_image_rgb):
    """Test reading geotiff image in RGB mode via satpy.Scene()."""
    scn = Scene(reader="generic_image", filenames=[test_image_rgb])
    scn.load(["image"])
    assert scn["image"].area == AREA_DEFINITION
    _assert_image_common(scn, 3, DATA_DATE, DATA_DATE, np.float32)


def test_geotiff_scene_rgba(test_image_rgba):
    """Test reading geotiff image in RGBA mode via satpy.Scene()."""
    scn = Scene(reader="generic_image", filenames=[test_image_rgba])
    scn.load(["image"])
    _assert_image_common(scn, 3, None, None, np.float32)
    assert scn["image"].area == AREA_DEFINITION


def test_geotiff_scene_nan_fill_value(test_image_l_nan_fill_value):
    """Test reading geotiff image with fill value set via satpy.Scene()."""
    scn = Scene(reader="generic_image", filenames=[test_image_l_nan_fill_value])
    scn.load(["image"])
    assert np.sum(scn["image"].data[0][:10, :10].compute()) == 0
    _assert_image_common(scn, 1, None, None, np.uint8)

def test_geotiff_scene_nan(test_image_l_nan):
    """Test reading geotiff image with NaN values in it via satpy.Scene()."""
    scn = Scene(reader="generic_image", filenames=[test_image_l_nan])
    scn.load(["image"])
    assert np.all(np.isnan(scn["image"].data[0][:10, :10].compute()))
    _assert_image_common(scn, 1, None, None, np.float32)


def test_GenericImageFileHandler(test_image_rgba):
    """Test direct use of the reader."""
    from satpy.readers.generic_image import GenericImageFileHandler

    fname_info = {"start_time": DATA_DATE}
    ftype_info = {}
    reader = GenericImageFileHandler(test_image_rgba, fname_info, ftype_info)

    data_id = make_dataid(name="image")
    assert reader.file_content
    assert reader.finfo["filename"] == test_image_rgba
    assert reader.finfo["start_time"] == DATA_DATE
    assert reader.finfo["end_time"] == DATA_DATE
    assert reader.area == AREA_DEFINITION
    assert reader.get_area_def(None) == AREA_DEFINITION
    assert reader.start_time == DATA_DATE
    assert reader.end_time == DATA_DATE

    dataset = reader.get_dataset(data_id, {})
    assert isinstance(dataset, xr.DataArray)
    assert "spatial_ref" in dataset.coords
    assert np.all(np.isnan(dataset.data[:, :10, :10].compute()))


class FakeGenericImageFileHandler(GenericImageFileHandler):
    """Fake file handler."""

    def __init__(self, filename, filename_info, filetype_info, file_content, **kwargs):
        """Get fake file content from 'get_test_content'."""
        super(GenericImageFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.file_content = file_content
        self.dataset_name = None
        self.file_content.update(kwargs)


def test_GenericImageFileHandler_no_masking_for_float(rgba_dset):
    """Test direct use of the reader for float_data."""
    # do nothing if not integer
    float_data = rgba_dset / 255.
    reader = FakeGenericImageFileHandler("dummy", {}, {}, {"image": float_data})
    assert reader.get_dataset(make_dataid(name="image"), {}) is float_data


def test_GenericImageFileHandler_masking_for_integer(rgba_dset):
    """Test direct use of the reader for float_data."""
    # masking if integer
    data = rgba_dset.astype(np.uint32)
    assert data.bands.size == 4
    reader = FakeGenericImageFileHandler("dummy", {}, {}, {"image": data})
    ret_data = reader.get_dataset(make_dataid(name="image"), {})
    assert ret_data.bands.size == 3


def test_GenericImageFileHandler_datasetid(test_image_rgba):
    """Test direct use of the reader."""
    fname_info = {"start_time": DATA_DATE}
    ftype_info = {}
    reader = GenericImageFileHandler(test_image_rgba, fname_info, ftype_info)

    data_id = make_dataid(name="image-custom")
    assert reader.file_content
    dataset = reader.get_dataset(data_id, {})
    assert isinstance(dataset, xr.DataArray)


@pytest.fixture
def reader_l_nan_fill_value(test_image_l_nan_fill_value):
    """Create GenericImageFileHandler."""
    fname_info = {"start_time": DATA_DATE}
    ftype_info = {}
    return GenericImageFileHandler(test_image_l_nan_fill_value, fname_info, ftype_info)


def test_GenericImageFileHandler_nodata_nan_mask(reader_l_nan_fill_value):
    """Test nodata handling with direct use of the reader with nodata handling: nan_mask."""
    data_id = make_dataid(name="image-custom")
    assert reader_l_nan_fill_value.file_content
    info = {"nodata_handling": "nan_mask"}
    dataset = reader_l_nan_fill_value.get_dataset(data_id, info)
    assert isinstance(dataset, xr.DataArray)
    assert np.all(np.isnan(dataset.data[0][:10, :10].compute()))
    assert np.isnan(dataset.attrs["_FillValue"])


def test_GenericImageFileHandler_nodata_fill_value(reader_l_nan_fill_value):
    """Test nodata handling with direct use of the reader with nodata handling: fill_value."""
    info = {"nodata_handling": "fill_value"}
    data_id = make_dataid(name="image-custom")
    dataset = reader_l_nan_fill_value.get_dataset(data_id, info)
    assert isinstance(dataset, xr.DataArray)
    assert np.sum(dataset.data[0][:10, :10].compute()) == 0
    assert dataset.attrs["_FillValue"] == 0


def test_GenericImageFileHandler_nodata_nan_mask_default(reader_l_nan_fill_value):
    """Test nodata handling with direct use of the reader with default nodata handling."""
    data_id = make_dataid(name="image-custom")
    dataset = reader_l_nan_fill_value.get_dataset(data_id, {})
    assert isinstance(dataset, xr.DataArray)
    assert np.sum(dataset.data[0][:10, :10].compute()) == 0
    assert dataset.attrs["_FillValue"] == 0
