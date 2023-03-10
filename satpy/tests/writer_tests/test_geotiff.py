#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""Tests for the geotiff writer."""

from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


def _get_test_datasets_2d():
    """Create a single 2D test dataset."""
    ds1 = xr.DataArray(
        da.zeros((100, 200), chunks=50),
        dims=('y', 'x'),
        attrs={'name': 'test',
               'start_time': datetime.utcnow()}
    )
    return [ds1]


def _get_test_datasets_2d_nonlinear_enhancement():
    data_arrays = _get_test_datasets_2d()
    enh_history = [
        {"gamma": 2.0},
    ]
    for data_arr in data_arrays:
        data_arr.attrs["enhancement_history"] = enh_history
    return data_arrays


def _get_test_datasets_3d():
    """Create a single 3D test dataset."""
    ds1 = xr.DataArray(
        da.zeros((3, 100, 200), chunks=50),
        dims=('bands', 'y', 'x'),
        coords={'bands': ['R', 'G', 'B']},
        attrs={'name': 'test',
               'start_time': datetime.utcnow()}
    )
    return [ds1]


class TestGeoTIFFWriter:
    """Test the GeoTIFF Writer class."""

    def test_init(self):
        """Test creating the writer with no arguments."""
        from satpy.writers.geotiff import GeoTIFFWriter
        GeoTIFFWriter()

    @pytest.mark.parametrize(
        "input_func",
        [
            _get_test_datasets_2d,
            _get_test_datasets_3d
        ]
    )
    def test_simple_write(self, input_func, tmp_path):
        """Test basic writer operation."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = input_func()
        w = GeoTIFFWriter(base_dir=tmp_path)
        w.save_datasets(datasets)

    def test_simple_delayed_write(self, tmp_path):
        """Test writing can be delayed."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(base_dir=tmp_path)
        # when we switch to rio_save on XRImage then this will be sources
        # and targets
        res = w.save_datasets(datasets, compute=False)
        # this will fail if rasterio isn't installed
        assert isinstance(res, tuple)
        # two lists, sources and destinations
        assert len(res) == 2
        assert isinstance(res[0], list)
        assert isinstance(res[1], list)
        assert isinstance(res[0][0], da.Array)
        da.store(res[0], res[1])
        for target in res[1]:
            if hasattr(target, 'close'):
                target.close()

    def test_colormap_write(self, tmp_path):
        """Test writing an image with a colormap."""
        from trollimage.colormap import spectral
        from trollimage.xrimage import XRImage

        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(base_dir=tmp_path)
        # we'd have to customize enhancements to test this through
        # save_datasets. We'll use `save_image` as a workaround.
        img = XRImage(datasets[0])
        img.palettize(spectral)
        w.save_image(img, keep_palette=True)

    def test_float_write(self, tmp_path):
        """Test that geotiffs can be written as floats.

        NOTE: Does not actually check that the output is floats.

        """
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(base_dir=tmp_path,
                          enhance=False,
                          dtype=np.float32)
        w.save_datasets(datasets)

    def test_dtype_for_enhance_false(self, tmp_path):
        """Test that dtype of dataset is used if parameters enhance=False and dtype=None."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(base_dir=tmp_path, enhance=False)
        with mock.patch('satpy.writers.XRImage.save') as save_method:
            save_method.return_value = None
            w.save_datasets(datasets, compute=False)
            assert save_method.call_args[1]['dtype'] == np.float64

    def test_dtype_for_enhance_false_and_given_dtype(self, tmp_path):
        """Test that dtype of dataset is used if enhance=False and dtype=uint8."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(base_dir=tmp_path, enhance=False, dtype=np.uint8)
        with mock.patch('satpy.writers.XRImage.save') as save_method:
            save_method.return_value = None
            w.save_datasets(datasets, compute=False)
            assert save_method.call_args[1]['dtype'] == np.uint8

    def test_fill_value_from_config(self, tmp_path):
        """Test fill_value coming from the writer config."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(base_dir=tmp_path)
        w.info['fill_value'] = 128
        with mock.patch('satpy.writers.XRImage.save') as save_method:
            save_method.return_value = None
            w.save_datasets(datasets, compute=False)
            assert save_method.call_args[1]['fill_value'] == 128

    def test_tags(self, tmp_path):
        """Test tags being added."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(tags={'test1': 1}, base_dir=tmp_path)
        w.info['fill_value'] = 128
        with mock.patch('satpy.writers.XRImage.save') as save_method:
            save_method.return_value = None
            w.save_datasets(datasets, tags={'test2': 2}, compute=False)
            called_tags = save_method.call_args[1]['tags']
            assert called_tags == {'test1': 1, 'test2': 2}

    @pytest.mark.parametrize(
        "input_func",
        [
            _get_test_datasets_2d,
            _get_test_datasets_3d,
            _get_test_datasets_2d_nonlinear_enhancement,
        ]
    )
    @pytest.mark.parametrize(
        "save_kwargs",
        [
            {"include_scale_offset": True},
            {"scale_offset_tags": ("scale", "offset")},
        ]
    )
    def test_scale_offset(self, input_func, save_kwargs, tmp_path):
        """Test tags being added."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = input_func()
        w = GeoTIFFWriter(tags={'test1': 1}, base_dir=tmp_path)
        w.info['fill_value'] = 128
        with mock.patch('satpy.writers.XRImage.save') as save_method:
            save_method.return_value = None
            w.save_datasets(datasets, tags={'test2': 2}, compute=False, **save_kwargs)
        kwarg_name = "include_scale_offset_tags" if "include_scale_offset" in save_kwargs else "scale_offset_tags"
        kwarg_value = save_method.call_args[1].get(kwarg_name)
        assert kwarg_value is not None

    def test_tiled_value_from_config(self, tmp_path):
        """Test tiled value coming from the writer config."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = _get_test_datasets_2d()
        w = GeoTIFFWriter(base_dir=tmp_path)
        with mock.patch('satpy.writers.XRImage.save') as save_method:
            save_method.return_value = None
            w.save_datasets(datasets, compute=False)
            assert save_method.call_args[1]['tiled']
