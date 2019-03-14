#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Enhancements."""

import numpy as np
import xarray as xr
import xarray.ufuncs as xu
import dask
import dask.array as da
import logging

LOG = logging.getLogger(__name__)


def stretch(img, **kwargs):
    """Perform stretch."""
    return img.stretch(**kwargs)


def gamma(img, **kwargs):
    """Perform gamma correction."""
    return img.gamma(**kwargs)


def invert(img, *args):
    """Perform inversion."""
    return img.invert(*args)


def apply_enhancement(data, func, exclude=None, separate=False,
                      pass_dask=False):
    """Apply `func` to the provided data.

    Args:
        data (xarray.DataArray): Data to be modified inplace.
        func (callable): Function to be applied to an xarray
        exclude (iterable): Bands in the 'bands' dimension to not include
                            in the calculations.
        separate (bool): Apply `func` one band at a time. Default is False.
        pass_dask (bool): Pass the underlying dask array instead of the
                          xarray.DataArray.

    """
    attrs = data.attrs
    bands = data.coords['bands'].values
    if exclude is None:
        exclude = ['A'] if 'A' in bands else []

    if separate:
        data_arrs = []
        for idx, band_name in enumerate(bands):
            band_data = data.sel(bands=[band_name])
            if band_name in exclude:
                # don't modify alpha
                data_arrs.append(band_data)
                continue

            if pass_dask:
                dims = band_data.dims
                coords = band_data.coords
                d_arr = func(band_data.data, index=idx)
                band_data = xr.DataArray(d_arr, dims=dims, coords=coords)
            else:
                band_data = func(band_data, index=idx)
            data_arrs.append(band_data)
            # we assume that the func can add attrs
            attrs.update(band_data.attrs)

        data.data = xr.concat(data_arrs, dim='bands').data
        data.attrs = attrs
        return data
    else:
        band_data = data.sel(bands=[b for b in bands
                                    if b not in exclude])
        if pass_dask:
            dims = band_data.dims
            coords = band_data.coords
            d_arr = func(band_data.data)
            band_data = xr.DataArray(d_arr, dims=dims, coords=coords)
        else:
            band_data = func(band_data)

        attrs.update(band_data.attrs)
        # combine the new data with the excluded data
        new_data = xr.concat([band_data, data.sel(bands=exclude)],
                             dim='bands')
        data.data = new_data.sel(bands=bands).data
        data.attrs = attrs

    return data


# pointed to by generic.yaml
def crefl_scaling(img, **kwargs):
    LOG.debug("Applying the crefl_scaling")

    def func(band_data, index=None):
        idx = np.array(kwargs['idx']) / 255
        sc = np.array(kwargs['sc']) / 255
        band_data *= .01
        # Interpolate band on [0,1] using "lazy" arrays (put calculations off until the end).
        band_data = xr.DataArray(da.clip(band_data.data.map_blocks(np.interp, xp=idx, fp=sc), 0, 1),
                                 coords=band_data.coords, dims=band_data.dims, name=band_data.name,
                                 attrs=band_data.attrs)
        return band_data

    return apply_enhancement(img.data, func, separate=True)


def cira_stretch(img, **kwargs):
    """Logarithmic stretch adapted to human vision.

    Applicable only for visible channels.
    """
    LOG.debug("Applying the cira-stretch")

    def func(band_data):
        log_root = np.log10(0.0223)
        denom = (1.0 - log_root) * 0.75
        band_data *= 0.01
        band_data = band_data.clip(np.finfo(float).eps)
        band_data = xu.log10(band_data)
        band_data -= log_root
        band_data /= denom
        return band_data

    return apply_enhancement(img.data, func)


def _lookup_delayed(luts, band_data):
    # can't use luts.__getitem__ for some reason
    return luts[band_data]


def lookup(img, **kwargs):
    """Assign values to channels based on a table."""
    luts = np.array(kwargs['luts'], dtype=np.float32) / 255.0

    def func(band_data, luts=luts, index=-1):
        # NaN/null values will become 0
        lut = luts[:, index] if len(luts.shape) == 2 else luts
        band_data = band_data.clip(0, lut.size - 1).astype(np.uint8)

        new_delay = dask.delayed(_lookup_delayed)(lut, band_data)
        new_data = da.from_delayed(new_delay, shape=band_data.shape,
                                   dtype=luts.dtype)
        return new_data

    return apply_enhancement(img.data, func, separate=True, pass_dask=True)


def colorize(img, **kwargs):
    """Colorize the given image."""
    full_cmap = _merge_colormaps(kwargs)
    img.colorize(full_cmap)


def palettize(img, **kwargs):
    """Palettize the given image (no color interpolation)."""
    full_cmap = _merge_colormaps(kwargs)
    img.palettize(full_cmap)


def _merge_colormaps(kwargs):
    """Merge colormaps listed in kwargs."""
    from trollimage.colormap import Colormap
    full_cmap = None

    palette = kwargs['palettes']
    if isinstance(palette, Colormap):
        full_cmap = palette
    else:
        for itm in palette:
            cmap = create_colormap(itm)
            cmap.set_range(itm["min_value"], itm["max_value"])
            if full_cmap is None:
                full_cmap = cmap
            else:
                full_cmap = full_cmap + cmap

    return full_cmap


def create_colormap(palette):
    """Create colormap of the given numpy file, color vector or colormap."""
    from trollimage.colormap import Colormap
    fname = palette.get('filename', None)
    if fname:
        data = np.load(fname)
        cmap = []
        num = 1.0 * data.shape[0]
        for i in range(int(num)):
            cmap.append((i / num, (data[i, 0] / 255., data[i, 1] / 255.,
                                   data[i, 2] / 255.)))
        return Colormap(*cmap)

    colors = palette.get('colors', None)
    if isinstance(colors, list):
        cmap = []
        values = palette.get('values', None)
        for idx, color in enumerate(colors):
            if values:
                value = values[idx]
            else:
                value = idx / float(len(colors) - 1)
            cmap.append((value, tuple(color)))
        return Colormap(*cmap)

    if isinstance(colors, str):
        from trollimage import colormap
        import copy
        return copy.copy(getattr(colormap, colors))

    return None


def _three_d_effect_delayed(band_data, kernel, mode):
    from scipy.signal import convolve2d
    band_data = band_data.reshape(band_data.shape[1:])
    new_data = convolve2d(band_data, kernel, mode=mode)
    return new_data.reshape((1, band_data.shape[0], band_data.shape[1]))


def three_d_effect(img, **kwargs):
    """Create 3D effect using convolution"""
    w = kwargs.get('weight', 1)
    LOG.debug("Applying 3D effect with weight %.2f", w)
    kernel = np.array([[-w, 0, w],
                       [-w, 1, w],
                       [-w, 0, w]])
    mode = kwargs.get('convolve_mode', 'same')

    def func(band_data, kernel=kernel, mode=mode, index=None):
        del index

        delay = dask.delayed(_three_d_effect_delayed)(band_data, kernel, mode)
        new_data = da.from_delayed(delay, shape=band_data.shape, dtype=band_data.dtype)
        return new_data

    return apply_enhancement(img.data, func, separate=True, pass_dask=True)


def btemp_threshold(img, min_in, max_in, threshold, threshold_out=None, **kwargs):
    """Scale data linearly in two separate regions.

    This enhancement scales the input data linearly by splitting the data
    into two regions; min_in to threshold and threshold to max_in. These
    regions are mapped to 1 to threshold_out and threshold_out to 0
    respectively, resulting in the data being "flipped" around the
    threshold. A default threshold_out is set to `176.0 / 255.0` to
    match the behavior of the US National Weather Service's forecasting
    tool called AWIPS.

    Args:
        img (XRImage): Image object to be scaled
        min_in (float): Minimum input value to scale
        max_in (float): Maximum input value to scale
        threshold (float): Input value where to split data in to two regions
        threshold_out (float): Output value to map the input `threshold`
            to. Optional, defaults to 176.0 / 255.0.

    """
    threshold_out = threshold_out if threshold_out is not None else (176 / 255.0)
    low_factor = (threshold_out - 1.) / (min_in - threshold)
    low_offset = 1. + (low_factor * min_in)
    high_factor = threshold_out / (max_in - threshold)
    high_offset = high_factor * max_in

    def _bt_threshold(band_data):
        # expects dask array to be passed
        return da.where(band_data >= threshold,
                        high_offset - high_factor * band_data,
                        low_offset - low_factor * band_data)

    return apply_enhancement(img.data, _bt_threshold, pass_dask=True)
