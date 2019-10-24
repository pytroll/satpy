#!/usr/bin/env python
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
"""Enhancements."""

import numpy as np
import xarray as xr
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


def crefl_scaling(img, **kwargs):
    """Apply non-linear stretch used by CREFL-based RGBs."""
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
        band_data = np.log10(band_data)
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
    """Colorize the given image.

    Args:
        img: image to be colorized
    Kwargs:
        palettes: colormap(s) to use

    The `palettes` kwarg can be one of the following:
        - a trollimage.colormap.Colormap object
        - list of dictionaries with each of one of the following forms:
            - {'filename': '/path/to/colors.npy',
               'min_value': <float, min value to match colors to>,
               'max_value': <float, min value to match colors to>,
               'reverse': <bool, reverse the colormap if True (default: False)}
            - {'colors': <trollimage.colormap.Colormap instance>,
               'min_value': <float, min value to match colors to>,
               'max_value': <float, min value to match colors to>,
               'reverse': <bool, reverse the colormap if True (default: False)}
            - {'colors': <tuple of RGB(A) tuples>,
               'min_value': <float, min value to match colors to>,
               'max_value': <float, min value to match colors to>,
               'reverse': <bool, reverse the colormap if True (default: False)}
            - {'colors': <tuple of RGB(A) tuples>,
               'values': <tuple of values to match colors to>,
               'min_value': <float, min value to match colors to>,
               'max_value': <float, min value to match colors to>,
               'reverse': <bool, reverse the colormap if True (default: False)}

    If multiple palettes are supplied, they are concatenated before applied.

    """
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
            if full_cmap is None:
                full_cmap = cmap
            else:
                full_cmap = full_cmap + cmap

    return full_cmap


def create_colormap(palette):
    """Create colormap of the given numpy file, color vector, or colormap.

    Args:
        palette (dict): Information describing how to create a colormap
            object. See below for more details.

    **From a file**

    Colormaps can be loaded from ``.npy`` files as 2D raw arrays with rows for
    each color. The filename to load can be provided with the ``filename`` key
    in the provided palette information. The colormap is interpreted as 1 of 4
    different "colormap modes": ``RGB``, ``RGBA``, ``VRGB``, or ``VRGBA``. The
    colormap mode can be forced with the ``colormap_mode`` key in the provided
    palette information. If it is not provided then a default will be chosen
    based on the number of columns in the array (3: RGB, 4: VRGB, 5: VRGBA).

    The "V" in the possible colormap modes represents the control value of
    where that color should be applied. If "V" is not provided in the colormap
    data it defaults to the row index in the colormap array (0, 1, 2, ...)
    divided by the total number of colors to produce a number between 0 and 1.
    See the "Set Range" section below for more information.
    The remaining elements in the colormap array represent the Red (R),
    Green (G), and Blue (B) color to be mapped to.

    See the "Color Scale" section below for more information on the value
    range of provided numbers.

    **From a list**

    Colormaps can be loaded from lists of colors provided by the ``colors``
    key in the provided dictionary. Each element in the list represents a
    single color to be mapped to and can be 3 (RGB) or 4 (RGBA) elements long.
    By default the value or control point for a color is determined by the
    index in the list (0, 1, 2, ...) divided by the total number of colors
    to produce a number between 0 and 1. This can be overridden by providing a
    ``values`` key in the provided dictionary. See the "Set Range" section
    below for more information.

    See the "Color Scale" section below for more information on the value
    range of provided numbers.

    **From a builtin colormap**

    Colormaps can be loaded by name from the builtin colormaps in the
    ``trollimage``` package. Specify the name with the ``colors``
    key in the provided dictionary (ex. ``{'colors': 'blues'}``).
    See :doc:`trollimage:colormap` for the full list of available colormaps.

    **Color Scale**

    By default colors are expected to be in a 0-255 range. This
    can be overridden by specifying ``color_scale`` in the provided colormap
    information. A common alternative to 255 is ``1`` to specify floating
    point numbers between 0 and 1. The resulting Colormap uses the normalized
    color values (0-1).

    **Set Range**

    By default the control points or values of the Colormap are between 0 and
    1. This means that data values being mapped to a color must also be
    between 0 and 1. When this is not the case, the expected input range of
    the data can be used to configure the Colormap and change the control point
    values. To do this specify the input data range with ``min_value`` and
    ``max_value``. See :meth:`trollimage.colormap.Colormap.set_range` for more
    information.

    """
    from trollimage.colormap import Colormap
    fname = palette.get('filename', None)
    colors = palette.get('colors', None)
    # are colors between 0-255 or 0-1
    color_scale = palette.get('color_scale', 255)
    if fname:
        data = np.load(fname)
        cols = data.shape[1]
        default_modes = {
            3: 'RGB',
            4: 'VRGB',
            5: 'VRGBA'
        }
        default_mode = default_modes.get(cols)
        mode = palette.setdefault('colormap_mode', default_mode)
        if mode is None or len(mode) != cols:
            raise ValueError(
                "Unexpected colormap shape for mode '{}'".format(mode))

        rows = data.shape[0]
        if mode[0] == 'V':
            colors = data[:, 1:]
            if color_scale != 1:
                colors = data[:, 1:] / float(color_scale)
            values = data[:, 0]
        else:
            colors = data
            if color_scale != 1:
                colors = colors / float(color_scale)
            values = np.arange(rows) / float(rows - 1)
        cmap = Colormap(*zip(values, colors))
    elif isinstance(colors, (tuple, list)):
        cmap = []
        values = palette.get('values', None)
        for idx, color in enumerate(colors):
            if values is not None:
                value = values[idx]
            else:
                value = idx / float(len(colors) - 1)
            if color_scale != 1:
                color = tuple(elem / float(color_scale) for elem in color)
            cmap.append((value, tuple(color)))
        cmap = Colormap(*cmap)
    elif isinstance(colors, str):
        from trollimage import colormap
        import copy
        cmap = copy.copy(getattr(colormap, colors))
    else:
        raise ValueError("Unknown colormap format: {}".format(palette))

    if palette.get("reverse", False):
        cmap.reverse()
    if 'min_value' in palette and 'max_value' in palette:
        cmap.set_range(palette["min_value"], palette["max_value"])
    elif 'min_value' in palette or 'max_value' in palette:
        raise ValueError("Both 'min_value' and 'max_value' must be specified")

    return cmap


def _three_d_effect_delayed(band_data, kernel, mode):
    """Kernel for running delayed 3D effect creation."""
    from scipy.signal import convolve2d
    band_data = band_data.reshape(band_data.shape[1:])
    new_data = convolve2d(band_data, kernel, mode=mode)
    return new_data.reshape((1, band_data.shape[0], band_data.shape[1]))


def three_d_effect(img, **kwargs):
    """Create 3D effect using convolution."""
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
