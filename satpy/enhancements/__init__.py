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

import logging
import os
import warnings
from functools import partial
from numbers import Number

import dask
import dask.array as da
import numpy as np
import xarray as xr
from trollimage.xrimage import XRImage

from satpy._compat import ArrayLike
from satpy._config import get_config_path

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
    warnings.warn("'crefl_scaling' is deprecated, use 'piecewise_linear_stretch' instead.", DeprecationWarning)
    img.data.data = img.data.data / 100
    return piecewise_linear_stretch(img, xp=kwargs['idx'], fp=kwargs['sc'], reference_scale_factor=255)


def piecewise_linear_stretch(
        img: XRImage,
        xp: ArrayLike,
        fp: ArrayLike,
        reference_scale_factor: Number = None,
        **kwargs) -> xr.DataArray:
    """Apply 1D linear interpolation.

    This uses :func:`numpy.interp` mapped over the provided dask array chunks.

    Args:
        img: Image data to be scaled. It is assumed the data is already
            normalized between 0 and 1.
        xp: Input reference values of the image data points used for
            interpolation. This is passed directly to :func:`numpy.interp`.
        fp: Target reference values of the output image data points used for
            interpolation. This is passed directly to :func:`numpy.interp`.
        reference_scale_factor: Divide ``xp`` and ``fp`` by this value before
            using them for interpolation. This is a convenience to make
            matching normalized image data to interp coordinates or to avoid
            floating point precision errors in YAML configuration files.
            If not provided, ``xp`` and ``fp`` will not be modified.

    Examples:
        This example YAML uses a 'crude' stretch to pre-scale the RGB data
        and then uses reference points in a 0-255 range.

        .. code-block:: yaml

              true_color_linear_interpolation:
                sensor: abi
                standard_name: true_color
                operations:
                - name: reflectance_range
                  method: !!python/name:satpy.enhancements.stretch
                  kwargs: {stretch: 'crude', min_stretch: 0., max_stretch: 100.}
                - name: Linear interpolation
                  method: !!python/name:satpy.enhancements.piecewise_linear_stretch
                  kwargs:
                   xp: [0., 25., 55., 100., 255.]
                   fp: [0., 90., 140., 175., 255.]
                   reference_scale_factor: 255

        This example YAML does the same as the above on the C02 channel, but
        the interpolation reference points are already adjusted for the input
        reflectance (%) data and the output range (0 to 1).

        .. code-block:: yaml

              c02_linear_interpolation:
                sensor: abi
                standard_name: C02
                operations:
                - name: Linear interpolation
                  method: !!python/name:satpy.enhancements.piecewise_linear_stretch
                  kwargs:
                   xp: [0., 9.8039, 21.5686, 39.2157, 100.]
                   fp: [0., 0.3529, 0.5490, 0.6863, 1.0]

    """
    LOG.debug("Applying the piecewise_linear_stretch")
    if reference_scale_factor is not None:
        xp = np.asarray(xp) / reference_scale_factor
        fp = np.asarray(fp) / reference_scale_factor

    def func(band_data, xp, fp, index=None):
        # Interpolate band on [0,1] using "lazy" arrays (put calculations off until the end).
        band_data = xr.DataArray(da.clip(band_data.data.map_blocks(np.interp, xp=xp, fp=fp), 0, 1),
                                 coords=band_data.coords, dims=band_data.dims, name=band_data.name,
                                 attrs=band_data.attrs)
        return band_data

    func_with_kwargs = partial(func, xp=xp, fp=fp)
    return apply_enhancement(img.data, func_with_kwargs, separate=True)


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


def reinhard_to_srgb(img, saturation=1.25, white=100, **kwargs):
    """Stretch method based on the Reinhard algorithm, using luminance.

    Args:
        saturation: Saturation enhancement factor. Less is grayer. Neutral is 1.
        white: the reflectance luminance to set to white (in %).


    Reinhard, Erik & Stark, Michael & Shirley, Peter & Ferwerda, James. (2002).
    Photographic Tone Reproduction For Digital Images. ACM Transactions on Graphics.
    :doi: `21. 10.1145/566654.566575`
    """
    with xr.set_options(keep_attrs=True):
        # scale the data to [0, 1] interval
        rgb = img.data / 100
        white /= 100

        # extract color components
        r = rgb.sel(bands='R').data
        g = rgb.sel(bands='G').data
        b = rgb.sel(bands='B').data

        # saturate
        luma = _compute_luminance_from_rgb(r, g, b)
        rgb = (luma + (rgb - luma) * saturation).clip(0)

        # reinhard
        reinhard_luma = (luma / (1 + luma)) * (1 + luma/(white**2))
        coef = reinhard_luma / luma
        rgb = rgb * coef

        # srgb gamma
        rgb.data = _srgb_gamma(rgb.data)
        img.data = rgb

    return img.data


def _compute_luminance_from_rgb(r, g, b):
    """Compute the luminance of the image."""
    return r * 0.2126 + g * 0.7152 + b * 0.0722


def _srgb_gamma(arr):
    """Apply the srgb gamma."""
    return da.where(arr < 0.0031308, arr * 12.92, 1.055 * arr ** 0.41666 - 0.055)


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

    Colormaps can be loaded from ``.npy``, ``.npz``, or comma-separated text
    files. Numpy (npy/npz) files should be 2D arrays with rows for each color.
    Comma-separated files should have a row for each color with each column
    representing a single value/channel. The filename to load can be provided
    with the ``filename`` key in the provided palette information. A filename
    ending with ``.npy`` or ``.npz`` is read as a numpy file with
    :func:`numpy.load`. All other extensions are
    read as a comma-separated file. For ``.npz`` files the data must be stored
    as a positional list where the first element represents the colormap to
    use. See :func:`numpy.savez` for more information. The path to the
    colormap can be relative if it is stored in a directory specified by
    :ref:`config_path_setting`. Otherwise it should be an absolute path.

    The colormap is interpreted as 1 of 4 different "colormap modes":
    ``RGB``, ``RGBA``, ``VRGB``, or ``VRGBA``. The
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
    fname = palette.get('filename', None)
    colors = palette.get('colors', None)
    # are colors between 0-255 or 0-1
    color_scale = palette.get('color_scale', 255)
    if fname:
        cmap = _create_colormap_from_file(fname, palette, color_scale)
    elif isinstance(colors, (tuple, list)):
        cmap = _create_colormap_from_sequence(colors, palette, color_scale)
    elif isinstance(colors, str):
        import copy

        from trollimage import colormap
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


def _create_colormap_from_sequence(colors, palette, color_scale):
    from trollimage.colormap import Colormap
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
    return Colormap(*cmap)


def _create_colormap_from_file(filename, palette, color_scale):
    from trollimage.colormap import Colormap
    data = _read_colormap_data_from_file(filename)
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
    return Colormap(*zip(values, colors))


def _read_colormap_data_from_file(filename):
    if not os.path.exists(filename):
        filename = get_config_path(filename)
    ext = os.path.splitext(filename)[1]
    if ext in (".npy", ".npz"):
        file_content = np.load(filename)
        if ext == ".npz":
            # .npz is a collection
            # assume position list-like and get the first element
            file_content = file_content["arr_0"]
        return file_content
    # CSV
    return np.loadtxt(filename, delimiter=",")


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
