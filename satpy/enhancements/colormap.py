# Copyright (c) 2017-2023 Satpy developers
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

"""Lookups, colorization and colormaps."""

from __future__ import annotations

import logging
import os

import numpy as np

from satpy._config import get_config_path
from satpy.utils import find_in_ancillary

from .wrappers import exclude_alpha, on_separate_bands, using_map_blocks

LOG = logging.getLogger(__name__)


def lookup(img, **kwargs):
    """Assign values to channels based on a table."""
    luts = np.array(kwargs["luts"], dtype=np.float32) / 255.0
    return _lookup_table(img.data, luts=luts)


@exclude_alpha
@on_separate_bands
@using_map_blocks
def _lookup_table(band_data, luts=None, index=-1):
    # NaN/null values will become 0
    lut = luts[:, index] if len(luts.shape) == 2 else luts
    band_data = band_data.clip(0, lut.size - 1).astype(np.uint8)
    return lut[band_data]


def colorize(img, **kwargs):  # noqa: D417
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
            - {'dataset': <str, referring to dataset containing palette>,
               'color_scale': <int, value to be interpreted as white>,
               'min_value': <float, see above>,
               'max_value': <float, see above>}

    If multiple palettes are supplied, they are concatenated before applied.

    """
    full_cmap = _merge_colormaps(kwargs, img)
    img.colorize(full_cmap)


def palettize(img, **kwargs):
    """Palettize the given image (no color interpolation).

    Arguments as for :func:`colorize`.

    NB: to retain the palette when saving the resulting image, pass
    ``keep_palette=True`` to the save method (either via the Scene class or
    directly in trollimage).
    """
    full_cmap = _merge_colormaps(kwargs, img)
    img.palettize(full_cmap)


def _merge_colormaps(kwargs, img=None):
    """Merge colormaps listed in kwargs."""
    from trollimage.colormap import Colormap
    full_cmap = None

    palette = kwargs["palettes"]
    if isinstance(palette, Colormap):
        full_cmap = palette
    else:
        for itm in palette:
            cmap = create_colormap(itm, img)
            if full_cmap is None:
                full_cmap = cmap
            else:
                full_cmap = full_cmap + cmap

    return full_cmap


def create_colormap(palette, img=None):  # noqa: D417
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
    By default, the value or control point for a color is determined by the
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

    **From an auxiliary variable**

    If the colormap is defined in the same dataset as the data to which the
    colormap shall be applied, this can be indicated with
    ``{'dataset': 'palette_variable'}``, where ``'palette_variable'`` is the
    name of the variable containing the palette.  This variable must be an
    auxiliary variable to the dataset to which the colours are applied.  When
    using this, it is important that one should **not** set ``min_value`` and
    ``max_value`` as those will be taken from the ``valid_range`` attribute
    on the dataset and if those differ from ``min_value`` and ``max_value``,
    the resulting colors will not match the ones in the palette.

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

    **Set Alpha Range**

    The alpha channel of a created colormap can be added and/or modified by
    specifying ``min_alpha`` and ``max_alpha``.
    See :meth:`trollimage.colormap.Colormap.set_alpha_range`  for more info.

    """
    # are colors between 0-255 or 0-1
    color_scale = palette.get("color_scale", 255)
    cmap = _get_cmap_from_palette_info(palette, img, color_scale)

    _reverse_cmap(cmap, palette)
    _set_cmap_range(cmap, palette)
    _set_cmap_alpha_range(cmap, palette, color_scale)

    return cmap


def _reverse_cmap(cmap, palette):
    if palette.get("reverse", False):
        cmap.reverse()


def _set_cmap_range(cmap, palette):
    if "min_value" in palette and "max_value" in palette:
        cmap.set_range(palette["min_value"], palette["max_value"])
    elif "min_value" in palette or "max_value" in palette:
        raise ValueError("Both 'min_value' and 'max_value' must be specified (or neither).")


def _set_cmap_alpha_range(cmap, palette, color_scale):
    if "min_alpha" in palette and "max_alpha" in palette:
        cmap.set_alpha_range(palette["min_alpha"] / color_scale,
                             palette["max_alpha"] / color_scale)
    elif "min_alpha" in palette or "max_alpha" in palette:
        raise ValueError("Both 'min_alpha' and 'max_alpha' must be specified (or neither).")


def _get_cmap_from_palette_info(palette, img, color_scale):
    from trollimage.colormap import Colormap

    fname = palette.get("filename", None)
    colors = palette.get("colors", None)
    dataset = palette.get("dataset", None)
    if fname:
        if not os.path.exists(fname):
            fname = get_config_path(fname)
        cmap = Colormap.from_file(fname, palette.get("colormap_mode", None), color_scale)
    elif isinstance(colors, (tuple, list)):
        cmap = Colormap.from_sequence_of_colors(colors, palette.get("values", None), color_scale)
    elif isinstance(colors, str):
        cmap = Colormap.from_name(colors)
    elif isinstance(dataset, str):
        cmap = _create_colormap_from_dataset(img, dataset, color_scale)
    else:
        raise ValueError("Unknown colormap format: {}".format(palette))
    return cmap



def _create_colormap_from_dataset(img, dataset, color_scale):
    """Create a colormap from an auxiliary variable in a source file."""
    from trollimage.colormap import Colormap

    match = find_in_ancillary(img.data, dataset)
    return Colormap.from_array_with_metadata(
        match, img.data.dtype, color_scale,
        valid_range=img.data.attrs.get("valid_range"),
        scale_factor=img.data.attrs.get("scale_factor", 1),
        add_offset=img.data.attrs.get("add_offset", 0),
        remove_last=False)
