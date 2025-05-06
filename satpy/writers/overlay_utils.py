# Copyright (c) 2015-2023 Satpy developers
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
"""Helpers for adding overlays and decorations to images."""
from __future__ import annotations

import logging
import warnings

import numpy as np
import xarray as xr
from dask import array as da

from satpy.utils import get_legacy_chunk_size

LOG = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()


def _burn_overlay(img, image_metadata, area, cw_, overlays):
    """Burn the overlay in the image array."""
    del image_metadata
    cw_.add_overlay_from_dict(overlays, area, background=img)
    return img


def add_overlay(orig_img, area, coast_dir, color=None, width=None, resolution=None,
                level_coast=None, level_borders=None, fill_value=None,
                grid=None, overlays=None):
    """Add coastline, political borders and grid(graticules) to image.

    Uses ``color`` for feature colors where ``color`` is a 3-element tuple
    of integers between 0 and 255 representing (R, G, B).

    .. warning::

        This function currently loses the data mask (alpha band).

    ``resolution`` is chosen automatically if None (default),
    otherwise it should be one of:

    +-----+-------------------------+---------+
    | 'f' | Full resolution         | 0.04 km |
    +-----+-------------------------+---------+
    | 'h' | High resolution         | 0.2 km  |
    +-----+-------------------------+---------+
    | 'i' | Intermediate resolution | 1.0 km  |
    +-----+-------------------------+---------+
    | 'l' | Low resolution          | 5.0 km  |
    +-----+-------------------------+---------+
    | 'c' | Crude resolution        | 25  km  |
    +-----+-------------------------+---------+

    ``grid`` is a dictionary with key values as documented in detail in pycoast

    eg. overlay={'grid': {'major_lonlat': (10, 10),
                          'write_text': False,
                          'outline': (224, 224, 224),
                          'width': 0.5}}

    Here major_lonlat is plotted every 10 deg for both longitude and latitude,
    no labels for the grid lines are plotted, the color used for the grid lines
    is light gray, and the width of the gratucules is 0.5 pixels.

    For grid if aggdraw is used, font option is mandatory, if not
    ``write_text`` is set to False::

        font = aggdraw.Font('black', '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',
                            opacity=127, size=16)

    """
    if area is None:
        raise ValueError("Area of image is None, can't add overlay.")

    from pycoast import ContourWriterAGG
    from pyresample.area_config import get_area_def

    if isinstance(area, str):
        area = get_area_def(area)
    LOG.info("Add coastlines and political borders to image.")

    old_args = [color, width, resolution, grid, level_coast, level_borders]
    if any(arg is not None for arg in old_args):
        warnings.warn(
            "'color', 'width', 'resolution', 'grid', 'level_coast', 'level_borders'"
            " arguments will be deprecated soon. Please use 'overlays' instead.",
            DeprecationWarning,
            stacklevel=2
        )
    if hasattr(orig_img, "convert"):
        # image must be in RGB space to work with pycoast/pydecorate
        res_mode = ("RGBA" if orig_img.final_mode(fill_value).endswith("A") else "RGB")
        orig_img = orig_img.convert(res_mode)
    elif not orig_img.mode.startswith("RGB"):
        raise RuntimeError("'trollimage' 1.6+ required to support adding "
                           "overlays/decorations to non-RGB data.")

    if overlays is None:
        overlays = _create_overlays_dict(color, width, grid, level_coast, level_borders)

    cw_ = ContourWriterAGG(coast_dir)
    new_image = orig_img.apply_pil(_burn_overlay, res_mode,
                                   None, {"fill_value": fill_value},
                                   (area, cw_, overlays), None)
    return new_image


def _create_overlays_dict(color, width, grid, level_coast, level_borders):
    """Fill in the overlays dict."""
    overlays = dict()
    # fill with sensible defaults
    general_params = {"outline": color or (0, 0, 0),
                      "width": width or 0.5}
    for key, val in general_params.items():
        if val is not None:
            overlays.setdefault("coasts", {}).setdefault(key, val)
            overlays.setdefault("borders", {}).setdefault(key, val)
    if level_coast is None:
        level_coast = 1
    overlays.setdefault("coasts", {}).setdefault("level", level_coast)
    if level_borders is None:
        level_borders = 1
    overlays.setdefault("borders", {}).setdefault("level", level_borders)
    if grid is not None:
        if "major_lonlat" in grid and grid["major_lonlat"]:
            major_lonlat = grid.pop("major_lonlat")
            minor_lonlat = grid.pop("minor_lonlat", major_lonlat)
            grid.update({"Dlonlat": major_lonlat, "dlonlat": minor_lonlat})
        for key, val in grid.items():
            overlays.setdefault("grid", {}).setdefault(key, val)
    return overlays


def add_text(orig, dc, img, text):
    """Add text to an image using the pydecorate package.

    All the features of pydecorate's ``add_text`` are available.
    See documentation of :doc:`pydecorate:index` for more info.

    """
    from trollimage.xrimage import XRImage

    LOG.info("Add text to image.")

    dc.add_text(**text)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    new_data = xr.DataArray(arr, dims=["y", "x", "bands"],
                            coords={"y": orig.data.coords["y"],
                                    "x": orig.data.coords["x"],
                                    "bands": list(img.mode)},
                            attrs=orig.data.attrs)
    return XRImage(new_data)


def add_logo(orig, dc, img, logo):
    """Add logos or other images to an image using the pydecorate package.

    All the features of pydecorate's ``add_logo`` are available.
    See documentation of :doc:`pydecorate:index` for more info.

    """
    from trollimage.xrimage import XRImage

    LOG.info("Add logo to image.")

    dc.add_logo(**logo)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    new_data = xr.DataArray(arr, dims=["y", "x", "bands"],
                            coords={"y": orig.data.coords["y"],
                                    "x": orig.data.coords["x"],
                                    "bands": list(img.mode)},
                            attrs=orig.data.attrs)
    return XRImage(new_data)


def add_scale(orig, dc, img, scale):
    """Add scale to an image using the pydecorate package.

    All the features of pydecorate's ``add_scale`` are available.
    See documentation of :doc:`pydecorate:index` for more info.

    """
    from trollimage.xrimage import XRImage

    LOG.info("Add scale to image.")

    dc.add_scale(**scale)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    new_data = xr.DataArray(arr, dims=["y", "x", "bands"],
                            coords={"y": orig.data.coords["y"],
                                    "x": orig.data.coords["x"],
                                    "bands": list(img.mode)},
                            attrs=orig.data.attrs)
    return XRImage(new_data)


def add_decorate(orig, fill_value=None, **decorate):
    """Decorate an image with text and/or logos/images.

    This call adds text/logos in order as given in the input to keep the
    alignment features available in pydecorate.

    An example of the decorate config::

        decorate = {
            'decorate': [
                {'logo': {'logo_path': <path to a logo>, 'height': 143, 'bg': 'white', 'bg_opacity': 255}},
                {'text': {'txt': start_time_txt,
                          'align': {'top_bottom': 'bottom', 'left_right': 'right'},
                          'font': <path to ttf font>,
                          'font_size': 22,
                          'height': 30,
                          'bg': 'black',
                          'bg_opacity': 255,
                          'line': 'white'}}
            ]
        }

    Any numbers of text/logo in any order can be added to the decorate list,
    but the order of the list is kept as described above.

    Note that a feature given in one element, eg. bg (which is the background color)
    will also apply on the next elements  unless a new value is given.

    align is a special keyword telling where in the image to start adding features, top_bottom is either top or bottom
    and left_right is either left or right.
    """
    LOG.info("Decorate image.")

    # Need to create this here to possible keep the alignment
    # when adding text and/or logo with pydecorate
    if hasattr(orig, "convert"):
        # image must be in RGB space to work with pycoast/pydecorate
        orig = orig.convert("RGBA" if orig.mode.endswith("A") else "RGB")
    elif not orig.mode.startswith("RGB"):
        raise RuntimeError("'trollimage' 1.6+ required to support adding "
                           "overlays/decorations to non-RGB data.")
    img_orig = orig.pil_image(fill_value=fill_value)
    from pydecorate import DecoratorAGG
    dc = DecoratorAGG(img_orig)

    # decorate need to be a list to maintain the alignment
    # as ordered in the list
    img = orig
    if "decorate" in decorate:
        for dec in decorate["decorate"]:
            if "logo" in dec:
                img = add_logo(img, dc, img_orig, logo=dec["logo"])
            elif "text" in dec:
                img = add_text(img, dc, img_orig, text=dec["text"])
            elif "scale" in dec:
                img = add_scale(img, dc, img_orig, scale=dec["scale"])
    return img
