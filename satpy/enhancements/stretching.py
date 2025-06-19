# Copyright (c) 2017-2025 Satpy developers
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
"""Stretching."""
from __future__ import annotations

import logging
import typing
from collections import namedtuple
from numbers import Number
from typing import Optional

import dask.array as da
import numpy as np
import xarray as xr

from satpy._compat import ArrayLike

from .wrappers import exclude_alpha, on_dask_array, using_map_blocks

if typing.TYPE_CHECKING:
    from trollimage.xrimage import XRImage

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


def piecewise_linear_stretch(  # noqa: D417
        img: XRImage,
        xp: ArrayLike,
        fp: ArrayLike,
        reference_scale_factor: Optional[Number] = None,
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

    return _piecewise_linear(img.data, xp=xp, fp=fp)


@exclude_alpha
@using_map_blocks
def _piecewise_linear(band_data, xp, fp):
    # Interpolate band on [0,1] using "lazy" arrays (put calculations off until the end).
    interp_data = np.interp(band_data, xp=xp, fp=fp)
    interp_data = np.clip(interp_data, 0, 1, out=interp_data)
    return interp_data


def cira_stretch(img, **kwargs):
    """Logarithmic stretch adapted to human vision.

    Applicable only for visible channels.
    """
    LOG.debug("Applying the cira-stretch")
    return _cira_stretch(img.data)


@exclude_alpha
def _cira_stretch(band_data):
    dtype = band_data.dtype
    log_root = np.log10(0.0223, dtype=dtype)
    denom = (1.0 - log_root) * 0.75
    band_data *= 0.01
    band_data = band_data.clip(np.finfo(float).eps)
    band_data = np.log10(band_data, dtype=dtype)
    band_data -= log_root
    band_data /= denom
    return band_data


def reinhard_to_srgb(img, saturation=1.25, white=100, **kwargs):  # noqa: D417
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
        r = rgb.sel(bands="R").data
        g = rgb.sel(bands="G").data
        b = rgb.sel(bands="B").data

        # saturate
        luma = _compute_luminance_from_rgb(r, g, b)
        rgb = (luma + (rgb - luma) * saturation).clip(0)

        # reinhard
        reinhard_luma = (luma / (1 + luma)) * (1 + luma / (white ** 2))
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


def btemp_threshold(img, min_in, max_in, threshold, threshold_out=None, **kwargs):  # noqa: D417
    """Scale data linearly in two separate regions.

    This enhancement scales the input data linearly by splitting the data
    into two regions; min_in to threshold and threshold to max_in. These
    regions are mapped to 1 to threshold_out and threshold_out to 0
    respectively, resulting in the data being "flipped" around the
    threshold. A default threshold_out is set to `176.0 / 255.0` to
    match the behavior of the US National Weather Service's forecasting
    tool called AWIPS.

    Args:
        img (trollimage.xrimage.XRImage): Image object to be scaled
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

    Coeffs = namedtuple("Coeffs", "factor offset")
    high = Coeffs(high_factor, high_offset)
    low = Coeffs(low_factor, low_offset)

    return _bt_threshold(img.data,
                         threshold=threshold,
                         high_coeffs=high,
                         low_coeffs=low)


@exclude_alpha
@using_map_blocks
def _bt_threshold(band_data, threshold, high_coeffs, low_coeffs):
    # expects dask array to be passed
    return np.where(band_data >= threshold,
                    high_coeffs.offset - high_coeffs.factor * band_data,
                    low_coeffs.offset - low_coeffs.factor * band_data)


def jma_true_color_reproduction(img):
    """Apply CIE XYZ matrix and return True Color Reproduction data.

    Himawari-8 True Color Reproduction Approach Based on the CIE XYZ Color System
    Hidehiko MURATA, Kotaro SAITOH, and Yasuhiko SUMIDA
    Meteorological Satellite Center, Japan Meteorological Agency
    NOAA National Environmental Satellite, Data, and Information Service
    Colorado State University—CIRA
    https://www.jma.go.jp/jma/jma-eng/satellite/introduction/TCR.html
    """
    _jma_true_color_reproduction(img.data,
                                 platform=img.data.attrs["platform_name"])


@exclude_alpha
@on_dask_array
def _jma_true_color_reproduction(img_data, platform=None):
    """Convert from AHI RGB space to sRGB space.

    The conversion matrices for this are supplied per-platform.
    The matrices are computed using the method described in the paper:
    'True Color Imagery Rendering for Himawari-8 with a Color Reproduction Approach
    Based on the CIE XYZ Color System' (:doi:`10.2151/jmsj.2018-049`).

    """
    # Conversion matrix dictionaries specifying sensor and platform.
    ccm_dict = {"himawari-8": np.array([[1.1629, 0.1539, -0.2175],
                                        [-0.0252, 0.8725, 0.1300],
                                        [-0.0204, -0.1100, 1.0633]]),

                "himawari-9": np.array([[1.1619, 0.1542, -0.2168],
                                        [-0.0271, 0.8749, 0.1295],
                                        [-0.0202, -0.1103, 1.0634]]),

                "goes-16": np.array([[1.1425, 0.1819, -0.2250],
                                     [-0.0951, 0.9363, 0.1360],
                                     [-0.0113, -0.1179, 1.0621]]),
                "goes-17": np.array([[1.1437, 0.1818, -0.2262],
                                     [-0.0952, 0.9354, 0.1371],
                                     [-0.0113, -0.1178, 1.0620]]),
                "goes-18": np.array([[1.1629, 0.1539, -0.2175],
                                     [-0.0252, 0.8725, 0.1300],
                                     [-0.0204, -0.1100, 1.0633]]),
                "goes-19": np.array([[0.9481, 0.3706, -0.2194],
                                     [-0.0150, 0.8605, 0.1317],
                                     [-0.0174, -0.1009, 1.0512]]),

                "mtg-i1": np.array([[0.9007, 0.2086, -0.0100],
                                    [-0.0475, 1.0662, -0.0414],
                                    [-0.0123, -0.1342, 1.0794]]),

                "geo-kompsat-2a": np.array([[1.1661, 0.1489, -0.2157],
                                            [-0.0255, 0.8745, 0.1282],
                                            [-0.0205, -0.1103, 1.0637]]),
                }

    # A conversion matrix, sensor name and platform name is required
    if platform is None:
        raise ValueError("Missing platform name.")

    # Get the satellite-specific conversion matrix
    try:
        ccm = ccm_dict[platform.lower()]
    except KeyError:
        raise KeyError(f"No conversion matrix found for platform {platform}")

    output = da.dot(img_data.T, ccm.T)
    return output.T
