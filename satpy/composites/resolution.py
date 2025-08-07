# Copyright (c) 2015-2025 Satpy developers
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

"""Sharpening compositors."""

from __future__ import annotations

import logging

import dask.array as da
import numpy as np
import xarray as xr

from .core import GenericCompositor, IncompatibleAreas, enhance2dataset

LOG = logging.getLogger(__name__)


class RatioSharpenedRGB(GenericCompositor):
    """Sharpen RGB bands with ratio of a high resolution band to a lower resolution version.

    Any pixels where the ratio is computed to be negative or infinity, it is
    reset to 1. Additionally, the ratio is limited to 1.5 on the high end to
    avoid high changes due to small discrepancies in instrument detector
    footprint. Note that the input data to this compositor must already be
    resampled so all data arrays are the same shape.

    Example::

        R_lo -  1000m resolution - shape=(2000, 2000)
        G - 1000m resolution - shape=(2000, 2000)
        B - 1000m resolution - shape=(2000, 2000)
        R_hi -  500m resolution - shape=(4000, 4000)

        ratio = R_hi / R_lo
        new_R = R_hi
        new_G = G * ratio
        new_B = B * ratio

    In some cases, there could be multiple high resolution bands::

        R_lo -  1000m resolution - shape=(2000, 2000)
        G_hi - 500m resolution - shape=(4000, 4000)
        B - 1000m resolution - shape=(2000, 2000)
        R_hi -  500m resolution - shape=(4000, 4000)

    To avoid the green band getting involved in calculating ratio or sharpening,
    add "neutral_resolution_band: green" in the YAML config file. This way
    only the blue band will get sharpened::

        ratio = R_hi / R_lo
        new_R = R_hi
        new_G = G_hi
        new_B = B * ratio

    """

    def __init__(self, *args, **kwargs):
        """Instanciate the ration sharpener."""
        self.high_resolution_color = kwargs.pop("high_resolution_band", "red")
        self.neutral_resolution_color = kwargs.pop("neutral_resolution_band", None)
        if self.high_resolution_color not in ["red", "green", "blue", None]:
            raise ValueError("RatioSharpenedRGB.high_resolution_band must "
                             "be one of ['red', 'green', 'blue', None]. Not "
                             "'{}'".format(self.high_resolution_color))
        if self.neutral_resolution_color not in ["red", "green", "blue", None]:
            raise ValueError("RatioSharpenedRGB.neutral_resolution_band must "
                             "be one of ['red', 'green', 'blue', None]. Not "
                             "'{}'".format(self.neutral_resolution_color))
        super(RatioSharpenedRGB, self).__init__(*args, **kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        """Sharpen low resolution datasets by multiplying by the ratio of ``high_res / low_res``.

        The resulting RGB has the units attribute removed.
        """
        if len(datasets) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(datasets), ))
        if not all(x.shape == datasets[0].shape for x in datasets[1:]) or \
                (optional_datasets and
                 optional_datasets[0].shape != datasets[0].shape):
            raise IncompatibleAreas("RatioSharpening requires datasets of "
                                    "the same size. Must resample first.")

        optional_datasets = tuple() if optional_datasets is None else optional_datasets
        datasets = self.match_data_arrays(datasets + optional_datasets)
        red, green, blue, new_attrs = self._get_and_sharpen_rgb_data_arrays_and_meta(datasets, optional_datasets)
        combined_info = self._combined_sharpened_info(info, new_attrs)
        res = super(RatioSharpenedRGB, self).__call__((red, green, blue,), **combined_info)
        res.attrs.pop("units", None)
        return res

    def _get_and_sharpen_rgb_data_arrays_and_meta(self, datasets, optional_datasets):
        new_attrs = {}
        low_res_red = datasets[0]
        low_res_green = datasets[1]
        low_res_blue = datasets[2]
        if optional_datasets and self.high_resolution_color is not None:
            LOG.debug("Sharpening image with high resolution {} band".format(self.high_resolution_color))
            high_res = datasets[3]
            if "rows_per_scan" in high_res.attrs:
                new_attrs.setdefault("rows_per_scan", high_res.attrs["rows_per_scan"])
            new_attrs.setdefault("resolution", high_res.attrs["resolution"])

        else:
            LOG.debug("No sharpening band specified for ratio sharpening")
            high_res = None

        bands = {"red": low_res_red, "green": low_res_green, "blue": low_res_blue}
        if high_res is not None:
            self._sharpen_bands_with_high_res(bands, high_res)

        return bands["red"], bands["green"], bands["blue"], new_attrs

    def _sharpen_bands_with_high_res(self, bands, high_res):
        ratio = da.map_blocks(
            _get_sharpening_ratio,
            high_res.data,
            bands[self.high_resolution_color].data,
            meta=np.array((), dtype=high_res.dtype),
            dtype=high_res.dtype,
            chunks=high_res.chunks,
        )

        bands[self.high_resolution_color] = high_res

        with xr.set_options(keep_attrs=True):
            for color in bands.keys():
                if color != self.neutral_resolution_color and color != self.high_resolution_color:
                    bands[color] = bands[color] * ratio

    def _combined_sharpened_info(self, info, new_attrs):
        combined_info = {}
        combined_info.update(info)
        combined_info.update(new_attrs)
        # Update that information with configured information (including name)
        combined_info.update(self.attrs)
        # Force certain pieces of metadata that we *know* to be true
        combined_info.setdefault("standard_name", "true_color")
        return combined_info


def _get_sharpening_ratio(high_res, low_res):
    with np.errstate(divide="ignore"):
        ratio = high_res / low_res
    # make ratio a no-op (multiply by 1) where the ratio is NaN, infinity,
    # or it is negative.
    ratio[~np.isfinite(ratio) | (ratio < 0)] = 1.0
    # we don't need ridiculously high ratios, they just make bright pixels
    np.clip(ratio, 0, 1.5, out=ratio)
    return ratio


def _mean4(data, offset=(0, 0), block_id=None):
    rows, cols = data.shape
    # we assume that the chunks except the first ones are aligned
    if block_id[0] == 0:
        row_offset = offset[0] % 2
    else:
        row_offset = 0
    if block_id[1] == 0:
        col_offset = offset[1] % 2
    else:
        col_offset = 0
    row_after = (row_offset + rows) % 2
    col_after = (col_offset + cols) % 2
    pad = ((row_offset, row_after), (col_offset, col_after))

    rows2 = rows + row_offset + row_after
    cols2 = cols + col_offset + col_after

    av_data = np.pad(data, pad, "edge")
    new_shape = (int(rows2 / 2.), 2, int(cols2 / 2.), 2)
    with np.errstate(invalid="ignore"):
        data_mean = np.nanmean(av_data.reshape(new_shape), axis=(1, 3))
    data_mean = np.repeat(np.repeat(data_mean, 2, axis=0), 2, axis=1)
    data_mean = data_mean[row_offset:row_offset + rows, col_offset:col_offset + cols]
    return data_mean


class SelfSharpenedRGB(RatioSharpenedRGB):
    """Sharpen RGB with ratio of a band with a strided-version of itself.

    Example::

        R -  500m resolution - shape=(4000, 4000)
        G - 1000m resolution - shape=(2000, 2000)
        B - 1000m resolution - shape=(2000, 2000)

        ratio = R / four_element_average(R)
        new_R = R
        new_G = G * ratio
        new_B = B * ratio

    """

    @staticmethod
    def four_element_average_dask(d):
        """Average every 4 elements (2x2) in a 2D array."""
        try:
            offset = d.attrs["area"].crop_offset
        except (KeyError, AttributeError):
            offset = (0, 0)

        res = d.data.map_blocks(_mean4, offset=offset, dtype=d.dtype, meta=np.ndarray((), dtype=d.dtype))
        return xr.DataArray(res, attrs=d.attrs, dims=d.dims, coords=d.coords)

    def __call__(self, datasets, optional_datasets=None, **attrs):
        """Generate the composite."""
        colors = ["red", "green", "blue"]
        if self.high_resolution_color not in colors:
            raise ValueError("SelfSharpenedRGB requires at least one high resolution band, not "
                             "'{}'".format(self.high_resolution_color))

        high_res = datasets[colors.index(self.high_resolution_color)]
        high_mean = self.four_element_average_dask(high_res)
        red = high_mean if self.high_resolution_color == "red" else datasets[0]
        green = high_mean if self.high_resolution_color == "green" else datasets[1]
        blue = high_mean if self.high_resolution_color == "blue" else datasets[2]
        return super(SelfSharpenedRGB, self).__call__((red, green, blue), optional_datasets=(high_res,), **attrs)


class LuminanceSharpeningCompositor(GenericCompositor):
    """Create a high resolution composite by sharpening a low resolution using high resolution luminance.

    This is done by converting to YCbCr colorspace, replacing Y, and convertin back to RGB.
    """

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        from trollimage.image import rgb2ycbcr, ycbcr2rgb
        projectables = self.match_data_arrays(projectables)
        luminance = projectables[0].copy()
        luminance /= 100.
        # Limit between min(luminance) ... 1.0
        luminance = da.where(luminance > 1., 1., luminance)

        # Get the enhanced version of the composite to be sharpened
        rgb_img = enhance2dataset(projectables[1])

        # This all will be eventually replaced with trollimage convert() method
        # ycbcr_img = rgb_img.convert('YCbCr')
        # ycbcr_img.data[0, :, :] = luminance
        # rgb_img = ycbcr_img.convert('RGB')

        # Replace luminance of the IR composite
        y__, cb_, cr_ = rgb2ycbcr(rgb_img.data[0, :, :],
                                  rgb_img.data[1, :, :],
                                  rgb_img.data[2, :, :])

        r__, g__, b__ = ycbcr2rgb(luminance, cb_, cr_)
        y_size, x_size = r__.shape
        r__ = da.reshape(r__, (1, y_size, x_size))
        g__ = da.reshape(g__, (1, y_size, x_size))
        b__ = da.reshape(b__, (1, y_size, x_size))

        rgb_img.data = da.vstack((r__, g__, b__))
        return super(LuminanceSharpeningCompositor, self).__call__(rgb_img, *args, **kwargs)


class SandwichCompositor(GenericCompositor):
    """Make a sandwich product."""

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        luminance = projectables[0]
        luminance = luminance / 100.
        # Limit between min(luminance) ... 1.0
        luminance = luminance.clip(max=1.)

        # Get the enhanced version of the RGB composite to be sharpened
        rgb_img = enhance2dataset(projectables[1])
        # Ignore alpha band when applying luminance
        rgb_img = rgb_img.where(rgb_img.bands == "A", rgb_img * luminance)
        return super(SandwichCompositor, self).__call__(rgb_img, *args, **kwargs)
