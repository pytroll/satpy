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

"""Compositors filling one composite with others."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import dask.array as da
import numpy as np
import xarray as xr

from satpy.dataset import combine_metadata

from .core import (
    GenericCompositor,
    SingleBandCompositor,
    add_bands,
    enhance2dataset,
)

LOG = logging.getLogger(__name__)


class FillingCompositor(GenericCompositor):
    """Make a regular RGB, filling the RGB bands with the first provided dataset's values."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        projectables[1] = projectables[1].fillna(projectables[0])
        projectables[2] = projectables[2].fillna(projectables[0])
        projectables[3] = projectables[3].fillna(projectables[0])
        return super(FillingCompositor, self).__call__(projectables[1:], **info)


class Filler(GenericCompositor):
    """Fix holes in projectable 1 with data from projectable 2."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        filled_projectable = projectables[0].fillna(projectables[1])
        return super(Filler, self).__call__([filled_projectable], **info)


class MultiFiller(SingleBandCompositor):
    """Fix holes in projectable 1 with data from the next projectables."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        filled_projectable = projectables[0]
        for next_projectable in projectables[1:]:
            filled_projectable = filled_projectable.fillna(next_projectable)
        if "optional_datasets" in info.keys():
            for next_projectable in info["optional_datasets"]:
                filled_projectable = filled_projectable.fillna(next_projectable)

        return super().__call__([filled_projectable], **info)


class DayNightCompositor(GenericCompositor):
    """A compositor that blends day data with night data.

    Using the `day_night` flag it is also possible to provide only a day product
    or only a night product and mask out (make transparent) the opposite portion
    of the image (night or day). See the documentation below for more details.
    """

    def __init__(self, name, lim_low=85., lim_high=88., day_night="day_night", include_alpha=True, **kwargs):  # noqa: D417
        """Collect custom configuration values.

        Args:
            lim_low (float): lower limit of Sun zenith angle for the
                             blending of the given channels
            lim_high (float): upper limit of Sun zenith angle for the
                             blending of the given channels
            day_night (str): "day_night" means both day and night portions will be kept
                                "day_only" means only day portion will be kept
                                "night_only" means only night portion will be kept
            include_alpha (bool): This only affects the "day only" or "night only" result.
                                  True means an alpha band will be added to the output image for transparency.
                                  False means the output is a single-band image with undesired pixels being masked out
                                  (replaced with NaNs).

        """
        self.lim_low = lim_low
        self.lim_high = lim_high
        self.day_night = day_night
        self.include_alpha = include_alpha
        self._has_sza = False
        super().__init__(name, **kwargs)

    def __call__(
            self,
            datasets: Sequence[xr.DataArray],
            optional_datasets: Optional[Sequence[xr.DataArray]] = None,
            **attrs
    ) -> xr.DataArray:
        """Generate the composite."""
        datasets = self.match_data_arrays(datasets)
        # At least one composite is requested.
        foreground_data = datasets[0]
        weights = self._get_coszen_blending_weights(datasets)
        # Apply enhancements to the foreground data
        foreground_data = enhance2dataset(foreground_data)

        if "only" in self.day_night:
            fg_attrs = foreground_data.attrs.copy()
            day_data, night_data, weights = self._get_data_for_single_side_product(foreground_data, weights)
        else:
            day_data, night_data, fg_attrs = self._get_data_for_combined_product(foreground_data, datasets[1])

        # The computed coszen is for the full area, so it needs to be masked for missing and off-swath data
        if self.include_alpha and not self._has_sza:
            weights = self._mask_weights_with_data(weights, day_data, night_data)

        if "only" not in self.day_night:
            # Replace missing channel data with zeros
            day_data = zero_missing_data(day_data, night_data)
            night_data = zero_missing_data(night_data, day_data)

        data = self._weight_data(day_data, night_data, weights, fg_attrs)

        return super(DayNightCompositor, self).__call__(
            data,
            optional_datasets=optional_datasets,
            **attrs
        )

    def _get_coszen_blending_weights(
            self,
            projectables: Sequence[xr.DataArray],
    ) -> xr.DataArray:
        lim_low = float(np.cos(np.deg2rad(self.lim_low)))
        lim_high = float(np.cos(np.deg2rad(self.lim_high)))
        try:
            coszen = np.cos(np.deg2rad(projectables[2 if self.day_night == "day_night" else 1]))
            self._has_sza = True
        except IndexError:
            from satpy.modifiers.angles import get_cos_sza
            LOG.debug("Computing sun zenith angles.")
            # Get chunking that matches the data
            coszen = get_cos_sza(projectables[0])
        # Calculate blending weights
        coszen -= min(lim_high, lim_low)
        coszen /= abs(lim_low - lim_high)
        return coszen.clip(0, 1)

    def _get_data_for_single_side_product(
            self,
            foreground_data: xr.DataArray,
            weights: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        # Only one portion (day or night) is selected. One composite is requested.
        # Add alpha band to single L/RGB composite to make the masked-out portion transparent when needed
        # L -> LA
        # RGB -> RGBA
        if self.include_alpha:
            foreground_data = add_alpha_bands(foreground_data)
        else:
            weights = self._mask_weights(weights)

        day_data, night_data = self._get_day_night_data_for_single_side_product(foreground_data)
        return day_data, night_data, weights

    def _mask_weights(self, weights):
        if "day" in self.day_night:
            return weights.where(weights != 0, np.nan)
        return weights.where(weights != 1, np.nan)

    def _get_day_night_data_for_single_side_product(self, foreground_data):
        if "day" in self.day_night:
            return foreground_data, foreground_data.dtype.type(0)
        return foreground_data.dtype.type(0), foreground_data

    def _get_data_for_combined_product(self, day_data, night_data):
        # Apply enhancements also to night-side data
        night_data = enhance2dataset(night_data)

        # Adjust bands so that they match
        # L/RGB -> RGB/RGB
        # LA/RGB -> RGBA/RGBA
        # RGB/RGBA -> RGBA/RGBA
        day_data = add_bands(day_data, night_data["bands"])
        night_data = add_bands(night_data, day_data["bands"])

        # Get merged metadata
        attrs = combine_metadata(day_data, night_data)

        return day_data, night_data, attrs

    def _mask_weights_with_data(
            self,
            weights: xr.DataArray,
            day_data: xr.DataArray,
            night_data: xr.DataArray,
    ) -> xr.DataArray:
        data_a = _get_single_channel(day_data)
        data_b = _get_single_channel(night_data)
        if "only" in self.day_night:
            mask = _get_weight_mask_for_single_side_product(data_a, data_b)
        else:
            mask = _get_weight_mask_for_daynight_product(weights, data_a, data_b)

        return weights.where(mask, np.nan)

    def _weight_data(
            self,
            day_data: xr.DataArray,
            night_data: xr.DataArray,
            weights: xr.DataArray,
            attrs: dict,
    ) -> list[xr.DataArray]:
        weights = self._fill_weights(weights)
        data = self._merge_bands_with_weights(day_data, night_data, weights, attrs)
        return data

    def _fill_weights(self, weights):
        if not self.include_alpha:
            fill = 1 if self.day_night == "night_only" else 0
            weights = weights.where(~np.isnan(weights), fill)
        return weights

    def _merge_bands_with_weights(self, day_data, night_data, weights, attrs):
        data = []
        for b in _get_band_names(day_data, night_data):
            day_band = _get_single_band_data(day_data, b)
            night_band = _get_single_band_data(night_data, b)
            # For day-only and night-only products only the alpha channel is weighted
            # If there's no alpha band, weight the actual data
            if self._is_weightable(b):
                day_band = day_band * weights
                night_band = night_band * (1 - weights)
            band = day_band + night_band
            band.attrs = attrs
            data.append(band)
        return data

    def _is_weightable(self, band):
        return (band == "A") or ("only" not in self.day_night) or not self.include_alpha


def _get_band_names(day_data, night_data):
    try:
        bands = day_data["bands"]
    except (IndexError, TypeError):
        bands = night_data["bands"]
    return bands


def _get_single_band_data(data, band):
    try:
        return data.sel(bands=band)
    except AttributeError:
        return data


def _get_single_channel(data: xr.DataArray) -> xr.DataArray:
    try:
        data = data[0, :, :]
        # remove coordinates that may be band-specific (ex. "bands")
        # and we don't care about anymore
        data = data.reset_coords(drop=True)
    except (IndexError, TypeError):
        pass
    return data


def _get_weight_mask_for_single_side_product(data_a, data_b):
    if data_b.shape:
        return ~da.isnan(data_b)
    return ~da.isnan(data_a)


def _get_weight_mask_for_daynight_product(weights, data_a, data_b):
    mask1 = (weights > 0) & ~np.isnan(data_a)
    mask2 = (weights < 1) & ~np.isnan(data_b)
    return mask1 | mask2


def add_alpha_bands(data):
    """Only used for DayNightCompositor.

    Add an alpha band to L or RGB composite as prerequisites for the following band matching
    to make the masked-out area transparent.
    """
    if "A" not in data["bands"].data:
        new_data = [data.sel(bands=band) for band in data["bands"].data]
        # Create alpha band based on a copy of the first "real" band
        alpha = new_data[0].copy()
        alpha.data = da.ones((data.sizes["y"],
                              data.sizes["x"]),
                             chunks=new_data[0].chunks,
                             dtype=data.dtype)
        # Rename band to indicate it's alpha
        alpha["bands"] = "A"
        new_data.append(alpha)
        new_data = xr.concat(new_data, dim="bands")
        new_data.attrs["mode"] = data.attrs["mode"] + "A"
        data = new_data
    return data


def zero_missing_data(data1, data2):
    """Replace NaN values with zeros in data1 if the data is valid in data2."""
    nans = np.logical_and(np.isnan(data1), np.logical_not(np.isnan(data2)))
    return data1.where(~nans, 0)


class BackgroundCompositor(GenericCompositor):
    """A compositor that overlays one composite on top of another.

    The output image mode will be determined by both foreground and background. Generally, when the background has
    an alpha band, the output image will also have one.

    ============  ============  ========
    Foreground     Background    Result
    ============  ============  ========
    L             L             L
    ------------  ------------  --------
    L             LA            LA
    ------------  ------------  --------
    L             RGB           RGB
    ------------  ------------  --------
    L             RGBA          RGBA
    ------------  ------------  --------
    LA            L             L
    ------------  ------------  --------
    LA            LA            LA
    ------------  ------------  --------
    LA            RGB           RGB
    ------------  ------------  --------
    LA            RGBA          RGBA
    ------------  ------------  --------
    RGB           L             RGB
    ------------  ------------  --------
    RGB           LA            RGBA
    ------------  ------------  --------
    RGB           RGB           RGB
    ------------  ------------  --------
    RGB           RGBA          RGBA
    ------------  ------------  --------
    RGBA          L             RGB
    ------------  ------------  --------
    RGBA          LA            RGBA
    ------------  ------------  --------
    RGBA          RGB           RGB
    ------------  ------------  --------
    RGBA          RGBA          RGBA
    ============  ============  ========

    """

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        projectables = self.match_data_arrays(projectables)

        # Get enhanced datasets
        foreground = enhance2dataset(projectables[0], convert_p=True)
        background = enhance2dataset(projectables[1], convert_p=True)
        before_bg_mode = background.attrs["mode"]

        # Adjust bands so that they have the same mode
        foreground = add_bands(foreground, background["bands"])
        background = add_bands(background, foreground["bands"])

        # It's important whether the alpha band of background is initially generated, e.g. by CloudCompositor
        # The result will be used to determine the output image mode
        initial_bg_alpha = "A" in before_bg_mode

        attrs = self._combine_metadata_with_mode_and_sensor(foreground, background)
        if "A" not in foreground.attrs["mode"] and "A" not in background.attrs["mode"]:
            data = self._simple_overlay(foreground, background)
        else:
            data = self._get_merged_image_data(foreground, background, initial_bg_alpha=initial_bg_alpha)
        for data_arr in data:
            data_arr.attrs = attrs
        res = super(BackgroundCompositor, self).__call__(data, **kwargs)
        attrs.update(res.attrs)
        res.attrs = attrs
        return res

    def _combine_metadata_with_mode_and_sensor(self,
                                               foreground: xr.DataArray,
                                               background: xr.DataArray
                                               ) -> dict:
        # Get merged metadata
        attrs = combine_metadata(foreground, background)
        # 'mode' is no longer valid after we've remove the 'A'
        # let the base class __call__ determine mode
        attrs.pop("mode", None)
        if attrs.get("sensor") is None:
            # sensor can be a set
            attrs["sensor"] = self._get_sensors([foreground, background])
        return attrs

    @staticmethod
    def _get_merged_image_data(foreground: xr.DataArray,
                               background: xr.DataArray,
                               initial_bg_alpha: bool,
                               ) -> list[xr.DataArray]:
        # For more info about alpha compositing please review https://en.wikipedia.org/wiki/Alpha_compositing
        alpha_fore = _get_alpha(foreground)
        alpha_back = _get_alpha(background)
        new_alpha = alpha_fore + alpha_back * (1 - alpha_fore)

        data = []

        # Pass the image data (alpha band will be dropped temporally) to the writer
        output_mode = background.attrs["mode"].replace("A", "")

        for band in output_mode:
            fg_band = foreground.sel(bands=band)
            bg_band = background.sel(bands=band)
            # Do the alpha compositing
            chan = (fg_band * alpha_fore + bg_band * alpha_back * (1 - alpha_fore)) / new_alpha
            # Fill the NaN area with background
            chan = xr.where(chan.isnull(), bg_band * alpha_back, chan)
            chan["bands"] = band
            data.append(chan)

        # If background has an initial alpha band, it will also be passed to the writer
        if initial_bg_alpha:
            new_alpha["bands"] = "A"
            data.append(new_alpha)

        return data

    @staticmethod
    def _simple_overlay(foreground: xr.DataArray,
                        background: xr.DataArray,) -> list[xr.DataArray]:
        # This is for the case when no alpha bands are involved
        # Just simply lay the foreground upon background
        data_arr = xr.where(foreground.isnull(), background, foreground)
        # Split to separate bands so the mode is correct
        data = [data_arr.sel(bands=b) for b in data_arr["bands"]]

        return data


def _get_alpha(dataset: xr.DataArray):
    # 1. This function is only used by _get_merged_image_data
    # 2. Both foreground and background have been through add_bands, so they have the same mode
    # 3. If none of them has alpha band, they will be passed to _simple_overlay not _get_merged_image_data
    # So any dataset(whether foreground or background) passed to this function has an alpha band for certain
    # We will use it directly
    alpha = dataset.sel(bands="A")
    # There could be NaNs in the alpha
    # Replace them with 0 to prevent cases like 1 + nan = nan, so they won't affect new_alpha
    alpha = xr.where(alpha.isnull(), 0, alpha)

    return alpha
