#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Implementation of a composite that highlights on layer using another."""

import logging

import xarray as xr

from satpy.composites import GenericCompositor
from satpy.writers import get_enhanced_image

LOG = logging.getLogger(__name__)


class HighlightCompositor(GenericCompositor):
    """Highlight pixels of a layer by an amount determined by a secondary layer.

    The highlighting is applied per channel to either add or subtract an
    intensity from the primary image. In the addition case, the code is
    essentially doing::

        highlight_factor = (highlight_data - min_highlight) / (max_highlight - min_highlight)
        channel_result = primary_data + highlight_factor * max_factor

    The ``max_factor`` is defined per channel and can be positive for an
    additive effect, negative for a subtractive effect, or zero for no
    effect.

    """

    def __init__(self, name, min_highlight=0.0, max_highlight=10.0,
                 max_factor=(0.8, 0.8, -0.8, 0), **kwargs):
        """Initialize composite with highlight factor options.

        Args:
            min_highlight (float or tuple): Minimum raw value of the "highlight"
                data that will be used for linearly scaling the data along with
                ``max_highlight``.
                If a tuple, min highlight value will be scaled by the solar
                zenith angle between [0] and [1] as a linear function of SZA
                between [2] and [3].
                For example, ``(310, 290, 0, 90)`` will set min_highlight to
                310 for pixels with sun directly overhear and 290 with sun
                at the horizon.
            max_highlight (float): Maximum raw value of the "highlight" data
                that will be used for linearly scaling the data along with
                ``min_highlight``.
            max_factor (tuple): Maximum effect that the highlight data can
                have on each channel of the primary image data. This will be
                multiplied by the linearly scaled highlight data and then
                added or subtracted from the highlight channels. See class
                docstring for more information. By default this is set to
                ``(0.8, 0.8, -0.8, 0)`` meaning the Red and Green channel
                will be added to by at most 0.8, the Blue channel will be
                subtracted from by at most 0.8, and the Alpha channel will
                not be affected.

        """
        self.min_highlight = min_highlight
        self.max_highlight = max_highlight
        self.max_factor = max_factor
        super().__init__(name, **kwargs)

    @staticmethod
    def _get_enhanced_background_data(background_layer):
        img = get_enhanced_image(background_layer)
        img.data = img.data.clip(0.0, 1.0)
        img = img.convert('RGBA')
        return img.data

    def _get_highlight_factor(self, highlight_data):
        import numpy as np
        if type(self.min_highlight) is list:
            from pyorbital.astronomy import sun_zenith_angle as calc_sza
            LOG.debug("Computing sun zenith angles.")
            # Get chunking that matches the data
            try:
                chunks = highlight_data.sel(bands=highlight_data['bands'][0]).chunks
            except KeyError:
                chunks = highlight_data.chunks
            min_t = self.min_highlight[0]
            tdiff = self.min_highlight[1] - self.min_highlight[0]
            max_sza = self.min_highlight[3]
            min_sza = self.min_highlight[2]
            lons, lats = highlight_data.attrs["area"].get_lonlats(chunks=chunks)
            sza = xr.DataArray(calc_sza(highlight_data.attrs["start_time"],
                                        lons, lats),
                               dims=['y', 'x'],
                               coords=[highlight_data['y'], highlight_data['x']])
            sza = sza.where(sza > min_sza, min_sza)
            sza = sza.where(sza < max_sza, max_sza)

            highlighter = (sza - min_sza) / (max_sza - min_sza)
            min_highlight = min_t + tdiff * highlighter
            print(np.nanmean(sza), np.nanmean(highlighter), np.nanmean(min_highlight), min_t, tdiff, min_sza, max_sza)
        factor = (highlight_data - min_highlight) / (self.max_highlight - min_highlight)
        factor = factor.where(factor > 0, 0)
        factor = factor.where(factor.notnull(), 0)
        return factor

    def _apply_highlight_effect(self, background_data, factor):
        new_channels = []
        for max_factor, band_name in zip(self.max_factor, "RGBA"):
            new_channel = background_data.sel(bands=[band_name])
            if max_factor != 0 or max_factor is not None:
                new_channel = new_channel + factor * max_factor
            new_channels.append(new_channel)
        return new_channels

    def _update_attrs(self, new_data, background_layer, highlight_layer):
        new_data.attrs = background_layer.attrs.copy()
        new_data.attrs['units'] = 1
        new_sensors = self._get_sensors((highlight_layer, background_layer))
        new_data.attrs.update({
            'sensor': new_sensors,
        })

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Create RGBA image with highlighted pixels."""
        highlight_product, background_layer = self.match_data_arrays(projectables)
        background_data = self._get_enhanced_background_data(background_layer)

        # Adjust the colors of background by highlight layer
        factor = self._get_highlight_factor(highlight_product)
        new_channels = self._apply_highlight_effect(background_data, factor)
        new_data = xr.concat(new_channels, dim='bands')
        self._update_attrs(new_data, background_layer,
                           highlight_product)
        return super(HighlightCompositor, self).__call__((new_data,), **attrs)
