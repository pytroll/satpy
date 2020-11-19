#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Composite classes for the GLM instrument."""

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

    See the ``highlight_channel`` option to control if the effect is additive,
    subtractive, or not applied at all.

    """

    def __init__(self, name, min_highlight=0.0, max_highlight=10.0, max_factor=0.8,
                 highlight_channel=(True, True, False, None), **kwargs):
        """Initialize composite with highlight factor options.

        Args:
            min_highlight (float): Minimum raw value of the "highlight" data
                that will be used for linearly scaling the data along with
                ``max_hightlight``.
            max_highlight (float): Maximum raw value of the "highlight" data
                that will be used for linearly scaling the data along with
                ``min_hightlight``.
            max_factor (float): Maximum effect that the highlight data can
                have on the primary image data. This will be multiplied by
                the linearly scaled highlight data and then added or
                subtracted from the highlight channels. See class docstring
                for more information.
            highlight_channel (tuple): Series of booleans or None for every
                channel in the RGBA image (4). True means apply the highlight
                effect by adding, False means apply the highlight effect by
                subtracting, and None means don't apply the highlight. By
                default this will add to the Red and Green channels, subtract
                from the Blue channel, and not effect the Alpha channel. This
                results in yellow highlights in the resulting image.

        """
        self.min_highlight = min_highlight
        self.max_highlight = max_highlight
        self.max_factor = max_factor
        self.highlight_channel = highlight_channel
        super().__init__(name, **kwargs)

    def _get_highlight_factor(self, highlight_data):
        factor = (highlight_data - self.min_highlight) / (self.max_highlight - self.min_highlight)
        factor = factor.where(factor.notnull(), 0) * self.max_factor
        return factor

    def _apply_highlight_effect(self, background_data, factor):
        new_channels = []
        for highlight_effect, band_name in zip(self.highlight_channel, "RGBA"):
            if highlight_effect:
                channel_factor = factor
            elif highlight_effect is None:
                channel_factor = None
            else:
                channel_factor = -factor

            new_channel = background_data.sel(bands=[band_name])
            if channel_factor is not None:
                new_channel = new_channel + channel_factor
            new_channels.append(new_channel)
        return new_channels

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Create B/W image with highlighted pixels."""
        highlight_product, background_layer = self.match_data_arrays(projectables)

        # Enhance the background as normal (assume B/W image)
        attrs = background_layer.attrs
        img = get_enhanced_image(background_layer)
        # Clip image data to interval [0.0, 1.0]
        img.data = img.data.clip(0.0, 1.0)
        # Convert to RGBA so we can adjust the colors later
        img = img.convert('RGBA')
        background_data = img.data

        # Adjust the colors of background by highlight layer
        factor = self._get_highlight_factor(highlight_product)
        new_channels = self._apply_highlight_effect(background_data, factor)
        new_data = xr.concat(new_channels, dim='bands')
        new_data.attrs = attrs
        new_data.attrs['units'] = 1
        new_data.attrs.update(attrs)
        new_sensors = self._get_sensors((highlight_product, background_layer))
        new_data.attrs.update({
            'sensor': new_sensors,
        })

        return super(HighlightCompositor, self).__call__((new_data,), **attrs)
