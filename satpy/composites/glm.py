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
    """Highlight pixels of a layer by an amount determined by a secondary layer."""

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
        min_hightlight = 0
        max_hightlight = 10
        max_highlight_change = 0.8  # maximum of a 5% difference in pixel value
        factor = (highlight_product - min_hightlight) / (max_hightlight - min_hightlight)
        factor = factor.where(factor.notnull(), 0)
        new_r = background_data.sel(bands=['R']) + factor * max_highlight_change
        new_g = background_data.sel(bands=['G']) + factor * max_highlight_change
        new_b = background_data.sel(bands=['B']) - factor * max_highlight_change
        new_a = background_data.sel(bands=['A'])
        new_data = xr.concat((new_r, new_g, new_b, new_a), dim='bands')
        new_data.attrs = attrs
        new_sensors = self._get_sensors((highlight_product, background_layer))
        new_data.attrs['units'] = 1
        new_data.attrs.update(attrs)
        new_data.attrs.update({
            'sensor': new_sensors,
        })

        return super(HighlightCompositor, self).__call__((new_data,), **attrs)
