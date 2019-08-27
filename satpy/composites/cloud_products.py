#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2019 Satpy developers
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
"""Compositors for cloud products.
"""

import numpy as np
import xarray as xr

from satpy.composites import ColormapCompositor
from satpy.composites import GenericCompositor


class CloudTopHeightCompositor(ColormapCompositor):
    """Colorize with a palette, put cloud-free pixels as black."""

    @staticmethod
    def build_colormap(palette, info):
        """Create the colormap from the `raw_palette` and the valid_range."""

        from trollimage.colormap import Colormap
        if 'palette_meanings' in palette.attrs:
            palette_indices = palette.attrs['palette_meanings']
        else:
            palette_indices = range(len(palette))

        sqpalette = np.asanyarray(palette).squeeze() / 255.0
        tups = [(val, tuple(tup))
                for (val, tup) in zip(palette_indices, sqpalette)]
        colormap = Colormap(*tups)
        if 'palette_meanings' not in palette.attrs:
            sf = info.get('scale_factor', np.array(1))
            colormap.set_range(
                *(np.array(info['valid_range']) * sf + info.get('add_offset', 0)))

        return colormap, sqpalette

    def __call__(self, projectables, **info):
        """Create the composite."""
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables), ))
        data, palette, status = projectables
        colormap, palette = self.build_colormap(palette, data.attrs)
        channels, colors = colormap.palettize(np.asanyarray(data.squeeze()))
        channels = palette[channels]
        mask_nan = data.notnull()
        mask_cloud_free = (status + 1) % 2
        chans = []
        for idx in range(channels.shape[-1]):
            chan = xr.DataArray(channels[:, :, idx].reshape(data.shape),
                                dims=data.dims, coords=data.coords,
                                attrs=data.attrs).where(mask_nan)
            # Set cloud-free pixels as black
            chans.append(chan.where(mask_cloud_free, 0).where(status != status.attrs['_FillValue']))

        res = super(CloudTopHeightCompositor, self).__call__(chans, **data.attrs)
        res.attrs['_FillValue'] = np.nan
        return res


class PrecipCloudsRGB(GenericCompositor):

    def __call__(self, projectables, *args, **kwargs):
        """Make an RGB image out of the three probability categories of the NWCSAF precip product."""

        projectables = self.match_data_arrays(projectables)
        light = projectables[0]
        moderate = projectables[1]
        intense = projectables[2]
        status_flag = projectables[3]

        if np.bitwise_and(status_flag, 4).any():
            # AMSU is used
            maxs1 = 70
            maxs2 = 70
            maxs3 = 100
        else:
            # avhrr only
            maxs1 = 30
            maxs2 = 50
            maxs3 = 40

        scalef3 = 1.0 / maxs3 - 1 / 255.0
        scalef2 = 1.0 / maxs2 - 1 / 255.0
        scalef1 = 1.0 / maxs1 - 1 / 255.0

        p1data = (light*scalef1).where(light != 0)
        p1data = p1data.where(light != light.attrs['_FillValue'])
        p1data.attrs = light.attrs
        data = moderate*scalef2
        p2data = data.where(moderate != 0)
        p2data = p2data.where(moderate != moderate.attrs['_FillValue'])
        p2data.attrs = moderate.attrs
        data = intense*scalef3
        p3data = data.where(intense != 0)
        p3data = p3data.where(intense != intense.attrs['_FillValue'])
        p3data.attrs = intense.attrs

        res = super(PrecipCloudsRGB, self).__call__((p3data, p2data, p1data),
                                                    *args, **kwargs)
        return res
