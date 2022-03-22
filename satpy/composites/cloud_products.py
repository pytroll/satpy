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
"""Compositors for cloud products."""

import numpy as np

from satpy.composites import ColormapCompositor, GenericCompositor


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

        squeezed_palette = np.asanyarray(palette).squeeze() / 255.0
        tups = [(val, tuple(tup))
                for (val, tup) in zip(palette_indices, squeezed_palette)]
        colormap = Colormap(*tups)
        if 'palette_meanings' not in palette.attrs:
            sf = info.get('scale_factor', np.array(1))
            colormap.set_range(
                *(np.array(info['valid_range']) * sf + info.get('add_offset', 0)))

        return colormap, squeezed_palette

    def __call__(self, projectables, **info):
        """Create the composite."""
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables), ))
        data, palette, status = projectables
        fill_value_color = palette.attrs.get("fill_value_color", [0, 0, 0])
        colormap, palette = self.build_colormap(palette, data.attrs)
        mapped_channels = colormap.colorize(data.data)
        valid = status != status.attrs['_FillValue']
        # cloud-free pixels are marked invalid (fill_value in ctth_alti) but have status set to 1.
        status_not_cloud_free = status % 2 == 0
        not_cloud_free = np.logical_or(status_not_cloud_free, np.logical_not(valid))

        channels = []
        for (channel, cloud_free_color) in zip(mapped_channels, fill_value_color):
            channel_data = self._create_masked_dataarray_like(channel, data, valid)
            # Set cloud-free pixels as fill_value_color
            channels.append(channel_data.where(not_cloud_free, cloud_free_color))

        res = GenericCompositor.__call__(self, channels, **data.attrs)
        res.attrs['_FillValue'] = np.nan
        return res


class PrecipCloudsRGB(GenericCompositor):
    """Precipitation clouds compositor."""

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
