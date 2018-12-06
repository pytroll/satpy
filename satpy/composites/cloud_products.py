#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2018 PyTroll developers

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Compositors for cloud products.
"""

import numpy as np
import xarray as xr

from satpy.composites import ColormapCompositor


class CloudTopHeightCompositor(ColormapCompositor):
    """Colorize with a palette, put cloud-free pixels as black."""

    @staticmethod
    def build_colormap(palette, info):
        """Create the colormap from the `raw_palette` and the valid_range."""

        from trollimage.colormap import Colormap

        palette = np.asanyarray(palette).squeeze()
        tups = [(val, tuple(tup))
                for (val, tup) in enumerate(palette)]
        colormap = Colormap(*tups)

        sf = info.get('scale_factor', np.array(1))
        colormap.set_range(
            *(np.array(info['valid_range']) * sf + info.get('add_offset', 0)))

        return colormap

    def __call__(self, projectables, **info):
        """Create the composite."""
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" %
                             (len(projectables), ))
        data, palette, status = projectables
        palette = np.asanyarray(palette).squeeze() / 255.0
        colormap = self.build_colormap(palette, data.attrs)
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
            chans.append(chan.where(mask_cloud_free, 0))

        res = super(CloudTopHeightCompositor, self).__call__(chans, **data.attrs)
        res.attrs['_FillValue'] = np.nan
        return res
