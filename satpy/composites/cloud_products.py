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

from satpy.composites import GenericCompositor, SingleBandCompositor


class CloudCompositorWithoutCloudfree(SingleBandCompositor):
    """Put cloud-free pixels as fill_value_color in palette."""

    def __call__(self, projectables, **info):
        """Create the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))
        data, status = projectables
        valid = status != status.attrs['_FillValue']
        status_cloud_free = status % 2 == 1  # bit 0 is set
        cloud_free = np.logical_and(valid, status_cloud_free)
        if "bad_optical_conditions" in status.attrs.get("flag_meanings", "") and data.name == "cmic_cre":
            bad_optical_conditions = np.bitwise_and(np.right_shift(status, 1), 1)
            cloud_free = np.logical_and(cloud_free, np.logical_not(bad_optical_conditions))
        # Where condition is true keep data, in other place update to scaled_FillValue:
        data = data.where(np.logical_not(cloud_free), data.attrs["scaled_FillValue"])
        # Update not cloudfree product and nodata to NaN (already done for scaled vars in the reader)
        # Keep cloudfree or valid product
        data = data.where(np.logical_or(cloud_free, data != data.attrs["scaled_FillValue"]), np.nan)
        res = SingleBandCompositor.__call__(self, [data], **data.attrs)
        res.attrs['_FillValue'] = np.nan
        return res


class CloudCompositorCommonMask(SingleBandCompositor):
    """Put cloud-free pixels as fill_value_color in palette."""

    def __call__(self, projectables, **info):
        """Create the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))
        data, cma = projectables
        valid_cma = cma != cma.attrs['_FillValue']
        valid_prod = data != data.attrs['_FillValue']
        valid_prod = np.logical_and(valid_prod, np.logical_not(np.isnan(data)))
        # Update valid_cma and not valid_prod means: keep not valid cma or valid prod
        data = data.where(np.logical_or(np.logical_not(valid_cma), valid_prod),
                          data.attrs["scaled_FillValue"])
        data = data.where(np.logical_or(valid_prod, valid_cma), np.nan)
        res = SingleBandCompositor.__call__(self, [data], **data.attrs)
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
