# Copyright (c) 2022- Satpy developers
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
"""Composites showing atmospheric features."""

import dask.array as da
import xarray as xr

from . import GenericCompositor


class ESSLMoisture(GenericCompositor):
    """Low level moisture by European Severe Storms Laboratory (ESSL).

    Must be passed exactly two projectables.  The first one should correspond
    to a channel at around 0.86 µm, the second one at 0.91 µm.

    This composite was developed by ESSL.
    """

    low = 1.1  # FIXME: configurable
    high = 1.6  # FIXME: configurable

    def __call__(self, projectables, others=None, **info):
        """Generate the ESSL low level moisture composite."""
        (nir_086, nir_091) = projectables
        with xr.set_options(keep_attrs=True):
            ratio = nir_091 / nir_086
            ratio = self._scale_and_clip(ratio)
            red = self._calc_red(ratio)
            green = self._calc_green(ratio)
            blue = self._calc_blue(ratio)
        return super().__call__([red, green, blue], **info)

    def _scale_and_clip(self, ratio):
        """Scale ratio values to [0, 1] and clip values outside this range."""
        scaled = (ratio - self.low) / (self.high - self.low)
        scaled.data = da.clip(scaled.data, 0, 1)
        return scaled

    def _calc_red(self, ratio):
        """Calculate values for red based on scaled and clipped ratio."""
        red_a = 1.375 - 2.67 * ratio
        red_b = -0.75 + ratio
        red = xr.where(red_a > red_b, red_a, red_b)
        red.data = da.clip(red.data, 0, 1)
        return red

    def _calc_green(self, ratio):
        """Calculate values for green based on scaled and clipped ratio."""
        green = 1 - (8/7) * ratio
        green.data = da.clip(green.data, 0, 1)
        return green

    def _calc_blue(self, ratio):
        """Calculate values for blue based on scaled and clipped ratio."""
        blue_a = 0.75 - 1.5 * ratio
        blue_b = 0.25 - (ratio - 0.75)**2
        blue = xr.where(blue_a > blue_b, blue_a, blue_b)
        blue.data = da.clip(blue.data, 0, 1)
        return blue
