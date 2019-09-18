#!/usr/bin/env python
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
"""Enhancement functions specific to the ABI sensor."""

from satpy.enhancements import apply_enhancement


def cimss_true_color_contrast(img, **kwargs):
    """Scale data based on CIMSS True Color recipe for AWIPS."""
    def func(img_data):
        """Perform per-chunk enhancement.

        Code ported from Kaba Bah's AWIPS python plugin for creating the
        CIMSS Natural (True) Color image in AWIPS. AWIPS provides that python
        code the image data on a 0-255 scale. Satpy gives this function the
        data on a 0-1.0 scale (assuming linear stretching and sqrt
        enhancements have already been applied).

        """
        max_value = 1.0
        acont = (255.0 / 10.0) / 255.0
        amax = (255.0 + 4.0) / 255.0
        amid = 1.0 / 2.0
        afact = (amax * (acont + max_value) / (max_value * (amax - acont)))
        aband = (afact * (img_data - amid) + amid)
        aband[aband <= 10 / 255.0] = 0
        aband[aband >= 1.0] = 1.0
        return aband

    apply_enhancement(img.data, func, pass_dask=True)
