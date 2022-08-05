#!/usr/bin/env python
# Copyright (c) 2021 Satpy developers
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
"""Enhancement functions specific to the AHI sensor."""
import dask.array as da
import numpy as np

from satpy.enhancements import exclude_alpha, on_dask_array


def jma_true_color_reproduction(img, **kwargs):
    """Apply CIE XYZ matrix and return True Color Reproduction data.

    Himawari-8 True Color Reproduction Approach Based on the CIE XYZ Color System
    Hidehiko MURATA, Kotaro SAITOH, and Yasuhiko SUMIDA
    Meteorological Satellite Center, Japan Meteorological Agency
    NOAA National Environmental Satellite, Data, and Information Service
    Colorado State University—CIRA
    https://www.jma.go.jp/jma/jma-eng/satellite/introduction/TCR.html
    """
    _jma_true_color_reproduction(img.data)


@exclude_alpha
@on_dask_array
def _jma_true_color_reproduction(img_data):
    ccm = np.array([
        [1.1759, 0.0561, -0.1322],
        [-0.0386, 0.9587, 0.0559],
        [-0.0189, -0.1161, 1.0777]
    ])
    output = da.dot(img_data.T, ccm.T)
    return output.T
