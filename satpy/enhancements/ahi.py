# Copyright (c) 2017-2025 Satpy developers
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

"""AHI specific enhancements."""

from __future__ import annotations

import logging

import dask.array as da
import numpy as np

from .wrappers import exclude_alpha, on_dask_array

LOG = logging.getLogger(__name__)


def jma_true_color_reproduction(img):
    """Apply CIE XYZ matrix and return True Color Reproduction data.

    Himawari-8 True Color Reproduction Approach Based on the CIE XYZ Color System
    Hidehiko MURATA, Kotaro SAITOH, and Yasuhiko SUMIDA
    Meteorological Satellite Center, Japan Meteorological Agency
    NOAA National Environmental Satellite, Data, and Information Service
    Colorado State University—CIRA
    https://www.jma.go.jp/jma/jma-eng/satellite/introduction/TCR.html
    """
    _jma_true_color_reproduction(img.data,
                                 platform=img.data.attrs["platform_name"])


@exclude_alpha
@on_dask_array
def _jma_true_color_reproduction(img_data, platform=None):
    """Convert from AHI RGB space to sRGB space.

    The conversion matrices for this are supplied per-platform.
    The matrices are computed using the method described in the paper:
    'True Color Imagery Rendering for Himawari-8 with a Color Reproduction Approach
    Based on the CIE XYZ Color System' (:doi:`10.2151/jmsj.2018-049`).

    """
    # Conversion matrix dictionaries specifying sensor and platform.
    ccm_dict = {"himawari-8": np.array([[1.1629, 0.1539, -0.2175],
                                        [-0.0252, 0.8725, 0.1300],
                                        [-0.0204, -0.1100, 1.0633]]),

                "himawari-9": np.array([[1.1619, 0.1542, -0.2168],
                                        [-0.0271, 0.8749, 0.1295],
                                        [-0.0202, -0.1103, 1.0634]]),

                "goes-16": np.array([[1.1425, 0.1819, -0.2250],
                                     [-0.0951, 0.9363, 0.1360],
                                     [-0.0113, -0.1179, 1.0621]]),
                "goes-17": np.array([[1.1437, 0.1818, -0.2262],
                                     [-0.0952, 0.9354, 0.1371],
                                     [-0.0113, -0.1178, 1.0620]]),
                "goes-18": np.array([[1.1629, 0.1539, -0.2175],
                                     [-0.0252, 0.8725, 0.1300],
                                     [-0.0204, -0.1100, 1.0633]]),
                "goes-19": np.array([[0.9481, 0.3706, -0.2194],
                                     [-0.0150, 0.8605, 0.1317],
                                     [-0.0174, -0.1009, 1.0512]]),

                "mtg-i1": np.array([[0.9007, 0.2086, -0.0100],
                                    [-0.0475, 1.0662, -0.0414],
                                    [-0.0123, -0.1342, 1.0794]]),

                "geo-kompsat-2a": np.array([[1.1661, 0.1489, -0.2157],
                                            [-0.0255, 0.8745, 0.1282],
                                            [-0.0205, -0.1103, 1.0637]]),
                }

    # A conversion matrix, sensor name and platform name is required
    if platform is None:
        raise ValueError("Missing platform name.")

    # Get the satellite-specific conversion matrix
    try:
        ccm = ccm_dict[platform.lower()]
    except KeyError:
        raise KeyError(f"No conversion matrix found for platform {platform}")

    output = da.dot(img_data.T, ccm.T)
    return output.T
