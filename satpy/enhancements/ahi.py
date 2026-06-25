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
    ccm_dict = {"himawari-8": np.array([[1.1495, 0.1534, -0.2203],
                                        [-0.0245, 0.8741, 0.1321],
                                        [-0.0203, -0.1102, 1.0685]]),

                "himawari-9": np.array([[1.1485, 0.1537, -0.2196],
                                        [-0.0264, 0.8765, 0.1316],
                                        [-0.0200, -0.1105, 1.0686]]),

                "goes-16": np.array([[1.1293, 0.1810, -0.2277],
                                     [-0.0940, 0.9377, 0.1380],
                                     [-0.0111, -0.1180, 1.0673]]),
                "goes-17": np.array([[1.1306, 0.1809, -0.2289],
                                     [-0.0941, 0.9367, 0.1391],
                                     [-0.0112, -0.1179, 1.0672]]),
                "goes-18": np.array([[1.1328, 0.1815, -0.2317],
                                     [-0.0943, 0.9342, 0.1417],
                                     [-0.0112, -0.1176, 1.0669]]),
                "goes-19": np.array([[1.1253, 0.1818, -0.2244],
                                     [-0.0937, 0.9403, 0.1351],
                                     [-0.0111, -0.1184, 1.0676]]),

                "meteosat-12": np.array([[0.8860, 0.2081, -0.0115],
                                    [-0.0468, 1.0691, -0.0406],
                                    [-0.0121, -0.1345, 1.0847]]),

                "geo-kompsat-2a": np.array([[1.1527, 0.1484, -0.2185],
                                            [-0.0248, 0.8762, 0.1303],
                                            [-0.0203, -0.1105, 1.0689]]),

                "sentinel-3a": np.array([[2.3828, 0.0046, -1.3048],
                                         [-0.2172, 0.0342, 1.1647],
                                         [-0.0211, -0.0043, 0.9636]]),
                "sentinel-3b": np.array([[2.3718, 0.0058, -1.2950],
                                        [-0.2161, 0.0434, 1.1544],
                                        [-0.0211, -0.0055, 0.9646]]),

                "sentinel-2a": np.array([[2.0301, 0.0231, -0.9706],
                                         [-0.1801, 0.1302, 1.0316],
                                         [-0.0186, -0.0164, 0.9732]]),
                "sentinel-2b": np.array([[1.9942, 0.0289, -0.9405],
                                         [-0.1774, 0.1617, 0.9974],
                                         [-0.0182, -0.0204, 0.9767]]),
                "sentinel-2c": np.array([[1.7759, 0.0666, -0.7598],
                                         [-0.1589, 0.3731, 0.7675],
                                         [-0.0161, -0.0470, 1.0013]]),
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
