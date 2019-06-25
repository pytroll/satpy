#!/usr/bin/env python
# Copyright (c) 2018-2019 Satpy developers
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
from trollimage.colormap import Colormap
import numpy as np
from satpy.enhancements import apply_enhancement


def water_detection(img, **kwargs):
    """Palettizes images from VIIRS flood data.

    This modifies the image's data so the correct colors
    can be applied to it, and then palettizes the image.
    """
    palette = kwargs['palettes']
    palette['colors'] = tuple(map(tuple, palette['colors']))

    def func(img_data):
        data = np.asarray(img_data)
        data[data == 150] = 31
        data[data == 199] = 18
        data[data >= 200] = data[data >= 200] - 100

        return data

    apply_enhancement(img.data, func, pass_dask=True)
    cm = Colormap(*palette['colors'])
    img.palettize(cm)
