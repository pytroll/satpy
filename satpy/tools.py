# -*- coding: utf-8 -*-
# Copyright (c) 2014, 2016
#
# Author(s):
#
#   Panu Lahtinen <pnuu+git@iki.fi>
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

'''Helper functions for eg. performing Sun zenith angle correction.
'''

import numpy as np


def sunzen_corr_cos(data, cos_zen, limit=88.):
    '''Perform Sun zenith angle correction to the given *data* using
    cosine of the zenith angle (*cos_zen*).  The correction is limited
    to *limit* degrees (default: 88.0 degrees).  For larger zenith
    angles, the correction is the same as at the *limit*.  Both *data*
    and *cos_zen* are given as 2-dimensional Numpy arrays or Numpy
    MaskedArrays, and they should have equal shapes.
    '''

    # Convert the zenith angle limit to cosine of zenith angle
    limit = np.cos(np.radians(limit))

    # Cosine correction
    lim_y, lim_x = np.where(cos_zen > limit)
    data[lim_y, lim_x] /= cos_zen[lim_y, lim_x]
    # Use constant value (the limit) for larger zenith
    # angles
    lim_y, lim_x = np.where(cos_zen <= limit)
    data[lim_y, lim_x] /= limit

    return data
