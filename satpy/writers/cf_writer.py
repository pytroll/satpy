#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Writer for netCDF4/CF.
"""

import logging

import cf
import numpy as np

from satpy.writers import Writer

LOG = logging.getLogger(__name__)


class CFWriter(Writer):

    def save_datasets(self, datasets, filename, **kwargs):
        """Save all datasets to one or more files)
        """
        fields = []
        shapes = {}
        for dataset in datasets:
            if dataset.shape in shapes:
                domain = shapes[dataset.shape]
            else:
                lines, pixels = dataset.shape
                # Create a grid_latitude dimension coordinate
                line_coord = cf.DimensionCoordinate(data=cf.Data(
                    np.arange(lines), '1'))
                pixel_coord = cf.DimensionCoordinate(data=cf.Data(
                    np.arange(pixels), '1'))
                domain = cf.Domain(dim={'lines': line_coord,
                                        'pixels': pixel_coord}, )
                shapes[dataset.shape] = domain
            data = cf.Data(dataset, dataset.info['units'])
            properties = {'standard_name': dataset.info['standard_name'],
                          'name': dataset.info['name']}
            fields.append(cf.Field(properties=properties,
                                   data=data,
                                   axes=['lines', 'pixels'],
                                   domain=domain))
        cf.write(fields, filename, fmt='NETCDF4')
