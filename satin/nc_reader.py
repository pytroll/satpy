#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.
"""Very simple netcdf reader for mpop.
"""
from netCDF4 import Dataset
import datetime
import saturn.runner
import numpy as np

def load_from_nc4(filename):
    """Load data from a netcdf4 file.
    """
    rootgrp = Dataset(filename, 'r')

    klass = saturn.runner.get_class(rootgrp.Platform,
                                    rootgrp.Number,
                                    rootgrp.Service)
    time_slot = datetime.datetime.strptime(rootgrp.Time,
                                           "%Y-%m-%d %H:%M:%S UTC")
    if rootgrp.Area_Name == "":
        area_id = None
    else:
        area_id = rootgrp.Area_Name

    scene = klass(time_slot=time_slot, area_id=area_id)

    for var_name in rootgrp.variables:
        
        var = rootgrp.variables[var_name]

        data = var[:, :].astype(var.dtype)
        if hasattr(var, "valid_range"):
            scene[var.standard_name] = np.ma.masked_outside(data,
                                                            var.valid_range[0],
                                                            var.valid_range[1])
        else:
            scene[var.standard_name] = data
        
        if not hasattr(scene[var.standard_name], 'info'):
            scene[var.standard_name].info = {}
        scene[var.standard_name].info['var_data'] = \
                                                  scene[var.standard_name].data
        scene[var.standard_name].info['var_name'] = var_name
        scene[var.standard_name].info['var_dim_names'] = var.dimensions
        if var.Area_Name == "":
            scene[var.standard_name].area_id = None
        else:
            scene[var.standard_name].area_id = var.Area_Name
        for attr in var.ncattrs():
            scene[var.standard_name].info[attr] = getattr(var, attr)
    
    if not hasattr(scene, 'info'):
        scene.info = {}
        
    for attr in rootgrp.ncattrs():
        scene.info[attr] = getattr(rootgrp, attr)

    scene.info['var_children'] = []
    rootgrp.close()

    return scene


