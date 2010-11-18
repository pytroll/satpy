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

# TODO
# - complete projection list and attribute list
# - handle other units than "m" for coordinates
# - handle units for data
# - pluginize

import datetime
import logging
from ConfigParser import NoSectionError

import numpy as np
from netCDF4 import Dataset

from pp.instruments.visir import VisirScene
from pp.satellites import get_satellite_class


LOG = logging.getLogger("nc4/cf reader")

# To be complete, get from appendix F of cf conventions
MAPPING_ATTRIBUTES = {'grid_mapping_name': "proj",
                      'latitude_projection_of_origin': "lat_0",
                      'longitude_projection_of_origin': "lon_0",
                      'perspective_point_height': "h",
                      'semi_major_axis': "a",
                      'semi_minor_axis': "b"
                      }
# To be completed, get from appendix F of cf conventions
PROJNAME = {"vertical_perspective": "near_sided_perspective"
    }


def load_from_nc4(filename):
    """Load data from a netcdf4 file.
    """
    rootgrp = Dataset(filename, 'r')

    time_slot = rootgrp.variables["time"].getValue()[0]

    time_slot = (datetime.timedelta(seconds=time_slot) +
                 datetime.datetime(1970,1,1))
    area = None

    try:
        klass = get_satellite_class(rootgrp.platform_name,
                                    rootgrp.platform_number,
                                    rootgrp.service)
        scene = klass(time_slot=time_slot, area=area)
    except NoSectionError:
        klass = VisirScene
        scene = klass(time_slot=time_slot, area=area)
        scene.satname = rootgrp.platform_name
        scene.number = rootgrp.platform_number
        scene.service = rootgrp.service


    for var_name, var in rootgrp.variables.items():

        if var_name.startswith("band_data"):
            resolution = var.resolution
            str_res = str(resolution) + "m"
            
            names = rootgrp.variables["bandname"+str_res][:]

            data = var[:, :, :].astype(var.dtype)

            data = np.ma.masked_outside(data,
                                        var.valid_range[0],
                                        var.valid_range[1])

                                        
            for i, name in enumerate(names):
                try:
                    scene[name] = (data[:, :, i] *
                                   rootgrp.variables["scale"+str_res][i] +
                                   rootgrp.variables["offset"+str_res][i])
                    #FIXME complete this
                    #scene[name].info
                except KeyError:
                    from pp.channel import Channel
                    wv_var = rootgrp.variables["nominal_wavelength"+str_res]
                    wb_var = rootgrp.variables[getattr(wv_var, "bounds")]
                    minmax = wb_var[i]
                    scene.channels.append(Channel(name,
                                                  resolution,
                                                  (minmax[0],
                                                   wv_var[i][0],
                                                   minmax[1])))
                    scene[name] = (data[:, :, i] *
                                   rootgrp.variables["scale"+str_res][i] +
                                   rootgrp.variables["offset"+str_res][i])
                    
                    # build the channel on the fly

            try:
                area_var = getattr(var,"grid_mapping")
                area_var = rootgrp.variables[area_var]
                proj4_dict = {}
                for attr, projattr in MAPPING_ATTRIBUTES.items():
                    try: 
                        the_attr = getattr(area_var, attr)
                        if projattr == "proj":
                            proj4_dict[projattr] = PROJNAME[the_attr]
                        else:
                            proj4_dict[projattr] = the_attr
                    except AttributeError:
                        pass

                x__ = rootgrp.variables["x"+str_res][:]
                y__ = rootgrp.variables["y"+str_res][:]

                x_pixel_size = abs((x__[1] - x__[0]))
                y_pixel_size = abs((y__[1] - y__[0]))

                llx = x__[0] - x_pixel_size / 2.0
                lly = y__[-1] - y_pixel_size / 2.0
                urx = x__[-1] + x_pixel_size / 2.0
                ury = y__[0] + y_pixel_size / 2.0

                area_extent = (llx, lly, urx, ury)

                try:
                    # create the pyresample areadef
                    from pyresample.geometry import AreaDefinition
                    area = AreaDefinition("myareaid", "myareaname",
                                          "myprojid", proj4_dict,
                                          data.shape[1], data.shape[0],
                                          area_extent)

                except ImportError:
                    LOG.warning("Pyresample not found, "
                                "cannot load area descrition")

            except AttributeError:
                LOG.debug("No grid mapping found.")
                
            try:
                area_var = getattr(var,"coordinates")
                coordinates_vars = area_var.split(" ")
                lons = None
                lats = None
                for coord_var_name in coordinates_vars:
                    coord_var = rootgrp.variables[coord_var_name]
                    units = getattr(coord_var, "units")
                    if(coord_var_name.lower().startswith("lon") or
                       units.lower().endswith("east") or 
                       units.lower().endswith("west")):
                        lons = coord_var[:]
                    elif(coord_var_name.lower().startswith("lat") or
                         units.lower().endswith("north") or 
                         units.lower().endswith("south")):
                        lats = coord_var[:]
                if lons and lats:
                    try:
                        from pyresample.geometry import SwathDefinition
                        area = SwathDefinition(lons=lons, lats=lats)

                    except ImportError:
                        LOG.warning("Pyresample not found, "
                                    "cannot load area descrition")
                
            except AttributeError:
                LOG.debug("No lon/lat found.")
            
    for attr in rootgrp.ncattrs():
        scene.info[attr] = getattr(rootgrp, attr)



                         

#         data = var[:, :].astype(var.dtype)
#         if hasattr(var, "valid_range"):
#             scene[var.standard_name] = np.ma.masked_outside(data,
#                                                             var.valid_range[0],
#                                                             var.valid_range[1])
#         else:
#             scene[var.standard_name] = data
        
#         if not hasattr(scene[var.standard_name], 'info'):
#             scene[var.standard_name].info = {}
#         scene[var.standard_name].info['var_data'] = \
#                                                   scene[var.standard_name].data
#         scene[var.standard_name].info['var_name'] = var_name
#         scene[var.standard_name].info['var_dim_names'] = var.dimensions
#         if var.area_name == "":
#             scene[var.standard_name].area_id = None
#         else:
#             scene[var.standard_name].area_id = var.area_name
#         for attr in var.ncattrs():
#             scene[var.standard_name].info[attr] = getattr(var, attr)
    
#     if not hasattr(scene, 'info'):
#         scene.info = {}
        
#     for attr in rootgrp.ncattrs():
#         scene.info[attr] = getattr(rootgrp, attr)

#     scene.info['var_children'] = []
#     rootgrp.close()

    return scene


