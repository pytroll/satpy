#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014.

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
import warnings
from ConfigParser import NoSectionError

import numpy as np
from netCDF4 import Dataset, num2date

from mpop.instruments.visir import VisirCompositer
from mpop.satellites import GenericFactory
from mpop.satout.cfscene import TIME_UNITS
from mpop.utils import get_logger


LOG = get_logger("netcdf4/cf reader")

# To be complete, get from appendix F of cf conventions
MAPPING_ATTRIBUTES = {'grid_mapping_name': "proj",
                      'standard_parallel': ["lat_1", "lat_2"],
                      'latitude_of_projection_origin': "lat_0",
                      'longitude_of_projection_origin': "lon_0",
                      'longitude_of_central_meridian': "lon_0",
                      'perspective_point_height': "h",
                      'false_easting': "x_0",
                      'false_northing': "y_0",
                      'semi_major_axis': "a",
                      'semi_minor_axis': "b",
                      'inverse_flattening': "rf",
                      'ellipsoid': "ellps", # not in CF conventions...
                      }

# To be completed, get from appendix F of cf conventions
PROJNAME = {"vertical_perspective": "nsper",
            "geostationary": "geos",
            "albers_conical_equal_area": "aea",
            "azimuthal_equidistant": "aeqd",
            "equirectangular": "eqc",
            "transverse_mercator": "tmerc",
            "stereographic": "stere",
            "general_oblique_transformation": "ob_tran"
    }

def _load02(filename):
    """Load data from a netcdf4 file, cf-satellite v0.2 (2012-02-03).
    """

    
    rootgrp = Dataset(filename, 'r')
    
    # processed variables
    processed = set()

    # Currently MPOP does not like unicode (so much).
    satellite_name, satellite_number = [str(i) for i in rootgrp.platform.rsplit("-", 1)]

    time_slot = rootgrp.variables["time"].getValue()[0]
    time_slot = num2date(time_slot, TIME_UNITS)

    processed |= set(["time"])



    try:
        service = str(rootgrp.service)
    except AttributeError:
        service = ""

    instrument_name = str(rootgrp.instrument)

    try:
        orbit = str(rootgrp.orbit)
    except AttributeError:
        orbit = None

    try:
        scene = GenericFactory.create_scene(satellite_name,
                                            satellite_number,
                                            instrument_name,
                                            time_slot,
                                            orbit,
                                            None,
                                            service)
    except NoSectionError:
        scene = VisirCompositer(time_slot=time_slot)
        scene.satname = satellite_name
        scene.number = satellite_number
        scene.service = service

    dim_chart = {}

    for var_name, var in rootgrp.variables.items():
        varname = None
        try:
            varname = var.standard_name
        except AttributeError:
            try:
                varname = var.long_name
            except AttributeError:
                pass

        if varname in ["band_data", "Band data"]:
            LOG.debug("Found some data: " + var_name)
            dims = var.dimensions

            for dim in dims:
                dim_chart[dim] = var_name

            for cnt, dim in enumerate(dims):
                if dim.startswith("band"):
                    break

            data = var
            data.set_auto_maskandscale(False)
                
            area = None
            try:
                area_var_name = getattr(var,"grid_mapping")
                area_var = rootgrp.variables[area_var_name]
                proj4_dict = {}
                for attr, projattr in MAPPING_ATTRIBUTES.items():
                    try: 
                        the_attr = getattr(area_var, attr)
                        if projattr == "proj":
                            proj4_dict[projattr] = PROJNAME[the_attr]
                        elif(isinstance(projattr, (list, tuple))):
                            try:
                                for i, subattr in enumerate(the_attr):
                                    proj4_dict[projattr[i]] = subattr
                            except TypeError:
                                proj4_dict[projattr[0]] = the_attr
                        else:
                            proj4_dict[projattr] = the_attr
                    except AttributeError:
                        pass
                y_name, x_name = dims[:cnt] + dims[cnt + 1:]
                x__ = rootgrp.variables[x_name][:]
                y__ = rootgrp.variables[y_name][:]

                if proj4_dict["proj"] == "geos":
                    x__ *= proj4_dict["h"]
                    y__ *= proj4_dict["h"]

                x_pixel_size = abs((np.diff(x__)).mean())
                y_pixel_size = abs((np.diff(y__)).mean())

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
                                          len(x__), len(y__),
                                          area_extent)

                except ImportError:
                    LOG.warning("Pyresample not found, "
                                "cannot load area descrition")
                processed |= set([area_var_name, x_name, y_name])
                LOG.debug("Grid mapping found and used.")
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
                if lons.any() and lats.any():
                    try:
                        from pyresample.geometry import SwathDefinition
                        area = SwathDefinition(lons=lons, lats=lats)

                    except ImportError:
                        LOG.warning("Pyresample not found, "
                                    "cannot load area descrition")
                
                processed |= set(coordinates_vars)
                LOG.debug("Lon/lat found and used.")
            except AttributeError:
                LOG.debug("No lon/lat found.")         
            
            names = rootgrp.variables[dim][:]
            scales = data.scale_factor
            offsets = data.add_offset
            if len(names) == 1:
                scales = np.array([scales])
                offsets = np.array([offsets])
            LOG.info("Scales and offsets: %s %s %s" % (str(names), str(scales), str(offsets)))
            for nbr, name in enumerate(names):
                name = str(name)
                try:
                    if cnt == 0:
                        chn_data = data[nbr, :, :].squeeze()
                    if cnt == 1:
                        chn_data = data[:, nbr, :].squeeze()
                    if cnt == 2:
                        chn_data = data[:, :, nbr].squeeze()
                    scene[name] = (np.ma.masked_equal(chn_data, data._FillValue)
                                   * scales[nbr] + offsets[nbr])

                    scene[name].info["units"] = var.units
                except KeyError:
                    from mpop.channel import Channel
                    scene.channels.append(Channel(name))
                
                if area is not None:
                    scene[name].area = area

            processed |= set([var_name, dim])

    non_processed = set(rootgrp.variables.keys()) - processed

    for var_name in non_processed:
        var = rootgrp.variables[var_name]
        if not (hasattr(var, "standard_name") or
                hasattr(var, "long_name")):
            LOG.info("Delayed processing of " + var_name)
            continue

        dims = var.dimensions
        if len(dims) != 1:
            LOG.info("Don't know what to do with " + var_name)
            continue

        dim = dims[0]
        if var.standard_name == "radiation_wavelength":
        
            names = rootgrp.variables[dim][:]
            for nbr, name in enumerate(names):
                name = str(name)
                scene[name].wavelength_range[1] = var[nbr]
            try:
                bnds = rootgrp.variables[var.bounds][:]
                for nbr, name in enumerate(names):
                    name = str(name)
                    scene[name].wavelength_range[0] = bnds[nbr, 0]
                    scene[name].wavelength_range[2] = bnds[nbr, 1]
                processed |= set([var.bounds])
            except AttributeError:
                pass

            processed |= set([var_name])


    
    non_processed = set(rootgrp.variables.keys()) - processed
    if len(non_processed) > 0:
        LOG.warning("Remaining non-processed variables: " + str(non_processed))
        
    return scene
    
def load_from_nc4(filename):
    """Load data from a netcdf4 file, cf-satellite v0.1
    """

    rootgrp = Dataset(filename, 'r')

    try:
        rootgrp.satellite_number
        warnings.warn("You are loading old style netcdf files...", DeprecationWarning)
    except AttributeError:
        return _load02(filename)
    

    if not isinstance(rootgrp.satellite_number, str):
        satellite_number = "%02d" % rootgrp.satellite_number
    else:
        satellite_number = str(rootgrp.satellite_number)

    time_slot = rootgrp.variables["time"].getValue()[0]

    time_slot = num2date(time_slot, TIME_UNITS)

    service = str(rootgrp.service)

    satellite_name = str(rootgrp.satellite_name)
    instrument_name = str(rootgrp.instrument_name)

    try:
        orbit = str(rootgrp.orbit)
    except AttributeError:
        orbit = None

    try:
        scene = GenericFactory.create_scene(satellite_name,
                                            satellite_number,
                                            instrument_name,
                                            time_slot,
                                            orbit,
                                            None,
                                            service)
    except NoSectionError:
        scene = VisirCompositer(time_slot=time_slot)
        scene.satname = satellite_name
        scene.number = satellite_number
        scene.service = service


    for var_name, var in rootgrp.variables.items():
        area = None

        if var_name.startswith("band_data"):
            resolution = var.resolution
            str_res = str(int(resolution)) + "m"
            
            names = rootgrp.variables["bandname"+str_res][:]

            data = var[:, :, :].astype(var.dtype)

            data = np.ma.masked_outside(data,
                                        var.valid_range[0],
                                        var.valid_range[1])

            try:
                area_var = getattr(var,"grid_mapping")
                area_var = rootgrp.variables[area_var]
                proj4_dict = {}
                for attr, projattr in MAPPING_ATTRIBUTES.items():
                    try: 
                        the_attr = getattr(area_var, attr)
                        if projattr == "proj":
                            proj4_dict[projattr] = PROJNAME[the_attr]
                        elif(isinstance(projattr, (list, tuple))):
                            try:
                                for i, subattr in enumerate(the_attr):
                                    proj4_dict[projattr[i]] = subattr
                            except TypeError:
                                proj4_dict[projattr[0]] = the_attr
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
            
            for i, name in enumerate(names):
                name = str(name)
                if var.dimensions[0].startswith("band"):
                    chn_data = data[i, :, :]
                elif var.dimensions[1].startswith("band"):
                    chn_data = data[:, i, :]
                elif var.dimensions[2].startswith("band"):
                    chn_data = data[:, :, i]
                else:
                    raise ValueError("Invalid dimension names for band data")
                try:
                    scene[name] = (chn_data *
                                   rootgrp.variables["scale"+str_res][i] +
                                   rootgrp.variables["offset"+str_res][i])
                    #FIXME complete this
                    #scene[name].info
                except KeyError:
                    # build the channel on the fly

                    from mpop.channel import Channel
                    wv_var = rootgrp.variables["nominal_wavelength"+str_res]
                    wb_var = rootgrp.variables[getattr(wv_var, "bounds")]
                    minmax = wb_var[i]
                    scene.channels.append(Channel(name,
                                                  resolution,
                                                  (minmax[0],
                                                   wv_var[i][0],
                                                   minmax[1])))
                    scene[name] = (chn_data *
                                   rootgrp.variables["scale"+str_res][i] +
                                   rootgrp.variables["offset"+str_res][i])
                    
                if area is not None:
                    scene[name].area = area
        area = None

    for attr in rootgrp.ncattrs():
        scene.info[attr] = getattr(rootgrp, attr)
    scene.add_to_history("Loaded from netcdf4/cf by mpop")

    return scene
