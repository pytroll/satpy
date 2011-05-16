#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# Author(s):
 
#   Kristian Rune Larssen <krl@dmi.dk>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>

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

"""The :mod:`satout.cfscene` module provide a proxy class and utilites for
conversion of mpop scene to cf conventions.
"""

import numpy as np
from netCDF4 import date2num

from mpop.channel import Channel

#CF_DATA_TYPE = np.int16
CF_FLOAT_TYPE = np.float64
TIME_UNITS = "seconds since 1970-01-01 00:00:00"

class InfoObject(object):
    """Simple data and info container.
    """
    info = {}
    data = None

class CFScene(object):
    """Scene proxy class for cf conventions. The constructor should be called
    with the *scene* to transform as argument.
    """
    info = {}
    
    def __init__(self, scene, dtype=np.int16):
        if not issubclass(dtype, np.integer):
            raise TypeError('Only integer saving allowed for CF data')
        
        self.info = scene.info.copy()
        CF_DATA_TYPE = dtype
        
        # Other global attributes
        self.info["Conventions"] = "CF-1.4"
        self.info["satellite_name"] = scene.satname
        try:
            self.info["satellite_number"] = int(scene.number)
        except ValueError:
            self.info["satellite_number"] = str(scene.number)
        self.info["instrument_name"] = scene.instrument_name
        self.info["service"] = scene.variant
        
        self.time = InfoObject()
        self.time.data = date2num(scene.time_slot,
                                  TIME_UNITS)
        self.time.info = {"var_name": "time",
                          "var_data": self.time.data,
                          "var_dim_names": (),
                          "long_name": "Nominal time of the image",
                          "standard_name": "time",
                          "units": TIME_UNITS} 

        resolutions = []
        for chn in scene:
            #print "CHN: ",chn
            if not chn.is_loaded():
                continue
            
            #print type(chn)
            if not isinstance(chn, Channel):
                setattr(self, chn.name, chn)
                #print "INFO: ", chn.info
                continue

            chn_max = chn.data.max()
            chn_min = chn.data.min()
           
            scale = ((chn_max - chn_min) / 
                    (np.iinfo(CF_DATA_TYPE).max - np.iinfo(CF_DATA_TYPE).min - 1))
            offset = chn_max - (np.iinfo(CF_DATA_TYPE).max * scale)
             
            fill_value = np.iinfo(CF_DATA_TYPE).min
            valid_min = int((chn_min - offset) / scale)            
            valid_max = int((chn_max - offset) / scale)
            
            data = ((chn.data.data - offset) / scale).astype(CF_DATA_TYPE)
            data[chn.data.mask] = fill_value            
            
            str_res = str(int(chn.resolution)) + "m"

            if chn.resolution in resolutions:
                # resolution has been used before
                band = getattr(self, "band" + str_res)

                # data
                band.data = np.dstack((band.data, data))
                band.info["var_data"] = band.data
                
                # bandname
                bandname = getattr(self, "bandname" + str_res)
                bandname.data = np.concatenate((bandname.data,
                                                np.array([chn.name])))
                bandname.info["var_data"] = bandname.data

                # offset
                offset_attr = getattr(self, "offset" + str_res)
                offset_attr.data = np.concatenate((offset_attr.data,
                                                   np.array([offset])))
                offset_attr.info["var_data"] = offset_attr.data

                # scale
                scale_attr = getattr(self, "scale" + str_res)
                scale_attr.data = np.concatenate((scale_attr.data,
                                                  np.array([scale])))
                scale_attr.info["var_data"] = scale_attr.data

                # units
                units = getattr(self, "units" + str_res)
                units.data = np.concatenate((units.data,
                                             np.array([chn.info["units"]])))
                units.info["var_data"] = units.data
                
                # wavelength bounds
                bwl = getattr(self, "wl_bnds" + str_res)
                bwl.data = np.vstack((bwl.data,
                                      np.array([chn.wavelength_range[0],
                                                chn.wavelength_range[2]])))
                bwl.info["var_data"] = bwl.data

                # nominal_wavelength
                nwl = getattr(self, "nominal_wavelength" + str_res)
                nwl.data = np.concatenate((nwl.data,
                                           np.array([chn.wavelength_range[1]])))
                nwl.info["var_data"] = nwl.data

            else:
                # first encounter of this resolution
                resolutions += [chn.resolution]
                
                # data

                band = InfoObject()
                band.data = data
                band.info = {"var_name": "band_data"+str_res,
                             "var_data": band.data,
                             'var_dim_names': ('y'+str_res,
                                               'x'+str_res,
                                               "band"+str_res),
                             "standard_name": "band_data",
                             "valid_range": np.array([valid_min, valid_max]),
                             "resolution": chn.resolution}


                # bandname
                
                bandname = InfoObject()
                bandname.data = np.array([chn.name], 'O')
                bandname.info = {"var_name": "bandname"+str_res,
                                 "var_data": bandname.data,
                                 "var_dim_names": ("band"+str_res,),
                                 "standard_name": "band_name"}
                setattr(self, "bandname" + str_res, bandname)
                
                # offset
                off_attr = InfoObject()
                off_attr.data = np.array([offset])
                off_attr.info = {"var_name": "offset"+str_res,
                                 "var_data": off_attr.data,
                                 "var_dim_names": ("band"+str_res,),
                                 "standard_name": "linear_calibration_offset"}
                setattr(self, "offset" + str_res, off_attr) 

                # scale
                sca_attr = InfoObject()
                sca_attr.data = np.array([scale])
                sca_attr.info = {"var_name": "scale"+str_res,
                                 "var_data": sca_attr.data,
                                 "var_dim_names": ("band"+str_res,),
                                 "standard_name": ("linear_calibration"
                                                   "_scale_factor")}
                setattr(self, "scale" + str_res, sca_attr) 
                
                # units
                units = InfoObject()
                units.data = np.array([chn.info["units"]], 'O')
                units.info = {"var_name": "units"+str_res,
                              "var_data": units.data,
                              "var_dim_names": ("band"+str_res,),
                              "standard_name": "band_units"}
                setattr(self, "units" + str_res, units)
                
                # wavelength bounds
                wlbnds = InfoObject()
                wlbnds.data = np.array([[chn.wavelength_range[0],
                                         chn.wavelength_range[2]]])
                wlbnds.info = {"var_name": "wl_bnds"+str_res,
                               "var_data": wlbnds.data,
                               "var_dim_names": ("band"+str_res, "nv")}
                setattr(self, wlbnds.info["var_name"], wlbnds)
                
                # nominal_wavelength
                nomwl = InfoObject()
                nomwl.data = np.array([chn.wavelength_range[1]])
                nomwl.info = {"var_name": "nominal_wavelength"+str_res,
                              "var_data": nomwl.data,
                              "var_dim_names": ("band"+str_res,),
                              "standard_name": "radiation_wavelength",
                              "units": "um",
                              "bounds": wlbnds.info["var_name"]}
                setattr(self, "nominal_wavelength" + str_res, nomwl)

                # grid mapping or lon lats
                
                try:
                    area = InfoObject()
                    area.data = 0
                    area.info = {"var_name": chn.area.area_id,
                                 "var_data": area.data,
                                 "var_dim_names": ()}
                    area.info.update(proj2cf(chn.area.proj_dict))
                    setattr(self, area.info["var_name"], area)

                    x__ = InfoObject()
                    x__.data = chn.area.projection_x_coords[0, :]
                    x__.info = {"var_name": "x"+str_res,
                                "var_data": x__.data,
                                "var_dim_names": ("x"+str_res,),
                                "units": "m",
                                "standard_name": "projection_x_coordinate",
                                "long_name": "x coordinate of projection"}
                    setattr(self, x__.info["var_name"], x__)

                    y__ = InfoObject()
                    y__.data = chn.area.projection_y_coords[:, 0]
                    y__.info = {"var_name": "y"+str_res,
                                "var_data": y__.data,
                                "var_dim_names": ("y"+str_res,),
                                "units": "m",
                                "standard_name": "projection_y_coordinate",
                                "long_name": "y coordinate of projection"}
                    setattr(self, y__.info["var_name"], y__)
                    
                    band.info["grid_mapping"] = area.info["var_name"]
                except AttributeError:
                    lons = InfoObject()
                    try:
                        lons.data = chn.area.lons[:]
                    except AttributeError:
                        pass

                    lons.info = {"var_name": "lon"+str_res,
                                 "var_data": lons.data,
                                 "var_dim_names": ("y"+str_res,
                                                   "x"+str_res),
                                 "units": "degrees east",
                                 "long_name": "longitude coordinate",
                                 "standard_name": "longitude"}
                    if lons.data is not None:
                        setattr(self, lons.info["var_name"], lons)

                    lats = InfoObject()
                    try:
                        lats.data = chn.area.lats[:]
                    except AttributeError:
                        pass
                    
                    lats.info = {"var_name": "lon"+str_res,
                                 "var_data": lats.data,
                                 "var_dim_names": ("y"+str_res,
                                                   "x"+str_res),
                                 "units": "degrees north",
                                 "long_name": "latitude coordinate",
                                 "standard_name": "latitude"}
                    if lats.data is not None:
                        setattr(self, lats.info["var_name"], lats)
                    
                    if lats.data is not None and lons.data is not None:
                        band.info["coordinates"] = (lats.info["var_name"]+" "+
                                                    lons.info["var_name"])

                setattr(self, "band" + str_res, band)
                



def proj2cf(proj_dict):
    """Return the cf grid mapping from a proj dict.

    Description of the cf grid mapping:
    http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/ch05s06.html
    
    Table of the available grid mappings:
    http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.4/apf.html
    """

    cases = {"geos": geos2cf,
             "stere": stere2cf,
             "merc": merc2cf,
             "aea": aea2cf}

    return cases[proj_dict["proj"]](proj_dict)

def geos2cf(proj_dict):
    """Return the cf grid mapping from a geos proj dict.
    """

    return {"grid_mapping_name": "vertical_perspective",
            "latitude_of_projection_origin": 0.0,
            "longitude_of_projection_origin": eval(proj_dict["lon_0"]),
            "semi_major_axis": eval(proj_dict["a"]),
            "semi_minor_axis": eval(proj_dict["b"]),
            "perspective_point_height": eval(proj_dict["h"])
            }

def stere2cf(proj_dict):
    """Return the cf grid mapping from a geos proj dict.
    """

    raise NotImplementedError

def merc2cf(proj_dict):
    """Return the cf grid mapping from a mercator proj dict.
    """

    raise NotImplementedError

def aea2cf(proj_dict):
    """Return the cf grid mapping from a Albers Equal Area proj dict.
    """

    #standard_parallels = []
    #for item in ['lat_1', 'lat_2']:
    #    if item in proj_dict:
    #        standard_parallels.append(eval(proj_dict[item]))
    standard_parallel1 = eval(proj_dict['lat_1'])
    if 'lat_2' in proj_dict:
        standard_parallel2 = eval(proj_dict['lat_2'])
    else:
        standard_parallel2 = None
        
    lat_0 = 0.0
    if 'lat_0' in proj_dict:
        lat_0 = eval(proj_dict['lat_0'])

    x_0 = 0.0
    if 'x_0' in proj_dict:
        x_0 = eval(proj_dict['x_0'])

    y_0 = 0.0
    if 'y_0' in proj_dict:
        y_0 = eval(proj_dict['y_0'])
    
    retv = {"grid_mapping_name": "albers_conical_equal_area",
            "standard_parallel1": standard_parallel1,
            "latitude_of_projection_origin": lat_0,
            "longitude_of_central_meridian": eval(proj_dict["lon_0"]),
            "false_easting": x_0,
            "false_northing": y_0
            }
    if standard_parallel2:
        retv['standard_parallel2'] = standard_parallel2

    return retv

