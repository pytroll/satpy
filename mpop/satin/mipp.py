#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
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

"""Interface to Eumetcast level 1.5 HRIT/LRIT format. Uses the MIPP reader.
"""
import ConfigParser
import datetime
import os

import numpy as np
import xrit.sat
from xrit import CalibrationError

from mpop import CONFIG_PATH
from mpop.satin.logger import LOG


def load(satscene, calibrate=True):
    """Read data from file and load it into *satscene*. The *calibrate*
    argument is passed to mipp (should be 0 for off, 1 for default, and 2 for
    radiances only).
    """    
    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2"):
        options[option] = value

    for section in conf.sections():
        if(section.startswith(satscene.instrument_name) and
           not (section == "satellite") and
           not section.endswith("-level2") and
           not section.endswith("-level1") and
           not section.endswith("-granules")):
            options[section] = conf.items(section)
    CASES.get(satscene.instrument_name, load_generic)(satscene,
                                                      options,
                                                      calibrate)

def load_generic(satscene, options, calibrate=True):
    """Read seviri data from file and load it into *instrument_instance*.
    """
    os.environ["PPP_CONFIG_DIR"] = CONFIG_PATH

    LOG.debug("Channels to load from seviri: %s"%satscene.channels_to_load)
    
    # Compulsory global attribudes
    satscene.info["title"] = (satscene.satname.capitalize() + satscene.number +
                              " satellite, " +
                              satscene.instrument_name.capitalize() +
                              " instrument.")
    satscene.info["institution"] = "Original data disseminated by EumetCast."
    satscene.add_to_history("HRIT/LRIT data read by mipp/mpop.")
    satscene.info["references"] = "No reference."
    satscene.info["comments"] = "No comment."

    entire = False


    # Slicing
    try:
        from pyresample import geometry
        from mpop.projector import get_area_def

        if satscene.area is None:
            entire = True
        else:
            if not satscene.area_def:
                satscene.area_def = get_area_def(satscene.area_id)

            if(satscene.area_def.proj_dict["proj"] != "geos" or
               satscene.area_def.proj_dict["lon_0"] != "0.0"):
                raise ValueError("Slicing area must be in "
                                 "geos0.0 projection.")
                
            area_xres = satscene.area_def.pixel_size_x
            area_yres = satscene.area_def.pixel_size_y
        
    except ImportError:
        LOG.warning("Pyresample not found, has to load entire data !")
        entire = True

    xsize = None
    ysize = None

    # Loading
    for chn in satscene.channels_to_load:
        for option, value in options.items():
            if not option.startswith(satscene.instrument_name):
                continue
            try:
                for item in value:
                    if item[0] == "name":
                        name = eval(item[1])
                    if item[0] == "size":
                        size = eval(item[1])
                    if item[0] == "resolution":
                        resolution = eval(item[1])
                if chn == name:
                    ysize, xsize = size
                    if not entire:
                        if resolution != area_xres:
                            LOG.warning("Area resolution not corresponding to channel. Adapting area.")
                            area_def = geometry.AreaDefinition(
                                satscene.area.area_id,
                                satscene.area.name,
                                satscene.area.proj_id,
                                satscene.area.proj_dict,
                                np.round(satscene.area.x_size * area_xres /
                                         resolution),
                                np.round(satscene.area.y_size * area_yres /
                                         resolution),
                                satscene.area.area_extent,
                                satscene.area.nprocs)
                        else:
                            area_def = satscene.area_def
                        area_x_size = area_def.x_size
                        area_y_size = area_def.y_size
                        ll_x = area_def.projection_x_coords[0, 0]
                        ll_y = area_def.projection_y_coords[area_y_size - 1, 0]
                        ur_x = area_def.projection_x_coords[0, area_x_size - 1]
                        ur_y = area_def.projection_y_coords[0, 0]
                    break
            except (KeyError, TypeError):
                continue


        if not entire:

            xres = float(resolution)
            yres = float(resolution)

            line_start = (ysize - 1) - int((ur_y / yres) + ysize / 2.0)
            line_end = (ysize - 1) - int((ll_y / yres) + ysize / 2.0)
            col_start = int((ll_x / xres) + xsize / 2.0)
            col_end = int((ur_x / xres) + xsize / 2.0)

            LOG.debug("Requesting lines " + str(line_start) + " to " +
                      str(line_end) + ", cols " + str(col_start) + " to " +
                      str(col_end))
        else:

            col_start, line_start, col_end, line_end = (0, 0, xsize - 1, ysize - 1)

        try:
            metadata, data = (xrit.sat.load(satscene.fullname,
                                            satscene.time_slot,
                                            chn,
                                            mask=True,
                                            calibrate=calibrate)
                              [line_start:line_end + 1,
                               col_start:col_end + 1])            
        except CalibrationError:
            LOG.warning("Loading non calibrated data since calibration failed.")
            metadata, data = (xrit.sat.load(satscene.fullname,
                                            satscene.time_slot,
                                            chn,
                                            mask=True,
                                            calibrate=False)
                              [line_start:line_end + 1,
                               col_start:col_end + 1])
            
        satscene[chn] = data

        satscene[chn].info['units'] = metadata.calibration_unit

        # Setting up area
        if isinstance(satscene.area, str):
            satscene[chn].area = satscene.area
        else:
            try:
                satscene[chn].area = geometry.AreaDefinition(
                    satscene.area.area_id + str(data.shape),
                    satscene.area.name,
                    satscene.area.proj_id,
                    satscene.area.proj_dict,
                    data.shape[1],
                    data.shape[0],
                    satscene.area.area_extent,
                    satscene.area.nprocs)
            except AttributeError:
                # Build an area on the fly from the mipp metadata
                proj_params = getattr(metadata, "proj4_params").split(" ")
                proj_dict = {}
                for param in proj_params:
                    key, val = param.split("=")
                    proj_dict[key] = val
                xres = metadata.pixel_size[1]
                yres = metadata.pixel_size[0]
                mid_x_area_extent = (xsize * xres) / 2.0
                mid_y_area_extent = (ysize * yres) / 2.0
                area_extent = (col_start * xres - mid_x_area_extent,
                               line_start * yres - mid_y_area_extent,
                               col_end * xres - mid_x_area_extent,
                               line_end * xres - mid_y_area_extent)
                satscene[chn].area = geometry.AreaDefinition(
                    satscene.satname + satscene.instrument_name +
                    str(col_start) + str(col_end) +
                    str(line_start) + str(line_end) +
                    str(data.shape),
                    "On-the-fly area",
                    proj_dict["proj"],
                    proj_dict,
                    data.shape[1],
                    data.shape[0],
                    area_extent)


def load_seviri(satscene, options, calibrate=True):
    """Read seviri data from file and load it into *satscene*.
    """
    os.environ["PPP_CONFIG_DIR"] = CONFIG_PATH

    LOG.debug("Channels to load from seviri: %s"%satscene.channels_to_load)
    
    # Compulsory global attribudes
    satscene.info["title"] = (satscene.satname.capitalize() + satscene.number +
                              " satellite, " +
                              satscene.instrument_name.capitalize() +
                              " instrument.")
    satscene.info["institution"] = "Original data disseminated by EumetCast."
    satscene.add_to_history("HRIT/LRIT data read by mipp/mpop.")
    satscene.info["references"] = "No reference."
    satscene.info["comments"] = "No comment."

    if satscene.area:
        from pyresample import geometry
        from mpop.projector import get_area_def
        if not satscene.area_def:
            satscene.area = get_area_def(satscene.area_id)
        area_extent = satscene.area.area_extent

        if(satscene.area_def.proj_dict["proj"] != "geos" or
           satscene.area_def.proj_dict["lon_0"] != "0.0"):
            raise ValueError("Slicing area must be in "
                             "geos0.0 projection.")
    else:
        area_extent = (-5567248.074173444, -5570248.4773392612,
                        5570248.4773392612, 5567248.074173444)
    
    for chn in satscene.channels_to_load:
        try:
            image = xrit.sat.load(satscene.fullname,
                                  satscene.time_slot,
                                  chn,
                                  mask=True,
                                  calibrate=calibrate)
            metadata, data = image.area_extent(area_extent)
        except CalibrationError:
            LOG.warning("Loading non calibrated data since calibration failed.")
            image = xrit.sat.load(satscene.fullname,
                                  satscene.time_slot,
                                  chn,
                                  mask=True,
                                  calibrate=False)
            metadata, data = image.area_extent(area_extent)


        satscene[chn] = data

        satscene[chn].info['units'] = metadata.calibration_unit
        
        # Setting up area
        if isinstance(satscene.area, str):
            satscene[chn].area = satscene.area
        else:
            try:
                satscene[chn].area = geometry.AreaDefinition(
                    satscene.area.area_id + str(data.shape),
                    satscene.area.name,
                    satscene.area.proj_id,
                    satscene.area.proj_dict,
                    data.shape[1],
                    data.shape[0],
                    metadata.actual_area_extent,
                    satscene.area.nprocs)
            except AttributeError:
                # Build an area on the fly from the mipp metadata
                proj_params = getattr(metadata, "proj4_params").split(" ")
                proj_dict = {}
                for param in proj_params:
                    key, val = param.split("=")
                    proj_dict[key] = val
                satscene[chn].area = geometry.AreaDefinition(
                    satscene.satname + satscene.instrument_name +
                    str(metadata.actual_area_extent) +
                    str(data.shape),
                    "On-the-fly area",
                    proj_dict["proj"],
                    proj_dict,
                    data.shape[1],
                    data.shape[0],
                    metadata.actual_area_extent)

CASES = {"seviri": load_seviri}

