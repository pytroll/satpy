#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

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
"""Interface to AAPP level 1b format. Uses the AHAMAP reader.
"""

from ConfigParser import ConfigParser
from satin import CONFIG_PATH
import os.path
import glob
import numpy as np
import math
from satin.logger import LOG

# Using ahamap
import avhrr

EPSILON = 0.001

def load(satscene):
    """Read data from file and load it into *satscene*.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2",
                                    raw = True):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options)

def load_avhrr(satscene, options):
    """Read avhrr data from file and load it into *satscene*.
    """
    
    if "filename" not in options:
        raise IOError("No filename given, cannot load.")
    values = {"orbit": satscene.orbit,
              "satname": satscene.satname,
              "number": satscene.number,
              "instrument": satscene.instrument_name,
              "fullname": satscene.fullname
              }
    filename = os.path.join(
        options["dir"],
        (satscene.time_slot.strftime(options["filename"])%values))

    file_list = glob.glob(satscene.time_slot.strftime(filename))

    if len(file_list) > 1:
        raise IOError("More than one l1b file matching!")
    elif len(file_list) == 0:
        raise IOError("No l1b file matching!")

    
    
    avh = avhrr.avhrr(filename)
    avh.get_unprojected()
    instrument_data = avh.build_raw()

    available_channels = set([])
    data_channels = {}

    for chn in instrument_data.data:
        channel_name = chn.info.info["channel_id"][3:].upper()
        available_channels |= set([channel_name])
        data_channels[channel_name] = chn.data

    for chn in satscene.channels_to_load:
        if chn in available_channels:
            if chn in ["1", "2", "3A"]:
                gain = instrument_data.info["vis_gain"]
                intercept = instrument_data.info["vis_intercept"]
            else:
                gain = instrument_data.info["ir_gain"]
                intercept = instrument_data.info["ir_intercept"]

            chn_array = np.ma.array(data_channels[chn])
            missing_data = instrument_data.info["missing_data"]
            chn_array = np.ma.masked_inside(chn_array,
                                            missing_data - EPSILON,
                                            missing_data + EPSILON)

            no_data = instrument_data.info["nodata"]
            chn_array = np.ma.masked_inside(chn_array,
                                            no_data - EPSILON,
                                            no_data + EPSILON)

            satscene[chn] = chn_array
            satscene[chn].data =  satscene[chn].data * gain + intercept

        else:
            LOG.warning("Channel "+str(chn)+" not available, not loaded.")

    satscene.lat = instrument_data.latdata / math.pi * 180
    satscene.lon = instrument_data.londata / math.pi * 180


def get_lat_lon(satscene, resolution):
    """Read lat and lon.
    """
    del resolution
    
    return LAT_LON_CASES[satscene.instrument_name](satscene, None)

def get_lat_lon_avhrr(satscene, options):
    """Read lat and lon.
    """
    del options
    
    return satscene.lat, satscene.lon


LAT_LON_CASES = {
    "avhrr": get_lat_lon_avhrr
    }

CASES = {
    "avhrr": load_avhrr
    }

