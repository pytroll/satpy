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
"""Interface to Eumetcast level 1.5 format. Uses the MSG NWCLIB reader.
"""
from ConfigParser import ConfigParser
from satin import CONFIG_PATH
import os.path
import numpy as np
import satin.pynwclib as nwclib
import os

CONF = ConfigParser()
CONF.read(os.path.join(CONFIG_PATH, "meteosat09.cfg"))

MODE = os.getenv("SMHI_MODE")
if MODE is None:
    MODE = "offline"

MSG_DIR = CONF.get(MODE, 'msg_dir')
MSG_LIB = CONF.get(MODE, 'msg_lib')
MSG_BIN = CONF.get(MODE, 'msg_bin')

os.environ['SAFNWC'] = MSG_DIR
os.environ['SAFNWC_BIN'] = MSG_BIN
os.environ['SAFNWC_LIB'] = MSG_LIB
os.environ['PATH'] = os.environ['PATH']+":"+os.environ['SAFNWC_BIN']
os.environ['LD_LIBRARY_PATH'] = (os.environ['LD_LIBRARY_PATH']+
                                 ":"+os.environ['SAFNWC_LIB'])
os.environ['BUFR_TABLES'] = (os.environ['SAFNWC']+
                             "/src/bufr_000360/bufrtables/")
os.environ['LOCAL_DEFINITION_TEMPLATES'] = (os.environ['SAFNWC']+
                                            "/src/gribex_000360/gribtemplates/")


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

def load_seviri(satscene, options):
    """Read seviri data from file and load it into *satscene*.
    """
    del options
    # Do not reload data.
    satscene.channels_to_load -= set([chn.name for chn in
                                      satscene.loaded_channels()])
    
    data = nwclib.get_channels(satscene.time_slot.strftime("%Y%m%d%H%M"), 
                               satscene.area_id, 
                               list(satscene.channels_to_load),
                               False)
    for chn in data:
        satscene[chn] = np.ma.array(data[chn]["CAL"], mask = data[chn]["MASK"])
        if chn != "HRVIS":
            satscene[chn].area_id = satscene.area_id
        else:
            satscene[chn].area_id = "HR" + satscene.area_id

    if(len(satscene.channels_to_load) > 1 and
       "HRVIS" in satscene.channels_to_load):
        satscene.area_id = None

def get_lat_lon(satscene, resolution):
    """Read data from file and load it into *satscene*.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname() + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name, raw = True):
        options[option] = value
    return LAT_LON_CASES[satscene.instrument_name](satscene,
                                                   resolution,
                                                   options)

def get_lat_lon_seviri(satscene, resolution, options):
    """Read seviri data from file and load it into *satscene*.
    """
    del options
    channel = satscene[0.6, resolution]
    return nwclib.lat_lon_from_region(satscene.area_id, channel.name)

CASES = {
    "seviri": load_seviri
    }

LAT_LON_CASES = {
    "seviri": get_lat_lon_seviri
    }
