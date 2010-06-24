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
"""Interface to Eumetcast level 1.5 format. Uses the MIPP reader.
"""


from ConfigParser import ConfigParser
from satin import CONFIG_PATH
import xrit.sat
from satin.logger import LOG
import os
import numpy as np

def load(satscene):
    """Read data from file and load it into *satscene*.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2"):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options)

def load_mviri(satscene, options):
    """Read mviri data from file and load it into *instrument_instance*.
    """
    os.environ["PPP_CONFIG_DIR"] = CONFIG_PATH

    LOG.debug("Channels to load from mviri: %s"%satscene.channels_to_load)
    satscene.info = {}
    satscene.info["Satellite"] = "meteosat07"
    for chn in satscene.channels_to_load:
        metadata, data = xrit.sat.load_meteosat07(satscene.time_slot,
                                                  chn,
                                                  mask = True)()
        satscene[chn].info = {"bla": "alb",
                              'var_name' : chn,
                              'var_data' : data,
                              '_FillValue' : -99999,
                              'var_dim_names': ('x', 'y')}
        satscene[chn] = data

        if chn == "00_7":
            satscene[chn].area_id = "HR" + satscene.area_id
        else:
            satscene[chn].area_id = satscene.area_id

    if(len(satscene.channels_to_load) > 1 and
       "00_7" in satscene.channels_to_load):
        satscene.area_id = None

def load_seviri(satscene, options):
    """Read seviri data from file and load it into *instrument_instance*.
    """
    os.environ["PPP_CONFIG_DIR"] = CONFIG_PATH

    LOG.debug("Channels to load from seviri: %s"%satscene.channels_to_load)
    satscene.info = {}
    satscene.info["Platform"] = satscene.satname
    satscene.info["Number"] = satscene.number
    satscene.info["Variant"] = satscene.variant
    for chn in satscene.channels_to_load:
        metadata, data = xrit.sat.load_meteosat09(satscene.time_slot,
                                                  chn,
                                                  mask = True,
                                                  calibrate = True)()
        satscene[chn].info = {'var_name' : chn,
                              'var_data' : data,
                              'valid_range' : np.array([data.min(), data.max()]),
                              'var_dim_names': ('x', 'y')}

        satscene[chn] = data

        if chn == "HRV":
            satscene[chn].area_id = "HR" + satscene.area_id
        else:
            satscene[chn].area_id = satscene.area_id

    for key in metadata.__dict__:
        if (not isinstance(metadata.__dict__[key],
                           (int, long, float, complex, str, np.ndarray)) or
            isinstance(metadata.__dict__[key], bool)):
            satscene.info[key] = str(metadata.__dict__[key])
        else:
            satscene.info[key] = metadata.__dict__[key]

            
        
    if(len(satscene.channels_to_load) > 1 and
       "HRV" in satscene.channels_to_load):
        satscene.area_id = None

CASES = {
    "mviri": load_mviri,
    "seviri": load_seviri
    }
