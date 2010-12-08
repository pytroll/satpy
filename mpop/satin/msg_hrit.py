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
import os.path
from ConfigParser import ConfigParser

import mpop.satin.pynwclib as nwclib
import numpy as np

from mpop import CONFIG_PATH


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

    new_names = ("VIS006", "VIS008", "IR_016", "IR_039", "WV_062", "WV_073",
                 "IR_087", "IR_097", "IR_108", "IR_120", "IR_134", "HRV")
    old_names = ("VIS06", "VIS08", "IR16", "IR39", "WV62", "WV73",
                 "IR87", "IR97", "IR108", "IR120", "IR134", "HRVIS")

    new_old = {}
    old_new = {}

    for old, new in zip(old_names, new_names):
        new_old[new] = old
        old_new[old] = new

    channels_to_load = [new_old[chn] for chn in satscene.channels_to_load]
    
    data = nwclib.get_channels(satscene.time_slot.strftime("%Y%m%d%H%M"), 
                               satscene.area_id, 
                               list(channels_to_load),
                               False)
    for chn in data:
        new_chn = old_new[chn]

        if new_chn != "HRV":
            satscene[new_chn].area_id = satscene.area_id
        else:
            satscene[new_chn].area_id = "HR" + satscene.area_id

        satscene[new_chn] = np.ma.array(data[chn]["CAL"], mask = data[chn]["MASK"])
        satscene[new_chn].info = {
            'var_name' : new_chn,
            'var_data' : satscene[new_chn].data,
            'var_dim_names': ('x'+str(satscene[new_chn].resolution),'y'+str(satscene[new_chn].resolution)),
            'valid_range' : np.array([satscene[new_chn].data.min(),
                                      satscene[new_chn].data.max()]),
            'standard_name' : new_chn,
            'Area_Name': satscene[new_chn].area_id or "",
            'scale_factor' : 1.0, 
            'add_offset' : 0.0,
                }

    if(len(satscene.channels_to_load) > 1 and
       "HRV" in satscene.channels_to_load):
        satscene.area_id = None

    satscene.info = {
        'var_children' : [   #{'var_name' : 'lat', 'var_callback': Functor(satscene.get_lat, low_res), 'var_dim_names': ('x','y') },
                             #{'var_name' : 'lon', 'var_callback' : Functor(satscene.get_lon, low_res) , 'var_dim_names': ('x','y') },
                           ## {'var_name' : 'lat_hrvis', 'var_data' : satscene.lat[high_res]}, 
                           ## {'var_name' : 'lon_hrvis', 'var_data' : satscene.lon[high_res]}, 
                        ],
        'Satellite' : satscene.fullname,
        'Platform' : satscene.satname,
        'Number' : satscene.number,
        'Service' : satscene.variant,
        'Antenna' : 'Fixed',
        'Receiver' : 'DMI (SMHI)' ,
        'Time' : satscene.time_slot.strftime("%Y-%m-%d %H:%M:%S UTC"), 
        'Area_Name' : satscene.area_id or "", 
        'Projection' : 'proj4-name GEOS(lon)',
        #'Columns' : satscene.channels[0].shape[1], 
        #'Lines' : satscene.channels[0].shape[0], 
        'SampleX' : 1.0, 
        'SampleY' : 1.0, 
        'title' : 'SEVIRI HRIT', 
        # from configurations file
        #'Conventions' : CONF.get('netcdf', 'Conventions'),
        #'history' :  "%s : %s \n" % (datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'),
        #                            "mpop") ,
        #'institution' : CONF.get('netcdf', 'institution'),
        #'source' :  CONF.get('netcdf', 'source'), 
        #'references' : CONF.get('netcdf', 'references'), 
        #'comment' : CONF.get('netcdf', 'comment'), 
         ## 'AreaStartPix' =     
        }

def get_lat_lon(satscene, resolution):
    """Read data from file and load it into *satscene*.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    #for option, value in conf.items(satscene.instrument_name, raw = True):
    #    options[option] = value
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
