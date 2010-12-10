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

"""mpop.satellites is the module englobes all satellite specific modules. In
itself, it hold the mighty :meth:`mpop.satellites.get_satellite_class` method.
"""
import os.path
from ConfigParser import ConfigParser

import mpop.utils
import mpop.satellites.meteosat09
from mpop import CONFIG_PATH


LOG = mpop.utils.get_logger("satellites")

def get_satellite_class(satellite, number, variant=""):
    """Get the class for a given satellite, defined by the three strings
    *satellite*, *number*, and *variant*. If no class is found, an attempt is
    made to build the class from a corresponding configuration file, see
    :func:`build_satellite_class`. Several classes can be returned if a given
    satellite has several instruments.
    """
    classes = []
    for i in dir(mpop.satellites):
        module_name = "mpop.satellites."+i
        for j in dir(eval(module_name)):
            if(hasattr(eval(module_name+"."+j), "satname") and
               hasattr(eval(module_name+"."+j), "number") and
               satellite == eval(module_name+"."+j+".satname") and
               number == eval(module_name+"."+j+".number")):
                if(hasattr(eval(module_name+"."+j), "variant") and
                   variant == eval(module_name+"."+j+".variant")):
                    classes += [eval(module_name+"."+j)]
    if classes != []:
        if len(classes) == 1:
            return classes[0]
        else:
            return classes
    else:
        return build_satellite_class(satellite, number, variant)

def build_instrument(instrument_name, channel_list):
    """Automatically generate an instrument class from its *instrument_name* and
    *channel_list*.
    """

    from mpop.instruments.visir import VisirScene
    instrument_class = type(instrument_name.capitalize() + "Scene",
                            (VisirScene,),
                            {"channel_list": channel_list,
                             "instrument_name": instrument_name})
    return instrument_class
                     
def build_satellite_class(satellite, number, variant=""):
    """Build a class for the given satellite (defined by the three strings
    *satellite*, *number*, and *variant*) on the fly, using a config file. The
    function returns as many classes as there are instruments defined in the
    configuration files. They inherit from the corresponding instrument class,
    which is also created on the fly is no predefined module for this
    instrument is available.
    """

    fullname = variant + satellite + number
    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, fullname + ".cfg"))
    instruments = eval(conf.get("satellite", "instruments"))
    sat_classes = []
    for instrument in instruments:
        try:
            mod = __import__("mpop.instruments." + instrument,
                             globals(), locals(),
                             [instrument.capitalize() + 'Scene'])
            instrument_class = getattr(mod, instrument.capitalize() + 'Scene')
        except ImportError:
            ch_list = []
            for section in conf.sections():
                if(not section.endswith("level1") and
                   not section.endswith("level2") and
                   not section.endswith("level3") and
                   not section.endswith("granules") and
                   section.startswith(instrument)):
                    ch_list += [[eval(conf.get(section, "name")),
                                 eval(conf.get(section, "frequency")),
                                 eval(conf.get(section, "resolution"))]]
                                 
            instrument_class = build_instrument(instrument, ch_list)

        sat_class = type(variant.capitalize() +
                         satellite.capitalize() +
                         number.capitalize() +
                         instrument.capitalize() +
                         "Scene",
                         (instrument_class,),
                         {"satname": satellite,
                          "number": number,
                          "variant": variant})
            
        sat_classes += [sat_class]
        
    if len(sat_classes) == 1:
        return sat_classes[0]
    return sat_classes
