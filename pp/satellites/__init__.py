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

"""Load relevant modules.
"""

import pp.satellites.meteosat07
import pp.satellites.meteosat09
import pp.satellites.noaa15
import pp.satellites.noaa17
import pp.satellites.noaa18
import pp.satellites.noaa19
import pp.satellites.metop02
import pp.satellites.globalmetop02
import pp.satellites.earsmetop02
import pp.satellites.aqua
import pp.satellites

from pp import CONFIG_PATH
from ConfigParser import ConfigParser
import os.path
import logging

LOG = logging.getLogger("pp.satellites")

def get_satellite_class(satellite, number, variant=""):
    """Get the class for a given satellite, defined by the three strings
    *satellite*, *number*, and *variant*. If no class is found, an attempt is
    made to build the class from a corresponding configuration file.
    """
    for i in dir(pp.satellites):
        module_name = "pp.satellites."+i
        for j in dir(eval(module_name)):
            if(hasattr(eval(module_name+"."+j), "satname") and
               hasattr(eval(module_name+"."+j), "number") and
               satellite == eval(module_name+"."+j+".satname") and
               number == eval(module_name+"."+j+".number")):
                if(variant is not None and
                   hasattr(eval(module_name+"."+j), "variant") and
                   variant == eval(module_name+"."+j+".variant")):
                    return eval(module_name+"."+j)
    return build_satellite_class(satellite, number, variant)

def build_instrument(name, channels):
    """Automatically generate an instrument class from its *name* and
    *channels*.
    """

    from pp.instruments.visir import VisirScene
    class Instrument(VisirScene):
        """Generic instrument, built on the fly.
        """
        channel_list = channels
        instrument_name = name
    return Instrument
                     
def build_satellite_class(satellite, num, var=""):
    """Build a class for the given satellite (defined by the three strings
    *satellite*, *num*, and *var*) on the fly, using a config file. The
    function returns as many classes as there are instruments defined in the
    configuration files. They inherit from the corresponding instrument class,
    which is also created on the fly is no predefined module for this
    instrument is available.
    """

    fullname = var + satellite + num
    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, fullname + ".cfg"))
    instruments = eval(conf.get("satellite", "instruments"))
    sat_classes = []
    for instrument in instruments:
        try:
            mod = __import__("pp.instruments." +
                             instrument, globals(), locals(),
                             [instrument.capitalize() + 'Scene'])
            instrument_class = getattr(mod, instrument.capitalize() + 'Scene')
        except ImportError:
            ch_list = []
            for section in conf.sections():
                if(not section.endswith("level1") and
                   not section.endswith("level2") and
                   not section.endswith("granules") and
                   section.startswith(instrument)):
                    ch_list += [[eval(conf.get(section, "name")),
                                 eval(conf.get(section, "frequency")),
                                 eval(conf.get(section, "resolution"))]]
                                 
            instrument_class = build_instrument(instrument, ch_list)
        
        class Satellite(instrument_class):
            """Generic satellite, built on the fly.
            """
            satname = satellite
            number = num
            variant = var
            
        sat_classes += [Satellite]
        
    if len(sat_classes) == 1:
        return sat_classes[0]
    return sat_classes
