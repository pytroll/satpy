"""Load relevant modules.
"""

import meteosat07
import meteosat09
import noaa15
import noaa17
import noaa18
import noaa19
import metop02
import globalmetop02
import earsmetop02
import aqua
import pp.satellites

from pp import CONFIG_PATH
from ConfigParser import ConfigParser
import os.path
import logging

LOG = logging.getLogger("pp.satellites")

def get_satellite_class(satellite, number, variant):
    """Get the class for a given satellite.
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
                     
def build_satellite_class(satellite, num, var):
    """Build a class for the given satellite on the fly, using a config file.
    """

    fullname = var + satellite + num
    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, fullname + ".cfg"))
    LOG.debug("Build new class from " +
              os.path.join(CONFIG_PATH, fullname + ".cfg"))
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
