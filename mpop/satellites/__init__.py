#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011.

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
import weakref
from ConfigParser import ConfigParser, NoSectionError, NoOptionError

import mpop.utils
from mpop import CONFIG_PATH
from mpop.scene import SatelliteInstrumentScene

LOG = mpop.utils.get_logger("satellites")

def get_custom_composites(name):
    """Get the home made methods for building composites for a given satellite
    or instrument *name*.
    """
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, "mpop.cfg"))
    try:
        module_name = conf.get("composites", "module")
    except (NoSectionError, NoOptionError):
        return []

    try:
        module = __import__(module_name, globals(), locals(), [name])
    except ImportError:
        return []

    try:
        return getattr(module, name)
    except AttributeError:
        return []
        


def get_sat_instr_compositer((satellite, number, variant), instrument):
    """Get the compositer class for a given satellite, defined by the three
    strings *satellite*, *number*, and *variant*, and *instrument*. The class
    is then filled with custom composites if there are any (see
    :func:`get_custom_composites`). If no class is found, an attempt is made to
    build the class from a corresponding configuration file, see
    :func:`build_sat_instr_compositer`.
    """

    module_name = variant + satellite + number
    class_name = (variant.capitalize() + satellite.capitalize() +
                  number.capitalize() + instrument.capitalize())

    try:
        module = __import__(module_name, globals(), locals(), [class_name])
        klass = getattr(module, class_name)
        for k in get_custom_composites(variant + satellite +
                                       number + instrument):
            klass.add_method(k)
        return klass
    except (ImportError, AttributeError):
        return build_sat_instr_compositer((satellite, number,
                                                      variant),
                                          instrument)
        

def build_instrument_compositer(instrument_name):
    """Automatically generate an instrument compositer class from its
    *instrument_name*. The class is then filled with custom composites if there
    are any (see :func:`get_custom_composites`)
    """

    from mpop.instruments.visir import VisirCompositer
    instrument_class = type(instrument_name.capitalize() + "Compositer",
                            (VisirCompositer,),
                            {"instrument_name": instrument_name})
    for i in get_custom_composites(instrument_name):
        instrument_class.add_method(i)
        
    return instrument_class
                     
def build_sat_instr_compositer((satellite, number, variant), instrument):
    """Build a compositer class for the given satellite (defined by the three
    strings *satellite*, *number*, and *variant*) and *instrument* on the fly,
    using data from a corresponding config file. They inherit from the
    corresponding instrument class, which is also created on the fly is no
    predefined module (containing a compositer) for this instrument is
    available (see :func:`build_instrument_compositer`).
    """

    fullname = variant + satellite + number
    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, fullname + ".cfg"))

    try:
        mod = __import__("mpop.instruments." + instrument,
                         globals(), locals(),
                         [instrument.capitalize() + 'Compositer'])
        instrument_class = getattr(mod, instrument.capitalize() + 'Compositer')
        for i in get_custom_composites(instrument):
            instrument_class.add_method(i)

    except (ImportError, AttributeError):
        instrument_class = build_instrument_compositer(instrument)
                
    sat_class = type(variant.capitalize() +
                     satellite.capitalize() +
                     number.capitalize() +
                     instrument.capitalize() +
                     "Compositer",
                     (instrument_class,),
                     {})

    for i in get_custom_composites(variant + satellite +
                                   number + instrument):
        sat_class.add_method(i)
            
    return sat_class


class GeostationaryFactory(object):
    """Factory for geostationary satellite scenes.
    """


    @staticmethod
    def create_scene(satname, satnumber, instrument, time_slot, area=None, 
                     variant=''):
        """Create a compound satellite scene.
        """
        
        return GenericFactory.create_scene(satname, satnumber, instrument,
                                           time_slot, None, area, variant)

class PolarFactory(object):
    """Factory for polar satellite scenes.
    """


    @staticmethod
    def create_scene(satname, satnumber, instrument, time_slot, orbit=None,
                     area=None, variant=''):
        """Create a compound satellite scene.
        """
        
        return GenericFactory.create_scene(satname, satnumber, instrument,
                                           time_slot, orbit, area, variant)

class GenericFactory(object):
    """Factory for generic satellite scenes.
    """


    @staticmethod
    def create_scene(satname, satnumber, instrument, time_slot, orbit,
                     area=None, variant=''):
        """Create a compound satellite scene.
        """
        
        satellite = (satname, satnumber, variant)
        
        instrument_scene = SatelliteInstrumentScene(satellite=satellite,
                                                    instrument=instrument,
                                                    area=area,
                                                    orbit=orbit,
                                                    time_slot=time_slot)
        
        compositer = get_sat_instr_compositer(satellite, instrument)
        instrument_scene._CompositerClass = compositer
        
        if compositer is not None:
            # Pass weak ref to compositor to allow garbage collection
            instrument_scene.image = compositer(weakref.proxy(instrument_scene))
        return instrument_scene 
