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

"""This module defines satellite scenes. They are defined as generic classes,
to be inherited when needed.

A scene is a set of :mod:`pp.channel`s for a given time, and sometimes also for
a given area.
"""

import numpy as np
from pp.channel import Channel, NotLoadedError
from pp.logger import LOG
import copy
from pp.projector import Projector
import ConfigParser
import os.path
from pp import CONFIG_PATH
import datetime

class SatelliteScene(object):
    """This is the satellite scene class. It is a capture of the satellite
    (channels) data at given *time_slot* and *area*.
    """

    #: Name of the satellite
    satname = ""

    #: Number of the satellite
    number = ""
    
    #: Variant of the satellite (often the communication channel it comes from)
    variant = ""

    #: Time of the snapshot
    time_slot = None

    #: Orbit number of the satellite
    orbit = None

    #: Area on which the scene is defined.
    area_id = None

    #: Metadata information
    info = {}
    
    def __init__(self, time_slot = None, area_id = None, orbit = None):

        if(time_slot is not None and
           not isinstance(time_slot, datetime.datetime)):
            raise TypeError("Time_slot must be a datetime.datetime instance.")
        
        self.time_slot = time_slot

        
        if(area_id is not None and
           not isinstance(area_id, str)):
            raise TypeError("Area must be a string.")
        
        self.area_id = area_id


        if(orbit is not None and
           not isinstance(orbit, str)):
            raise TypeError("Orbit must be a string.")
        
        self.orbit = orbit


        self.lat = None
        self.lon = None


    @property
    def fullname(self):
        """Full name of the satellite, that is platform name and number
        (eg"metop02").
        """
        return self.variant + self.satname + self.number

class SatelliteInstrumentScene(SatelliteScene):
    """This is the satellite instrument class. It is an abstract channel
    container.
    """
    channels = []
    channel_list = []

    #. Instrument name
    instrument_name = None

    def __init__(self, time_slot = None, area_id = None, orbit = None):
        SatelliteScene.__init__(self, time_slot, area_id, orbit)
        self.channels = []
        
        for name, w_range, resolution in self.channel_list:
            self.channels.append(Channel(name = name,
                                         wavelength_range = w_range,
                                         resolution = resolution))
        self.channels_to_load = set([])

    def __getitem__(self, key, aslist = False):
        if(isinstance(key, float)):
            channels = [chn for chn in self.channels
                        if(hasattr(chn, "wavelength_range") and
                           chn.wavelength_range[0] <= key and
                           chn.wavelength_range[2] >= key)]
            channels = sorted(channels,
                              lambda ch1,ch2:
                                  ch1.__cmp__(ch2, key))
            
        elif(isinstance(key, str)):
            channels = [chn for chn in self.channels
                        if chn.name == key]
            channels = sorted(channels)

        elif(isinstance(key, int)):
            channels = [chn for chn in self.channels
                        if chn.resolution == key]
            channels = sorted(channels)

        elif(isinstance(key, (tuple, list))):
            if len(key) == 0:
                raise KeyError("Key list must contain at least one element.")
            channels = self.__getitem__(key[0], aslist = True)
            if(len(key) > 1 and len(channels) > 0):
                dummy_instance = SatelliteInstrumentScene()
                dummy_instance.channels = channels
                channels = dummy_instance.__getitem__(key[1:], aslist = True)
        else:
            raise TypeError("Malformed key.")

        if len(channels) == 0:
            raise KeyError("No channel corresponding to "+str(key)+".")
        elif aslist:
            return channels
        else:
            return channels[0]

    def __setitem__(self, key, data):
        self[key].data = data

    def __str__(self):
        return "\n".join([str(chn) for chn in self.channels])



    def load(self, channels = None):
        """Load instrument data into the *channels*. *Channels* is a list or a
        tuple containing channels we will load data into. If None, all channels
        are loaded.
        """
        if channels is None:
            for chn in self.channel_list:
                self.channels_to_load |= set([chn[0]])

        elif(isinstance(channels, (list, tuple, set))):
            for chn in channels:
                try:
                    self.channels_to_load |= set([self[chn].name])
                except KeyError:
                    LOG.warning("Channel "+str(chn)+" not found,"
                                " will not load from raw data.")
        else:
            raise TypeError("Channels must be a list/"
                            "tuple/set of channel keys!")

        conf = ConfigParser.ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, self.fullname + ".cfg"))

        reader_name = conf.get(self.instrument_name + "-level2", 'format')
        try:
            reader_name = eval(reader_name)
        except NameError:
            reader_name = str(reader_name)
            
        reader = "satin."+reader_name
        try:
            reader_module = __import__(reader, globals(), locals(), ['load'])
            reader_module.load(self)
        except ImportError:
            raise ImportError("No "+reader+" reader found.")


    def get_lat_lon(self, resolution):
        """Get the latitude and longitude grids of the current region for the
        given *resolution*.
        """
        if not isinstance(resolution, int):
            raise TypeError("Resolution must be an integer number of meters.")

        conf = ConfigParser.ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, self.fullname + ".cfg"))

        reader_name = conf.get(self.instrument_name + "-level2", 'format')
        reader = "satin." + reader_name
        try:
            reader_module = __import__(reader, globals(), locals(), ['load'])
            return reader_module.get_lat_lon(self, resolution)
        except ImportError:
            raise ImportError("No "+reader+" reader found.")

    def check_channels(self, *channels):
        """Check if the *channels* are loaded, raise an error otherwise.
        """
        for chan in channels:
            if not self[chan].is_loaded():
                raise NotLoadedError("Required channel %s not loaded,"
                                     " aborting."%chan)

        return True

    def loaded_channels(self):
        """Return the set of loaded_channels.
        """
        return set([chan for chan in self.channels if chan.is_loaded()])

    def project(self, dest_area, channels=None, precompute=False, mode="quick"):
        """Make a copy of the current snapshot projected onto the
        *dest_area*. Available areas are defined in the region configuration
        file (ACPG). *channels* tells which channels are to be projected, and
        if None, all channels are projected and copied over to the return
        snapshot.

        Note: channels have to be loaded to be projected, otherwise an
        exception is raised.
        """
        
        _channels = set([])

        if channels is None:
            for chn in self.loaded_channels():
                _channels |= set([chn])

        elif(isinstance(channels, (list, tuple, set))):
            for chn in channels:
                try:
                    _channels |= set([self[chn]])
                except KeyError:
                    LOG.warning("Channel "+str(chn)+" not found,"
                                "thus not projected.")
        else:
            raise TypeError("Channels must be a list/"
                            "tuple/set of channel keys!")

        if self.area_id == dest_area:
            return self

        res = copy.copy(self)
        res.area_id = dest_area
        res.info["Area_Name"] = dest_area
        res.channels = []

        if not _channels <= self.loaded_channels():
            LOG.warning("Cannot project nonloaded channels: %s."
                        %(_channels - self.loaded_channels()))
            LOG.info("Will project the other channels though.")

        cov = {}

        for chn in _channels:
            if chn.area_id is None:
                if self.area_id is None:
                    area_name = ("swath_" + self.fullname + "_" +
                                 str(self.time_slot))
                    chn.area_id = area_name
                else:
                    chn.area_id = self.area_id
                    
            if chn.area_id == dest_area:
                res.channels.append(chn)
            else:
                if chn.area_id not in cov:
                    if chn.area_id.startswith("swath_"):
                        cov[chn.area_id] = \
                            Projector(chn.area_id,
                                      dest_area,
                                      self.get_lat_lon(chn.resolution),
                                      precompute=precompute,
                                      mode=mode)
                    else:
                        cov[chn.area_id] = Projector(chn.area_id,
                                                     dest_area,
                                                     precompute=precompute,
                                                     mode=mode)
                try:
                    res.channels.append(chn.project(cov[chn.area_id]))
                    res.channels[-1].area_id =  None
                    res.channels[-1].info["Area_Name"] =  ""
                except NotLoadedError:
                    LOG.warning("Channel "+str(chn.name)+" not loaded,"
                                "thus not projected.")

        return res
            
                
def assemble_swaths(swath_list):
    """Assemble the scene objects listed in *swath_list* into one.
    """
    channels = set([])
    for swt in swath_list:
        channels |= set([chn.name for chn in swt.loaded_channels()])
    
    new_swath = copy.deepcopy(swath_list[0])
    loaded_channels = set([chn.name for chn in new_swath.loaded_channels()])
    dummy = np.ma.masked_all_like(list(new_swath.loaded_channels())[0].data)
    
    for chn in channels - loaded_channels:
        new_swath[chn] = dummy
        
    for swt in swath_list[1:]:
        for chn in new_swath.loaded_channels():
            if swt[chn.name].is_loaded():
                chn.data = np.ma.concatenate((chn.data,
                                              swt[chn.name].data))
            else:
                chn.data = np.ma.concatenate((chn.data, dummy))
                
        new_swath.lon = np.ma.concatenate((new_swath.lon,
                                           swt.lon))
        new_swath.lat = np.ma.concatenate((new_swath.lat,
                                           swt.lat))

    return new_swath
