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

"""The :mod:`mpop.scene` module defines satellite scenes. They are defined as generic
classes, to be inherited when needed.

A scene is a set of :mod:`mpop.channel` objects for a given time, and sometimes
also for a given area.
"""
import ConfigParser
import copy
import datetime
import os.path

import numpy as np

from mpop import CONFIG_PATH
from mpop.channel import Channel, NotLoadedError
from mpop.logger import LOG

class Satellite(object):
    """This is the satellite class. It contains information on the satellite.
    """
    #: Name of the satellite
    satname = ""

    #: Number of the satellite
    number = ""
    
    #: Variant of the satellite (often the communication channel it comes from)
    variant = ""

    def __init__(self, satname, number, variant):
        self.satname = satname
        self.number = number
        self.variant = variant

    @property
    def fullname(self):
        """Full name of the satellite, that is platform name and number
        (eg "metop02").
        """
        return self.variant + self.satname + self.number

class SatelliteScene(Satellite):
    """This is the satellite scene class. It is a capture of the satellite
    (channels) data at given *time_slot* and *area_id*/*area*.
    """

    #: Time of the snapshot
    time_slot = None

    #: Orbit number of the satellite
    orbit = None

    #: Name of the area on which the scene is defined.
    area_id = None

    #: Area on which the scene is defined.
    area_def = None

    #: Metadata information
    info = {}
    
    def __init__(self, time_slot=None, area_id=None, area=None, orbit=None):

        if(time_slot is not None and
           not isinstance(time_slot, datetime.datetime)):
            raise TypeError("Time_slot must be a datetime.datetime instance.")
        
        self.time_slot = time_slot

        
        if(area_id is not None):
            from warnings import warn
            warn("The *area_id* attribute is deprecated."
                 "Please use *area* instead.",
                 DeprecationWarning)
            if(not isinstance(area_id, str)):
                raise TypeError("Area must be a string.")

        self.area = area_id

        if area is not None:
            self.area = area
        
        if(orbit is not None and
           not isinstance(orbit, str)):
            raise TypeError("Orbit must be a string.")
        
        self.orbit = orbit


        self.lat = None
        self.lon = None


    def get_area(self):
        """Getter for area.
        """
        return self.area_def or self.area_id

    def set_area(self, area):
        """Setter for area.
        """
        if (area is None):
            self.area_def = None
            self.area_id = None
        elif(isinstance(area, str)):
            self.area_id = area
            self.area_def = None
        else:
            try:
                dummy = area.area_extent
                dummy = area.x_size
                dummy = area.y_size
                dummy = area.proj_id
                dummy = area.proj_dict
                self.area_def = area
                self.area_id = None
            except AttributeError:
                try:
                    dummy = area.lons
                    dummy = area.lats
                    self.area_def = area
                    self.area_id = None
                except AttributeError:
                    raise TypeError("Malformed area argument. "
                                    "Should be a string or an area object.")

    area = property(get_area, set_area)

class SatelliteInstrumentScene(SatelliteScene):
    """This is the satellite instrument class. It is an abstract channel
    container, from which all concrete satellite scenes should be derived.

    The constructor accepts as optional arguments the *time_slot* of the scene,
    the *area* on which the scene is defined (this can be use for slicing of
    big datasets, or can be set automatically when loading), and *orbit* which
    is a string giving the orbit number.
    """
    channels = []
    channel_list = []

    #. Instrument name
    instrument_name = None

    def __init__(self, time_slot=None, area_id=None, area=None, orbit=None):
        SatelliteScene.__init__(self, time_slot, area_id, area, orbit)
        self.channels = []
        
        for name, w_range, resolution in self.channel_list:
            self.channels.append(Channel(name=name,
                                         wavelength_range=w_range,
                                         resolution=resolution))
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

    def __iter__(self):
        return self.channels.__iter__()



    def load(self, channels=None, load_again=False, **kwargs):
        """Load instrument data into the *channels*. *Channels* is a list or a
        tuple containing channels we will load data into, designated by there
        center wavelength (float), resolution (integer) or name (string). If
        None, all channels are loaded.

        The *load_again* boolean flag allows to reload the channels even they
        have already been loaded, to mirror changes on disk for example. This
        is false by default.

        The other keyword arguments are passed as is to the reader
        plugin. Check the corresponding documentation for more details.
        """

        # Set up the list of channels to load.
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

        if not load_again:
            self.channels_to_load -= set([chn.name
                                          for chn in self.loaded_channels()])

        if len(self.channels_to_load) == 0:
            return

        
        # find the plugin to use from the config file
        conf = ConfigParser.ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, self.fullname + ".cfg"))

        reader_name = conf.get(self.instrument_name + "-level2", 'format')
        try:
            reader_name = eval(reader_name)
        except NameError:
            reader_name = str(reader_name)
            
        # read the data
        reader = "mpop.satin."+reader_name
        try:
            reader_module = __import__(reader, globals(), locals(), ['load'])
            reader_module.load(self, **kwargs)
        except ImportError:
            LOG.exception("ImportError while loading the reader")
            raise ImportError("No "+reader+" reader found.")



    def get_lat_lon(self, resolution):
        """Get the latitude and longitude grids of the current region for the
        given *resolution*.
        """

        from warnings import warn
        warn("The `get_lat_lon` function is deprecated."
             "Please use the area's `get_lonlats` method instead.",
             DeprecationWarning)

        
        if not isinstance(resolution, int):
            raise TypeError("Resolution must be an integer number of meters.")

        conf = ConfigParser.ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, self.fullname + ".cfg"))

        reader_name = conf.get(self.instrument_name + "-level2", 'format')
        try:
            reader_name = eval(reader_name)
        except NameError:
            reader_name = str(reader_name)
        reader = "mpop.satin." + reader_name
        try:
            reader_module = __import__(reader,
                                       globals(), locals(),
                                       ['get_lat_lon'])
            return reader_module.get_lat_lon(self, resolution)
        except ImportError:
            raise ImportError("No "+reader+" reader found.")

    def save(self, filename, to_format="netcdf4", compression=True, 
             data_type=np.int16):
        """Saves the current scene into a file of format *to_format*. Supported
        formats are:
        
        - *netcdf4*: NetCDF4 with CF conventions.
        """

        writer = "satout." + to_format
        try:
            writer_module = __import__(writer, globals(), locals(), ["save"])
        except ImportError, err:
            raise ImportError("Cannot load "+writer+" writer: "+str(err))

        return writer_module.save(self, filename, compression=compression, 
                                  data_type=data_type)

    def unload(self, channels=None):
        """Unloads *channels* from
        memory. :meth:`mpop.scene.SatelliteInstrumentScene.load` must be called
        again to reload the data.
        """
        for chn in channels:
            try:
                self[chn].data = None
            except AttributeError:
                LOG.warning("Can't unload channel" + str(chn))
        
        
    def add_to_history(self, message):
        """Adds a message to history info.
        """
        import datetime
        timestr = datetime.datetime.utcnow().isoformat()
        timed_message = str(timestr + " - " + message)
        if not self.info.get("history", ""):
            self.info["history"] = timed_message
        else:
            self.info["history"] += "\n" + timed_message
            

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
        # Lazy import in case pyresample is missing
        try:
            from mpop.projector import Projector
        except ImportError:
            LOG.exception("Cannot load reprojection module. "
                          "Is pyresample/pyproj missing ?")
            return self
        
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


        res = copy.copy(self)
        
        res.area = dest_area
        res.channels = []

        if not _channels <= self.loaded_channels():
            LOG.warning("Cannot project nonloaded channels: %s."
                        %(_channels - self.loaded_channels()))
            LOG.info("Will project the other channels though.")

        cov = {}

        for chn in _channels:
            if chn.area is None:
                if self.area is None:
                    area_name = ("swath_" + self.fullname + "_" +
                                 str(self.time_slot) + "_"
                                 + str(chn.shape))
                    chn.area = area_name
                else:
                    try:
                        from pyresample.geometry import AreaDefinition
                        chn.area = AreaDefinition(
                            self.area.area_id + str(chn.shape),
                            self.area.name,
                            self.area.proj_id,
                            self.area.proj_dict,
                            chn.shape[1],
                            chn.shape[0],
                            self.area.area_extent,
                            self.area.nprocs)

                    except AttributeError:
                        try:
                            dummy = self.area.lons
                            dummy = self.area.lats
                            chn.area = self.area
                            area_name = ("swath_" + self.fullname + "_" +
                                         str(self.time_slot) + "_"
                                         + str(chn.shape))
                            chn.area.area_id = area_name
                        except AttributeError:
                            chn.area = self.area + str(chn.shape)
                    except ImportError:
                        chn.area = self.area + str(chn.shape) 
                    
            if chn.area == dest_area:
                res.channels.append(chn)
            else:
                if chn.area not in cov:
                    if(isinstance(chn.area, str) and
                       chn.area.startswith("swath_")):
                        cov[chn.area] = \
                            Projector(chn.area,
                                      dest_area,
                                      self.get_lat_lon(chn.resolution),
                                      mode=mode)
                    else:
                        cov[chn.area] = Projector(chn.area,
                                                  dest_area,
                                                  mode=mode)
                    if precompute:
                        cov[chn.area].save()
                try:
                    res.channels.append(chn.project(cov[chn.area]))
                except NotLoadedError:
                    LOG.warning("Channel "+str(chn.name)+" not loaded, "
                                "thus not projected.")

        return res

def assemble_swaths(swath_list):
    """Assemble the scene objects listed in *swath_list* and returns the
    resulting scene object.
    """
    channels = set([])
    for swt in swath_list:
        channels |= set([chn.name for chn in swt.loaded_channels()])
    
    new_swath = copy.deepcopy(swath_list[0])
    loaded_channels = set([chn.name for chn in new_swath.loaded_channels()])
    all_mask = np.ma.masked_all_like(list(new_swath.loaded_channels())[0].data)
    
    for chn in channels - loaded_channels:
        new_swath[chn] = all_mask
        
    for swt in swath_list[1:]:
        for chn in new_swath.loaded_channels():
            if swt[chn.name].is_loaded():
                chn.data = np.ma.concatenate((chn.data,
                                              swt[chn.name].data))
            else:
                chn.data = np.ma.concatenate((chn.data, all_mask))
                
        new_swath.lon = np.ma.concatenate((new_swath.lon,
                                           swt.lon))
        new_swath.lat = np.ma.concatenate((new_swath.lat,
                                           swt.lat))

    return new_swath
