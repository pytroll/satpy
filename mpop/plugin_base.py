#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2011 SMHI

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`mpop.plugin_base` module defines the plugin API.
"""
from ConfigParser import ConfigParser
from mpop.projectable import Projectable
import weakref
import numbers

class Plugin(object):
    """The base plugin class. It is not to be used as is, it has to be
    inherited by other classes.
    """
    pass

class Reader(Plugin):
    """Reader plugins. They should have a *pformat* attribute, and implement
    the *load* method. This is an abstract class to be inherited.
    """
    def __init__(self, name, config_file, **kwargs):
        """The reader plugin takes as input a satellite scene to fill in.
        
        Arguments:
        - `scene`: the scene to fill.
        """
        #TODO make config_file optional
        Plugin.__init__(self)
        self.name = name
        self.config_file = config_file
        self.description = kwargs.pop("description", "")
        self.file_patterns = kwargs.pop("file_patterns", None)
        self.filenames = set(kwargs.pop("filenames", []))
        self.start_time = kwargs.pop("start_time", None)
        self.end_time = kwargs.pop("end_time", None)
        self.area = kwargs.pop("area", None)
        self.channels = {}

        self.load_config()

    def add_filenames(self, *filenames):
        self.filenames |= set(filenames)

    @property
    def channel_names(self):
        """Names of all channels configured for this reader.
        """
        return sorted(self.channels.keys())

    @property
    def sensor_names(self):
        """Sensors supported by this reader.
        """
        return set([sensor_name for chn_info in self.channels.values()
                    for sensor_name in chn_info["sensor"].split(",")])

    def load_config(self):
        conf = ConfigParser()
        conf.read(self.config_file)
        # Assumes only one section with "reader:" prefix
        info = {}
        for section_name in conf.sections():
            # Can't load the reader section because any options not specified in keywords don't use the default
            # if section_name.startswith("reader:"):
            #     info.update(dict(conf.items(section_name)))
            if section_name.startswith("channel:"):
                channel_info = self.parse_channel_section(section_name, dict(conf.items(section_name)))
                self.channels[channel_info["uid"]] = channel_info
        return info

    def parse_channel_section(self, section_name, section_options):
        # Allow subclasses to make up their own rules about channels, but this is a good starting point
        if "file_patterns" in section_options:
            section_options["file_patterns"] = section_options["file_patterns"].split(",")
        if "wavelength_range" in section_options:
            section_options["wavelength_range"] = [float(wl) for wl in section_options["wavelength_range"].split(",")]
        return section_options

    def get_channel(self, key):
        """Get the channel corresponding to *key*, either by name or centerwavelength.
        """
        if isinstance(key, numbers.Number):
            channels = [chn for chn in self.channels.values()
                        if("wavelength_range" in chn and
                           chn["wavelength_range"][0] <= key <=chn["wavelength_range"][2])]
            channels = sorted(channels,
                              lambda ch1, ch2:
                              cmp(abs(ch1["wavelength_range"][1] - key),
                                  abs(ch2["wavelength_range"][1] - key)))

            if not channels:
                raise KeyError("Can't find any projectable at %gum" % key)
            return channels[0]
        # get by name
        else:
            return self.channels[key]
        raise KeyError("No channel corresponding to " + str(key) + ".")

    def load(self, channels_to_load):
        """Loads the *channels_to_load* into the scene object.
        """
        raise NotImplementedError


class Writer(Plugin):
    """Writer plugins. They must implement the *save* method. This is an
    abstract class to be inherited.
    """
    ptype = "writer"
    
    def __init__(self, scene):
        """The writer saves the *scene* to *filename*.
        
        Arguments:
        - `scene`: the scene to save.
        - `filename`: the place to save it.
        """
        Plugin.__init__(self)
        self._scene = weakref.proxy(scene)

    def save(self, filename):
        """Saves the scene to a given *filename*.
        """
        raise NotImplementedError
