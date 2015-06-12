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
import weakref

class Plugin(object):
    """The base plugin class. It is not to be used as is, it has to be
    inherited by other classes.
    """
    pass

class Reader(Plugin):
    """Reader plugins. They should have a *pformat* attribute, and implement
    the *load* method. This is an abstract class to be inherited.
    """
    ptype = "reader"

    #TODO make config_file optional
    def __init__(self, config_file, **kwargs):
        """The reader plugin takes as input a satellite scene to fill in.
        
        Arguments:
        - `scene`: the scene to fill.
        """
        Plugin.__init__(self)
        self.config_file = config_file
        self.info = self.load_config()
        # Update info after loading the config in case a user provided option overrides config file
        self.info.update(kwargs)

    def load_config(self):
        conf = ConfigParser()
        conf.read(self.config_file)
        # Assumes only one section with "reader:" prefix
        info = {
            "channels": {},
        }
        for section_name in conf.sections():
            if section_name.startswith("reader:"):
                info.update(dict(conf.items(section_name)))
            elif section_name.startswith("channel:"):
                channel_info = self.parse_channel_section(section_name, dict(conf.items(section_name)))
                info["channels"][channel_info["uid"]] = channel_info
        return info

    def parse_channel_section(self, section_name, section_options):
        # Allow subclasses to make up their own rules about channels, but this is a good starting point
        if "file_patterns" in section_options:
            section_options["file_patterns"] = section_options["file_patterns"].split(",")
        return section_options

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
