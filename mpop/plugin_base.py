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

"""Plugin handling.
"""
import os.path
import sys
import weakref

from mpop import BASE_PATH
from mpop.satin.logger import LOG

class Plugin(object):
    """The base plugin class. It is not to be used as is, it has to be
    inherited by other classes.
    """
    pass

class Reader(Plugin):
    """Reader plugins. They should have a *pformat* attribute, and implement
    the *load* method.
    """
    ptype = "reader"
    def __init__(self, scene):
        """The reader plugin takes as input a satellite scene to fill in.
        
        Arguments:
        - `scene`: the scene to fill.
        """
        Plugin.__init__(self)
        self._scene = weakref.proxy(scene)

    def load(self, channels_to_load):
        """Loads the *channels_to_load* into the scene object.
        
        Arguments:
        - `channels_to_load`:
        """
        raise NotImplementedError


class Writer(Plugin):
    """Writer plugins. They must implement the *save* method.
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
        
        Arguments:
        - `filename`:
        """
        raise NotImplementedError


def get_plugin_types():
    """Get the list of available plugin types.
    """
    return [x.ptype for x in Plugin.__subclasses__()]

def _get_plugin_type(ptype):
    """Get the class of a given plugin type.
    """
    for k in Plugin.__subclasses__():
        if k.ptype == ptype:
            return k

def get_plugin(ptype, pformat):
    """Get the class of a plugin, given its *ptype* and *pformat*.
    """
    ptype_class = _get_plugin_type(ptype)
    for k in ptype_class.__subclasses__():
        if k.pformat == pformat:
            return k

def get_plugin_formats(ptype):
    """Get the different formats for plugins of a given *ptype*.
    """
    return [x.pformat for x in _get_plugin_type(ptype).__subclasses__()]


def load_plugins(directory):
    """Load all the plugins from *directory*.
    """
    sys.path.append(directory)
    for name in os.listdir(directory):
        if name.endswith(".py") and not name.startswith("__"):
            modulename = name[:-3]
            try:
                __import__(modulename)
                LOG.info("Imported plugin file "+
                         os.path.join(directory, name)+
                         ".")
            except Exception, exc:
                LOG.warning("Could not read plugin file "+
                            os.path.join(directory, name)+
                            ": "+str(exc))

# Load plugins on module import

import ConfigParser
from mpop import CONFIG_PATH

CONF = ConfigParser.ConfigParser()
CONF.read(os.path.join(CONFIG_PATH, "mpop.cfg"))

DIRECTORIES = [x.strip() for x in CONF.get("plugins", "dir").split(",")]

for plugin_dir in DIRECTORIES:
    if os.path.split(plugin_dir)[0] == '':
        load_plugins(os.path.join(BASE_PATH, "mpop", plugin_dir))
    else:
        load_plugins(plugin_dir)

                
