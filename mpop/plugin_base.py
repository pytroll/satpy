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
    def __init__(self, scene):
        """The reader plugin takes as input a satellite scene to fill in.
        
        Arguments:
        - `scene`: the scene to fill.
        """
        Plugin.__init__(self)
        self._scene = weakref.proxy(scene)

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
