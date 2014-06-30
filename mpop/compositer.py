#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Lars Ørum Rasmussen <ras@dmi.dk>

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

"""This module defines the generic Compositer class that builds rgb composites
from satellite channels.
"""

import types

class Compositer(object):
    def __init__(self, scene):
        self._data_holder = scene

    @property
    def area(self):
        """Get the current area.
        """
        return self._data_holder.area

    @property
    def time_slot(self):
        """Get the current time slot.
        """
        return self._data_holder.time_slot

    @property
    def fullname(self):
        """Get the full satellite name (ex 'meteosat10') of the scene.
        """
        return self._data_holder.fullname

    def __getitem__(self, *args, **kwargs):
        return self._data_holder.__getitem__(*args, **kwargs)

    def check_channels(self, *args):
        self._data_holder.check_channels(*args)

    @classmethod
    def remove_attribute(cls, name):
        """Remove an attribute from the class.
        """
        return delattr(cls, name)

    @classmethod
    def add_method(cls, func):
        """Add a method to the class.
        """
        return setattr(cls, func.__name__, func)

    def add_method_to_instance(self, func):
        """Add a method to the instance.
        """
        return setattr(self, func.__name__,
                       types.MethodType(func, self.__class__))
