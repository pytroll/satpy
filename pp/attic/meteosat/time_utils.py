#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""This module adds a few time utilities to the Meteosat plugin. It is based on
the datetime.datetime object.
"""

import datetime
import numpy

def time_slots(n_slots, dtime = 15):
    """Returns an array of time slots for the previous satellite
     shots, where the time of the last slot is close to now, and the
     start slot is *n_slots* earlier. *dt* is the time difference in
     minutes between two successive time slots. *dt* is supposed to be
     a divisor of 60.
     """
    deltat = datetime.timedelta(minutes = dtime)
    now = datetime.datetime.utcnow()
    snap = now.replace(minute = (now.minute // dtime) * dtime, 
                       second = 0, 
                       microsecond = 0)

    slots = numpy.empty(n_slots, datetime.datetime)
    slot = snap

    for i in range(n_slots):
        slots[- int(i + 1)] =  slot
        slot = slot - deltat

    return slots

def time_string(time_slot):
    """Return the standart MSG time string corresponding to the
    datetime *time_slot* object."""
    return time_slot.strftime("%Y%m%d%H%M")

