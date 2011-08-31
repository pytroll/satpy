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

"""Minimal script for polar data (from aapp level 1b).

We take the case of level 1b data (calibrated and geolocalized) from noaa 19,
as output from AAPP.

- Install mpop and pyresample
- Don't forget to set up the PPP_CONFIG_DIR variable to point to your
  configuration files.
- Edit the noaa19.cfg configuration file (a template is provided in case
  you don't have one) with your data directory:

  .. code-block:: ini
  
  [avhrr-level2]
  filename = hrpt_%(satellite)s_%Y%m%d_%H%M_%(orbit)s.l1b
  dir = /data/polar/
  format = aapp1b

- Here is an example of a minimal script that has to be called as soon as a new
  swath has arrived

"""

from mpop.satellites import PolarFactory

import sys
from datetime import datetime

if sys.version_info < (2, 5):
    import time
    def strptime(string, fmt=None):
        """This function is available in the datetime module only
        from Python >= 2.5.
        """

        return datetime(*time.strptime(string, fmt)[:6])

else:
    strptime = datetime.strptime

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: " + sys.argv[0] + " time_string orbit"
        sys.exit()

    time_string = sys.argv[1]
    orbit = sys.argv[2] 
    time_slot = strptime(time_string, "%Y%m%d%H%M")
    global_data = PolarFactory.create_scene("noaa", "19",
                                            "avhrr", time_slot, orbit)

    global_data.load()

    areas = ["euro4", "scan2"]

    for area in areas:

        local_data = global_data.project(area)

        img = local_data.image.overview()
        img.save("overview_" + area + "_" + time_string + ".png")

        img = local_data.image.cloudtop()
        img.save("cloudtop_" + area + "_" + time_string + ".png")
