#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2011, 2014 SMHI

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

"""Minimal script for geostationary production.


We take the case of HRIT data from meteosat 9, as send through eumetcast.

- Install mipp, mpop, and pyresample
- Don't forget to set up the PPP_CONFIG_DIR variable to point to your
  configuration files.
- Edit the meteosat09.cfg configuration file (a template is provided in case
  you don't have one) with your HRIT directory:

  .. code-block:: ini

   [seviri-level1]
   format = 'xrit/MSG'
   dir='/data/hrit_in'
   filename='H-000-MSG?__-MSG?________-%(channel)s-%(segment)s-%Y%m%d%H%M-__'
   filename_pro='H-000-MSG?__-MSG?________-_________-%(segment)s-%Y%m%d%H%M-__'
   filename_epi='H-000-MSG?__-MSG?________-_________-%(segment)s-%Y%m%d%H%M-__'

  where `/data/hrit_in` has to be changed to anything that suits your
  environment.

- Here is an example of a minimal script that has to be called as soon as an
  MSG slot has arrived (usually, watching the arrival of the epilogue file
  suffices)

"""

from mpop.utils import debug_on
debug_on()
from mpop.satellites import GeostationaryFactory

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
    if len(sys.argv) < 2:
        print "Usage: " + sys.argv[0] + " time_string"
        sys.exit()

    time_string = sys.argv[1]
    time_slot = strptime(time_string, "%Y%m%d%H%M")
    global_data = GeostationaryFactory.create_scene("meteosat", "09",
                                                    "seviri", time_slot)

    global_data.load()

    areas = ["euro4", "scan2"]

    for area in areas:

        local_data = global_data.project(area)

        img = local_data.image.overview()
        img.save("overview_" + area + "_" + time_string + ".png")

        img = local_data.image.fog()
        img.save("fog_" + area + "_" + time_string + ".png")
