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

"""Minimal script for assembling and processing segmented polar data.

We take the case of level 1b data (calibrated and geolocalized) from noaa 19,
as output from AAPP.

- Install satpy and pyresample
- Don't forget to set up the PPP_CONFIG_DIR variable to point to your
  configuration files.
- Edit the gdsmetop02.cfg configuration file (a template is provided in case
  you don't have one) with your data directory:
  
  .. code-block:: ini
  
   [avhrr-granules]
   type=eps_avhrr
   granularity=60
   full_scan_period=0.1667
   scan_width=2048
   dir=/data/prod/satellit/ears/avhrr
   filename=AVHR_xxx_1B_M02_%Y%m%d%H%M*


- Here is a minimal script that monitors a directory and builds composites:
"""
import sys
from datetime import timedelta, datetime
import glob
import os
import time

from satpy.saturn.gatherer import Granule, Gatherer


def get_files_newer_than(directory, time_stamp):
    """Get the list of files from the *directory* which are newer than a given
 *time_stamp*.
    """
    filelist = glob.glob(os.path.join(directory, "*"))
    return [filename for filename in filelist
            if datetime.fromtimestamp(os.stat(filename)[8]) > time_stamp]


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: " + sys.argv[0] + " directory wait_for_more"
        sys.exit()
        
    directory = sys.argv[1]
    # if we wait for files in the directory forever or not
    wait_for_more = eval(sys.argv[2])

    areas = ["euro4", "scan2"]

    gatherer = None

    time_stamp = datetime(1970, 1, 1)
    
    while True:

        # Scanning directory
        
        new_time_stamp = datetime.now()
        filenames = get_files_newer_than(directory, time_stamp)
        time_stamp = new_time_stamp

        # Adding files to the gatherer
        
        for filename in filenames:
            granule = Granule(filename)
            if gatherer is None:
                gatherer = Gatherer(areas_of_interest=areas,
                                    timeliness=timedelta(minutes=150),
                                    satname=granule.satname,
                                    number=granule.number,
                                    variant=granule.variant)
            gatherer.add(granule)

        # Build finished swath and process them.
            
        for swath in gatherer.finished_swaths:
            global_data = swath.concatenate()

            local_data = global_data.project(swath.area)

            time_string = global_data.time_slot.strftime("%Y%m%d%H%M")

            area_id = swath.area.area_id
            
            img = local_data.image.overview()
            img.save("overview_" + area_id + "_" + time_string + ".png")

            img = local_data.image.natural()
            img.save("natural_" + area_id + "_" + time_string + ".png")

        if not wait_for_more:
            break
        
        # wait 60 seconds before restarting
        time.sleep(60)
