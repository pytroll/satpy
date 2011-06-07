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
"""
import sys
from datetime import timedelta, datetime
import glob
import os

from mpop.saturn.gatherer import Granule, Gatherer


def get_files_newer_than(directory, time_stamp):
    """Get the list of files from the *directory* which are newer than a given
 *time_stamp*.
    """
    filelist = glob.glob(os.path.join(directory, "*"))
    return [filename for filename in filelist
            if datetime.fromtimestamp(os.stat(filename)[8]) > time_stamp]


if __name__ == '__main__':

    directory = sys.argv[1]

    areas = ["euro4", "scan2"]

    gatherer = None

    time_stamp = datetime(1970, 1, 1)
    
    while True:
        new_time_stamp = datetime.now()
        filenames = get_files_newer_than(directory, time_stamp)
        time_stamp = new_time_stamp
        
        for filename in filenames:
            granule = Granule(filename)
            if gatherer is None:
                gatherer = Gatherer(areas_of_interest=areas,
                                    timeliness=timedelta(minutes=150),
                                    satellite=granule.satname,
                                    number=granule.number,
                                    variant=granule.variant)
            gatherer.add(granule)
            
        for swath in gatherer.finished_swaths:
            global_data = swath.concatenate()

            local_data = global_data.project(swath.area)

            time_string = global_data.time_slot.strftime("%Y%m%d%H%M")

            img = local_data.image.overview()
            img.save("overview_" + swath.area + "_" + time_string + ".png")

            img = local_data.image.cloudtop()
            img.save("cloudtop_" + swath.area + "_" + time_string + ".png")
        
