#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""This is the main module for terra handling.
"""

import logging
import numpy as np
import ConfigParser
import os.path
import glob

from pp.satellite.satellite import SatelliteSnapshot
from pp.terra import BASE_PATH


CONF = ConfigParser.ConfigParser()
CONF.read(os.path.join(BASE_PATH, "etc", "terra.cfg"))

L1_DIR = CONF.get('dirs_in', 'l1_dir')

LOG = logging.getLogger("pp.terra")

TERRA_MODIS = [["1", (0.6200, 0.6450, 0.6700), 250],
               ["2", (0.8410, 0.8585, 0.8760), 250],
               ["3", (0.4590, 0.4690, 0.4790), 500],
               ["4", (0.5450, 0.5550, 0.5650), 500],
               ["5", (1.2300, 1.2400, 1.2500), 500],
               ["6", (1.6280, 1.6400, 1.6520), 500],
               ["7", (2.1050, 2.1300, 2.1550), 500],
               ["8", (0.4050, 0.4125, 0.4200), 1000],
               ["9", (0.4380, 0.4430, 0.4480), 1000],
               ["10", (0.4830, 0.4880, 0.4930), 1000],
               ["11", (0.5260, 0.5310, 0.5360), 1000],
               ["12", (0.5460, 0.5510, 0.5560), 1000],
               ["13hi", (0.6620, 0.6670, 0.6720), 1000],
               ["13lo", (0.6620, 0.6670, 0.6720), 1000],
               ["14hi", (0.6730, 0.6780, 0.6830), 1000],
               ["14lo", (0.6730, 0.6780, 0.6830), 1000],
               ["15", (0.7430, 0.7480, 0.7530), 1000],
               ["16", (0.8620, 0.8695, 0.8770), 1000],
               ["17", (0.8900, 0.9050, 0.9200), 1000],
               ["18", (0.9310, 0.9360, 0.9410), 1000],
               ["19", (0.9150, 0.9400, 0.9650), 1000],
               ["20", (3.6600, 3.7500, 3.8400), 1000],
               ["21", (3.9290, 3.9590, 3.9890), 1000],
               ["22", (3.9290, 3.9590, 3.9890), 1000],
               ["23", (4.0200, 4.0500, 4.0800), 1000],
               ["24", (4.4330, 4.4655, 4.4980), 1000],
               ["25", (4.4820, 4.5155, 4.5490), 1000],
               ["26", (1.3600, 1.3750, 1.3900), 1000],
               ["27", (6.5350, 6.7150, 6.8950), 1000],
               ["28", (7.1750, 7.3250, 7.4750), 1000],
               ["29", (8.4000, 8.5500, 8.7000), 1000],
               ["30", (9.5800, 9.7300, 9.8800), 1000],
               ["31", (10.7800, 11.0300, 11.2800), 1000],
               ["32", (11.7700, 12.0200, 12.2700), 1000],
               ["33", (13.1850, 13.3350, 13.4850), 1000],
               ["34", (13.4850, 13.6350, 13.7850), 1000],
               ["35", (13.7850, 13.9350, 14.0850), 1000],
               ["36", (14.0850, 14.2350, 14.3850), 1000]]

class TerraModisSnapshot(SatelliteSnapshot):
    """A snapshot of Terra's modis instrument.
    """

    def __init__(self, *args, **kwargs):

        self.channel_list = TERRA_MODIS
        
        super(TerraModisSnapshot, self).__init__(*args, **kwargs)

        self.satname = "terra"
        self.number = kwargs.get("number", 0)

        if self.number != 0:
            number_string = str(self.number)
        else:
            number_string = "??"

        # L1B file
        
        filename = (L1_DIR+"/MOD021KM.A%Y%j.%H%M.???.?????????????.h5")

        file_list = glob.glob(self.time_slot.strftime(filename))

        if len(file_list) > 1:
            raise IOError("More than one l1b file matching!")
        elif len(file_list) == 0:
            raise IOError("No l1b file matching!")

        self.file = file_list[0]

        # Geolocation file

        filename = (L1_DIR+"/MOD03.A%Y%j.%H%M.???.?????????????.h5")

        file_list = glob.glob(self.time_slot.strftime(filename))

        if len(file_list) > 1:
            raise IOError("More than one geolocation file matching!")
        elif len(file_list) == 0:
            raise IOError("No geolocation file matching!")
        
        self.geo_file = file_list[0]

        # Area name

        self.area = self.time_slot.strftime("%Y%m%d%H%M")

    def load(self, channels = None):
        """Load data into the *channels*. Channels* is a list or a tuple
        containing channels we will load data into. If None, all channels are
        loaded.

        Note that with the current version of Ahamap (1.55), on which this
        library depends, it is not possible to load one channel at the time, so
        the user should read as many channels as needed at once.
        """
        import pp.terra.ppsPrepareModis as modis
        
        super(TerraModisSnapshot, self).load(channels)

        print self.file

        s250 = modis.getREFL250Agg(self.file)
        s500 = modis.getREFL500Agg(self.file)
        s1000 = modis.getREFL1000(self.file)
        temp_bright = modis.getTB(self.file)

        refl250 = s250[1]
        refl500 = s500[1]
        refl1000 = s1000[1]
        b_t1000 = temp_bright
        
        i = 0
        for (name, lamb, res) in self.channel_list:
            if i < 2:
                self[name] = np.ma.array(refl250[i, :, :] * 100,
                                         mask = (refl250[i, :, :] == -99.9))
            elif i < 7:
                self[name] = np.ma.array(refl500[i - 2, :, :] * 100,
                                         mask = (refl500[i - 2, :, :] == -99.9))
            elif i < 21:
                self[name] = np.ma.array(refl1000[i - 7, :, :] * 100,
                                         mask = (refl1000[i - 7, :, :] == -99.9))
            elif i < 27:
                self[name] = np.ma.array(b_t1000[i - 21, :, :] * 100,
                                         mask = (b_t1000[i - 21, :, :] == -99.9))
            elif i == 27:
                self[name] = np.ma.array(refl1000[14, :, :] * 100,
                                         mask = (refl1000[14, :, :] == -99.9))
            else:
                self[name] = np.ma.array(b_t1000[i - 22, :, :] * 100,
                                         mask = (b_t1000[i - 22, :, :] == -99.9))
            i = i + 1

    def get_lat_lon(self, resolution):
        """Get the latitude and longitude grids of the current region for the
        given *resolution*.
        """
        import pp.terra.ppsPrepareModis as modis
        
        geo_info = modis.getGeo2(self.geo_file)
        geo_gain = geo_info[10]
        
        return geo_info[0] * geo_gain, geo_info[1] * geo_gain
