#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""This modules describes the seviri instrument.
"""

import numpy as np
from pp.instruments.visir import VisirScene
import imageo.geo_image as geo_image
from pp.logger import LOG

SEVIRI = [["VIS006", (0.56, 0.635, 0.71), 3000],
          ["VIS008", (0.74, 0.81, 0.88), 3000],
          ["IR_016", (1.50, 1.64, 1.78), 3000],
          ["IR_039", (3.48, 3.92, 4.36), 3000],
          ["WV_062", (5.35, 6.25, 7.15), 3000],
          ["WV_073", (6.85, 7.35, 7.85), 3000],
          ["IR_087", (8.30, 8.70, 9.10), 3000],
          ["IR_097", (9.38, 9.66, 9.94), 3000],
          ["IR_108", (9.80, 10.80, 11.80), 3000],
          ["IR_120", (11.00, 12.00, 13.00), 3000],
          ["IR_134", (12.40, 13.40, 14.40), 3000],
          ["HRV", (0.50, 0.70, 0.90), 1000]]

class SeviriScene(VisirScene):
    """This class sets up the Seviri instrument channel list.
    """

    channel_list = SEVIRI
    instrument_name = "seviri"

    def co2corr(self):
        """CO2 correction of the brightness temperature of the MSG 3.9um
        channel::

        .. math::
        
          T4_CO2corr = (BT(IR3.9)^4 + Rcorr)^0.25
          Rcorr = BT(IR10.8)^4 - (BT(IR10.8)-dt_CO2)^4
          dt_CO2 = (BT(IR10.8)-BT(IR13.4))/4.0

          
        """
        try:
            self.check_channels(3.9, 10.8, 13.4)
        except RuntimeError:
            LOG.warning("CO2 correction not performed, channel data missing.")
            return



        bt039 = self[3.9].data
        bt108 = self[10.8].data
        bt134 = self[13.4].data
        
        dt_co2 = (bt108-bt134)/4.0
        rcorr = bt108 ** 4 - (bt108-dt_co2) ** 4
        
        
        t4_co2corr = bt039 ** 4 + rcorr
        t4_co2corr = np.ma.where(t4_co2corr > 0.0, t4_co2corr, 0)
        t4_co2corr = t4_co2corr ** 0.25
        
        return t4_co2corr

    co2corr.prerequisites = set([3.9, 10.8, 13.4])

    def cloudtop(self):
        """Make a Cloudtop RGB image composite from Seviri channels.
        """
        self.check_channels("_IR39Corr", 10.8, 12.0)

        ch1 = -self["_IR39Corr"].data
        ch2 = -self[10.8].data
        ch3 = -self[12.0].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")

        img.enhance(stretch=(0.005, 0.005))

        return img
    
    cloudtop.prerequisites = set(["_IR39Corr", 10.8, 12.0])

    def night_fog(self):
        """Make a Night Fog RGB image composite from Seviri channels.
        """
        self.check_channels("_IR39Corr", 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self["_IR39Corr"].data
        ch3 = self[10.8].data
        
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode="RGB",
                                 crange=((-4, 2),
                                         (0, 6),
                                         (243, 293)))
        
        img.enhance(gamma = (1.0, 2.0, 1.0))

        return img

    night_fog.prerequisites = set(["_IR39Corr", 10.8, 12.0])


    def hr_overview(self):
        """Make a High Resolution Overview RGB image composite from Seviri
        channels.
        """
        self.check_channels(0.635, 0.85, 10.8, "HRV")

        ch1 = self[0.635].check_range()
        ch2 = self[0.85].check_range()
        ch3 = -self[10.8].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")

        img.enhance(stretch="crude")
        img.enhance(gamma=[1.6, 1.6, 1.1])
        
        luminance = geo_image.GeoImage((self["HRV"].data),
                                       self.area,
                                       self.time_slot,
                                       crange=(0, 100),
                                       mode="L")

        luminance.enhance(gamma=2.0)

        img.replace_luminance(luminance.channels[0])
        
        return img

    hr_overview.prerequisites = set(["HRV", 0.635, 0.85, 10.8])

    def hr_visual(self):
        """Make a High Resolution visual BW image composite from Seviri
        channels.
        """
        self.check_channels("HRV")

        img = geo_image.GeoImage(self["HRV"].data,
                                 self.area,
                                 self.time_slot,
                                 fill_value=0,
                                 mode="L")
        img.enhance(stretch="crude")
        return img

    hr_visual.prerequisites = set(["HRV"])
