#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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

import mpop.imageo.geo_image as geo_image
from mpop.instruments.visir import VisirScene
from mpop.logger import LOG


MODIS = [["Rrs_412", (0.405, 0.412, 0.420), 1000],
         ["Rrs_443", (0.438, 0.443, 0.488), 1000],
         ["Rrs_469", (0.459, 0.469, 0.479), 1000],
         ["Rrs_488", (0.483, 0.488, 0.493), 1000],
         ["Rrs_531", (0.526, 0.531, 0.536), 1000],
         ["Rrs_547", (0.546, 0.547, 0.556), 1000],
         ["Rrs_555", (0.545, 0.555, 0.565), 1000],
         ["Rrs_645", (0.620, 0.645, 0.670), 1000],
         ["Rrs_667", (0.662, 0.667, 0.672), 1000]
         ]

class ModisScene(VisirScene):
    """This class sets up the Modis instrument channel list.
    """

    channel_list = MODIS
    instrument_name = "modis"


    def ocean_colour(self):
        """Make a daytime ocean colour RGB composite from Modis channels.
        """

        self.check_channels(0.645, 0.555, 0.469)

        ch1 = self[0.645].data
        ch2 = self[0.555].data
        ch3 = self[0.469].data
        
        cranges = ((-0.015, 0.115), (-0.015, 0.115), (-0.015, 0.115))
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode="RGB",
                                 crange=cranges)        

        return img

    ocean_colour.prerequisites = set([0.645, 0.555, 0.469])
