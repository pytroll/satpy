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

"""This modules describes the modis instrument.
It provides MODIS specific methods for RGB-compositing.
"""
import numpy as np

import mpop.imageo.geo_image as geo_image
from mpop.instruments.visir import VisirCompositer
from mpop.logger import LOG

MODIS = [["8", (0.405, 0.4125, 0.420), 1000],
         ["9", (0.438, 0.443, 0.488), 1000],
         ["3", (0.459, 0.469, 0.479), 1000],
         ["10", (0.483, 0.488, 0.493), 1000],
         ["11", (0.526, 0.531, 0.536), 1000],
         ["12", (0.546, 0.551, 0.556), 1000],
         ["4", (0.545, 0.555, 0.565), 1000],
         ["1", (0.620, 0.645, 0.670), 1000],
         ["13", (0.662, 0.667, 0.672), 1000]
         ]

# MODIS
#
class ModisCompositer(VisirCompositer):
    """This class sets up the Modis channel list and
    instrument specific composite methods.
    """

    channel_list = MODIS
    instrument_name = "modis"

    def truecolor(self):
        """Make a daytime true color RGB composite from Modis channels.
        """

        self.check_channels(0.645, 0.555, 0.469)

        ch1 = self[0.645].data / 100.
        ch2 = self[0.555].data / 100.
        ch3 = self[0.469].data / 100.
        
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value = None,
                                 mode="RGB")

        img.enhance(stretch="histogram")

        return img

    truecolor.prerequisites = set([0.645, 0.555, 0.469])
