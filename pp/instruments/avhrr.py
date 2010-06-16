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

"""This modules describes the avhrr instrument.
"""

from pp.instruments.visir import VisirScene
from pp.channel import NotLoadedError
from imageo import geo_image

AVHRR = [["1", (0.58, 0.635, 0.68), 1090],
         ["2", (0.725, 0.81, 1.00), 1090],
         ["3A", (1.58, 1.61, 1.64), 1090],
         ["3B", (3.55, 3.74, 3.93), 1090],
         ["4", (10.30, 10.80, 11.30), 1090],
         ["5", (11.50, 12.00, 12.50), 1090]]

class AvhrrScene(VisirScene):
    """This class sets up the Avhrr instrument channel list.
    """
    channel_list = AVHRR
    instrument_name = "avhrr"

        
    def cloudtop(self):
        """Make a Cloudtop RGB image composite. Use channel 3a (1.63μm) if
        channel 3b (3.75μm) is not available.
        """
        self.check_channels(10.8, 12.0)

        ch2 = -self[10.8].data
        ch3 = -self[12.0].data

        try:
            self.check_channels(3.75)
            ch1b = -self[3.75].data

            imgb = geo_image.GeoImage((ch1b, ch2, ch3),
                                      self.area_id,
                                      self.time_slot,
                                      fill_value = (0, 0, 0),
                                      mode = "RGB")
            
            imgb.enhance(stretch = (0.005, 0.005))

        except (NotLoadedError, KeyError):
            imgb = None

        try:
            self.check_channels(1.63)
            ch1a = self[1.63].data
            imga = geo_image.GeoImage((ch1a, ch2, ch3),
                                      self.area_id,
                                      self.time_slot,
                                      fill_value = (0, 0, 0),
                                      mode = "RGB")

            imga.enhance(stretch = (0.005, 0.005))

        except (NotLoadedError, KeyError):
            imga = None
        

        if imga and imgb:
            imgb.merge(imga)
            return imgb
        else:
            return (imgb or imga)
            
    cloudtop.prerequisites = set([1.63, 3.75, 10.8, 12.0])
