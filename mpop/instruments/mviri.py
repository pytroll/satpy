#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011.

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

"""This modules describes the mviri instrument.
"""
import mpop.instruments.visir
from mpop.imageo import geo_image


class MviriCompositer(mpop.instruments.visir.VisirCompositer):
    """This class sets up the Seviri instrument channel list.
    """

    instrument_name = "mviri"
    

    def overview(self):
        """Make an overview RGB image composite.
        """
        self.check_channels(0.635, 0.85, 10.8)

        ch1 = self[0.635].check_range(10)
        ch2 = self[0.85].check_range(10)
        ch3 = -self[10.8].data
        
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")
        
        img.enhance(stretch = "crude")
        img.enhance(gamma = 1.6)

        return img
    
    overview.prerequisites = set([0.635, 0.85, 10.8])

    def ir108(self):
        """Make a black and white image of the IR 10.8um channel.
        """
        self.check_channels(10.8)

        img = geo_image.GeoImage(self[10.8].data,
                                 self.area,
                                 self.time_slot,
                                 fill_value=0,
                                 mode="L")
        
        img.enhance(inverse=True)
        img.enhance(stretch="crude")

        return img

    ir108.prerequisites = set([10.8])
