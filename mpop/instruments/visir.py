#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011.

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Lars Ørum Rasmussen <ras@dmi.dk>

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

"""This module defines the generic VISIR instrument class.
"""
from mpop.scene import SatelliteInstrumentScene
from mpop.imageo import geo_image
from mpop.compositer import Compositer

#pylint: disable=W0612
# remove warnings for unused prerequisites

class VisirCompositer(Compositer):

    def channel_image(self, channel, fill_value=0):
        """Make a black and white image of the *channel*.
        """
        self.check_channels(channel)

        img = geo_image.GeoImage(self[channel].data,
                                 self[channel].area,
                                 self.time_slot,
                                 fill_value=fill_value,
                                 mode="L")
        img.enhance(stretch="crude")
        return img

    def overview(self):
        """Make an overview RGB image composite.
        """
        self.check_channels(0.635, 0.85, 10.8)

        ch1 = self[0.635].check_range()
        ch2 = self[0.85].check_range()
        ch3 = -self[10.8].data
        
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")
        
        img.enhance(stretch="crude")
        img.enhance(gamma=1.6)

        return img

    overview.prerequisites = set([0.635, 0.85, 10.8])

    def natural(self):
        """Make a Natural Colors RGB image composite.
        """
        self.check_channels(0.635, 0.85, 1.63)
        
        ch1 = self[1.63].check_range()
        ch2 = self[0.85].check_range()
        ch3 = self[0.635].check_range()

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((0, 90),
                                         (0, 90),
                                         (0, 90)))



        img.enhance(gamma=1.8)

        return img
    
    natural.prerequisites = set([0.635, 0.85, 1.63])

    def airmass(self):
        """Make an airmass RGB image composite.
        
        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | WV6.2 - WV7.3      |     -25 to 0 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR9.7 - IR10.8     |     -40 to 5 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | WV6.2              |   243 to 208 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        """
        self.check_channels(6.7, 7.3, 9.7, 10.8)

        ch1 = self[6.7].data - self[7.3].data
        ch2 = self[9.7].data - self[10.8].data
        ch3 = self[6.7].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-25, 0),
                                         (-40, 5),
                                         (243, 208)))
        return img
            
    airmass.prerequisites = set([6.7, 7.3, 9.7, 10.8])

    def convection(self):
        """Make a Severe Convection RGB image composite.
        """
        self.check_channels(0.635, 1.63, 3.75, 6.7, 7.3, 10.8)

        ch1 = self[6.7].data - self[7.3].data
        ch2 = self[3.75].data - self[10.8].data
        ch3 = self[1.63].check_range() - self[0.635].check_range()

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-30, 0),
                                         (0, 55),
                                         (-70, 20)))


        return img

    convection.prerequisites = set([0.635, 1.63, 3.75, 6.7, 7.3, 10.8])


    def dust(self):
        """Make a Dust RGB image composite.
        """
        self.check_channels(8.7, 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self[8.7].data
        ch3 = self[10.8].data
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-4, 2),
                                         (0, 15),
                                         (261, 289)))

        img.enhance(gamma=(1.0, 2.5, 1.0))
        
        return img

    dust.prerequisites = set([8.7, 10.8, 12.0])

    def ash(self):
        """Make a Ash RGB image composite.
        """
        self.check_channels(8.7, 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self[8.7].data
        ch3 = self[10.8].data
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-4, 2),
                                         (-4, 5),
                                         (243, 303)))

        return img

    ash.prerequisites = set([8.7, 10.8, 12.0])


    def fog(self):
        """Make a Fog RGB image composite.
        """
        self.check_channels(8.7, 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self[8.7].data
        ch3 = self[10.8].data
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-4, 2),
                                         (0, 6),
                                         (243, 283)))

        img.enhance(gamma=(1.0, 2.0, 1.0))
        
        return img

    fog.prerequisites = set([8.7, 10.8, 12.0])

    def night_fog(self):
        """Make a Night Fog RGB image composite.
        """
        self.check_channels(3.75, 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self[3.75].data
        ch3 = self[10.8].data
        
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-4, 2),
                                         (0, 6),
                                         (243, 293)))
        
        img.enhance(gamma=(1.0, 2.0, 1.0))

        return img

    night_fog.prerequisites = set([3.75, 10.8, 12.0])

    def cloudtop(self):
        """Make a Cloudtop RGB image composite.
        """
        self.check_channels(3.75, 10.8, 12.0)

        ch1 = -self[3.75].data
        ch2 = -self[10.8].data
        ch3 = -self[12.0].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")

        img.enhance(stretch=(0.005, 0.005))

        return img

    cloudtop.prerequisites = set([3.75, 10.8, 12.0])
#pylint: enable=W0612

