#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
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

"""SMHI's custom composites, to use with mpop.
"""
import mpop.imageo.palettes
import numpy as np
from mpop.channel import NotLoadedError
from mpop.imageo import geo_image, image_processing
from mpop.satin.logger import LOG


# VIIRS
# Since there is overlap between I-bands and M-bands we need to 
# specifically re-define some of the RGB composites already defined
# in the standard visir.py module. So, the same names, like "overview"
# can be invoked and based on M-bands only.
#
def overview(self):
    """Make an Overview RGB image composite from VIIRS
    channels.
    """
    self.check_channels('M05', 'M07', 'M15')
    
    ch1 = self['M05'].check_range()
    ch2 = self['M07'].check_range()
    ch3 = -self['M15'].data
        
    img = geo_image.GeoImage((ch1, ch2, ch3),
                             self.area,
                             self.time_slot,
                             fill_value=(0, 0, 0),
                             mode="RGB")
        
    img.enhance(stretch="crude")
    img.enhance(gamma=1.6)
    
    return img

overview.prerequisites = set(['M05', 'M07', 'M15'])


def trucolor(self):
    """Make a True Color RGB image composite from
    M-bands only.
    """
    self.check_channels('M02', 'M04', 'M05')
    
    ch1 = self['M05'].check_range()
    ch2 = self['M04'].check_range()
    ch3 = self['M02'].check_range()

    img = geo_image.GeoImage((ch1, ch2, ch3),
                             self.area,
                             self.time_slot,
                             fill_value=(0, 0, 0),
                             mode="RGB")

    img.enhance(stretch="crude")
    img.enhance(gamma=2.0)
    
    return img

trucolor.prerequisites = set(['M02', 'M04', 'M05'])

def natural(self):
    """Make a Natural Colors RGB image composite from
    M-bands only.
    """
    self.check_channels('M05', 'M07', 'M10')
    
    ch1 = self['M10'].check_range()
    ch2 = self['M07'].check_range()
    ch3 = self['M05'].check_range()

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

natural.prerequisites = set(['M05', 'M07', 'M10'])


def vis06(self):
    """Make a black and white image of the VIS 0.635um channel.
    """
    return self.channel_image(0.6)

vis06.prerequisites = set(['M05'])


def green_snow(self):
    """Make a Green Snow RGB image composite.
    """
    self.check_channels('M07', 'M10', 'M15')

    ch1 = self['M10'].check_range()
    ch2 = self['M07'].check_range()
    ch3 = -self['M15'].data
    
    img = geo_image.GeoImage((ch1, ch2, ch3),
                             self.area,
                             self.time_slot,
                             fill_value=(0, 0, 0),
                             mode="RGB")

    img.enhance(stretch = "crude")
    img.enhance(gamma = 1.6)

    return img

green_snow.prerequisites = set(['M07', 'M10', 'M15'])


def red_snow(self):
    """Make a Red Snow RGB image composite.
    """
    self.check_channels('M0', 'M10', 'M15')
    
    ch1 = self['M05'].check_range()
    ch2 = self['M10'].check_range()
    ch3 = -self['M15'].data
    
    img = geo_image.GeoImage((ch1, ch2, ch3),
                             self.area,
                             self.time_slot,
                             fill_value=(0, 0, 0),
                             mode="RGB")

    img.enhance(stretch = "crude")
        
    return img

red_snow.prerequisites = set(['M05', 'M10', 'M15'])


def night_fog(self):
    """Make a Night Fog RGB image composite.
    """
    self.check_channels('M12', 'M15', 'M16')
    
    ch1 = self['M16'].data - self['M15'].data
    ch2 = self['M15'].data - self['M12'].data
    ch3 = self['M15'].data
    
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

night_fog.prerequisites = set(['M12', 'M15', 'M16'])


def cloudtop(self):
    """Make a Cloudtop RGB image composite.
    """
    self.check_channels('M12', 'M15', 'M16')
    
    ch1 = -self['M12'].data
    ch2 = -self['M15'].data
    ch3 = -self['M16'].data
    
    img = geo_image.GeoImage((ch1, ch2, ch3),
                             self.area,
                             self.time_slot,
                             fill_value=(0, 0, 0),
                             mode="RGB")

    img.enhance(stretch=(0.005, 0.005))

    return img

cloudtop.prerequisites = set(['M12', 'M15', 'M16'])



viirs = [overview, natural, 
         vis06, trucolor,
         green_snow, red_snow,
         night_fog, cloudtop
         ]

