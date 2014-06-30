#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2013, 2014.

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
from mpop.imageo import geo_image
from mpop.compositer import Compositer

import numpy as np

try:
    from pyorbital.astronomy import sun_zenith_angle as sza
except ImportError:
    sza = None

#pylint: disable=W0612
# remove warnings for unused prerequisites

class VisirCompositer(Compositer):
    """Compositer for Visual-IR instruments
    """

    def __call__(self, *channels, **keys):
        """Build a geoimage.
        e.g.:
        img = l.image(0.6, 0.8, -10.8, mode="RGB")
        """

        data = []
        area = None
        inv = []
        new_channels = []

        for channel in channels:
            if isinstance(channel, str):
                if channel.startswith("-"):
                    inv.append(True)
                    channel = channel[1:]
                else:
                    inv.append(False)
            else:
                if channel < 0:
                    inv.append(True)
                    channel = -channel
                else:
                    inv.append(False)

            new_channels.append(channel)

            data.append(self[channel].data)

            new_area = self[channel].area
            if area and (new_area != area):
                raise ValueError("Channels should have the same area")
            else:
                area = new_area

        self.check_channels(*new_channels)

        img = geo_image.GeoImage(data,
                                 area=area,
                                 time_slot=self.time_slot,
                                 fill_value=keys.get("fill_value", None),
                                 crange=keys.get("crange", None),
                                 mode=keys.get("mode", None))

        img.enhance(inverse=inv,
                    gamma=keys.get("gamma", 1.0),
                    stretch=keys.get("stretch", "no"))

        return img

    def channel_image(self, channel, fill_value=0):
        """Make a black and white image of the *channel*.

        Linear stretch without clipping is applied by default.
        """
        self.check_channels(channel)

        img = geo_image.GeoImage(self[channel].data,
                                 self[channel].area,
                                 self.time_slot,
                                 fill_value=fill_value,
                                 mode="L")
        img.enhance(stretch="crude")
        return img

    def overview(self, stretch='crude', gamma=1.6):
        """Make an overview RGB image composite.

        +--------------------+--------------------+
        | Channels           | Gamma (default)    |
        +====================+====================+
        | VIS0.6             | gamma 1.6          |
        +--------------------+--------------------+
        | VIS0.8             | gamma 1.6          |
        +--------------------+--------------------+
        | IR10.8 (inverted)  | gamma 1.6          |
        +--------------------+--------------------+

        Linear stretch without clipping is applied.
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

        if stretch:
            img.enhance(stretch=stretch)
        if gamma:
            img.enhance(gamma=gamma)

        return img

    overview.prerequisites = set([0.635, 0.85, 10.8])



    def overview_sun(self, stretch='crude', gamma=1.6):
        """Make an overview RGB image composite normalising with cosine to the
        sun zenith angle.
        """
        self.check_channels(0.635, 0.85, 10.8)

        lonlats = self[10.8].area.get_lonlats()

        sunz = sza(self.time_slot, lonlats[0], lonlats[1])
        sunz = np.ma.masked_outside(sunz, 0.0, 88.0)
        sunzmask = sunz.mask
        sunz = sunz.filled(88.)

        costheta = np.cos(np.deg2rad(sunz))

        red = np.ma.masked_where(sunzmask, self[0.635].data / costheta)
        green = np.ma.masked_where(sunzmask, self[0.85].data / costheta)
        blue = -self[10.8].data
        
        img = geo_image.GeoImage((red, green, blue),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")

        if stretch:
            img.enhance(stretch=stretch)
        if gamma:
            img.enhance(gamma=gamma)

        return img

    overview_sun.prerequisites = set([0.635, 0.85, 10.8])



    def night_overview(self, stretch='histogram', gamma=None):
        """Make an overview RGB image composite using IR channels.

        +--------------------+--------------------+
        | Channels           | Gamma              |
        +====================+====================+
        | IR3.9 (inverted)   | gamma 1            |
        +--------------------+--------------------+
        | IR10.8 (inverted)  | gamma 1            |
        +--------------------+--------------------+
        | IR12.0 (inverted)  | gamma 1            |
        +--------------------+--------------------+

        Histogram equalization is applied for each channel.
        """
        return self.cloudtop(stretch=stretch, gamma=gamma)

    night_overview.prerequisites = set([3.75, 10.8, 12.0])


    def natural(self, stretch=None, gamma=1.8):
        """Make a Natural Colors RGB image composite.

        +--------------------+--------------------+--------------------+
        | Channels           | Range (reflectance)| Gamma (default)    |
        +====================+====================+====================+
        | IR1.6              | 0 - 90             | gamma 1.8          |
        +--------------------+--------------------+--------------------+
        | VIS0.8             | 0 - 90             | gamma 1.8          |
        +--------------------+--------------------+--------------------+
        | VIS0.6             | 0 - 90             | gamma 1.8          |
        +--------------------+--------------------+--------------------+
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

        if stretch:
            img.enhance(stretch=stretch)
        if gamma:
            img.enhance(gamma=gamma)

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


    def vis06(self):
        """Make a black and white image of the VIS 0.635um channel.

        Linear stretch without clipping is applied.
        """
        return self.channel_image(0.6)

    vis06.prerequisites = set([0.635])

    def ir108(self):
        """Make a black and white image of the IR 10.8um channel.

        Channel is inverted. Temperature range from -70 °C (white) to
        +57.5 °C (black) is shown.
        """
        self.check_channels(10.8)

        img = geo_image.GeoImage(self[10.8].data,
                                 self.area,
                                 self.time_slot,
                                 fill_value=0,
                                 mode="L",
                                 crange=(-70 + 273.15, 57.5 + 273.15))
        img.enhance(inverse=True)
        return img

    ir108.prerequisites = set([10.8])

    def wv_high(self):
        """Make a black and white image of the IR 6.7um channel.

        Channel inverted and a linear stretch is applied with 0.5 %
        clipping at both ends.
        """
        self.check_channels(6.7)

        img = geo_image.GeoImage(self[6.7].data,
                                 self.area,
                                 self.time_slot,
                                 fill_value=0,
                                 mode="L")
        img.enhance(inverse=True, stretch="linear")
        return img

    wv_high.prerequisites = set([6.7])

    def wv_low(self):
        """Make a black and white image of the IR 7.3um channel.

        Channel data inverted and a linear stretch is applied with 0.5
        % clipping at both ends.
        """
        self.check_channels(7.3)

        img = geo_image.GeoImage(self[7.3].data,
                                 self.area,
                                 self.time_slot,
                                 fill_value=0,
                                 mode="L")
        img.enhance(inverse=True, stretch="linear")
        return img

    wv_low.prerequisites = set([7.3])

    def green_snow(self):
        """Make a Green Snow RGB image composite.

        +--------------------+--------------------+
        | Channels           | Gamma              |
        +====================+====================+
        | IR1.6              | gamma 1.6          |
        +--------------------+--------------------+
        | VIS0.6             | gamma 1.6          |
        +--------------------+--------------------+
        | IR10.8 (inverted)  | gamma 1.6          |
        +--------------------+--------------------+

        Linear stretch without clipping.
        """
        self.check_channels(0.635, 1.63, 10.8)

        ch1 = self[1.63].check_range()
        ch2 = self[0.635].check_range()
        ch3 = -self[10.8].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")

        img.enhance(stretch="crude")
        img.enhance(gamma=1.6)

        return img

    green_snow.prerequisites = set([0.635, 1.63, 10.8])

    def red_snow(self):
        """Make a Red Snow RGB image composite.

        +--------------------+--------------------+
        | Channels           | Gamma              |
        +====================+====================+
        | VIS0.6             | gamma 1.6          |
        +--------------------+--------------------+
        | IR1.6              | gamma 1.6          |
        +--------------------+--------------------+
        | IR10.8 (inverted)  | gamma 1.6          |
        +--------------------+--------------------+

        Linear stretch without clipping.
        """
        self.check_channels(0.635, 1.63, 10.8)        

        ch1 = self[0.635].check_range()
        ch2 = self[1.63].check_range()
        ch3 = -self[10.8].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB")

        img.enhance(stretch = "crude")

        return img

    red_snow.prerequisites = set([0.635, 1.63, 10.8])

    def convection(self):
        """Make a Severe Convection RGB image composite.

        +--------------------+--------------------+--------------------+
        | Channels           | Span               | Gamma              |
        +====================+====================+====================+
        | WV6.2 - WV7.3      |     -30 to 0 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR3.9 - IR10.8     |      0 to 55 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR1.6 - VIS0.6     |    -70 to 20 %     | gamma 1            |
        +--------------------+--------------------+--------------------+
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

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | IR12.0 - IR10.8    |     -4 to 2 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.8 - IR8.7     |     0 to 15 K      | gamma 2.5          |
        +--------------------+--------------------+--------------------+
        | IR10.8             |   261 to 289 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
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

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | IR12.0 - IR10.8    |     -4 to 2 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.8 - IR8.7     |     -4 to 5 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.8             |   243 to 303 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
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

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | IR12.0 - IR10.8    |     -4 to 2 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.8 - IR8.7     |      0 to 6 K      | gamma 2.0          |
        +--------------------+--------------------+--------------------+
        | IR10.8             |   243 to 283 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
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

        +--------------------+--------------------+--------------------+
        | Channels           | Temp               | Gamma              |
        +====================+====================+====================+
        | IR12.0 - IR10.8    |     -4 to 2 K      | gamma 1            |
        +--------------------+--------------------+--------------------+
        | IR10.8 - IR3.9     |      0 to 6 K      | gamma 2.0          |
        +--------------------+--------------------+--------------------+
        | IR10.8             |   243 to 293 K     | gamma 1            |
        +--------------------+--------------------+--------------------+
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

    def cloudtop(self, stretch=(0.005, 0.005), gamma=None):
        """Make a Cloudtop RGB image composite.

        +--------------------+--------------------+
        | Channels           | Gamma              |
        +====================+====================+
        | IR3.9 (inverted)   | gamma 1            |
        +--------------------+--------------------+
        | IR10.8 (inverted)  | gamma 1            |
        +--------------------+--------------------+
        | IR12.0 (inverted)  | gamma 1            |
        +--------------------+--------------------+

        Linear stretch with 0.5 % clipping at both ends.
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

        if stretch:
            img.enhance(stretch=stretch)
        if gamma:
            img.enhance(gamma=gamma)

        return img

    cloudtop.prerequisites = set([3.75, 10.8, 12.0])

#pylint: enable=W0612

