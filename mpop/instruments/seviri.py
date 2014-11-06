#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2013, 2014.

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
from mpop.instruments.visir import VisirCompositer
import logging

LOG = logging.getLogger(__name__)

import os.path

try:
    from pyorbital.astronomy import sun_zenith_angle as sza
except ImportError:
    sza = None

METEOSAT = {'meteosat10': 'met10',
            'meteosat9': 'met9',
            'meteosat8': 'met8',
            'meteosat11': 'met11',
            }


class SeviriCompositer(VisirCompositer):

    """This class sets up the Seviri instrument channel list.
    """

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
            self.check_channels(3.75, 10.8, 13.4)
        except RuntimeError:
            LOG.warning("CO2 correction not performed, channel data missing.")
            return

        bt039 = self[3.9].data
        bt108 = self[10.8].data
        bt134 = self[13.4].data

        dt_co2 = (bt108 - bt134) / 4.0
        rcorr = bt108 ** 4 - (bt108 - dt_co2) ** 4

        t4_co2corr = bt039 ** 4 + rcorr
        t4_co2corr = np.ma.where(t4_co2corr > 0.0, t4_co2corr, 0)
        t4_co2corr = t4_co2corr ** 0.25

        return t4_co2corr

    co2corr.prerequisites = set([3.75, 10.8, 13.4])

    def co2corr_chan(self):
        """CO2 correction of the brightness temperature of the MSG 3.9um
        channel, adding it as a channel::

        .. math::

          T4_CO2corr = (BT(IR3.9)^4 + Rcorr)^0.25
          Rcorr = BT(IR10.8)^4 - (BT(IR10.8)-dt_CO2)^4
          dt_CO2 = (BT(IR10.8)-BT(IR13.4))/4.0


        """

        if "_IR39Corr" in [chn.name for chn in self._data_holder.channels]:
            return

        self.check_channels(3.75, 10.8, 13.4)

        dt_co2 = (self[10.8] - self[13.4]) / 4.0
        rcorr = self[10.8] ** 4 - (self[10.8] - dt_co2) ** 4
        t4_co2corr = self[3.9] ** 4 + rcorr
        t4_co2corr.data = np.ma.where(
            t4_co2corr.data > 0.0, t4_co2corr.data, 0)
        t4_co2corr = t4_co2corr ** 0.25

        t4_co2corr.name = "_IR39Corr"
        t4_co2corr.area = self[3.9].area
        t4_co2corr.wavelength_range = self[3.9].wavelength_range
        t4_co2corr.resolution = self[3.9].resolution

        self._data_holder.channels.append(t4_co2corr)

    co2corr_chan.prerequisites = set([3.75, 10.8, 13.4])

    def refl39_chan(self):
        """Derive the solar (reflectance) part of the 3.9um channel including a
        correction of the limb cooling (co2 correction), adding it as a
        channel.

        """

        if "_IR39Refl" in [chn.name for chn in self._data_holder.channels]:
            return

        if not sza:
            LOG.warning("3.9 reflectance derivation is not possible..." +
                        "\nCheck that pyspectral and pyorbital are " +
                        "installed and available!")
            return

        self.check_channels(3.75, 10.8, 13.4)

        platform_name = METEOSAT.get(self.fullname, 'unknown')
        if platform_name == 'unknown':
            LOG.error("Failed setting correct platform name for pyspectral! " +
                      "Satellite = " + str(self.fullname))
        LOG.debug("Satellite = " + str(self.fullname))

        r39 = self[3.9].get_reflectance(self[10.8].data,
                                        sun_zenith=None,
                                        tb13_4=self[13.4].data,)

        if r39 is None:
            raise RuntimeError("Couldn't derive 3.x reflectance. " +
                               "Check if pyspectral is installed!")

        r39channel = self[3.9] * 1.0
        r39channel.data = np.ma.where(r39channel.data > 0.0,
                                      r39 * 100, 0)
        r39channel.name = "_IR39Refl"
        r39channel.area = self[3.9].area
        r39channel.wavelength_range = self[3.9].wavelength_range
        r39channel.resolution = self[3.9].resolution

        self._data_holder.channels.append(r39channel)

    refl39_chan.prerequisites = set([3.75, 10.8, 13.4])

    def convection_co2(self):
        """Make a Severe Convection RGB image composite on SEVIRI compensating
        for the CO2 absorption in the 3.9 micron channel.
        """
        self.co2corr_chan()
        self.check_channels("_IR39Corr", 0.635, 1.63, 6.7, 7.3, 10.8)

        ch1 = self[6.7].data - self[7.3].data
        ch2 = self["_IR39Corr"].data - self[10.8].data
        ch3 = self[1.63].check_range() - self[0.635].check_range()

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-30, 0),
                                         (0, 55),
                                         (-70, 20)))

        img.enhance(gamma=(1.0, 0.5, 1.0))

        return img

    convection_co2.prerequisites = (co2corr_chan.prerequisites |
                                    set([0.635, 1.63, 6.7, 7.3, 10.8]))

    def cloudtop(self, stretch=(0.005, 0.005), gamma=None):
        """Make a Cloudtop RGB image composite from Seviri channels.
        """
        self.co2corr_chan()
        self.check_channels("_IR39Corr", 10.8, 12.0)

        ch1 = -self["_IR39Corr"].data
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

    cloudtop.prerequisites = co2corr_chan.prerequisites | set([10.8, 12.0])

    def night_overview(self, stretch='histogram', gamma=None):
        """See cloudtop.
        """
        return self.cloudtop(stretch=stretch, gamma=gamma)

    night_overview.prerequisites = cloudtop.prerequisites

    def night_fog(self):
        """Make a Night Fog RGB image composite from Seviri channels.
        """
        self.co2corr_chan()
        self.check_channels("_IR39Corr", 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self["_IR39Corr"].data
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

    night_fog.prerequisites = co2corr_chan.prerequisites | set([10.8, 12.0])

    def night_microphysics(self):
        """Make a Night Microphysics RGB image composite from Seviri channels.
        This is a Eumetsat variant of night_fog.
        See e.g http://oiswww.eumetsat.int/~idds/html/doc/best_practices.pdf
        """
        self.check_channels(3.9, 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self[3.9].data
        ch3 = self[10.8].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value=(0, 0, 0),
                                 mode="RGB",
                                 crange=((-4, 2),
                                         (0, 10),
                                         (243, 293)))

        return img

    night_microphysics.prerequisites = set([3.9, 10.8, 12.0])

    def snow(self):
        """Make a 'Snow' RGB as suggested in the MSG interpretation guide
        (rgbpart04.ppt). It is kind of special as it requires the derivation of
        the daytime component of the mixed Terrestrial/Solar 3.9 micron
        channel. Furthermore the sun zenith angle is used.
        """

        self.refl39_chan()
        self.check_channels("_IR39Refl", 0.8, 1.63, 3.75)

        # We calculate the sun zenith angle again. Should be reused if already
        # calculated/available...
        # FIXME!
        lonlats = self[3.9].area.get_lonlats()
        sunz = sza(self.time_slot, lonlats[0], lonlats[1])
        sunz = np.ma.masked_outside(sunz, 0.0, 88.0)
        sunzmask = sunz.mask
        sunz = sunz.filled(88.)

        costheta = np.cos(np.deg2rad(sunz))

        red = np.ma.masked_where(sunzmask, self[0.8].data / costheta)
        green = np.ma.masked_where(sunzmask, self[1.6].data / costheta)

        img = geo_image.GeoImage((red, green, self['_IR39Refl'].data),
                                 self.area,
                                 self.time_slot,
                                 crange=((0, 100), (0, 70), (0, 30)),
                                 fill_value=None, mode="RGB")
        img.gamma((1.7, 1.7, 1.7))

        return img

    snow.prerequisites = refl39_chan.prerequisites | set(
        [0.8, 1.63, 3.75])

    def day_microphysics(self, wintertime=False):
        """Make a 'Day Microphysics' RGB as suggested in the MSG interpretation guide
        (rgbpart04.ppt). It is kind of special as it requires the derivation of
        the daytime component of the mixed Terrestrial/Solar 3.9 micron
        channel. Furthermore the sun zenith angle is used.
        """

        self.refl39_chan()
        self.check_channels(0.8, "_IR39Refl", 10.8)

        # We calculate the sun zenith angle again. Should be reused if already
        # calculated/available...
        # FIXME!
        lonlats = self[3.9].area.get_lonlats()
        sunz = sza(self.time_slot, lonlats[0], lonlats[1])
        sunz = np.ma.masked_outside(sunz, 0.0, 88.0)
        sunzmask = sunz.mask
        sunz = sunz.filled(88.)

        costheta = np.cos(np.deg2rad(sunz))

        if wintertime:
            crange = ((0, 100), (0, 25), (203, 323))
        else:
            crange = ((0, 100), (0, 60), (203, 323))

        red = np.ma.masked_where(sunzmask,
                                 self[0.8].data / costheta)
        green = np.ma.masked_where(sunzmask,
                                   self['_IR39Refl'].data)
        blue = np.ma.masked_where(sunzmask, self[10.8].data)
        img = geo_image.GeoImage((red, green, blue),
                                 self.area,
                                 self.time_slot,
                                 crange=crange,
                                 fill_value=None, mode="RGB")
        if wintertime:
            img.gamma((1.0, 1.5, 1.0))
        else:
            img.gamma((1.0, 2.5, 1.0))  # Summertime settings....

        return img

    day_microphysics.prerequisites = refl39_chan.prerequisites | set(
        [0.8, 10.8])
