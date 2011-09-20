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

"""This modules describes the seviri instrument.
"""
import numpy as np

import mpop.imageo.geo_image as geo_image
from mpop.instruments.visir import VisirCompositer
from mpop.logger import LOG


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
        
        dt_co2 = (bt108-bt134)/4.0
        rcorr = bt108 ** 4 - (bt108-dt_co2) ** 4
        
        
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
        t4_co2corr.data = np.ma.where(t4_co2corr.data > 0.0, t4_co2corr.data, 0)
        t4_co2corr = t4_co2corr ** 0.25

        t4_co2corr.name = "_IR39Corr"
        t4_co2corr.area = self[3.9].area
        t4_co2corr.wavelength_range = self[3.9].wavelength_range
        t4_co2corr.resolution = self[3.9].resolution
        
        self._data_holder.channels.append(t4_co2corr)

    co2corr_chan.prerequisites = set([3.75, 10.8, 13.4])

    def cloudtop(self):
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

        img.enhance(stretch=(0.005, 0.005))

        return img
    
    cloudtop.prerequisites = co2corr_chan.prerequisites | set([10.8, 12.0])

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
                                 fill_value = (0, 0, 0),
                                 mode="RGB",
                                 crange=((-4, 2),
                                         (0, 6),
                                         (243, 293)))
        
        img.enhance(gamma = (1.0, 2.0, 1.0))

        return img

    night_fog.prerequisites = co2corr_chan.prerequisites | set([10.8, 12.0])
