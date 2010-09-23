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

"""This module is the loader for Meteosat09 scenes.
"""

from pp.channel import Channel
from pp.instruments.seviri import SeviriScene
from pp.logger import LOG
import imageo.palettes
import imageo.image_processing
import imageo.geo_image as geo_image
import numpy as np

class Meteosat09SeviriScene(SeviriScene):
    """This class implements the MeteoSat scene as captured by the seviri
    instrument. It's constructor accepts the same arguments as
    :class:`pp.scene.SatelliteScene`.
    """
    satname = "meteosat"
    number = "09"

    def load(self, channels = None):
        """Load data into the *channels*. *Channels* is a list or a tuple
        containing channels we will load data into. If None, all channels are
        loaded.
        """
        LOG.info("Loading channels...")

        do_correct = channels is not None and "_IR39Corr" in channels

        channels_to_load = set(channels)

        if do_correct:
            for chn in self.co2corr.prerequisites:
                channels_to_load |= set([self[chn].name])

        channels_to_load -= set(["_IR39Corr", "CTTH", "CloudType"])

        SeviriScene.load(self, channels_to_load)

        if channels is not None:
            if "_IR39Corr" in channels:
                area = self[3000].area
                self.channels.append(
                    Channel(name="_IR39Corr",
                            wavelength_range=(3.48, 3.92, 4.36),
                            resolution=3000))
                self["_IR39Corr"].area = area
                
            if("CTTH" in channels and
               "CTTH" not in self.channels):
                import satin.msg_ctth
                self.channels.append(
                    satin.msg_ctth.ctth_channel(self.time_slot, self.area))
                
            if("CloudType" in channels and
               "CloudType" not in self.channels):
                import satin.msg_ctype
                self.channels.append(
                    satin.msg_ctype.cloudtype_channel(self.time_slot,
                                                      self.area))
                # This is made necessary by the MSG string terminator bug
                try:
                    area_id = self.area.area_id
                except AttributeError:
                    area_id = self.area
                self["CloudType"].region_name = area_id


        if do_correct:
            self["_IR39Corr"] = self.co2corr()
        LOG.info("Loading channels done.")

    def pge02(self):
        """Make a Cloudtype RGB image composite, using paletted colors.
        """
        self.check_channels("CloudType")

        ch1 = self["CloudType"].cloudtype.data
        palette = imageo.palettes.cms_modified()

        img = geo_image.GeoImage(ch1,
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0),
                                 mode = "P",
                                 palette = palette)

        return img

    pge02.prerequisites = set(["CloudType"])

    def pge02b(self):
        """Make a Cloudtype RGB image composite, depicting low clouds, land and
        sea with palette colors, and the rest as in the IR 10.8 channel.
        """
        self.check_channels(10.8, "CloudType")

        ctype = self["CloudType"].cloudtype.data
        clouds = self[10.8].data

        palette = imageo.palettes.vv_legend()
        clouds = imageo.image_processing.crude_stretch(clouds)
        clouds = imageo.image_processing.gamma_correction(clouds, 1.6)
        clouds = 1 - clouds
        clouds = (clouds * 248 + 7).astype(np.uint8)
        clouds = np.ma.where(ctype <= 2, ctype, clouds)
        clouds = np.ma.where(np.logical_and(4 < ctype,
                                            ctype < 9),
                             ctype - 2,
                             clouds)
        
        img = geo_image.GeoImage(clouds,
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode = "P",
                                 palette = palette)

        return img

    pge02b.prerequisites = set(["CloudType", 10.8])

    def pge02b_with_overlay(self):
        """Same as :meth:`pge02b`, with borders overlay.
        """
        self.check_channels(10.8, "CloudType")
        
        img = self.pge02b()
        
        img.add_overlay()

        return img

    pge02b_with_overlay.prerequisites = set(["CloudType", 10.8])

    def pge02c(self):
        """Make an RGB composite showing clouds as depicted with the IR 10.8um
        channel, and cloudfree areas with land in green and sea in blue.
        """
        self.check_channels(10.8, "CloudType")
        
        ctype = self["CloudType"].cloudtype.data
        clouds = self[10.8].data

        palette = imageo.palettes.tv_legend()

        clouds = (clouds - 205.0) / (295.0 - 205.0)
        clouds = (1 - clouds).clip(0, 1)
        clouds = (clouds * 250 + 5).astype(np.uint8)
        clouds = np.ma.where(ctype <= 4, ctype, clouds)

        img = geo_image.GeoImage(clouds,
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode = "P",
                                 palette = palette)
        
        return img

    pge02c.prerequisites = set(["CloudType", 10.8])
        
    def pge02c_with_overlay(self):
        """Same as :meth:`pge02c` with borders overlay.
        """
        self.check_channels(10.8, "CloudType")

        img = self.pge02c()

        img.add_overlay()

        return img

    pge02c_with_overlay.prerequisites = set(["CloudType", 10.8])

    def pge02d(self):
        """Same as :meth:`pge02c` with transparent cloud-free areas, and
        semi-transparent thin clouds.
        """
        self.check_channels(10.8, "CloudType")

        ctype = self["CloudType"].cloudtype.data

        img = self.pge02c()
        img.fill_value = None
        
        alpha = np.ma.where(ctype < 5, 0.0, 1.0)
        alpha = np.ma.where(ctype == 15, 0.5, alpha)
        alpha = np.ma.where(ctype == 19, 0.5, alpha)

        img.putalpha(alpha)
        
        return img
        
    pge02d.prerequisites = set(["CloudType", 10.8])
        
    def pge02e(self):
        """Same as :meth:`pge02d` with clouds as in :meth:`overview`.
        """
        self.check_channels(0.6, 0.8, 10.8, "CloudType")

        img = self.overview()
        img.fill_value = None
        
        ctype = self["CloudType"].cloudtype.data

        alpha = np.ma.where(ctype < 5, 0.0, 1.0)
        alpha = np.ma.where(ctype == 15, 0.5, alpha)
        alpha = np.ma.where(ctype == 19, 0.5, alpha)

        img.putalpha(alpha)
        
        return img

    pge02e.prerequisites = set(["CloudType", 0.6, 0.8, 10.8])    



    def pge03(self):
        """Make an RGB composite of the CTTH.
        """
        self.check_channels("CTTH")
        # This is ugly. format should not matter here.
        # FIXME
        import satin.msg_ctth

        ctth = satin.msg_ctth.msg_ctth2ppsformat(self["CTTH"])
        
        arr = (ctth.height*ctth.h_gain+ctth.h_intercept)
        ctth_data = np.where(ctth.height == ctth.h_nodata, 0, arr / 500.0 + 1)
        ctth_data = np.ma.array(ctth_data)

        palette = imageo.palettes.ctth_height()

        img = geo_image.GeoImage(ctth_data.astype(np.uint8),
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode = "P",
                                 palette = palette)

        return img
    
    pge03.prerequisites = set(["CTTH"])    

    def cloudtype(self):
        """Return the cloudtype.
        """
        return self["CloudType"]
    
    cloudtype.prerequisites = set(["CloudType"])

    def nordrad(self):
        """Return the cloudtype in NordRad format.
        """
        import satin.msg_ctype
        datestr = self.time_slot.strftime("%Y%m%d%H%M")
        return satin.msg_ctype.NordRadCType(self["CloudType"], datestr)
    
    nordrad.prerequisites = set(["CloudType"])

    def ctth(self):
        """Return the ctth.
        """
        return self["CTTH"]
    
    ctth.prerequisites = set(["CTTH"])
