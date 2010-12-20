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
from mpop.channel import NotLoadedError
from mpop.imageo import geo_image
from mpop.instruments.visir import VisirScene
from mpop.utils import get_logger
import numpy as np

LOG = get_logger("mpop/avhrr")

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
                                      self.area,
                                      self.time_slot,
                                      fill_value=(0, 0, 0),
                                      mode="RGB")
            
            imgb.enhance(stretch=(0.005, 0.005))

        except (NotLoadedError, KeyError):
            imgb = None

        try:
            self.check_channels(1.63)
            ch1a = self[1.63].data
            imga = geo_image.GeoImage((ch1a, ch2, ch3),
                                      self.area,
                                      self.time_slot,
                                      fill_value=(0, 0, 0),
                                      mode="RGB")

            imga.enhance(stretch=(0.005, 0.005))

        except (NotLoadedError, KeyError):
            imga = None
        

        if imga and imgb:
            imgb.merge(imga)
            return imgb
        else:
            return (imgb or imga)
            
    cloudtop.prerequisites = set([1.63, 3.75, 10.8, 12.0])

    def pge02b(self):
        """Make a Cloudtype RGB image composite, depicting low clouds, land and
        sea with palette colors, and the rest as in the IR 10.8 channel.
        """
        self.check_channels(10.8, "CloudType")

        ctype = self["CloudType"].cloudtype
        clouds = self[10.8].data

        palette = mpop.imageo.palettes.vv_legend()
        clouds = mpop.imageo.image_processing.crude_stretch(clouds)
        clouds = mpop.imageo.image_processing.gamma_correction(clouds, 1.6)
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



    def load(self, channels=None, load_again=False, **kwargs):
        """Load data into the *channels*. *Channels* is a list or a tuple
        containing channels we will load data into. If None, all channels are
        loaded.
        """
        LOG.info("Loading channels...")

        if channels is not None:
            channels_to_load = set(channels)
            channels_to_load -= set(["CTTH", "CloudType"])
        else:
            channels_to_load = None
            
        VisirScene.load(self, channels_to_load, load_again, **kwargs)



        if channels is not None:
            if("CTTH" in channels and
               "CTTH" not in self.channels):
                import mpop.satin.msg_ctth
                self.channels.append(
                    mpop.satin.msg_ctth.ctth_channel(self.time_slot, self.area))
                
            if("CloudType" in channels and
               "CloudType" not in self.channels):
                import ConfigParser
                import os.path
                from mpop import CONFIG_PATH
                conf = ConfigParser.ConfigParser()
                conf.read(os.path.join(CONFIG_PATH, self.fullname+".cfg"))
                directory = conf.get(self.instrument_name+"-level3", "dir")
                filename = conf.get(self.instrument_name+"-level3", "filename",
                                    raw=True)
                pathname = os.path.join(directory, filename)
                area_name = self.area_id or self.area.area_id
                filename = (self.time_slot.strftime(pathname)
                            %{"orbit": self.orbit,
                              "number": self.number,
                              "area": area_name})
                import epshdf
                ct_chan = PpsCloudType()
                ct_chan.copy(epshdf.read_cloudtype(filename))
                self.channels.append(ct_chan)

                self["CloudType"].area = self.area


        LOG.info("Loading channels done.")

