#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""This module implements the meteosat 09 geostationnary satellite and
its Seviri instrument.
"""

import os
import numpy as np
import logging
import glob
import ConfigParser

import pp.meteosat.py_msg as py_msg
from pp.satellite.satellite import SatelliteSnapshot, SatelliteChannel
import pp.geo_image.geo_image as geo_image
import pp.meteosat.time_utils as time_utils
import pp.meteosat.msg_ctype as msg_ctype
import pp.meteosat.msg_ctth as msg_ctth
from pp.meteosat.msg_ctype2radar import NordRadCType
import pp.satellite.palettes
from pp.meteosat import BASE_PATH


CONF = ConfigParser.ConfigParser()
CONF.read(os.path.join(BASE_PATH, "etc", "meteosat.cfg"))

MSG_DIR = CONF.get('dirs_in', 'msg_dir')
MSG_LIB = CONF.get('dirs_in', 'msg_lib')
MSG_BIN = CONF.get('dirs_in', 'msg_bin')
CTYPE_DIR = CONF.get('dirs_in', 'ctype_dir')
CTTH_DIR = CONF.get('dirs_in', 'ctth_dir')

MSG_PGE_EXTENTIONS = ["PLAX.CTTH.0.h5", "PLAX.CLIM.0.h5", "h5"]

LOG = logging.getLogger("pp.meteosat")

os.environ['SAFNWC'] = MSG_DIR
os.environ['SAFNWC_BIN'] = MSG_BIN
os.environ['SAFNWC_LIB'] = MSG_LIB
os.environ['PATH'] = os.environ['PATH']+":"+os.environ['SAFNWC_BIN']
os.environ['LD_LIBRARY_PATH'] = (os.environ['LD_LIBRARY_PATH']+
                                 ":"+os.environ['SAFNWC_LIB'])
os.environ['BUFR_TABLES'] = (os.environ['SAFNWC']+
                             "/src/bufr_000360/bufrtables/")
os.environ['LOCAL_DEFINITION_TEMPLATES'] = (os.environ['SAFNWC']+
                                            "/src/gribex_000360/gribtemplates/")

MET09_SEVIRI = [["VIS06", (0.56, 0.635, 0.71), 3000],
                ["VIS08", (0.74, 0.81, 0.88), 3000],
                ["IR16", (1.50, 1.64, 1.78), 3000],
                ["IR39", (3.48, 3.92, 4.36), 3000],
                ["WV62", (5.35, 6.25, 7.15), 3000],
                ["WV73", (6.85, 7.35, 7.85), 3000],
                ["IR87", (8.30, 8.70, 9.10), 3000],
                ["IR97", (9.38, 9.66, 9.94), 3000],
                ["IR108", (9.80, 10.80, 11.80), 3000],
                ["IR120", (11.00, 12.00, 13.00), 3000],
                ["IR134", (12.40, 13.40, 14.40), 3000],
                ["HRVIS", (0.50, np.nan, 0.90), 1000]]

class MeteoSatSeviriSnapshot(SatelliteSnapshot):
    """This class implements the MeteoSat snapshot as captured by the seviri
    instrument. It's constructor accepts the same arguments as
    :class:`pp.satellite.SatelliteSnapshot`.
    """
    
    def __init__(self, *args, **kwargs):
        
        self.channel_list = MET09_SEVIRI
        
        super(MeteoSatSeviriSnapshot, self).__init__(*args, **kwargs)

        self.satname = "met09"
        

    def load(self, channels = None):
        """Load data into the *channels*. *Channels* is a list or a tuple
        containing channels we will load data into. If None, all channels are
        loaded.
        """
        LOG.info("Loading channels...")

        super(MeteoSatSeviriSnapshot, self).load(channels)

        do_correct = False

        if channels is not None:
            if "_IR39Corr" in channels:
                do_correct = True
                self.channels.append(
                    SatelliteChannel(name = "_IR39Corr",
                                     wavelength_range = (3.48, 3.92, 4.36),
                                     resolution = 3000))
            if("CTTH" in channels and
               "CTTH" not in self.channels):
                self.channels.append(self.ctth_channel())
                # This is made necessary by the MSG string terminator bug
                self["CTTH"].region_name = self.area
            if("CloudType" in channels and
               "CloudType" not in self.channels):
                self.channels.append(self.cloudtype_channel())
                # This is made necessary by the MSG string terminator bug
                self["CloudType"].region_name = self.area

        if do_correct:
            for chn in self.co2corr.prerequisites:
                self.channels_to_load |= set([self[chn].name])

        # Do not reload data.
        self.channels_to_load -= set([chn.name for chn in
                                      self.loaded_channels()])
        
        data = py_msg.get_channels(time_utils.time_string(self.time_slot), 
                                   self.area, 
                                   list(self.channels_to_load),
                                   False)

        for chn in data:
            self[chn] = np.ma.array(data[chn]["CAL"], mask = data[chn]["MASK"])

        if do_correct:
            self["_IR39Corr"] = self.co2corr()
        LOG.info("Loading channels done.")

    def ctth_channel(self):
        """Create and return a ctth channel.
        """
        time_string = time_utils.time_string(self.time_slot)

        prefix = ("SAFNWC_MSG?_CTTH_%s_%s"%(time_string, self.area))

        msgctth_filename = None
        for ext in MSG_PGE_EXTENTIONS:
            match_str = ("%s/%s.%s"
                         %(CTTH_DIR, prefix, ext))
            LOG.info("file-match: %s"%match_str)
            flist = glob.glob(match_str)

            if len(flist) > 1:
                LOG.error("More than one matching input file: N = %d"
                          %len(flist))
                raise RuntimeError("Cannot choose input CTTH file.")
            elif len(flist) == 0:
                LOG.warning("No matching input file.")
            else:
                # File found:
                LOG.info("MSG CTTH file found: %s"%flist[0])
                msgctth_filename = flist[0]
                break
             
        if msgctth_filename is not None:
            # Read the MSG file if not already done...
            LOG.info("Read MSG CTTH file: %s"%msgctth_filename)
            ctth = msg_ctth.MsgCTTH(resolution = 3000)
            ctth.read_msgCtth(msgctth_filename)
            LOG.debug("MSG CTTH area: %s"%ctth.region_name)
            return ctth
        else:
            LOG.error("No MSG CT input file found!")
            raise RuntimeError("No input CTTH file.")

    def cloudtype_channel(self):
        """Create and return a cloudtype channel.
        """
        time_string = time_utils.time_string(self.time_slot)

        prefix = ("SAFNWC_MSG?_CT___%s_%s"%(time_string, self.area))
        
        msgctype_filename = None
        
        for ext in MSG_PGE_EXTENTIONS:
            # Look for a ctype file
            match_str = ("%s/%s.%s"%(CTYPE_DIR, prefix, ext))
            LOG.info("file-match: %s"%match_str)
            flist = glob.glob(match_str)

            if len(flist) > 1:
                LOG.error("More than one matching input file: N = %d"
                          %len(flist))
                raise RuntimeError("Cannot choose input CType file.")
            elif len(flist) == 0:
                LOG.warning("No matching .%s input cloud type file."
                            %ext)
            else:
                # File found:
                LOG.info("MSG CT file found: %s"%flist[0])
                msgctype_filename = flist[0]
                break

        if msgctype_filename is not None:
            LOG.info("Read MSG CT file: %s"%msgctype_filename)
            ctype = msg_ctype.MsgCloudType(resolution = 3000)
            ctype.read_msg_ctype(msgctype_filename)
            LOG.debug("MSG CT area: %s"%ctype.region_name)
            return ctype
        else:
            LOG.error("No MSG CT input file found!")
            raise RuntimeError("No input CType file.")


    def co2corr(self):
        """CO2 correction of the brightness temperature of the MSG 3.9um
        channel::
        
          T4_CO2corr = (BT(IR3.9)^4 + Rcorr)^0.25
          Rcorr = BT(IR10.8)^4 - (BT(IR10.8)-dt_CO2)^4
          dt_CO2 = (BT(IR10.8)-BT(IR13.4))/4.0
        """
        try:
            self.check_channels(3.9, 10.8, 13.4)
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

    co2corr.prerequisites = set([3.9, 10.8, 13.4])

    def cloudtop(self):
        """Make a Cloudtop RGB image composite from Seviri channels.
        """
        self.check_channels("_IR39Corr", 10.8, 12.0)

        ch1 = -self["_IR39Corr"].data
        ch2 = -self[10.8].data
        ch3 = -self[12.0].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode = "RGB")

        img.enhance(stretch = (0.005, 0.005))

        return img
    
    cloudtop.prerequisites = set(["_IR39Corr", 10.8, 12.0])

    def night_fog(self):
        """Make a Night Fog RGB image composite from Seviri channels.
        """
        self.check_channels("_IR39Corr", 10.8, 12.0)

        ch1 = self[12.0].data - self[10.8].data
        ch2 = self[10.8].data - self["_IR39Corr"].data
        ch3 = self[10.8].data
        
        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode = "RGB",
                                 crange = ((-4, 2),
                                           (0, 6),
                                           (243, 293)))
        
        img.enhance(gamma = (1.0, 2.0, 1.0))
        img.clip()

        return img

    night_fog.prerequisites = set(["_IR39Corr", 10.8, 12.0])


    def hr_overview(self):
        """Make a High Resolution Overview RGB image composite from Seviri
        channels.
        """
        self.check_channels(0.6, 0.8, 10.8, "HRVIS")

        ch1 = self[0.6].check_range()
        ch2 = self[0.8].check_range()
        ch3 = -self[10.8].data

        img = geo_image.GeoImage((ch1, ch2, ch3),
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode = "RGB")

        img.enhance(stretch = "crude")
        img.enhance(gamma = [1.6, 1.6, 1.1])
        
        luminance = geo_image.GeoImage((self["HRVIS"].data),
                                       self.area,
                                       self.time_slot,
                                       mode = "L")

        luminance.enhance(stretch = "crude")
        luminance.enhance(gamma = 2.0)

        img.replace_luminance(luminance.channels[0])
        
        return img

    hr_overview.prerequisites = set(["HRVIS", 0.6, 0.8, 10.8])

    def pge03(self):
        """Make an RGB composite of the CTTH.
        """
        self.check_channels("CTTH")
        # This is ugly. format should not matter here.
        # FIXME
        import msg_ctth
        ctth = msg_ctth.msg_ctth2ppsformat(self["CTTH"])
        
        arr = (ctth.height*ctth.h_gain+ctth.h_intercept)
        ctth_data = np.where(ctth.height == ctth.h_nodata, 0, arr / 500.0 + 1)
        ctth_data = np.ma.array(ctth_data)

        palette = pp.satellite.palettes.ctth_height()

        img = geo_image.GeoImage(ctth_data.astype(np.uint8),
                                 self.area,
                                 self.time_slot,
                                 fill_value = (0, 0, 0),
                                 mode = "P",
                                 palette = palette)

        return img
        
    pge03.prerequisites = set(["CTTH"])    

    def get_lat_lon(self, resolution):
        """Get the latitude and longitude grids of the current region for the
        given *resolution*.
        """
        if not isinstance(resolution, int):
            raise TypeError("Resolution must be an integer number of meters.")
        channel = self[0.6, resolution]
        return py_msg.lat_lon_from_region(self.area, channel.name)

    def cloudtype(self):
        """Return the cloudtype.
        """
        return self["CloudType"]
    
    cloudtype.prerequisites = set(["CloudType"])

    def nordrad(self):
        """Return the cloudtype in NordRad format.
        """
        datestr = self.time_slot.strftime("%Y%m%d%H%M")
        return NordRadCType(self["CloudType"], datestr)
    
    nordrad.prerequisites = set(["CloudType"])

    def ctth(self):
        """Return the ctth.
        """
        return self["CTTH"]
    
    ctth.prerequisites = set(["CTTH"])


    

