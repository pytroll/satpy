#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2012.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>

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

"""Plugin for reading PPS's cloud products hdf files.
"""
import ConfigParser
import datetime
import os.path
from glob import glob

import mpop.channel
from mpop import CONFIG_PATH
from mpop.utils import get_logger


LOG = get_logger('satin/pps_hdf')

class PpsCloudType(mpop.channel.GenericChannel):
    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CloudType")
        self.region = None
        self.des = ""
        self.cloudtype_des = ""
        self.qualityflag_des = ""
        self.phaseflag_des = ""
        self.sec_1970 = 0
        self.satellite_id = ""
        self.cloudtype_lut = []
        self.qualityflag_lut = []
        self.phaseflag_lut = []
        self.cloudtype = None
        self.qualityflag = None
        self.phaseflag = None

    def copy(self, other):
        self.region = other.region
        self.des = other.des
        self.cloudtype_des = other.cloudtype_des
        self.qualityflag_des = other.qualityflag_des
        self.phaseflag_des = other.phaseflag_des
        self.sec_1970 = other.sec_1970
        self.satellite_id = other.satellite_id
        self.cloudtype_lut = other.cloudtype_lut
        self.qualityflag_lut = other.qualityflag_lut
        self.phaseflag_lut = other.phaseflag_lut
        self.cloudtype = other.cloudtype
        self.qualityflag = other.qualityflag
        self.phaseflag = other.phaseflag
        
    def read(self, filename):
        import epshdf
        self.copy(epshdf.read_cloudtype(filename))
        
    def is_loaded(self):
        return self.cloudtype is not None


class PpsCTTH(mpop.channel.GenericChannel):
    def __init__(self):
        mpop.channel.GenericChannel.__init__(self, "CTTH")
        self.region = None
        self.des = ""
        self.ctt_des = ""
        self.cth_des = ""
        self.ctp_des = ""
        self.cloudiness_des = ""
        self.processingflag_des = ""
        self.sec_1970 = 0
        self.satellite_id = ""
        self.processingflag_lut = []

        self.temperature = None
        self.t_gain = 1.0
        self.t_intercept = 0.0
        self.t_nodata = 255

        self.pressure = None
        self.p_gain = 1.0
        self.p_intercept = 0.0
        self.p_nodata = 255

        self.height = None
        self.h_gain = 1.0
        self.h_intercept = 0.0
        self.h_nodata = 255

        self.cloudiness = None
        self.c_nodata = 255
        self.processingflag = None

    def copy(self, other):
        self.region = other.region
        self.des = other.des
        self.ctt_des = other.ctt_des
        self.cth_des = other.cth_des
        self.ctp_des = other.ctp_des
        self.cloudiness_des = other.cloudiness_des
        self.processingflag_des = other.processingflag_des
        self.sec_1970 = other.sec_1970
        self.satellite_id = other.satellite_id
        self.processingflag_lut = other.processingflag_lut
        
        self.temperature = other.temperature
        self.t_gain = other.t_gain
        self.t_intercept = other.t_intercept
        self.t_nodata = other.t_nodata
        
        self.pressure = other.pressure
        self.p_gain = other.p_gain
        self.p_intercept = other.p_intercept
        self.p_nodata = other.p_nodata
        
        self.height = other.height
        self.h_gain = other.h_gain
        self.h_intercept = other.h_intercept
        self.h_nodata = other.h_nodata
        
        self.cloudiness = other.cloudiness
        self.c_nodata = other.c_nodata
        self.processingflag = other.processingflag
        
    def read(self, filename):
        import epshdf
        self.copy(epshdf.read_cloudtop(filename))

def load(scene, **kwargs):
    """Load data into the *channels*. *Channels* is a list or a tuple
    containing channels we will load data into. If None, all channels are
    loaded.
    """

    del kwargs

    if("CTTH" not in scene.channels_to_load and
       "CloudType" not in scene.channels_to_load):
        return
    
    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, scene.fullname+".cfg"))
    directory = conf.get(scene.instrument_name+"-level3", "dir")
    filename = conf.get(scene.instrument_name+"-level3", "filename",
                        raw=True)
    pathname = os.path.join(directory, filename)

    area_name = scene.area_id or scene.area.area_id

    times = [scene.time_slot,
             scene.time_slot + datetime.timedelta(minutes=1),
             scene.time_slot - datetime.timedelta(minutes=1)]
    
    if "CTTH" in scene.channels_to_load:
        for time_slot in times:
            filename = (time_slot.strftime(pathname)
                        %{"orbit": scene.orbit,
                          "number": scene.number,
                          "area": area_name,
                          "satellite": scene.fullname,
                          "product": "ctth"})
            flist = glob(filename)
            if len(flist) == 0:
                LOG.info("Can't find " + filename)
            elif len(flist) > 1:
                LOG.info("Too many files matching! " + str(flist))
                break
            else:
                filename = flist[0]
                break
        if not os.path.exists(filename):
            LOG.info("Can't find any CTTH file, skipping")
        else:
            ct_chan = PpsCTTH()
            ct_chan.read(filename)
            ct_chan.area = scene.area
            scene.channels.append(ct_chan)

    if "CloudType" in scene.channels_to_load:
        for time_slot in times:
            filename = (time_slot.strftime(pathname)
                        %{"orbit": scene.orbit,
                          "number": scene.number,
                          "area": area_name,
                          "satellite": scene.fullname,
                          "product": "cloudtype"})
            flist = glob(filename)
            if len(flist) == 0:
                LOG.info("Can't find " + filename)
            elif len(flist) > 1:
                LOG.info("Too many files matching! " + str(flist))
                break
            else:
                filename = flist[0]
                break
        if not os.path.exists(filename):
            LOG.info("Can't find any Cloudtype file, skipping")
        else:
            ct_chan = PpsCloudType()
            ct_chan.read(filename)
            ct_chan.area = scene.area
            scene.channels.append(ct_chan)

    LOG.info("Loading channels done.")
