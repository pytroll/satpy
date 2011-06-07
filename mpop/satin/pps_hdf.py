#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011.

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

import mpop.channel
from mpop import CONFIG_PATH
from mpop.utils import get_logger


LOG = get_logger('satin/pps_hdf')

def pcs_def_from_region(region):
    items = region.proj_dict.items()
    return ' '.join([ t[0] + '=' + t[1] for t in items])   

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

    def project(self, coverage):
        """Remaps the cloudtype channel.
        """
        import epshdf
        
        LOG.info("Projecting channel %s..."%(self.name))

        retv = PpsCloudType()

        area = coverage.out_area
        retv.region = epshdf.SafRegion()
        retv.region.xsize = area.x_size
        retv.region.ysize = area.y_size
        retv.region.id = area.area_id
        retv.region.pcs_id = area.proj_id
        retv.region.pcs_def = pcs_def_from_region(area)
        retv.region.area_extent = area.area_extent

        retv.cloudtype_des = self.cloudtype_des
        retv.cloudtype_lut = self.cloudtype_lut
        retv.cloudtype = coverage.project_array(self.cloudtype)

        retv.qualityflag_des = self.qualityflag_des
        retv.qualityflag_lut = self.qualityflag_lut
        retv.qualityflag = coverage.project_array(self.qualityflag)

        retv.phaseflag_des = self.phaseflag_des
        retv.phaseflag_lut = self.phaseflag_lut
        retv.phaseflag = coverage.project_array(self.phaseflag)

        retv.des = self.des
        retv.sec_1970 = self.sec_1970
        retv.satellite_id = self.satellite_id

        return retv

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
        
    def project(self, coverage):
        """Remaps the cloudtype channel.
        """
        import epshdf
        
        LOG.info("Projecting channel %s..."%(self.name))

        retv = PpsCTTH()

        area = coverage.out_area
        retv.region = epshdf.SafRegion()
        retv.region.xsize = area.x_size
        retv.region.ysize = area.y_size
        retv.region.id = area.area_id
        retv.region.pcs_id = area.proj_id
        retv.region.pcs_def = pcs_def_from_region(area)
        retv.region.area_extent = area.area_extent

        retv.ctt_des = self.ctt_des
        retv.temperature = coverage.project_array(self.temperature)
        retv.t_gain = self.t_gain
        retv.t_intercept = self.t_intercept
        retv.t_nodata = self.t_nodata

        retv.ctp_des = self.ctp_des
        retv.pressure = coverage.project_array(self.pressure)
        retv.p_gain = self.p_gain
        retv.p_intercept = self.p_intercept
        retv.p_nodata = self.p_nodata

        retv.cth_des = self.cth_des
        retv.height = coverage.project_array(self.height)
        retv.h_gain = self.h_gain
        retv.h_intercept = self.h_intercept
        retv.h_nodata = self.h_nodata

        retv.cloudiness = coverage.project_array(self.cloudiness)
        retv.c_nodata = self.c_nodata
        retv.cloudiness_des = self.cloudiness_des

        retv.processingflag = coverage.project_array(self.processingflag)
        retv.processingflag_des = self.processingflag_des
        retv.processingflag_lut = self.processingflag_lut


        retv.des = self.des
        retv.sec_1970 = self.sec_1970
        retv.satellite_id = self.satellite_id

        return retv

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
            if not os.path.exists(filename):
                LOG.info("Can't find " + filename)
            else:
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
            if not os.path.exists(filename):
                LOG.info("Can't find " + filename)
            else:
                break
        if not os.path.exists(filename):
            LOG.info("Can't find any Cloudtype file, skipping")
        else:
            ct_chan = PpsCloudType()
            ct_chan.read(filename)
            ct_chan.area = scene.area
            scene.channels.append(ct_chan)

    LOG.info("Loading channels done.")
