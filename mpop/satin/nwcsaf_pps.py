#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
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
import numpy as np

import h5py

LOG = get_logger('satin/nwcsaf_pps')

class PpsCloudType(mpop.channel.GenericChannel):
    def __init__(self, resolution=None):
        mpop.channel.GenericChannel.__init__(self, "CloudType")
        self.filled = False
        self.name = "CloudType"
        self.resolution = resolution
        self.shape = None

        self.version = ""
        self.region = None
        self.des = ""
        self.orbit_number = None
        self.cloudtype_des = ""
        self.qualityflag_des = ""
        self.phaseflag_des = ""
        self.sec_1970 = 0
        self.satellite_id = ""
        self.cloudtype_lut = []
        self.qualityflag_lut = []
        self.phaseflag_lut = []
        self.cloudtype = None
        self.data = None
        self.qualityflag = None
        self.phaseflag = None

        self.palette = None
        
    def read(self, filename):
        """Read the NWCSAF PPS Cloud Type"""
        h5f = h5py.File(filename, "r")
        # Global attributes:
        self.version = h5f.attrs['version']
        self.satellite_id = h5f.attrs['satellite_id']
        self.des = h5f.attrs['description']
        self.orbit_number = h5f.attrs['orbit_number']
        self.sec_1970 = h5f.attrs['sec_1970']

        # Data:
        nodata = 0
        ctype = np.ma.array(h5f['cloudtype'].value)
        self.cloudtype = np.ma.masked_equal(ctype, nodata)
        self.data = self.cloudtype
        mask = self.cloudtype.mask

        self.cloudtype_des = h5f['cloudtype'].attrs['description']
        qflags = h5f['quality_flag'].value
        self.qualityflag = np.ma.array(qflags, mask=mask)
        self.qualityflag_des = h5f['quality_flag'].attrs['description']
        phflags = h5f['phase_flag'].value
        self.phaseflag = np.ma.array(phflags, mask=mask)
        self.phaseflag_des = h5f['phase_flag'].attrs['description']
        # LUTs:
        self.cloudtype_lut = h5f['cloudtype'].attrs['output_value_namelist']
        self.qualityflag_lut = h5f['quality_flag'].attrs['output_value_namelist']
        self.phaseflag_lut = h5f['phase_flag'].attrs['output_value_namelist']
        
        self.palette = h5f['PALETTE'].value
        h5f.close()

        if not self.shape:
            self.shape = self.cloudtype.shape

        self.filled = True

    def project(self, coverage):
        """Remaps the NWCSAF/PPS level2 data to cartographic
        map-projection on a user defined area.
        """
        LOG.info("Projecting product %s..."%(self.name))
        retv = PpsCloudType(None)
        retv.cloudtype = coverage.project_array(self.cloudtype)
        retv.area = coverage.out_area
        retv.shape = retv.shape
        retv.resolution = self.resolution
        retv.orbit_number = self.orbit_number
        retv.satellite_id = self.satellite_id
        retv.info = self.info
        retv.filled = True
        valid_min = retv.cloudtype.min()
        valid_max = retv.cloudtype.max()
        retv.info['valid_range'] = np.array([valid_min, valid_max])
        retv.info['var_data'] = retv.cloudtype

        return retv

    def __str__(self):
        return ("'%s: shape %s, resolution %sm'"%
                (self.name, 
                 self.shape, 
                 self.resolution))   

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
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
        
    def read(self, filename):
        """Read the NWCSAF PPS Cloud Top Temperature & Height"""
        pass


def load(scene, **kwargs):
    """Load data into the *channels*. *Channels* is a list or a tuple
    containing channels we will load data into. If None, all channels are
    loaded.
    """

    del kwargs

    import glob

    if("CTTH" not in scene.channels_to_load and
       "CloudType" not in scene.channels_to_load):
        return
    
    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, scene.fullname+".cfg"))
    directory = conf.get(scene.instrument_name+"-level3", "dir")
    filename = conf.get(scene.instrument_name+"-level3", "filename",
                        raw=True)
    pathname_tmpl = os.path.join(directory, filename)

    lonlat_dir = conf.get(scene.instrument_name+"-level3", "lonlat_dir")
    lonlat_filename = conf.get(scene.instrument_name+"-level3", "lonlat_filename",
                               raw=True)
    lonlat_tmpl = os.path.join(lonlat_dir, lonlat_filename)

    area_name = "satproj"
    filename_tmpl = (scene.time_slot.strftime(lonlat_tmpl)
                     %{"orbit": scene.orbit,
                       "area": area_name,
                       "satellite": scene.satname})
    
    file_list = glob.glob(filename_tmpl)
    if len(file_list) > 1:
        raise IOError("More than one Geolocation file matching!")
    elif len(file_list) == 0:
        raise IOError("No NWCSAF/PPS Geolocation matching!: " + filename_tmpl)

    filename = file_list[0]
    if not os.path.exists(filename):
        raise IOError("Can't find geolocation file")
    else:
        lon, lat = get_lonlat(filename)

    lonlat_is_loaded = False

    if "CTTH" in scene.channels_to_load:
        filename_tmpl = (scene.time_slot.strftime(pathname_tmpl)
                         %{"orbit": scene.orbit,
                           "area": area_name,
                           "satellite": scene.satname,
                           "product": "ctth"})
    
        file_list = glob.glob(filename_tmpl)
        if len(file_list) > 1:
            raise IOError("More than 1 file matching!")
        elif len(file_list) == 0:
            raise IOError("No NWCSAF PPS CTTH matching!: " + filename_tmpl)

        filename = file_list[0]

        if not os.path.exists(filename):
            LOG.info("Can't find any CTTH file, skipping")
        else:
            ct_chan = PpsCTTH()
            ct_chan.read(filename)
            #ct_chan.area = scene.area
            scene.channels.append(ct_chan)

        if not lonlat_is_loaded:
            pass # FIXME!

            
    if "CloudType" in scene.channels_to_load:
        filename_tmpl = (scene.time_slot.strftime(pathname_tmpl)
                         %{"orbit": scene.orbit,
                           "area": area_name,
                           "satellite": scene.satname,
                           "product": "cloudtype"})

        file_list = glob.glob(filename_tmpl)
        if len(file_list) > 1:
            raise IOError("More than 1 file matching!")
        elif len(file_list) == 0:
            raise IOError("No NWCSAF PPS Cloudtype matching!: " + filename_tmpl)

        filename = file_list[0]

        if not os.path.exists(filename):
            LOG.info("Can't find any Cloudyype file, skipping")
        else:
            ct_chan = PpsCloudType()
            ct_chan.read(filename)
            ct_chan.area = scene.area
            scene.channels.append(ct_chan)

        if not lonlat_is_loaded:
            lons = np.ma.array(lon, mask=ct_chan.cloudtype.mask)
            lats = np.ma.array(lat, mask=ct_chan.cloudtype.mask)
            lonlat_is_loaded = True

    try:
        from pyresample import geometry
        scene.area = geometry.SwathDefinition(lons=lons, 
                                              lats=lats)
    except ImportError:
        scene.area = None
        scene.lat = lats
        scene.lon = lons

    LOG.info("Loading PPS parameters done.")


def get_lonlat(filename):
    """Read lon,lat from hdf5 file"""
    import h5py
    LOG.debug("Geo File = " + filename)

    h5f = h5py.File(filename, 'r')
    gain = h5f['where']['lon']['what'].attrs['gain']
    offset = h5f['where']['lon']['what'].attrs['offset']
    lons = h5f['where']['lon']['data'].value * gain + offset
    lats = h5f['where']['lat']['data'].value * gain + offset

    h5f.close()
    return lons, lats

