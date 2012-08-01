#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2012.

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

"""Plugin for reading PPS's cloud products hdf files.
"""
import ConfigParser
from ConfigParser import NoOptionError

from datetime import datetime, timedelta
import os.path

import mpop.channel
from mpop import CONFIG_PATH
from mpop.utils import get_logger
import numpy as np

import h5py

LOG = get_logger('satin/nwcsaf_pps')

class InfoObject(object):
    """Simple data and info container.
    """
    def __init__(self):
        self.info = {}
        self.data = None

def pack_signed(data, data_type):
    bits = np.iinfo(data_type).bits
    scale_factor = (data.max() - data.min()) / (2**bits - 2)
    add_offset = (data.max() - data.min()) / 2
    no_data = - 2**(bits - 1)
    pack = ((data - add_offset) / scale_factor).astype(data_type)
    return pack, scale_factor, add_offset, no_data


class NwcSafPpsChannel(mpop.channel.GenericChannel):

    def __init__(self, filename=None):
        mpop.channel.GenericChannel.__init__(self)
        self._md = {}
        self._projectables = []
        self._keys = []
        self._refs = {}
        self.shape = None
        if filename:
            self.read(filename)
            
    
    def read(self, filename):
        """Read product in hdf format from *filename*
        """
        h5f = h5py.File(filename, "r")

        # Read the global attributes

        self._md = dict(h5f.attrs)
        self._md["satellite"] = h5f.attrs['satellite_id']
        self._md["time_slot"] = (timedelta(seconds=long(h5f.attrs['sec_1970']))
                                 + datetime(1970, 1, 1, 0, 0))

        # Read the data and attributes
        #   This covers only one level of data. This could be made recursive.
        for key, dataset in h5f.iteritems():
            setattr(self, key, InfoObject())
            getattr(self, key).info = dict(dataset.attrs)
            for skey, value in dataset.attrs.iteritems():
                if isinstance(value, h5py.h5r.Reference):
                    self._refs[(key, skey)] = h5f[value].name.split("/")[1]
            try:
                getattr(self, key).data = dataset[:]
                is_palette = (dataset.attrs.get("CLASS", None) == "PALETTE")
                if(len(dataset.shape) > 1 and
                   not is_palette and
                   key not in ["lon", "lat"]):
                    self._projectables.append(key)
                    if self.shape is None:
                        self.shape = dataset.shape
                    elif self.shape != dataset.shape:
                        raise ValueError("Different variable shapes !")
                else:
                    self._keys.append(key)
            except TypeError:
                setattr(self, key, np.dtype(dataset))
                self._keys.append(key)

        h5f.close()

        # Setup geolocation

        try:
            from pyresample import geometry
        except ImportError:
            return

        if hasattr(self, "lon") and hasattr(self, "lat"):
            lons = self.lon.data * self.lon.info["gain"] + self.lon.info["intercept"]
            lats = self.lat.data * self.lat.info["gain"] + self.lat.info["intercept"]
            self.area = geometry.SwathDefinition(lons=lons, lats=lats)

        elif hasattr(self, "region") and self.region.data["area_extent"].any():
            region = self.region.data
            proj_dict = dict([elt.split('=')
                              for elt in region["pcs_def"].split(',')])
            self.area = geometry.AreaDefinition(region["id"],
                                                region["name"],
                                                region["proj_id"],
                                                proj_dict,
                                                region["xsize"],
                                                region["ysize"],
                                                region["area_extent"])
    def project(self, coverage):
        """Project what can be projected in the product.
        """

        import copy
        res = copy.copy(self)

        # Project the data
        for var in self._projectables:
            LOG.info("Projecting " + str(var))
            res.__dict__[var] = copy.copy(self.__dict__[var])
            res.__dict__[var].data = coverage.project_array(
                self.__dict__[var].data)

        # Take care of geolocation

        res.region = copy.copy(self.region)

        region = copy.copy(res.region.data)
        area = coverage.out_area
        try:
            # It's an area
            region["area_extent"] = np.array(area.area_extent)
            region["xsize"] = area.x_size
            region["ysize"] = area.y_size
            region["xscale"] = area.pixel_size_x
            region["yscale"] = area.pixel_size_y
            region["lon_0"] = area.proj_dict.get("lon_0", 0)
            region["lat_0"] = area.proj_dict.get("lat_0", 0)
            region["lat_ts"] = area.proj_dict.get("lat_ts", 0)
            region["name"] = area.name
            region["id"] = area.area_id
            region["pcs_id"] = area.proj_id
            pcs_def = ",".join([key + "=" + val
                                for key, val in area.proj_dict.iteritems()])
            region["pcs_def"] = pcs_def
            res.region.data = region

            # If switching to area representation, try removing lon and lat
            try:
                delattr(res, "lon")
                res._keys.remove("lon")
                delattr(res, "lat")
                res._keys.remove("lat")
            except AttributeError:
                pass
            
        except AttributeError:
            # It's a swath
            lons, scale_factor, add_offset, no_data = \
                  pack_signed(area.lons[:], np.int16)
            res.lon = InfoObject()
            res.lon.data = lons
            res.lon.info["description"] = "geographic longitude (deg)"
            res.lon.info["intercept"] = add_offset
            res.lon.info["gain"] = scale_factor
            res.lon.info["no_data_value"] = no_data
            if "lon" not in res._keys:
                res._keys.append("lon")

            lats, scale_factor, add_offset, no_data = \
                  pack_signed(area.lats[:], np.int16)
            res.lat = InfoObject()
            res.lat.data = lats
            res.lat.info["description"] = "geographic latitude (deg)"
            res.lat.info["intercept"] = add_offset
            res.lat.info["gain"] = scale_factor
            res.lat.info["no_data_value"] = no_data
            if "lat" not in res._keys:
                res._keys.append("lat")
            # Remove region parameters if switching from area
            region["area_extent"] = np.zeros(4)
            region["xsize"] = 0
            region["ysize"] = 0
            region["xscale"] = 0
            region["yscale"] = 0
            region["lon_0"] = 0
            region["lat_0"] = 0
            region["lat_ts"] = 0
            region["name"] = ""
            region["id"] = ""
            region["pcs_id"] = ""
            region["pcs_def"] = ""
            res.region.data = region
        return res

    def write(self, filename):
        """Write product in hdf format to *filename*
        """
        
        LOG.debug("Writing to " + filename)
        h5f = h5py.File(filename, "w")

        for dataset in self._projectables:
            dset = h5f.create_dataset(dataset, data=getattr(self, dataset).data,
                                      compression='gzip', compression_opts=6)
            for key, value in getattr(self, dataset).info.iteritems():
                dset.attrs[key] = value

        for thing in self._keys:
            try:
                dset = h5f.create_dataset(thing, data=getattr(self, thing).data,
                                          compression='gzip', compression_opts=6)
                for key, value in getattr(self, thing).info.iteritems():
                    dset.attrs[key] = value
            except AttributeError:
                h5f[thing] = getattr(self, thing)

        for key, value in self._md.iteritems():
            if key in ["time_slot", "satellite"]:
                continue
            h5f.attrs[key] = value

        for (key, skey), value in self._refs.iteritems():
            h5f[key].attrs[skey] = h5f[value].ref

        h5f.close()

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return len(self._projectables) > 0

class CloudType(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CloudType"

class CloudTopTemperatureHeight(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CTTH"

class CloudMask(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CMa"

class PrecipitationClouds(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "PC"

class CloudPhysicalProperties(NwcSafPpsChannel):

    def __init__(self):
        NwcSafPpsChannel.__init__(self)
        self.name = "CPP"



def load(scene, geofilename=None, **kwargs):
    del kwargs

    import glob

    lonlat_is_loaded = False

    products = []
    if "CTTH" in scene.channels_to_load:
        products.append("ctth")
    if "CloudType" in scene.channels_to_load:
        products.append("cloudtype")
    if "CMa" in scene.channels_to_load:
        products.append("cloudmask")
    if "PC" in scene.channels_to_load:
        products.append("precipclouds")
    if "CPP" in scene.channels_to_load:
        products.append("cpp")

    if len(products) == 0:
        return


    try:
        area_name = scene.area_id or scene.area.area_id
    except AttributeError:
        area_name = "satproj_?????_?????"


    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, scene.fullname+".cfg"))
    directory = conf.get(scene.instrument_name+"-level3", "dir")
    geodir = conf.get(scene.instrument_name+"-level3", "geodir")
    filename = conf.get(scene.instrument_name+"-level3", "filename",
                        raw=True)
    pathname_tmpl = os.path.join(directory, filename)

    if not geofilename:
        # Load geo file from config file:
        try:
            geoname_tmpl = conf.get(scene.instrument_name+"-level3", 
                                    "geofilename", raw=True)
            filename_tmpl = (scene.time_slot.strftime(geoname_tmpl)
                             %{"orbit": scene.orbit or "*",
                               "area": area_name,
                               "satellite": scene.satname + scene.number})

            file_list = glob.glob(os.path.join(geodir, filename_tmpl))
            if len(file_list) > 1:
                LOG.warning("More than 1 file matching for geoloaction")
            elif len(file_list) == 0:
                LOG.warning("No geolocation file matching!: " + filename_tmpl)
            else:
                geofilename = file_list[0]
        except NoOptionError:
            geofilename = None


    if geofilename:
        lons, lats = get_lonlat(geofilename)
        lonlat_is_loaded = True
    else:
        LOG.warning("No Geo file specified: No geolocation will be loaded")


    classes = {"ctth": CloudTopTemperatureHeight,
               "cloudtype": CloudType,
               "cloudmask": CloudMask,
               "precipclouds": PrecipitationClouds,
               "cpp": CloudPhysicalProperties
               }

    for product in products:
        LOG.debug("Loading " + product)
        filename_tmpl = (scene.time_slot.strftime(pathname_tmpl)
                         %{"orbit": scene.orbit or "*",
                           "area": area_name,
                           "satellite": scene.satname + scene.number,
                           "product": product})
    
        file_list = glob.glob(filename_tmpl)
        if len(file_list) > 1:
            LOG.warning("More than 1 file matching for " + product + "!")
        elif len(file_list) == 0:
            LOG.warning("No " + product + " matching!: " + filename_tmpl)
        else:
            filename = file_list[0]

            chn = classes[product]()
            chn.read(filename)
            scene.channels.append(chn)

    if lonlat_is_loaded:
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

