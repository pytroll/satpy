#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2012, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

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
"""Interface to PPS level1 radiance data in ODIM format.
"""
import glob
import os.path
from ConfigParser import ConfigParser

from datetime import datetime, timedelta
import math
import numpy as np

from mpop import CONFIG_PATH
import logging

LOG = logging.getLogger(__name__)

import h5py


EPSILON = 0.001


class InfoObject(object):

    """Simple data and info container.
    """

    def __init__(self):
        self.info = {}
        self.data = None


def pack_signed(data, data_type):
    bits = np.iinfo(data_type).bits
    scale_factor = (data.max() - data.min()) / (2 ** bits - 2)
    add_offset = (data.max() - data.min()) / 2
    no_data = - 2 ** (bits - 1)
    pack = ((data - add_offset) / scale_factor).astype(data_type)
    return pack, scale_factor, add_offset, no_data


def load(satscene, *args, **kwargs):
    """Read data from file and load it into *satscene*.
    """
    del args, kwargs
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2",
                                    raw=True):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options)


def load_channels(satscene, options):
    """Read avhrr/viirs/modis radiance (tb's and refl) data from file and load
    it into *satscene*.
    """

    if "filename" not in options:
        raise IOError("No filename given, cannot load.")

    LOG.debug("Start loading channels")

    if satscene.instrument_name in ['avhrr']:
        chns = satscene.channels_to_load & set(
            ["1", "2", "3A", "3B", "4", "5"])
    elif satscene.instrument_name in ['viirs']:
        chns = satscene.channels_to_load & set(["M05", "M07", "M10",
                                                "M12", "M14", "M15", "M16"])

    if len(chns) == 0:
        return

    values = {"orbit": satscene.orbit,
              "satname": satscene.satname,
              "number": satscene.number,
              "instrument": satscene.instrument_name,
              "satellite": satscene.fullname
              }

    filename = os.path.join(satscene.time_slot.strftime(options["dir"]) % values,
                            satscene.time_slot.strftime(options["filename"])
                            % values)

    file_list = glob.glob(filename)

    if len(file_list) > 1:
        raise IOError("More than one l1b file matching!")
    elif len(file_list) == 0:
        raise IOError("No PPS level 1 file matching!: " +
                      filename)

    filename = file_list[0]

    LOG.debug("Loading from " + filename)

    available_channels = set([])
    data_channels = {}
    info_channels = {}
    instrument_data = NwcSafPpsOdim(filename)

    idx = 1
    while hasattr(instrument_data, 'image%d' % idx):
        channel = getattr(instrument_data, 'image%d' % idx)
        channel_name = channel.info['channel'].upper()
        #channel_des = channel.info['description']
        available_channels |= set([channel_name])
        data_channels[channel_name] = channel.data
        info_channels[channel_name] = channel.info
        idx = idx + 1

    for chn in satscene.channels_to_load:
        if chn in available_channels:
            if info_channels[chn]['quantity'] in ["REFL"]:
                units = "%"
            elif info_channels[chn]['quantity'] in ["TB"]:
                units = "K"
            else:
                LOG.warning("Units not known! Unit = " +
                            str(info_channels[chn]['quantity']))

            gain = info_channels[chn]["gain"]
            intercept = info_channels[chn]["offset"]

            chn_array = np.ma.array(data_channels[chn])
            missing_data = info_channels[chn]["missingdata"]
            chn_array = np.ma.masked_inside(chn_array,
                                            missing_data - EPSILON,
                                            missing_data + EPSILON)
            no_data = info_channels[chn]["nodata"]
            chn_array = np.ma.masked_inside(chn_array,
                                            no_data - EPSILON,
                                            no_data + EPSILON)

            satscene[chn] = chn_array
            satscene[chn].data = np.ma.masked_less(satscene[chn].data *
                                                   gain +
                                                   intercept,
                                                   0)

            satscene[chn].info['units'] = units
        else:
            LOG.warning("Channel " + str(chn) + " not available, not loaded.")

    # Compulsory global attributes
    satscene.info["title"] = (satscene.satname.capitalize() + satscene.number +
                              " satellite, " +
                              satscene.instrument_name.capitalize() +
                              " instrument.")
    satscene.info["institution"] = "Data processed by EUMETSAT NWCSAF/PPS."
    satscene.add_to_history("PPS level 1 data read by mpop.")
    satscene.info["references"] = "No reference."
    satscene.info["comments"] = "No comment."

    lons = (instrument_data.lon.data * instrument_data.lon.info['gain'] +
            instrument_data.lon.info['offset'])
    lats = (instrument_data.lat.data * instrument_data.lat.info['gain'] +
            instrument_data.lat.info['offset'])

    try:
        from pyresample import geometry
        satscene.area = geometry.SwathDefinition(lons=lons, lats=lats)
    except ImportError:
        satscene.area = None
        satscene.lat = lats
        satscene.lon = lons


class NwcSafPpsOdim(object):

    def __init__(self, filename=None):
        self._how = {}
        self._what = {}
        self._projectables = []
        self._keys = []
        self._refs = {}
        self.lon = None
        self.lat = None
        self.shape = None
        if filename:
            self.read(filename)

    def read(self, filename):
        """Read data in hdf5 format from *filename*
        """

        h5f = h5py.File(filename, "r")

        # Read the /how attributes

        self._how = dict(h5f['how'].attrs)
        self._what = dict(h5f['what'].attrs)
        self._how["satellite"] = h5f['how'].attrs['platform']
        # Which one to use?:
        self._how["time_slot"] = (timedelta(seconds=long(h5f['how'].attrs['startepochs']))
                                  + datetime(1970, 1, 1, 0, 0))
        self._what["time_slot"] = datetime.strptime(h5f['what'].attrs['date'] +
                                                    h5f['what'].attrs['time'],
                                                    "%Y%m%d%H%M%S")

        # Read the data and attributes
        #   This covers only one level of data. This could be made recursive.
        for key, dataset in h5f.iteritems():
            if "how" in dataset.name or "what" in dataset.name:
                continue

            if "image" in dataset.name:
                setattr(self, key, InfoObject())
                getattr(self, key).info = dict(dataset.attrs)
                getattr(self, key).data = dataset['data'][:]

                if 'how' in dataset:
                    for skey, value in dataset['how'].attrs.iteritems():
                        getattr(self, key).info[skey] = value
                if 'what' in dataset:
                    for skey, value in dataset['what'].attrs.iteritems():
                        getattr(self, key).info[skey] = value

            if "where" in dataset.name:
                setattr(self, 'lon', InfoObject())
                getattr(self, 'lon').data = h5f['/where/lon/data'][:]
                getattr(self, 'lon').info = dict(dataset.attrs)
                for skey, value in dataset['lon/what'].attrs.iteritems():
                    getattr(self, 'lon').info[skey] = value

                setattr(self, 'lat', InfoObject())
                getattr(self, 'lat').data = h5f['/where/lat/data'][:]
                getattr(self, 'lat').info = dict(dataset.attrs)
                for skey, value in dataset['lat/what'].attrs.iteritems():
                    getattr(self, 'lat').info[skey] = value

                self.shape = self.lon.data.shape

        h5f.close()

        # Setup geolocation

        try:
            from pyresample import geometry
        except ImportError:
            return

        if hasattr(self, "lon") and hasattr(self, "lat"):
            lons = self.lon.data * \
                self.lon.info["gain"] + self.lon.info["gain"]
            lats = self.lat.data * \
                self.lat.info["gain"] + self.lat.info["gain"]
            self.area = geometry.SwathDefinition(lons=lons, lats=lats)
        else:
            LOG.warning("No longitudes or latitudes for data")

    # def project(self, coverage):
    #     """Project what can be projected in the product.
    #     """
    #     import copy
    #     res = copy.copy(self)

    #     area = coverage.out_area

    # Project the data
    #     for var in self._projectables:
    #         LOG.info("Projecting " + str(var))
    #         res.__dict__[var] = copy.copy(self.__dict__[var])
    #         res.__dict__[var].data = coverage.project_array(
    #             self.__dict__[var].data)

    # Take care of geolocation
    # We support only swath data with full lon,lat arrays
    #     lons, scale_factor, add_offset, no_data = \
    #         pack_signed(area.lons[:], np.int16)
    #     res.lon = InfoObject()
    #     res.lon.data = lons
    #     res.lon.info["description"] = "geographic longitude (deg)"
    #     res.lon.info["offset"] = add_offset
    #     res.lon.info["gain"] = scale_factor
    #     res.lon.info["nodata"] = no_data
    #     if "lon" not in res._keys:
    #         res._keys.append("lon")

    #     lats, scale_factor, add_offset, no_data = \
    #         pack_signed(area.lats[:], np.int16)
    #     res.lat = InfoObject()
    #     res.lat.data = lats
    #     res.lat.info["description"] = "geographic latitude (deg)"
    #     res.lat.info["offset"] = add_offset
    #     res.lat.info["gain"] = scale_factor
    #     res.lat.info["nodata"] = no_data
    #     if "lat" not in res._keys:
    #         res._keys.append("lat")

    #     return res

    # def is_loaded(self):
    #     """Tells if the channel contains loaded data.
    #     """
    #     return len(self._projectables) > 0

CASES = {
    "avhrr": load_channels,
    "viirs": load_channels
}
