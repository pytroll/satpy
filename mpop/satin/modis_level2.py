#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011, 2013, 2014.

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

"""Plugin for reading AQUA MODIS level 2 EOS HDF files downloaded from NASA FTP import
"""

import os.path
from ConfigParser import ConfigParser

import datetime
import numpy as np
from pyhdf.SD import SD

from mpop import CONFIG_PATH

import mpop.channel
#from mpop.projector import get_area_def

import logging

LOG = logging.getLogger(__name__)

EOS_SATELLITE = {'aqua': 'eos2',
                 'modisa': 'eos2',
                 'terra': 'eos1',
                 'modist': 'eos1'}

SCAN_LINE_ATTRS = ['year', 'day', 'msec',
                   'slat', 'slon', 'clat', 'clon',
                   'elat', 'elon', 'csol_z'
                   ]

GEO_PHYS_PRODUCTS = ['aot_869', 'chlor_a', 'chlor_oc5',
                     'poc', 'cdom_index', 'angstrom',
                     'pic', 'par', 'sst',
                     'nflh', 'ipar', 'Kd_490']

CHANNELS = ['Rrs_412',
            'Rrs_443',
            'Rrs_469',
            'Rrs_488',
            'Rrs_531',
            'Rrs_547',
            'Rrs_555',
            'Rrs_645',
            'Rrs_667',
            'Rrs_678',
            'nLw_412',
            'nLw_443',
            'nLw_469',
            'nLw_488',
            'nLw_531',
            'nLw_547',
            'nLw_555',
            'nLw_645',
            'nLw_667',
            'nLw_678',
            ]

# Flags and quality (the two latter only for SST products):
FLAGS_QUALITY = ['l2_flags', 'qual_sst', 'qual_sst4']

SENSOR_BAND_PARAMS = ['wavelength', 'F0', 'vcal_offset', 'vcal_gain',
                      'Tau_r', 'k_oz']

# Navigation control points and tilt - no LONLAT:
NAVIGATION_TILT = ['tilt', 'cntl_pt_cols', 'cntl_pt_rows']
# Geo-location - Longitude,latitude:
LONLAT = ['longitude', 'latitude']


class ModisEosHdfLevel2(mpop.channel.GenericChannel):

    """NASA EOS-HDF Modis data struct"""

    def __init__(self, prodname, resolution=None):
        mpop.channel.GenericChannel.__init__(self, prodname)
        self.filled = False
        self.name = prodname
        self.resolution = resolution

        self.info = {}
        self._eoshdf_info = {}
        self.shape = None
        self.satid = ""
        self.orbit = None
        self.attr = None

        #self.scanline_attrs = {}
        self.data = None

        self.starttime = None
        self.endtime = None

    def __str__(self):
        return ("'%s: shape %s, resolution %sm'" %
                (self.name,
                 self.shape,
                 self.resolution))

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

    def read(self, filename, **kwargs):
        """Read the data"""
        from pyhdf.SD import SD
        import datetime

        # print "*** >>> Read the hdf-eos file!"
        root = SD(filename)

        # Get all the Attributes:
        # Common Attributes, Data Time,
        # Data Structure and Scene Coordinates
        for key in root.attributes().keys():
            self._eoshdf_info[key] = root.attributes()[key]

        # Start Time - datetime object
        starttime = datetime.datetime.strptime(self._eoshdf_info['Start Time'][0:13],
                                               "%Y%j%H%M%S")
        msec = float(self._eoshdf_info['Start Time'][13:16]) / 1000.
        self.starttime = starttime + datetime.timedelta(seconds=msec)

        # End Time - datetime object
        endtime = datetime.datetime.strptime(self._eoshdf_info['End Time'][0:13],
                                             "%Y%j%H%M%S")
        msec = float(self._eoshdf_info['End Time'][13:16]) / 1000.
        self.endtime = endtime + datetime.timedelta(seconds=msec)

        # What is the leading 'H' doing here?
        sensor_name = self._eoshdf_info['Sensor Name'][1:-1].lower()
        try:
            self.satid = EOS_SATELLITE[sensor_name]
        except KeyError:
            LOG.error("Failed setting the satellite id - sat-name = ",
                      sensor_name)

        self.orbit = self._eoshdf_info['Orbit Number']
        self.shape = (self._eoshdf_info['Number of Scan Control Points'],
                      self._eoshdf_info['Number of Pixel Control Points'])

        # try:
        if 1:
            value = root.select(self.name)
            attr = value.attributes()
            data = value.get()

            self.attr = attr
            band = data
            if self.name in FLAGS_QUALITY:
                self.data = band
            else:
                nodata = attr['bad_value_scaled']
                self.data = (np.ma.masked_equal(band, nodata) *
                             attr['slope'] + attr['intercept'])

            value.endaccess()
        # except:
        #    pass

        root.end()
        self.filled = True

    def project(self, coverage):
        """Remaps the Modis EOS-HDF level2 ocean products to cartographic
        map-projection on a user defined area.
        """
        LOG.info("Projecting product %s..." % (self.name))
        #print("Inside project...")

        retv = ModisEosHdfLevel2(self.name)
        retv.data = coverage.project_array(self.data)
        retv.area = coverage.out_area
        retv.shape = retv.data.shape
        retv.resolution = self.resolution
        retv.info = self.info
        retv.filled = True
        valid_min = retv.data.min()
        valid_max = retv.data.max()
        retv.info['valid_range'] = np.array([valid_min, valid_max])
        retv.info['var_data'] = retv.data

        return retv


def load(satscene, **kwargs):
    """Read data from file and load it into *satscene*.  Load data into the
    *channels*. *Channels* is a list or a tuple containing channels we will
    load data into. If None, all channels are loaded.
    """
    del kwargs

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level3",
                                    raw=True):
        options[option] = value

    pathname = os.path.join(options["dir"], options['filename'])
    filename = satscene.time_slot.strftime(pathname)

    for prodname in GEO_PHYS_PRODUCTS + FLAGS_QUALITY:
        if prodname in satscene.channels_to_load:

            prod_chan = ModisEosHdfLevel2(prodname)
            prod_chan.read(filename)
            prod_chan.satid = satscene.satname.capitalize()
            prod_chan.resolution = 1000.0
            prod_chan.shape = prod_chan.data.shape

            # All this for the netCDF writer:
            prod_chan.info['var_name'] = prodname
            prod_chan.info['var_data'] = prod_chan.data
            resolution_str = str(int(prod_chan.resolution)) + 'm'
            prod_chan.info['var_dim_names'] = ('y' + resolution_str,
                                               'x' + resolution_str)
            prod_chan.info['long_name'] = prod_chan.attr['long_name'][:-1]
            try:
                prod_chan.info['standard_name'] = prod_chan.attr[
                    'standard_name'][:-1]
            except KeyError:
                pass
            valid_min = np.min(prod_chan.data)
            valid_max = np.max(prod_chan.data)
            prod_chan.info['valid_range'] = np.array([valid_min, valid_max])
            prod_chan.info['resolution'] = prod_chan.resolution

            if prodname == 'l2_flags':
                # l2 flags definitions
                for i in range(1, 33):
                    key = "f%02d_name" % i
                    prod_chan.info[key] = prod_chan.attr[key][:-1]

            satscene.channels.append(prod_chan)
            if prodname in CHANNELS:
                satscene[prodname].info['units'] = '%'
            else:
                satscene[prodname].info['units'] = prod_chan.attr['units'][:-1]

            LOG.info("Loading modis lvl2 product '%s' done" % prodname)

    # Check if there are any bands to load:
    channels_to_load = False
    for bandname in CHANNELS:
        if bandname in satscene.channels_to_load:
            channels_to_load = True
            break

    if channels_to_load:
        # print "FILE: ", filename
        eoshdf = SD(filename)
        # Get all the Attributes:
        # Common Attributes, Data Time,
        # Data Structure and Scene Coordinates
        info = {}
        for key in eoshdf.attributes().keys():
            info[key] = eoshdf.attributes()[key]

        dsets = eoshdf.datasets()
        selected_dsets = []

        for bandname in CHANNELS:
            if (bandname in satscene.channels_to_load and
                    bandname in dsets):

                value = eoshdf.select(bandname)
                selected_dsets.append(value)

                # Get only the selected datasets
                attr = value.attributes()
                band = value.get()

                nodata = attr['bad_value_scaled']
                mask = np.equal(band, nodata)
                satscene[bandname] = (np.ma.masked_where(mask, band) *
                                      attr['slope'] + attr['intercept'])

                satscene[bandname].info['units'] = '%'
                satscene[bandname].info['long_name'] = attr['long_name'][:-1]

        for dset in selected_dsets:
            dset.endaccess()

        LOG.info("Loading modis lvl2 Remote Sensing Reflectances done")
        eoshdf.end()

    lat, lon = get_lat_lon(satscene, None)

    from pyresample import geometry
    satscene.area = geometry.SwathDefinition(lons=lon, lats=lat)

    # print "Variant: ", satscene.variant
    satscene.variant = 'regional'  # Temporary fix!

    LOG.info("Loading modis data done.")


def get_lonlat(satscene, row, col):
    """Estimate lon and lat.
    """
    estimate = False
    try:
        latitude, longitude = get_lat_lon(satscene, None)

        lon = longitude[row, col]
        lat = latitude[row, col]
        if (longitude.mask[row, col] == False and
                latitude.mask[row, col] == False):
            estimate = False
    except TypeError:
        pass
    except IndexError:
        pass
    except IOError:
        estimate = True

    if not estimate:
        return lon, lat


def get_lat_lon(satscene, resolution):
    """Read lat and lon.
    """
    del resolution

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level3",
                                    raw=True):
        options[option] = value

    pathname = os.path.join(options["dir"], options['filename'])
    filename = satscene.time_slot.strftime(pathname)

    root = SD(filename)
    lon = root.select('longitude')
    longitude = lon.get()
    lat = root.select('latitude')
    latitude = lat.get()

    return latitude, longitude
