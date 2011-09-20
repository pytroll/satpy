#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#

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

"""Interface to VIIRS SDR format 
"""
import os.path
from ConfigParser import ConfigParser

import numpy as np
import h5py

from mpop import CONFIG_PATH
from mpop.satin.logger import LOG

EPSILON = 0.001

# ------------------------------------------------------------------------------
class ViirsMbandData(object):
    """Placeholder for the VIIRS M-band data.
    Reads the SDR data - one hdf5 file for each band
    """
    def __init__(self, filename):
        self.global_info = {}
        self.band_info = {}
        self.orbit = -9
        self.orbit_begin = -9
        self.orbit_end = -9
        self.band_id = 'unknown'
        self.data = None
        self.scale = 1.0    # gain
        self.offset = 0.0   # intercept
        self.nodata = 65533 # Where do we get the no-data/fillvalue
                            # in the SDR hdf5 file!? FIXME!
        self.filename = filename
        self.units = 'unknown'
        self.geo_filename = None

        self.latitude = None
        self.longitude = None


    def read(self):
        """Read one VIIRS M-band channel: Data and attributes (meta data)
        """

        h5f = h5py.File(self.filename, 'r')

        # Get the global header info first:
        for key in h5f.attrs.keys():
            self.global_info[key] = h5f.attrs[key][0, 0]
            if key == 'N_GEO_Ref':
                self.geo_filename = h5f.attrs[key][0, 0]
                
        if 'Data_Products' not in h5f:
            raise IOError("No group 'All_Data' in hdf5 file: " + 
                          self.filename)

        # Then get the band info (Data_Products attributes):
        bname = h5f['Data_Products'].keys()[0]
        for gran_aggr in h5f['Data_Products'][bname].keys():
            attributes = h5f['Data_Products'][bname][gran_aggr].attrs
            for key in attributes.keys():
                self.band_info[key] = attributes[key]
            if key == 'Band_ID':
                self.band_id = attributes[key][0, 0]
            if key == 'AggregateBeginningOrbitNumber':
                self.orbit_begin = attributes[key][0, 0]
            if key == 'AggregateEndingOrbitNumber':
                self.orbit_end = attributes[key][0, 0]

        self.orbit = self.orbit_begin

        if 'All_Data' not in h5f:
            raise IOError("No group 'All_Data' in hdf5 file: %s" % self.filename)
        
        keys = h5f['All_Data'].keys()
        if len(keys) > 1:
            raise IOError("Unexpected file content - " + 
                          "more than one sub-group under 'All_Data'")
        bname = keys[0]
        keys = h5f['All_Data'][bname].keys()
    
        # Get the M-band Tb or Reflectance:
        # First check if we have reflectances or brightness temperatures:
        tb_name = 'BrightnessTemperature'
        refl_name = 'Reflectance'

        if tb_name in keys:
            mband_data = h5f['All_Data'][bname][tb_name].value
            factors_name = tb_name + 'Factors'
            scale, offset = h5f['All_Data'][bname][factors_name].value
            self.units = 'K'
        elif refl_name in keys:
            mband_data = h5f['All_Data'][bname][refl_name].value
            factors_name = refl_name + 'Factors'
            scale, offset = h5f['All_Data'][bname][factors_name].value
            self.units = '%'
        else:
            raise IOError('Neither brightness temperatures nor ' + 
                          'reflectances in the SDR file!')

        self.scale = scale
        self.offset = offset

        band_array = np.ma.array(mband_data)
        band_array = np.ma.masked_inside(band_array,
                                         self.nodata - EPSILON,
                                         self.nodata + EPSILON)

        # Is it necessary to mask negatives?
        self.data =  np.ma.masked_less(band_array *
                                       self.scale +
                                       self.offset,
                                       0)

        h5f.close()


    def read_lonlat(self, geodir, **kwargs):
        """Read the geolocation from the GMODO/GMTCO file"""
        if 'filename' in kwargs:
            # Overwriting the geo-filename:
            self.geo_filename = kwargs['filename']

        if not self.geo_filename:
            print("Trying to read geo-location without knowledge of which" +
                  "geolocation file to read it from!")
            print("Do nothing...")
            return
        
        lon, lat = get_lonlat(os.path.join(geodir, 
                                           self.geo_filename))
        self.longitude = lon
        self.latitude = lat


# ------------------------------------------------------------------------------
def get_lonlat(filename):
    """Read lon,lat from hdf5 file"""
    import h5py
    print("Geo File = " + filename)

    h5f = h5py.File(filename, 'r')
    # Doing it a bit dirty for now - AD:
    lats = h5f['All_Data']['VIIRS-MOD-GEO_All']['Latitude'].value
    lons = h5f['All_Data']['VIIRS-MOD-GEO_All']['Longitude'].value
    
    h5f.close()
    return lons, lats


def load(satscene, *args, **kwargs):
    """Read data from file and load it into *satscene*.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level2",
                                    raw = True):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options)


def load_viirs_sdr(satscene, options):
    """Read viirs SDR reflectances and Tbs from file and load it into *satscene*.
    """
    import glob

    if "filename" not in options:
        raise IOError("No filename given, cannot load")

    values = {"orbit": satscene.orbit,
              "satname": satscene.satname,
              "instrument": satscene.instrument_name,
              "satellite": satscene.satname
              #"satellite": satscene.fullname
              }

    filename_tmpl = satscene.time_slot.strftime(options["filename"]) %values

    file_list = glob.glob(os.path.join(options["dir"], filename_tmpl))
    filenames = [ os.path.basename(s) for s in file_list ]

    if len(file_list) > 22: # 22 VIIRS bands (16 M-bands + 5 I-bands + DNB)
        raise IOError("More than 22 files matching!")
    elif len(file_list) == 0:
        raise IOError("No VIIRS file matching!: " + filename_tmpl)

    lonlat_is_loaded = False
    glob_info = {}

    for chn in satscene.channels_to_load:
        # Take only those files in the list matching the band:
        # (Filename starts with 'SV' and then the band-name)
        #print filenames
        #print chn
        fnames_band = [ s for s in filenames if s.find(chn) == 2 ]
        if len(fnames_band) == 0:
            continue

        filename_band = glob.glob(os.path.join(options["dir"], 
                                               fnames_band[0]))
        
        if len(filename_band) > 1:
            raise IOError("More than one file matching band-name %s" % chn)

        mband = ViirsMbandData(filename_band[0])
        mband.read()

        satscene[chn].data = mband.data
        satscene[chn].info['units'] = mband.units

        # We assume the same geolocation should apply to all M-bands!
        
        if not lonlat_is_loaded:
            mband.read_lonlat(options["dir"])
            lons = mband.longitude
            lats = mband.latitude
            lonlat_is_loaded = True

        if 'institution' not in glob_info:
            glob_info['institution'] = mband.global_info['N_Dataset_Source']
        if 'mission_name' not in glob_info:
            glob_info['mission_name'] = mband.global_info['Mission_Name']

    # Compulsory global attribudes
    satscene.info["title"] = (satscene.satname.capitalize() + 
                              " satellite, " +
                              satscene.instrument_name.capitalize() +
                              " instrument.")
    if 'institution' in glob_info:
        satscene.info["institution"] = glob_info['institution']

    if 'mission_name' in glob_info:
        satscene.add_to_history(glob_info['mission_name'] + 
                                " VIIRS SDR read by mpop") 
    else:
        satscene.add_to_history("NPP/JPSS VIIRS SDR read by mpop")

    satscene.info["references"] = "No reference."
    satscene.info["comments"] = "No comment."

    #satscene.lat = lats
    #satscene.lon = lons

    try:
        from pyresample import geometry
        satscene.area = geometry.SwathDefinition(lons=lons, lats=lats)
    except ImportError:
        satscene.area = None
        satscene.lat = lats
        satscene.lon = lons



CASES = {
    "viirs": load_viirs_sdr
    }
