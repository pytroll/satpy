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
class ViirsBandData(object):
    """Placeholder for the VIIRS M&I-band data.
    Reads the SDR data - one hdf5 file for each band.
    Not yet considering the Day-Night Band
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
        """Read one VIIRS M- or I-band channel: Data and attributes (meta data)
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
                    bid = attributes[key][0, 0]
                    if bname.find('DNB') > 0: # and bid == 'N/A':
                        self.band_id = 'DNB'
                    else:
                        self.band_id = bid
                if key == 'AggregateBeginningOrbitNumber':
                    self.orbit_begin = attributes[key][0, 0]
                if key == 'AggregateEndingOrbitNumber':
                    self.orbit_end = attributes[key][0, 0]

        # Orbit number is here defined as identical to the 
        # orbit number at beggining of aggregation:
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
        rad_name = 'Radiance' # Day/Night band

        if tb_name in keys:
            band_data = h5f['All_Data'][bname][tb_name].value
            factors_name = tb_name + 'Factors'
            scale, offset = h5f['All_Data'][bname][factors_name].value
            self.units = 'K'
        elif refl_name in keys:
            band_data = h5f['All_Data'][bname][refl_name].value
            factors_name = refl_name + 'Factors'
            scale, offset = h5f['All_Data'][bname][factors_name].value
            self.units = '%'
        elif refl_name not in keys and tb_name not in keys and rad_name in keys:
            band_data = h5f['All_Data'][bname][rad_name].value
            scale, offset = (10000., 0.) # The unit is W/sr cm-2 in the file!
            self.units = 'W sr-1 m-2'
        else:
            raise IOError('Neither brightness temperatures nor ' + 
                          'reflectances in the SDR file!')

        self.scale = scale
        self.offset = offset

        band_array = np.ma.array(band_data)
        band_array = np.ma.masked_inside(band_array,
                                         self.nodata - EPSILON,
                                         self.nodata + EPSILON)

        # Is it necessary to mask negatives?
        # The VIIRS reflectances are between 0 and 1.
        # mpop standard is '%'
        if self.units == '%':
            myscale = 100.0 # To get reflectances in percent!
        else:
            myscale = 1.0
        self.data =  np.ma.masked_less(myscale * (band_array *
                                                  self.scale +
                                                  self.offset),
                                       0)
        h5f.close()

    def read_lonlat(self, geodir, **kwargs):
        """Read the lons and lats from the seperate geolocation file.
        In case of M-bands: GMODO (Geoid) or GMTCO (terrain corrected).
        In case of I-bands: GIMGO (Geoid) or GITCO (terrain corrected).
        """
        if 'filename' in kwargs:
            # Overwriting the geo-filename:
            self.geo_filename = kwargs['filename']

        if not self.geo_filename:
            LOG.warning("Trying to read geo-location without" +
                        "knowledge of which geolocation file to read it from!")
            LOG.warning("Do nothing...")
            return
        
        lon, lat = get_lonlat(os.path.join(geodir, 
                                           self.geo_filename),
                              self.band_id)

        self.longitude = lon
        self.latitude = lat


# ------------------------------------------------------------------------------
def get_lonlat(filename, band_id):
    """Read lon,lat from hdf5 file"""
    import h5py
    LOG.debug("Geo File = " + filename)

    h5f = h5py.File(filename, 'r')
    # Doing it a bit dirty for now - AD:
    if band_id.find('M') == 0:
        lats = h5f['All_Data']['VIIRS-MOD-GEO_All']['Latitude'].value
        lons = h5f['All_Data']['VIIRS-MOD-GEO_All']['Longitude'].value
    elif band_id.find('I') == 0:
        lats = h5f['All_Data']['VIIRS-IMG-GEO_All']['Latitude'].value
        lons = h5f['All_Data']['VIIRS-IMG-GEO_All']['Longitude'].value
    elif band_id.find('D') == 0:
        lats = h5f['All_Data']['VIIRS-DNB-GEO_All']['Latitude'].value
        lons = h5f['All_Data']['VIIRS-DNB-GEO_All']['Longitude'].value
    else:
        raise IOError("Failed reading lon,lat: " + 
                      "Band-id not supported = %s" % (band_id))
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

    m_lats = None
    m_lons = None
    i_lats = None
    i_lons = None
    dnb_lats = None
    dnb_lons = None

    m_lonlat_is_loaded = False
    i_lonlat_is_loaded = False
    glob_info = {}

    for chn in satscene.channels_to_load:
        # Take only those files in the list matching the band:
        # (Filename starts with 'SV' and then the band-name)
        fnames_band = []

        try:
            fnames_band = [ s for s in filenames if s.find(chn) == 2 ]
        except TypeError:
            LOG.warning('Band frequency not available from VIIRS!')
            LOG.info('Asking for channel' + str(chn) + '!')

        if len(fnames_band) == 0:
            continue

        filename_band = glob.glob(os.path.join(options["dir"], 
                                               fnames_band[0]))
        
        if len(filename_band) > 1:
            raise IOError("More than one file matching band-name %s" % chn)

        band = ViirsBandData(filename_band[0])
        band.read()
        LOG.debug('Band id = ' + band.band_id)

        band_desc = None # I-band or M-band or Day/Night band?
        if band.band_id.find('I') == 0:
            band_desc = "I"
        elif band.band_id.find('M') == 0:
            band_desc = "M"
        elif band.band_id.find('D') == 0:
            band_desc = "DNB"

        if not band_desc:
            LOG.warning('Band name = ' + band.band_id)
            raise AttributeError('Band description not supported!')


        satscene[chn].data = band.data
        satscene[chn].info['units'] = band.units
        satscene[chn].info['band_id'] = band.band_id

        # We assume the same geolocation should apply to all M-bands!
        # ...and the same to all I-bands:
        
        if band_desc == "M" and not m_lonlat_is_loaded:
            band.read_lonlat(options["dir"])
            # Masking the geo-location using mask from an abitrary band:
            m_lons = np.ma.array(band.longitude, mask=band.data.mask)
            m_lats = np.ma.array(band.latitude, mask=band.data.mask)
            m_lonlat_is_loaded = True

        if band_desc == "I" and not i_lonlat_is_loaded:
            band.read_lonlat(options["dir"])
            # Masking the geo-location using mask from an abitrary band:
            i_lons = np.ma.array(band.longitude, mask=band.data.mask)
            i_lats = np.ma.array(band.latitude, mask=band.data.mask)
            i_lonlat_is_loaded = True

        if band_desc == "DNB":
            band.read_lonlat(options["dir"])
            # Masking the geo-location:
            dnb_lons = np.ma.array(band.longitude, mask=band.data.mask)
            dnb_lats = np.ma.array(band.latitude, mask=band.data.mask)


        if band_desc == "M":
            lons = m_lons
            lats = m_lats
        elif band_desc == "I":
            lons = i_lons
            lats = i_lats
        elif band_desc == "DNB":
            lons = dnb_lons
            lats = dnb_lats
            
        try:
            from pyresample import geometry
            satscene[chn].area = geometry.SwathDefinition(lons=lons, 
                                                          lats=lats)
        except ImportError:
            satscene[chn].area = None
            satscene[chn].lat = lats
            satscene[chn].lon = lons


        if 'institution' not in glob_info:
            glob_info['institution'] = band.global_info['N_Dataset_Source']
        if 'mission_name' not in glob_info:
            glob_info['mission_name'] = band.global_info['Mission_Name']

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


CASES = {
    "viirs": load_viirs_sdr
    }
