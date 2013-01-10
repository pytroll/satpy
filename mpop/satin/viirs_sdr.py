#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011, 2012.

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

Format documentation:
http://npp.gsfc.nasa.gov/science/sciencedocuments/082012/474-00001-03_CDFCBVolIII_RevC.pdf

"""
import os.path
from ConfigParser import ConfigParser

import numpy as np
import h5py
import hashlib

from mpop import CONFIG_PATH
from mpop.satin.logger import LOG
from mpop.utils import strftime

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
        self.filename = filename
        self.units = 'unknown'
        self.geo_filename = None

        self.latitude = None
        self.longitude = None


    def read(self, calibrate=1):
        """Read one VIIRS M- or I-band channel: Data and attributes (meta data)

        - *calibrate* set to 1 (default) returns reflectances for visual bands,
           tb for ir bands, and radiance for dnb.
           
        - *calibrate* set to 2 returns radiances.
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

        keys = h5f['Data_Products'].keys()
        idx = 0
        for key in keys:
            if key.find('SDR') >= 0:
                break
            idx = idx + 1

        # Then get the band info (Data_Products attributes):
        #bname = h5f['Data_Products'].keys()[0]
        bname = h5f['Data_Products'].keys()[idx]
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


        # Read the calibrated data

        if 'All_Data' not in h5f:
            raise IOError("No group 'All_Data' in hdf5 file:" + 
                          " %s" % self.filename)
        
        keys = h5f['All_Data'].keys()
        idx = 0
        for key in keys:
            if key.find('SDR') >= 0:
                break
            idx = idx + 1
        bname = keys[idx]
        keys = h5f['All_Data'][bname].keys()

        if calibrate == 1:
            # Get the M-band Tb or Reflectance:
            # First check if we have reflectances or brightness temperatures:
            tb_name = 'BrightnessTemperature'
            refl_name = 'Reflectance'
        elif calibrate == 2:
            tb_name = 'Radiance'
            refl_name = 'Radiance'
            
        rad_name = 'Radiance' # Day/Night band

        if tb_name in keys:
            band_data = h5f['All_Data'][bname][tb_name].value
            factors_name = tb_name + 'Factors'
            try:
                scale_factors = h5f['All_Data'][bname][factors_name].value
            except KeyError:
                scale_factors = 1.0, 0.0
            self.scale, self.offset = scale_factors[0:2]
            if calibrate == 1:
                self.units = 'K'
            elif calibrate == 2:
                self.units == 'W m-2 um-1 sr-1'
        elif refl_name in keys:
            band_data = h5f['All_Data'][bname][refl_name].value
            factors_name = refl_name + 'Factors'
            #self.scale, self.offset = h5f['All_Data'][bname][factors_name].value
            # In the data from CLASS this tuple is repeated 4 times!???
            # FIXME!
            self.scale, self.offset = h5f['All_Data'][bname][factors_name].value[0:2]
            if calibrate == 1:
                self.units = '%'
            elif calibrate == 2:
                self.units == 'W m-2 um-1 sr-1'
        elif refl_name not in keys and tb_name not in keys and rad_name in keys:
            band_data = h5f['All_Data'][bname][rad_name].value
            self.scale, self.offset = (10000., 0.) # The unit is W/sr cm-2 in the file!
            self.units = 'W sr-1 m-2'
        else:
            raise IOError('Neither brightness temperatures nor ' + 
                          'reflectances in the SDR file!')

        # Masking spurious data
        # according to documentation, mask integers >= 65328, floats <= -999.3
        if issubclass(band_data.dtype.type, np.integer):
            band_array = np.ma.masked_greater(band_data, 65528)
        if issubclass(band_data.dtype.type, np.floating):
            band_array = np.ma.masked_less(band_data, -999.2)

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
    LOG.debug("Geo File = " + filename)

    h5f = h5py.File(filename, 'r')
    # Doing it a bit dirty for now - AD:
    if band_id.find('M') == 0:
        try:
            lats = h5f['All_Data']['VIIRS-MOD-GEO-TC_All']['Latitude'].value
            lons = h5f['All_Data']['VIIRS-MOD-GEO-TC_All']['Longitude'].value
        except KeyError:
            lats = h5f['All_Data']['VIIRS-MOD-GEO_All']['Latitude'].value
            lons = h5f['All_Data']['VIIRS-MOD-GEO_All']['Longitude'].value
    elif band_id.find('I') == 0:
        try:
            lats = h5f['All_Data']['VIIRS-IMG-GEO-TC_All']['Latitude'].value
            lons = h5f['All_Data']['VIIRS-IMG-GEO-TC_All']['Longitude'].value
        except KeyError:
            lats = h5f['All_Data']['VIIRS-IMG-GEO_All']['Latitude'].value
            lons = h5f['All_Data']['VIIRS-IMG-GEO_All']['Longitude'].value
    elif band_id.find('D') == 0:
        lats = h5f['All_Data']['VIIRS-DNB-GEO_All']['Latitude'].value
        lons = h5f['All_Data']['VIIRS-DNB-GEO_All']['Longitude'].value
    else:
        raise IOError("Failed reading lon,lat: " + 
                      "Band-id not supported = %s" % (band_id))
    h5f.close()
    return (np.ma.masked_less(lons, -999, False), 
            np.ma.masked_less(lats, -999, False))


def load(satscene, *args, **kwargs):
    """Read data from file and load it into *satscene*.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level2",
                                    raw = True):
        options[option] = value

    CASES[satscene.instrument_name](satscene, options, *args, **kwargs)


def globify(filename):
    filename = filename.replace("%Y", "????")
    filename = filename.replace("%m", "??")
    filename = filename.replace("%d", "??")
    filename = filename.replace("%H", "??")
    filename = filename.replace("%M", "??")
    filename = filename.replace("%S", "??")
    return filename


def load_viirs_sdr(satscene, options, *args, **kwargs):
    """Read viirs SDR reflectances and Tbs from file and load it into
    *satscene*.
    """
    calibrate = kwargs.get('calibrate', 1)
    band_list = [ s.name for s in satscene.channels ]
    chns = satscene.channels_to_load & set(band_list)
    if len(chns) == 0:
        return

    import glob

    if "filename" not in options:
        raise IOError("No filename given, cannot load")

    values = {"orbit": satscene.orbit,
              "satname": satscene.satname,
              "instrument": satscene.instrument_name,
              "satellite": satscene.satname
              #"satellite": satscene.fullname
              }

    filename_tmpl = strftime(satscene.time_slot, options["filename"]) %values

    directory = strftime(satscene.time_slot, options["dir"]) % values

    if not os.path.exists(directory):
        directory = globify(options["dir"]) % values
        directories = glob.glob(directory)
        if len(directories) > 1:
            raise IOError("More than one directory for npp scene... " + 
                          "\nSearch path = %s\n\tPlease check npp.cfg file!" % directory)
        elif len(directories) == 0:
            raise IOError("No directory found for npp scene. " + 
                          "\nSearch path = %s\n\tPlease check npp.cfg file!" % directory)
        else:
            directory = directories[0]

    file_list = glob.glob(os.path.join(directory, filename_tmpl))
    filenames = [ os.path.basename(s) for s in file_list ]

    if len(file_list) > 22: # 22 VIIRS bands (16 M-bands + 5 I-bands + DNB)
        raise IOError("More than 22 files matching!")
    elif len(file_list) == 0:
        #LOG.warning("No VIIRS SDR file matching!: " + os.path.join(directory,
        #                                                           filename_tmpl))
        raise IOError("No VIIRS SDR file matching!: " + os.path.join(directory,
                                                                     filename_tmpl))
        return

    geo_filenames_tmpl = strftime(satscene.time_slot, options["geo_filenames"]) %values
    geofile_list = glob.glob(os.path.join(directory, geo_filenames_tmpl))

    m_lats = None
    m_lons = None
    i_lats = None
    i_lons = None

    m_lonlat_is_loaded = False
    i_lonlat_is_loaded = False
    glob_info = {}

    LOG.debug("Channels to load: " + str(satscene.channels_to_load))

    for chn in satscene.channels_to_load:
        # Take only those files in the list matching the band:
        # (Filename starts with 'SV' and then the band-name)
        fnames_band = []

        try:
            fnames_band = [ s for s in filenames if s.find('SV'+chn) >= 0 ]
        except TypeError:
            LOG.warning('Band frequency not available from VIIRS!')
            LOG.info('Asking for channel' + str(chn) + '!')

        LOG.debug("fnames_band = " + str(fnames_band))
        if len(fnames_band) == 0:
            continue

        filename_band = glob.glob(os.path.join(directory, 
                                               fnames_band[0]))
        
        if len(filename_band) > 1:
            raise IOError("More than one file matching band-name %s" % chn)


        band = ViirsBandData(filename_band[0])
        band.read(calibrate)
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

        if band_desc == "M":
            if not m_lonlat_is_loaded:
                mband_geos = [ s for s in geofile_list 
                             if os.path.basename(s).find('GMTCO') == 0 ]
                if len(mband_geos) == 1 and os.path.exists(mband_geos[0]):
                    band.read_lonlat(directory,
                                     filename=os.path.basename(mband_geos[0]))
                else:
                    band.read_lonlat(directory)
                m_lons = band.longitude
                m_lats = band.latitude
                m_lonlat_is_loaded = True
            else:
                band.longitude = m_lons
                band.latitude = m_lats

        if band_desc == "I":
            if not i_lonlat_is_loaded:
                iband_geos = [ s for s in geofile_list 
                             if os.path.basename(s).find('GITCO') == 0 ]
                if len(iband_geos) == 1 and os.path.exists(iband_geos[0]):
                    band.read_lonlat(directory,
                                     filename=os.path.basename(iband_geos[0]))
                else:
                    band.read_lonlat(directory)
                i_lons = band.longitude
                i_lats = band.latitude
                i_lonlat_is_loaded = True
            else:
                band.longitude = i_lons
                band.latitude = i_lats

        if band_desc == "DNB":
            dnb_geos = [ s for s in geofile_list 
                         if os.path.basename(s).find('GDNBO') == 0 ]
            if len(dnb_geos) == 1 and os.path.exists(dnb_geos[0]):
                band.read_lonlat(directory,
                                 filename=os.path.basename(dnb_geos[0]))
            else:
                band.read_lonlat(directory)
            dnb_lons = band.longitude
            dnb_lats = band.latitude

        band_uid = band_desc + hashlib.sha1(band.data.mask).hexdigest()
        
        try:
            from pyresample import geometry
        
            satscene[chn].area = geometry.SwathDefinition(
                lons=np.ma.array(band.longitude, mask=band.data.mask),
                lats=np.ma.array(band.latitude, mask=band.data.mask))

            area_name = ("swath_" + satscene.fullname + "_" +
                         str(satscene.time_slot) + "_"
                         + str(satscene[chn].data.shape) + "_" +
                         band_uid)
            satscene[chn].area.area_id = area_name
            satscene[chn].area_id = area_name
        except ImportError:
            satscene[chn].area = None
            satscene[chn].lat = np.ma.array(band.latitude, mask=band.data.mask)
            satscene[chn].lon = np.ma.array(band.longitude, mask=band.data.mask)

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
