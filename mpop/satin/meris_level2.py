#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011.

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

"""Plugin for reading Envisat MERIS level 2 files downloaded from ESA via ftp.
MERIS files are stored in the Envisat N1 format, and read using gdal plus an 
own reader to acquire the geolocation data.

MERIS bands:
------------
No. Band centre   Band width        Applications
        (nm)         (nm) 

1  412.5 10 Yellow substance and detrital pigments
2  442.5 10 Chlorophyll absorption maximum
3  490 10 Chlorophyll and other pigments
4  510 10 Suspended sediment, red tides
5  560 10 Chlorophyll absorption minimum
6  620 10 Suspended sediment
7  665 10 Chlorophyll absorption & fluorescence reference
8  681.25 7.5 Chlorophyll fluorescence peak
9  708.75 10 Fluorescence reference, atmosphere corrections
10 753.75 7.5 Vegetation, cloud, O2 absoption band reference
11 760.625 3.75 O2 R- branch absorption band
12 778.75 15 Atmosphere corrections
13 865 20 Atmosphere corrections
14 885 10 Vegetation, water vapour reference
15 900 10 Water vapour

This fixed set of bands was recommended by the Science Advisory Group (SAG). The level 2 ESA
products have been validated for this set of bands.

See meris.ProductHandbook.2_1.pdf for documentation on file format and content
"""

import sys
import os.path
import glob
from ConfigParser import ConfigParser

import datetime
import numpy as np

from osgeo.gdalconst import GA_ReadOnly
from osgeo import gdal
            
from mpop import CONFIG_PATH
from mpop.satin.logger import LOG
import mpop.channel

from mpop.plugin_base import Reader


EARTH_RADIUS = 6371000.0

CHANNELS = ['band-1',
            'band-2',
            'band-3',
            'band-4',
            'band-5',
            'band-6',
            'band-7',
            'band-8',
            'band-9',
            'band-10',
            'band-11',
            'band-12',
            'band-13',
            'band-14',
            'band-15']


# ------------------------------------------------------------------------    
if sys.version_info < (2, 5):
    import time
    def strptime(string, fmt=None):
        """This function is available in the datetime module only
        from Python >= 2.5.
        """

        return datetime.datetime(*time.strptime(string, fmt)[:6])
else:
    strptime = datetime.datetime.strptime


class MerisReader(Reader):
    """Reader for MERIS level2 data"""
    pformat = 'meris_level2'
    def __init__(self,  *args, **kwargs):
        Reader.__init__(self, *args, **kwargs)

        self.is_compressed = False
        self.compression = None # Supports only gzip compression

        self.gads = None
        self.satid = ""
        self.orbit = None
        self.info = {}

        conf = ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, self._scene.fullname + ".cfg"))
        options = {}
        for option, value in conf.items(self._scene.instrument_name+"-level3",
                                        raw = True):
            options[option] = value

        if "filename" not in options:
            raise IOError("No filename given, cannot load.")
        if "dir" not in options:
            raise IOError("No directory given, cannot load.")

        #MER_FRS_2PNPDE20101217_094256_000001993097_00295_45999_5414.N1.gz
        #filename = MER_FRS_2PNPDE%Y%m%d_%H%M*.N1.gz

        pathname = os.path.join(options["dir"], options['filename'])    
        print "Path = ", pathname
        LOG.debug("Looking for file %s" % self._scene.time_slot.strftime(pathname))
        file_list = glob.glob(self._scene.time_slot.strftime(pathname))
        print self._scene.time_slot.strftime(pathname)

        if len(file_list) > 1:
            raise IOError("More than one MERIS file matching!")
        elif len(file_list) == 0:
            raise IOError("No MERIS file matching!")

        self.filename = file_list[0]
        if self.filename and is_file_gzipped(self.filename):
            self.is_compressed = True
            self.compression = "gzip"
        
        LOG.info('File = ' + self.filename)
        LOG.info('Is compressed = ' + str(self.is_compressed))


    def load(self, *args, **kwargs):
        """load data"""
        del args
        load(self._scene, **kwargs)

    def get_gross_area(self):
        """Build a gross area of the MERIS scene based on its corners.
        """
        from pyresample.geometry import SwathDefinition

        # Get lon,lat of the four corner points.
        # Read the header:
        if not 'SPH_FIRST_FIRST_LONG' in self.info:
            self.read_header()

        fflon = self.info['SPH_FIRST_FIRST_LONG']
        fllon = self.info['SPH_FIRST_LAST_LONG']
        lflon = self.info['SPH_LAST_FIRST_LONG']
        lllon = self.info['SPH_LAST_LAST_LONG']

        fflat = self.info['SPH_FIRST_FIRST_LAT']
        fllat = self.info['SPH_FIRST_LAST_LAT']
        lflat = self.info['SPH_LAST_FIRST_LAT']
        lllat = self.info['SPH_LAST_LAST_LAT']

        top_left = fllon, fllat
        top_right = fflon, fflat
        bottom_left = lllon, lllat
        bottom_right = lflon, lflat

        LOG.info("Coreners: ")
        LOG.info("top-left: %f,%f" % (top_left[0], top_left[1]))
        LOG.info("top-right: %f,%f" % (top_right[0], top_right[1]))
        LOG.info("bottom-left: %f,%f" % (bottom_left[0], bottom_left[1]))
        LOG.info("bottom-right: %f,%f" % (bottom_right[0], bottom_right[1]))

        lons = np.array([[top_left[0], top_right[0]],
                         [bottom_left[0], bottom_right[0]]])
        lats = np.array([[top_left[1], top_right[1]],
                         [bottom_left[1], bottom_right[1]]])

        return SwathDefinition(lons, lats)
        

    def read_header(self):
        """Read the ascii header without using gdal.
        Supposed to be faster, especially if the file is compressed"""
        import gzip

        # For calibration:
        self.gads = GADS_ScalingFactorsAndOFfsets(self.filename)
        self.gads.get_products_scaling_factors()

        try:
            if self.is_compressed and self.compression == 'gzip':
                infile = gzip.open(self.filename, 'rb')
            else:
                infile = open(self.filename, 'rb')

            prefix = 'MPH_'
            for line in infile:
                if 'SPH_' in line:
                    prefix = 'SPH_'
                if 'DS_NAME' in line:
                    break
                if '=' in line:
                    items = line.split('=')
                    key = prefix + items[0].strip()
                    if 'LAT' in line or 'LONG' in line:
                        val = items[1].strip()
                    else:
                        try:
                            val = items[1].strip(('<samples>' + 
                                                  '<%>' + 
                                                  '<bytes>\n'))
                        except ValueError:
                            val = items[1].strip()
                    #print key, val
                    if 'LAT' in line or 'LONG' in line:
                        # Fix the lon and lat values properly:
                        val = get_lonlat_value(val)
                    if key not in self.info:
                        self.info[key] = val
        except Exception, exc:
            errmsg = `exc.__str__()`
            LOG.error("Failed extracting ascii header from file")
            LOG.error("Error = %s" % errmsg)
            raise exc

        if not self.orbit:
            self.orbit = int(self.info['MPH_ABS_ORBIT'])


# ------------------------------------------------------------------------    
class MerisLevel2Product(mpop.channel.GenericChannel):
    """ENVISAT MERIS data struct for products"""
    def __init__(self, filename, prodname, resolution = None):
        mpop.channel.GenericChannel.__init__(self, prodname)
        self.filled = False
        self.name = prodname
        self.resolution = resolution

        self.shape = None
        self.satid = ""
        self.orbit = None

        self.data = None

        self.starttime = None
        self.endtime = None

        self.filename = filename
        self.tempname = None

        self.is_compressed = False
        self.compression = None # Supports only gzip compression

        if self.filename and is_file_gzipped(self.filename):
            self.is_compressed = True
        
        # Level 2 products
        self.products = {}
        self.gads = None

    def read(self):
        """Read MERIS level 2 products from file"""
        
        self.gads = GADS_ScalingFactorsAndOFfsets(self.filename)
        self.gads.get_products_scaling_factors()

        data = get_product(self.filename, 'Chl_1', self.is_compressed)
        self.products['algal1'] = (data * self.gads.scale['Algal pigment index'] 
                                   + self.gads.offset['Algal pigment index'] )
        data = get_product(self.filename, 'Chl_2', self.is_compressed)
        self.products['algal2'] = (data * self.gads.scale['Algal pigment index'] 
                                   + self.gads.offset['Algal pigment index'] )

        print self.gads.offset
        print self.gads.scale

        if not self.shape:
            self.shape = self.products['algal2'].shape

        self.filled = True

    def project(self, coverage):
        """Remaps the MERIS level2 data to cartographic
        map-projection on a user defined area.
        """
        print "Projecting product %s..." % (self.name)
        LOG.info("Projecting product %s..."%(self.name))
        retv = MerisLevel2Product(None, self.name)
        retv.data = coverage.project_array(self.data)
        retv.area = coverage.out_area
        retv.shape = retv.data.shape
        retv.resolution = self.resolution
        retv.orbit = self.orbit
        retv.satid = self.satid
        retv.info = self.info
        retv.filled = True
        valid_min = retv.data.min()
        valid_max = retv.data.max()
        retv.info['valid_range'] = np.array([valid_min, valid_max])
        retv.info['var_data'] = retv.data

        return retv

    def close(self):
        """Clean up for temporary files"""
        import os
        if self.tempname and os.path.exists(self.tempname):
            os.remove(self.tempname)
            self.tempname = None

    def __str__(self):
        return ("'%s: shape %s, resolution %sm'"%
                (self.name, 
                 self.shape, 
                 self.resolution))   

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

    def __del__(self):
        """Cleanup"""
        self.close()


# ------------------------------------------------------------------------    
class MerisLevel2(mpop.channel.GenericChannel):
    """ENVISAT MERIS data struct"""
    def __init__(self, filename, resolution = None):
        mpop.channel.GenericChannel.__init__(self)
        self.filled = False
        self.resolution = resolution

        self.info = {}

        self.shape = None
        self.satid = ""
        self.orbit = None

        self.data = None

        self.starttime = None
        self.endtime = None

        self.bands = {}
        self.filename = filename
        self.tempname = None
        self.tiepoints = None
        self._gdal_dataset = None

        self.is_compressed = False
        self.compression = None # Supports only gzip compression

        if self.filename and is_file_gzipped(self.filename):
            self.is_compressed = True
            self.compression = "gzip"

        self.gads = None
        

    def read_header(self):
        """Read the ascii header without using gdal.
        Supposed to be faster, especially if the file is compressed"""
        import gzip

        # For calibration:
        self.gads = GADS_ScalingFactorsAndOFfsets(self.filename)
        self.gads.get_products_scaling_factors()

        try:
            if self.is_compressed and self.compression == 'gzip':
                infile = gzip.open(self.filename, 'rb')
            else:
                infile = open(self.filename, 'rb')

            prefix = 'MPH_'
            for line in infile:
                if 'SPH_' in line:
                    prefix = 'SPH_'
                if 'DS_NAME' in line:
                    break
                if '=' in line:
                    items = line.split('=')
                    key = prefix + items[0].strip()
                    if 'LAT' in line or 'LONG' in line:
                        val = items[1].strip()
                    else:
                        try:
                            val = items[1].strip(('<samples>' + 
                                                  '<%>' + 
                                                  '<bytes>\n'))
                        except ValueError:
                            val = items[1].strip()
                    #print key, val
                    if 'LAT' in line or 'LONG' in line:
                        # Fix the lon and lat values properly:
                        val = get_lonlat_value(val)
                    if key not in self.info:
                        self.info[key] = val
        except Exception, exc:
            errmsg = `exc.__str__()`
            LOG.error("Failed extracting ascii header from file")
            LOG.error("Error = %s" % errmsg)
            raise exc

        if not self.orbit:
            self.orbit = int(self.info['MPH_ABS_ORBIT'])

    def gdal_read(self):
        """Read the MERIS data: Info and Band data, but no geolocation"""
        import tempfile
        # Check if gzip compressed first, and uncompress if needed:
        if self.is_compressed:
            self.tempname = tempfile.mktemp()
            os.system("zcat %s > %s" % (self.filename, self.tempname))
            filename = self.tempname
        else:
            filename = self.filename
        
        # Open and read MERIS N1 file. Return the gdal dataset instance
        self._gdal_dataset = gdal.Open(filename, GA_ReadOnly)

        # Add the metdata dict to the info attribute:
        info = self._gdal_dataset.GetMetadata_Dict()
        for key in info:
            self.info[key] = info[key]        

        if not self.orbit:
            self.orbit = int(self.info['MPH_ABS_ORBIT'])

    def get_channeldata(self, **kwargs):
        """Get the 15 MERIS channels"""
        import numpy as np
        if not self._gdal_dataset :
            self.gdal_read()
 
        if 'bands' in kwargs:
            bands = kwargs['bands']
        else:
            bands = range(1,16)

        # Get the MERIS bands from the gdal dataset instance:
        for bnum in bands:
            band = self._gdal_dataset.GetRasterBand(bnum)
            arr = band.ReadAsArray()
            self.bands['band-%d' % bnum] = arr
            LOG.info("Max,Min = %d,%d" % (np.max(arr.ravel()), 
                                          np.min(arr.ravel())))

        if not self.shape:
            self.shape = arr.shape

        self.filled = True

    def get_tiepoints(self):
        """Read MERIS file and extract the ADS tiepoints"""
        if self.is_compressed and self.compression == 'gzip':
            self.tiepoints = get_tiepoints(self.filename)
        else:
            self.tiepoints = get_tiepoints(self.filename, False)

    def project(self, coverage):
        """Remaps the MERIS level2 data to cartographic
        map-projection on a user defined area.
        """
        print "Projecting product %s..." % (self.name)
        LOG.info("Projecting product %s..."%(self.name))
        retv = MerisLevel2(None) #self.name)        
        retv.data = coverage.project_array(self.data)
        retv.area = coverage.out_area
        retv.shape = retv.data.shape
        retv.resolution = self.resolution
        retv.orbit = self.orbit
        retv.satid = self.satid
        retv.info = self.info
        retv.filled = True
        valid_min = retv.data.min()
        valid_max = retv.data.max()
        retv.info['valid_range'] = np.array([valid_min, valid_max])
        retv.info['var_data'] = retv.data

        return retv

    def close(self):
        """Clean up for temporary files"""
        import os
        if self.tempname and os.path.exists(self.tempname):
            os.remove(self.tempname)
            self.tempname = None

    def __str__(self):
        idx = 0
        keys_vals = []
        for key in self.info:
            if idx > 5: 
                break
            keys_vals.append(str(key)+':'+str(self.info[key]))
            idx = idx + 1

        retv = "info =\n"+ "\n".join(keys_vals) + "\n..."
        
        if self.bands:
            retv = retv + "\nBands loaded: \n" + \
                "\n".join(
                [str(bname) +  ': ' + self.bands[bname].shape for bname in self.bands]
                )
        else:
            retv = retv + "\nNo bands loaded!"

        return retv

    #def __str__(self):
    #    return ("'%s: shape %s, resolution %sm'"%
    #            (self.name, 
    #             self.shape, 
    #             self.resolution))   

    def is_loaded(self):
        """Tells if the channel contains loaded data.
        """
        return self.filled

    def __del__(self):
        """Cleanup"""
        self.close()

# -----------------------------------------------------------
class GADS_ScalingFactorsAndOFfsets(object):
    """Global Annotation Data Set - Scaling Factors and General Info
    """
    def __init__(self, filename):
        self.scale = {}
        self.offset = {}

        self.gain_setting = None
        self.sampling_rate = None
        self.sun_spectral_flux = None # For band 1-15
        self.scale_rect_nir_refl = None
        self.offset_rect_nir_refl = None
        self.scale_rect_nr_refl = None
        self.offset_rect_nr_refl = None

        self.filename = filename
        print "File = ",filename
        self.file_is_gzipped = is_file_gzipped(filename)

    def get_products_scaling_factors(self):
        """Get the GADS scaling factors for the MERIS level-2 products"""
        import numpy as np
        import gzip

        if self.file_is_gzipped:
            infile = gzip.open(self.filename)
        else:
            infile = open(self.filename)
        for line in infile:
            if 'DS_NAME="Scaling Factor GADS' in line:
                break

        for line in infile:
            if 'DS_OFFSET=' in line:
                ds_offset = int(line.split('=')[1].split('<')[0])
                break

        for line in infile:
            if 'DS_SIZE=' in line:
                ds_size = int(line.split('=')[1].split('<')[0])
                break

        for line in infile:
            if 'NUM_DSR=' in line:
                num_dsr = int(line.split('=')[1].split('<')[0])
                break

        for line in infile:
            if 'DSR_SIZE=' in line:
                dsr_size = int(line.split('=')[1].split('<')[0])
                break

        infile.close()

        LOG.info("DS_OFFSET = %d" % (ds_offset))
        LOG.info("DS_SIZE = %d" % (ds_size))
        LOG.info("NUM_DSR = %d" % (num_dsr))
        LOG.info("DSR_SIZE = %d" % (dsr_size))

        if self.file_is_gzipped:
            infile = gzip.open(self.filename, 'r')
        else:
            infile = open(self.filename, 'r')

        dummy = infile.read(ds_offset)

        data = infile.read(ds_size)
        infile.close()

        offs = 0
        # Scaling factors:
        self.scale['altitude'] = funpack(data, offs)
        offs = offs + 4
        self.scale['roughness'] = funpack(data, offs)
        offs = offs + 4
        self.scale['zonal wind'] = funpack(data, offs)
        offs = offs + 4
        self.scale['meridional wind'] = funpack(data, offs)
        offs = offs + 4
        self.scale['atmospheric pressure'] = funpack(data, offs)
        offs = offs + 4
        self.scale['ozone'] = funpack(data, offs)
        offs = offs + 4
        self.scale['relative humidity'] = funpack(data, offs)
        offs = offs + 4
        self.scale['reflectances'] = []
        for i in range(13):
            self.scale['reflectances'].append(funpack(data, offs))
            offs = offs + 4

        self.scale['Algal pigment index'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Yellow substance'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Total Suspended sediment'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Aerosol epsilon'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Aerosol optical thickness'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Cloud optical thickness'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Surface pressure'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Water vapour'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Photosynthetically active radiation'] = funpack(data, offs)
        offs = offs + 4
        self.scale['TOA Vegetation index'] = funpack(data, offs)
        offs = offs + 4
        self.scale['BOA Vegetation index'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Cloud Albedo'] = funpack(data, offs)
        offs = offs + 4
        self.scale['Cloud Top Pressure'] = funpack(data, offs)
        offs = offs + 4

        # Offsets:
        self.offset['altitude'] = funpack(data, offs)
        offs = offs + 4
        self.offset['roughness'] = funpack(data, offs)
        offs = offs + 4
        self.offset['zonal wind'] = funpack(data, offs)
        offs = offs + 4
        self.offset['meridional wind'] = funpack(data, offs)
        offs = offs + 4
        self.offset['atmospheric pressure'] = funpack(data, offs)
        offs = offs + 4
        self.offset['ozone'] = funpack(data, offs)
        offs = offs + 4
        self.offset['relative humidity'] = funpack(data, offs)
        offs = offs + 4
        self.offset['reflectances'] = []
        for i in range(13):
            self.offset['reflectances'].append(funpack(data, offs))
            offs = offs + 4

        self.offset['Algal pigment index'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Yellow substance'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Total Suspended sediment'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Aerosol epsilon'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Aerosol optical thickness'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Cloud optical thickness'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Surface pressure'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Water vapour'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Photosynthetically active radiation'] = funpack(data, offs)
        offs = offs + 4
        self.offset['TOA Vegetation index'] = funpack(data, offs)
        offs = offs + 4
        self.offset['BOA Vegetation index'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Cloud Albedo'] = funpack(data, offs)
        offs = offs + 4
        self.offset['Cloud Top Pressure'] = funpack(data, offs)
        offs = offs + 4

        return

# -----------------------------------------------------------
def funpack(arr, offset):
    """unpack float"""
    import struct
    return struct.unpack("!f", arr[offset : offset + 4])[0]
    

class TiepointsADSR(object):
    """Data structure for the (tiepoint locations and corresponding aux data) 
    ADS Records. ADS = Annotation Data Set
    """
    def __init__(self):
        self.start_time = 0
        self.attachment_flag = ""
        self.latitudes = []
        self.longitudes = []
        self.dem = []
        self.drm_roughness = []
        self.dem_lat_corr = []
        self.dem_lon_corr = []
        self.sunz = []
        self.sun_azimuth = []
        self.satz = []
        self.sat_azimuth = []
        # ---- Below aux parameters probably not needed:
        self.zonal_winds = None
        self.meridional_winds = None
        self.sea_level_pressure = None
        self.total_ozone = None
        self.relative_humidity = None

class GeoTiePoints(object):
    """Geographical Tiepoint Container"""
    def __init__(self, lons, lats, tiepoint_grid):
        self.longitudes = lons
        self.latitudes = lats
        self.xdim = tiepoint_grid[0]
        self.ydim = tiepoint_grid[1]
        
    def interpolate(self):
        """Interpolate the tiepoints to the entire grid:
        From a set of evenly spaced tiepoints with lon,lat and the dimensions 
        of the grid, fill out the entire grid with lon,lats using 2d spline interpolation.
        """
        from pyresample import geometry
        from scipy import mgrid, ndimage
        import numpy as np

        geom = geometry.BaseDefinition(self.longitudes, self.latitudes)
        LOG.info( 
                "Transform tiepoint lons,lats to cartesian coordinates") 
        xyz = geom.get_cartesian_coords()

        rowdim = (self.longitudes.shape[0] - 1) * self.xdim + 1
        coldim = (self.longitudes.shape[1] - 1) * self.ydim + 1        

        x, y = np.mgrid[0:rowdim, 0:coldim]

        coords = np.array([x/float(self.xdim), y/float(self.ydim)])

        xnew = ndimage.map_coordinates(xyz[: , :, 0], coords, order=1, mode='reflect')
        ynew = ndimage.map_coordinates(xyz[: , :, 1], coords, order=1, mode='reflect')
        znew = ndimage.map_coordinates(xyz[: , :, 2], coords, order=1, mode='reflect')

        # Transform back to polar coordinates (lon,lat):
        LOG.info( "Transform back to lons,lats")
        wlons = get_lons_from_cartesian(xnew, ynew)
        wlats = get_lats_from_cartesian(znew)
    
        LOG.info( "Geo tiepoints interpolated to grid!")
        return wlons, wlats

# -------------------------------------------------------------------------
def get_lons_from_cartesian(x, y):
    """Get longitudes from cartesian coordinates"""
    import numpy as np
    return np.rad2deg(np.arccos(x/np.sqrt(x**2 + y**2)))*np.sign(y)
    
# -------------------------------------------------------------------------
def get_lats_from_cartesian(z):
    """Get latitudes from cartesian coordinates"""
    import numpy as np
    return 90 - np.rad2deg(np.arccos(z/EARTH_RADIUS))

# -----------------------------------------------------------
def tiepoints_interpolate(tielons, tielats, gridshape, method="linear"):
    """From a set of grid tiepoints with lon,lat and the dimensions 
    of the grid fill out the entire grid with lon,lats using 2d interpolation.
    """
    from pyresample import geometry
    from scipy.interpolate import griddata
    from numpy import (NaN, floor, meshgrid, mgrid, int32,
                       arccos, sign, rad2deg, sqrt)

    swath = geometry.BaseDefinition(tielons, tielats)

    rowdim = gridshape[0]
    coldim = gridshape[1]

    # Tiepoint grid is assumed to be 64x64 MERIS pixels!

    # Check dimensions:
    if ((tielons.shape[0]-1) * 64 + 1) != rowdim:
        LOG.error(("Non-consistent dimensions - rows: " + 
                   "num of tiepoints = %d " % tielons.shape[0] + 
                   "num of points in grid = %d" % rowdim))
        raise ValueError

    if ((tielons.shape[1]-1) * 64 + 1) != coldim:
        LOG.error(("Non-consistent dimensions - cols: " + 
                   "num of tiepoints = %d " % tielons.shape[1] + 
                   "num of points in grid = %d" % coldim))
        raise ValueError


    # Convert to cartesian coordinates        
    xyz = swath.get_cartesian_coords()
    col_indices_ties, row_indices_ties = range(0, rowdim, 64), range(0, rowdim, 64)
    col_indices, row_indices = meshgrid(col_indices_ties, 
                                        row_indices_ties)
    col_indices = col_indices.reshape(-1)
    row_indices = row_indices.reshape(-1)

    hcol_indices, hrow_indices = mgrid[0:col_indices_ties[-1] + 1,
                                       0:row_indices_ties[-1] + 1]

    hcol_indices = hcol_indices.reshape(-1)
    hrow_indices = hrow_indices.reshape(-1)

    # Interpolate x, y, and z    
    x_new = griddata((col_indices, row_indices),
                     xyz[:, :, 0].reshape(-1),
                     (hcol_indices, hrow_indices),
                     method=method)

    y_new = griddata((col_indices, row_indices),
                     xyz[:, :, 1].reshape(-1),
                     (hcol_indices, hrow_indices),
                     method=method)

    z_new = griddata((col_indices, row_indices),
                     xyz[:, :, 2].reshape(-1),
                     (hcol_indices, hrow_indices),
                     method=method)
    orig_col_size = col_indices_ties[-1] + 1
    orig_row_size = row_indices_ties[-1] + 1
    
    # Back to lon and lat
    lon = get_lons_from_cartesian(x_new, y_new)
    lon = lon.reshape(orig_col_size, orig_row_size).transpose()
    lat = get_lats_from_cartesian(z_new)
    lat = lat.reshape(orig_col_size, orig_row_size).transpose()
    
    LOG.info("Geo tiepoints interpolated to grid!")
    return lon, lat

# -----------------------------------------------------------
def is_inside_aoi(meris_info, boundary):
    """Check if the MERIS scene is inside the lonlat boundary box specified
    """
    # Now check if the ground track is inside the AOI:
    # Descending or ascending:
    descending = False
    if meris_info["SPH_FIRST_MID_LAT"] > meris_info["SPH_LAST_MID_LAT"]:
        descending = True

    if descending: 
        LOG.info("Satellite is in descending orbit")
        if meris_info["SPH_FIRST_MID_LONG"] < boundary["Northwest"][0] or \
                meris_info["SPH_FIRST_MID_LONG"] > boundary["Northeast"][0]:
            LOG.info("Outside")
        if meris_info["SPH_LAST_MID_LONG"] < boundary["Southwest"][0] or \
               meris_info["SPH_LAST_MID_LONG"] > boundary["Southeast"][0]:
            LOG.info("Outside")
            return False
    else:
        LOG.info("Satellite is in ascending orbit")
        if meris_info["SPH_FIRST_MID_LONG"] < boundary["Southwest"][0] or \
               meris_info["SPH_FIRST_MID_LONG"] > boundary["Southeast"][0]:
            LOG.info("Outside")
        if meris_info["SPH_LAST_MID_LONG"] < boundary["Northwest"][0] or \
               meris_info["SPH_LAST_MID_LONG"] > boundary["Northeast"][0]:
            LOG.info("Outside")
            return False

    # Inside area of interest:
    return True

# -----------------------------------------------------------
def get_lonlat_value(tmpstr):
    """Convert the lon,lat header entries to proper geolocation values
    (degrees N and degrees E)
    """
    import math
    
    ival = int(tmpstr.split('<')[0])
    deg_E_or_N = tmpstr.split('deg')[1][0]
    ipot = int(tmpstr.split('<')[1].strip('deg' + deg_E_or_N + '>').strip('10'))

    return ival * math.exp(ipot*math.log(10))


# -------------------------------------------------------------------------
def is_file_gzipped(filename):
    """Check if a file is gzip compressed
    Stupidly checking the extention only!"""
    import os
    bname = os.path.basename(filename)
    namel = bname.split('.')
    ext = namel[len(namel) - 1]
    if ext == 'gz':
        return True
    else:
        return False


# -------------------------------------------------------------------------
def get_product(filename, prodname, gzipped=True):
    """Get the MERIS level-2 products"""
    import struct
    import numpy as np
    import gzip

    if gzipped:
        infile = gzip.open(filename)
    else:
        infile = open(filename)
    for line in infile:
        #if 'DS_NAME="Norm. rho_surf - MDS(1)' in line:
        if prodname == 'Chl_1':
            if 'DS_NAME="Chl_1, TOAVI   - MDS(15)' in line:
                break
        elif prodname == 'YS':
            if 'DS_NAME="YS, SPM, Rect. Rho- MDS(16)' in line:
                break
        elif prodname == 'Chl_2':
            if 'DS_NAME="Chl_2, BOAVI   - MDS(17)' in line:
                break
        elif prodname == 'PAR':
            if 'DS_NAME="Press PAR Alb  - MDS(18)' in line:
                break
        elif prodname == 'Alpha':
            if 'DS_NAME="Alpha, OPT     - MDS(19)' in line:
                break
        elif prodname == 'Flags':
            if 'DS_NAME="Flags          - MDS(20)' in line:
                break

    for line in infile:
        if 'DS_OFFSET=' in line:
            ds_offset = int(line.split('=')[1].split('<')[0])
            break

    for line in infile:
        if 'DS_SIZE=' in line:
            ds_size = int(line.split('=')[1].split('<')[0])
            break

    for line in infile:
        if 'NUM_DSR=' in line:
            num_dsr = int(line.split('=')[1].split('<')[0])
            break

    for line in infile:
        if 'DSR_SIZE=' in line:
            dsr_size = int(line.split('=')[1].split('<')[0])
            break

    infile.close()

    LOG.info("DS_OFFSET = %d" % (ds_offset))
    LOG.info("DS_SIZE = %d" % (ds_size))
    LOG.info("NUM_DSR = %d" % (num_dsr))
    LOG.info("DSR_SIZE = %d" % (dsr_size))

    num_columns = 4481
    # Check dimensions:
    # FIXME!
    # ds_size / num_dsr = dsr_size

    if gzipped:
        infile = gzip.open(filename, 'r')
    else:
        infile = open(filename, 'r')

    dummy = infile.read(ds_offset)

    prod = infile.read(ds_size)
    infile.close()

    if prodname in ['Chl_1', 'Chl_2', 'PAR']:
        dtype = "!B" # Assume unsigned char!?
        dstep = 1
    elif prodname in ['YS', 'Alpha', 'Flags']:
        dtype = "!H" # Assume unsigned short!?
        dstep = 2
    else:
        raise IOError('Product name not supported! prodname = %s' % prodname)

    params = []
    for j in range(num_dsr):
        offs = j*dsr_size

        mjd_s = prod[offs : offs+12]
        flag = struct.unpack("!b", prod[offs + 12])[0]
        #print "Flag = ",flag
        offs = offs + 13

        for idx in range(offs, num_columns + offs, dstep):
            param = struct.unpack(dtype, prod[idx : idx + 1])[0]
            params.append(param)

    # Reshape the arrays
    shape = (num_dsr, num_columns)

    arr = np.reshape(params, shape)

    return arr
    
# -------------------------------------------------------------------------
def get_tiepoints(filename, gzipped=True):
    """Get the ADS tie points with geolocation data"""
    import struct
    import numpy as np
    import gzip

    if gzipped:
        infile = gzip.open(filename)
    else:
        infile = open(filename)
    for line in infile:
        if 'DS_NAME="Tie points ADS' in line:
            break

    for line in infile:
        if 'DS_OFFSET=' in line:
            ds_offset = int(line.split('=')[1].split('<')[0])
            break

    for line in infile:
        if 'DS_SIZE=' in line:
            ds_size = int(line.split('=')[1].split('<')[0])
            break

    for line in infile:
        if 'NUM_DSR=' in line:
            num_dsr = int(line.split('=')[1].split('<')[0])
            break

    for line in infile:
        if 'DSR_SIZE=' in line:
            dsr_size = int(line.split('=')[1].split('<')[0])
            break

    infile.close()

    LOG.info("DS_OFFSET = %d" % (ds_offset))
    LOG.info("DS_SIZE = %d" % (ds_size))
    LOG.info("NUM_DSR = %d" % (num_dsr))
    LOG.info("DSR_SIZE = %d" % (dsr_size))

    num_columns = 71
    # Check dimensions:
    # FIXME!
    # ds_size / num_dsr = dsr_size

    if gzipped:
        infile = gzip.open(filename, 'r')
    else:
        infile = open(filename, 'r')

    dummy = infile.read(ds_offset)
    ads_s = infile.read(ds_size)

    lats = []
    lons = []
    dems = []
    drms = []
    dem_lat_corr = []
    dem_lon_corr = []
    sun_zenith = []
    sun_azimuth = []
    sat_zenith = []
    sat_azimuth = []
    zwinds = []
    mwinds = []
    sea_lvl_pressure = []
    total_ozone = []
    rel_humidity = []

    for j in range(num_dsr):
        offs = j*dsr_size
        mjd_s = ads_s[offs : offs+12]
        flag = struct.unpack("!B", ads_s[offs + 12])[0]
        #print "Flag: ", flag
    
        offs = offs + 13
        for idx in range(offs, 284 + offs, 4):
            lat = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            lats.append(lat / 1000000.0)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            lon = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            lons.append(lon / 1000000.0)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            dem = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            dems.append(dem)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            drm = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            drms.append(drm)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            dem = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            dem_lat_corr.append(dem / 1000000.0)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            dem = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            dem_lon_corr.append(dem / 1000000.0)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            sunz = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            sun_zenith.append(sunz / 1000000.0)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            suna = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            sun_azimuth.append(suna / 1000000.0)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            satz = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            sat_zenith.append(satz / 1000000.0)

        offs += 284
        for idx in range(offs, 284 + offs, 4):
            sata = struct.unpack("!i", ads_s[idx : idx + 4])[0]
            sat_azimuth.append(sata / 1000000.0)

        offs += 284
        for idx in range(offs, 142 + offs, 2):
            zwi = struct.unpack("!h", ads_s[idx : idx + 2])[0]
            zwinds.append(zwi)

        offs += 142
        for idx in range(offs, 142 + offs, 2):
            mwi = struct.unpack("!h", ads_s[idx : idx + 2])[0]
            mwinds.append(mwi)

        offs += 142
        for idx in range(offs, 142 + offs, 2):
            slp = struct.unpack("!H", ads_s[idx : idx + 2])[0]
            sea_lvl_pressure.append(slp / 10.0)

        offs += 142
        for idx in range(offs, 142 + offs, 2):
            tot_o3 = struct.unpack("!H", ads_s[idx : idx + 2])[0]
            total_ozone.append(tot_o3 / 100.0)

        offs += 142
        for idx in range(offs, 142 + offs, 2):
            relh = struct.unpack("!H", ads_s[idx : idx + 2])[0]
            rel_humidity.append(relh / 10.0)

    # Reshape the arrays
    shape = (num_dsr, num_columns)
    tpoints = TiepointsADSR()
    tpoints.latitudes = np.reshape(lats, shape)
    tpoints.longitudes = np.reshape(lons, shape)
    tpoints.dem = np.array(dems)
    tpoints.drm_roughness = np.array(drms)
    tpoints.dem_lat_corr = np.array(dem_lat_corr)
    tpoints.dem_lon_corr = np.array(dem_lon_corr)
    tpoints.sunz = np.array(sun_zenith)
    tpoints.satz = np.array(sat_zenith)
    tpoints.sun_azimuth = np.array(sun_azimuth)
    tpoints.sat_azimuth = np.array(sat_azimuth)

    tpoints.zonal_winds = np.array(zwinds)
    tpoints.meridional_winds = np.array(mwinds)
    tpoints.sea_level_pressure = np.array(sea_lvl_pressure)
    tpoints.total_ozone = np.array(total_ozone)
    tpoints.relative_humidity = np.array(rel_humidity)
    
    infile.close()

    return tpoints


def load(satscene, **kwargs):
    """Read data from file and load it into *satscene*.  Load data into the
    *channels*. *Channels* is a list or a tuple containing channels we will
    load data into. If None, all channels are loaded.
    """

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level3",
                                    raw = True):
        options[option] = value

    if "filename" not in options:
        raise IOError("No filename given, cannot load.")
    if "dir" not in options:
        raise IOError("No directory given, cannot load.")

    #MER_FRS_2PNPDE20101217_094256_000001993097_00295_45999_5414.N1.gz
    #filename = MER_FRS_2PNPDE%Y%m%d_%H%M*.N1.gz

    pathname = os.path.join(options["dir"], options['filename'])    
    print "Path = ", pathname
    LOG.debug("Looking for file %s" % satscene.time_slot.strftime(pathname))
    file_list = glob.glob(satscene.time_slot.strftime(pathname))
    print satscene.time_slot.strftime(pathname)

    if len(file_list) > 1:
        raise IOError("More than one MERIS file matching!")
    elif len(file_list) == 0:
        raise IOError("No MERIS file matching!")

    filename = file_list[0]

    if "algal2" in satscene.channels_to_load:
        prodname = "algal2"

        merisdata = MerisLevel2Product(filename, prodname)
        merisdata.read()

        merisdata.data = merisdata.products[prodname]
        merisdata.satid = satscene.satname.capitalize()
        merisdata.resolution = 300.0
        merisdata.shape = merisdata.data.shape

        # All this for the netCDF writer:
        merisdata.info['var_name'] = prodname
        merisdata.info['var_data'] = merisdata.data
        resolution_str = str(int(merisdata.resolution))+'m'
        merisdata.info['var_dim_names'] = ('y'+resolution_str,
                                           'x'+resolution_str)

        nodata = 0 # ??? FIXME!
        mask = np.equal(merisdata.data, nodata)
        merisdata.data = np.ma.masked_where(mask, merisdata.data)
        satscene.channels.append(merisdata)

        satscene[prodname].info['units'] = ''


    try:
        # Empty the products dict to save memory!
        #del merisdata.products ???Need to do this?
        merisdata.products = {}
    except UnboundLocalError:
        pass

    # Check if there are any bands to load:
    channels_to_load = False
    for bandname in CHANNELS:
        if bandname in satscene.channels_to_load:
            channels_to_load = True
            break

    if channels_to_load:
        merisdata = MerisLevel2(filename)
        merisdata.read_header()
        merisdata.gdal_read()
        merisdata.get_tiepoints()

        for bandname in CHANNELS:
            if bandname in satscene.channels_to_load:

                band_idx = int(bandname.strip('band-'))
                merisdata.get_channeldata(bands=[band_idx])
                band = merisdata.bands[bandname]

                merisdata.satid = satscene.satname.capitalize()
                merisdata.resolution = 300.0
                merisdata.shape = band.shape

                # All this for the netCDF writer:
                merisdata.info['var_name'] = bandname
                merisdata.info['var_data'] = merisdata.data
                resolution_str = str(int(merisdata.resolution))+'m'
                merisdata.info['var_dim_names'] = ('y'+resolution_str,
                                                   'x'+resolution_str)

                nodata = 0 # ??? FIXME!
                mask = np.equal(band, nodata)
                counts = np.ma.masked_where(mask, band)
                # Should not be done for bands 11 and 15!:
                if band_idx not in [11, 15]:
                    scale = merisdata.gads.scale['reflectances'][band_idx - 1]
                    offset = merisdata.gads.offset['reflectances'][band_idx - 1]
                else:
                    scale = 1.0 # ???
                    offset = 0.0 # ???

                LOG.info("Band %d: Scale = %f, Offset = %f" % (
                        band_idx, scale, offset))
                satscene[bandname] = (counts * scale + offset)

                # Empty the bands dict to save memory!
                #del merisdata.bands ???Need to do this?
                merisdata.bands = {}
                
                satscene[bandname].info['units'] = '%'

    satscene.orbit = merisdata.orbit

    lat, lon = get_lat_lon(satscene, None)
    from pyresample import geometry
    satscene.area = geometry.SwathDefinition(lons=lon, lats=lat)

    LOG.info("Variant: " + satscene.variant)
    LOG.info("Loading meris data done.")


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
    """Read tiepoint lats and lons and interpolate.
    """
    del resolution    

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level3",
                                    raw = True):
        options[option] = value

    if "filename" not in options:
        raise IOError("No filename given, cannot load.")
    if "dir" not in options:
        raise IOError("No directory given, cannot load.")

    pathname = os.path.join(options["dir"], options['filename'])    
    LOG.debug("Looking for file %s" % satscene.time_slot.strftime(pathname))
    file_list = glob.glob(satscene.time_slot.strftime(pathname))

    if len(file_list) > 1:
        raise IOError("More than one MERIS file matching!")
    elif len(file_list) == 0:
        raise IOError("No MERIS file matching!")

    filename = file_list[0]

    merisdata = MerisLevel2(filename)
    merisdata.gdal_read()
    #merisdata.read_header()
    merisdata.get_tiepoints()

    if not merisdata.shape:
        print "set the shape..."
        shape = (int(merisdata.info['SPH_LINES_PER_TIE_PT']) * 
                 (merisdata.tiepoints.latitudes.shape[0] - 1) + 1, 
                 int(merisdata.info['SPH_LINE_LENGTH']))
    else:
        shape = merisdata.shape

    print "shape = ",shape

    # Interpolate and get longitudes and latitudes:
    #lons, lats = tiepoints_interpolate(merisdata.tiepoints.longitudes,
    #                                   merisdata.tiepoints.latitudes, 
    #                                   shape)

    # Interpolate and get longitudes and latitudes:
    tie_grid = (64, 64)
    geotie = GeoTiePoints(merisdata.tiepoints.longitudes,
                          merisdata.tiepoints.latitudes, 
                          tie_grid)
    lons, lats = geotie.interpolate()

    return lats, lons
