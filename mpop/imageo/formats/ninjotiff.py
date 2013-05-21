# -*- coding: utf-8 -*-
"""
ninjotiff.py

Created on Mon Apr 15 13:41:55 2013

Big parts of the tiff writer are (PFE) from 
https://github.com/davidh-ssec/polar2grid by David Hoese

License:
Copyright (C) 2013 Space Science and Engineering Center (SSEC),
 University of Wisconsin-Madison.
 Lars Ørum Rasmussen, DMI.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

Original scripts and automation included as part of this package are
distributed under the GNU GENERAL PUBLIC LICENSE agreement version 3.
Binary executable files included as part of this software package are
copyrighted and licensed by their respective organizations, and
distributed consistent with their licensing terms.
"""

import os
import copy
import logging
import calendar
from datetime import datetime
import numpy as np

import mpop.imageo.formats.libtiff as libtiff
from mpop.imageo.formats.libtiff import TIFF, TIFFFieldInfo, TIFFDataType, FIELD_CUSTOM

log = logging.getLogger(__name__)

#------------------------------------------------------------------------------- 
#
# Ninjo tiff tags from DWD
#
#------------------------------------------------------------------------------- 
# Geotiff tags
GTF_ModelPixelScale        = 33550
GTF_ModelTiepoint          = 33922 

NTD_Magic                  = 40000
NTD_SatelliteNameID        = 40001
NTD_DateID                 = 40002
NTD_CreationDateID         = 40003
NTD_ChannelID              = 40004
NTD_HeaderVersion          = 40005
NTD_FileName               = 40006
NTD_DataType               = 40007
NTD_SatelliteNumber        = 40008
NTD_ColorDepth             = 40009
NTD_DataSource             = 40010
NTD_XMinimum               = 40011
NTD_XMaximum               = 40012
NTD_YMinimum               = 40013
NTD_YMaximum               = 40014
NTD_Projection             = 40015
NTD_MeridianWest           = 40016
NTD_MeridianEast           = 40017
NTD_EarthRadiusLarge       = 40018
NTD_EarthRadiusSmall       = 40019
NTD_GeodeticDate           = 40020
NTD_ReferenceLatitude1     = 40021
NTD_ReferenceLatitude2     = 40022
NTD_CentralMeridian        = 40023
NTD_PhysicValue            = 40024
NTD_PhysicUnit             = 40025
NTD_MinGrayValue           = 40026
NTD_MaxGrayValue           = 40027
NTD_Gradient               = 40028
NTD_AxisIntercept          = 40029
NTD_ColorTable             = 40030
NTD_Description            = 40031
NTD_OverflightDirection    = 40032
NTD_GeoLatitude            = 40033
NTD_GeoLongitude           = 40034
NTD_Altitude               = 40035
NTD_AOSAsimuth             = 40036
NTD_LOSAsimuth             = 40037
NTD_MaxElevation           = 40038
NTD_OverflightTime         = 40039
NTD_IsBlackLineCorrection  = 40040
NTD_IsAtmosphereCorrected  = 40041
NTD_IsCalibrated           = 40042
NTD_IsNormalized           = 40043
NTD_OriginalHeader         = 40044
NTD_IsValueTableAvailable  = 40045
NTD_ValueTableStringField  = 40046
NTD_ValueTableFloatField   = 40047
NTD_TransparentPixel       = 50000

#
# model_pixel_scale_tag_count ? ... 
# Sometimes DWD product defines an array of length 2 (instead of 3 (as in geotiff)).
#
MODEL_PIXEL_SCALE_COUNT = int(os.environ.get("GEOTIFF_MODEL_PIXEL_SCALE_COUNT", 3))

ninjo_tags_dict = {
    # Geotiff tags
    GTF_ModelPixelScale:
        TIFFFieldInfo(GTF_ModelPixelScale, MODEL_PIXEL_SCALE_COUNT,
                      MODEL_PIXEL_SCALE_COUNT, TIFFDataType.TIFF_DOUBLE,
                      FIELD_CUSTOM, True, False, "ModelPixelScale" ),
    GTF_ModelTiepoint:
        TIFFFieldInfo(GTF_ModelTiepoint, 6, 6, TIFFDataType.TIFF_DOUBLE,
                      FIELD_CUSTOM, True, False, "ModelTiePoint" ),

    # DWD tags
    NTD_Magic: 
        TIFFFieldInfo(NTD_Magic, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "Magic" ), 
    NTD_SatelliteNameID:
        TIFFFieldInfo(NTD_SatelliteNameID, 1, 1, TIFFDataType.TIFF_LONG,
                      FIELD_CUSTOM, True, False, "SatelliteNameID" ),
    NTD_DateID:
        TIFFFieldInfo(NTD_DateID, 1, 1, TIFFDataType.TIFF_LONG,
                      FIELD_CUSTOM, True, False, "DateID" ),
    NTD_CreationDateID:
        TIFFFieldInfo(NTD_CreationDateID, 1, 1, TIFFDataType.TIFF_LONG,
                      FIELD_CUSTOM, True, False, "CreationDateID" ),
    NTD_ChannelID:
        TIFFFieldInfo(NTD_ChannelID, 1, 1, TIFFDataType.TIFF_LONG,
                      FIELD_CUSTOM, True, False, "ChannelID" ),
    NTD_HeaderVersion:
        TIFFFieldInfo(NTD_HeaderVersion, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "HeaderVersion" ),
    NTD_FileName:
        TIFFFieldInfo(NTD_FileName, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "FileName" ),
    NTD_DataType:
        TIFFFieldInfo(NTD_DataType, 5, 5, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "DataType" ), # 4 chars + NUL character
    NTD_SatelliteNumber:
        TIFFFieldInfo(NTD_SatelliteNumber, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "SatelliteNumber" ),
    NTD_ColorDepth:
        TIFFFieldInfo(NTD_ColorDepth, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "ColorDepth" ),
    NTD_DataSource:
        TIFFFieldInfo(NTD_DataSource, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "DataSource" ),
    NTD_XMinimum:
        TIFFFieldInfo(NTD_XMinimum, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "XMinimum" ),
    NTD_XMaximum:
        TIFFFieldInfo(NTD_XMaximum, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "XMaximum" ),
    NTD_YMinimum:
        TIFFFieldInfo(NTD_YMinimum, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "YMinimum" ),
    NTD_YMaximum:
        TIFFFieldInfo(NTD_YMaximum, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "YMaximum" ),
    NTD_Projection:
        TIFFFieldInfo(NTD_Projection, 5, 5, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "Projection" ), # 4 chars + NUL character
    NTD_MeridianWest:
        TIFFFieldInfo(NTD_MeridianWest, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "MeridianWest" ),
    NTD_MeridianEast:
        TIFFFieldInfo(NTD_MeridianEast, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "MeridianEast" ),
    NTD_EarthRadiusLarge:
        TIFFFieldInfo(NTD_EarthRadiusLarge, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "EarthRadiusLarge" ),
    NTD_EarthRadiusSmall:
        TIFFFieldInfo(NTD_EarthRadiusSmall, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "EarthRadiusSmall" ),
    NTD_GeodeticDate:
        TIFFFieldInfo(NTD_GeodeticDate, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "GeodeticDate" ),  # Max 20
    NTD_ReferenceLatitude1:
        TIFFFieldInfo(NTD_ReferenceLatitude1, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "ReferenceLatitude1" ),
    NTD_ReferenceLatitude2:
        TIFFFieldInfo(NTD_ReferenceLatitude2, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "ReferenceLatitude2" ),
    NTD_CentralMeridian:
        TIFFFieldInfo(NTD_CentralMeridian, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "CentralMeridian" ),
    NTD_PhysicValue:
        TIFFFieldInfo(NTD_PhysicValue, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "PhysicValue" ), # Max 10
    NTD_PhysicUnit:
        TIFFFieldInfo(NTD_PhysicUnit, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "PhysicUnit" ), # Max 10
    NTD_MinGrayValue:
        TIFFFieldInfo(NTD_MinGrayValue, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "MinGrayValue" ),
    NTD_MaxGrayValue:
        TIFFFieldInfo(NTD_MaxGrayValue, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "MaxGrayValue" ),
    NTD_Gradient:
        TIFFFieldInfo(NTD_Gradient, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "Gradient" ),
    NTD_AxisIntercept:
        TIFFFieldInfo(NTD_AxisIntercept, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "AxisIntercept" ),
    NTD_ColorTable:
        TIFFFieldInfo(NTD_ColorTable, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "ColorTable" ),
    NTD_Description:
        TIFFFieldInfo(NTD_Description, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "Description" ),
    NTD_OverflightDirection:
        TIFFFieldInfo(NTD_OverflightDirection, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "OverflightDirection" ),
    NTD_GeoLatitude:
        TIFFFieldInfo(NTD_GeoLatitude, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "GeoLatitude" ),
    NTD_GeoLongitude:
        TIFFFieldInfo(NTD_GeoLongitude, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "GeoLongitude" ),
    NTD_Altitude:
        TIFFFieldInfo(NTD_Altitude, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "Altitude" ),
    NTD_AOSAsimuth:
        TIFFFieldInfo(NTD_AOSAsimuth, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "AOSAsimuth" ),
    NTD_LOSAsimuth:
        TIFFFieldInfo(NTD_LOSAsimuth, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "LOSAsimuth" ),
    NTD_MaxElevation:
        TIFFFieldInfo(NTD_MaxElevation, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "MaxElevation" ),
    NTD_OverflightTime:
        TIFFFieldInfo(NTD_OverflightTime, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "OverflightTime" ),
    NTD_IsBlackLineCorrection:
        TIFFFieldInfo(NTD_IsBlackLineCorrection, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "IsBlackLineCorrection" ),
    NTD_IsAtmosphereCorrected:
        TIFFFieldInfo(NTD_IsAtmosphereCorrected, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "IsAtmosphereCorrected" ),
    NTD_IsCalibrated:
        TIFFFieldInfo(NTD_IsCalibrated, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "IsCalibrated" ),
    NTD_IsNormalized:
        TIFFFieldInfo(NTD_IsNormalized, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "IsNormalized" ),
    NTD_OriginalHeader:
        TIFFFieldInfo(NTD_OriginalHeader, -1, -1, TIFFDataType.TIFF_ASCII,
                          FIELD_CUSTOM, True, False, "OriginalHeader" ),
    NTD_IsValueTableAvailable:
        TIFFFieldInfo(NTD_IsValueTableAvailable, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "IsValueTableAvailable" ),
    NTD_ValueTableStringField:
        TIFFFieldInfo(NTD_ValueTableStringField, -1, -1, TIFFDataType.TIFF_ASCII,
                      FIELD_CUSTOM, True, False, "ValueTableStringField" ),
    NTD_ValueTableFloatField:
        TIFFFieldInfo(NTD_ValueTableFloatField, 1, 1, TIFFDataType.TIFF_FLOAT,
                      FIELD_CUSTOM, True, False, "ValueTableFloatField" ),

    NTD_TransparentPixel:
        TIFFFieldInfo(NTD_TransparentPixel, 1, 1, TIFFDataType.TIFF_SLONG,
                      FIELD_CUSTOM, True, False, "TransparentPixel" ),        
    }

# Add Ninjo tags to the libtiff library
_ninjo_tags_extender = libtiff.add_tags(ninjo_tags_dict.values())
ninjo_tags = sorted(ninjo_tags_dict.keys())

class _TIFF(object):
    """ Just an context wrapper.
    """
    def __init__(self, filename, mode='r'):
        """Open a tiff file.

        see: libtiff.TIFF.open()
        """
        self.tiff = TIFF.open(filename, mode)
        self.tiff.ninjo_tags_dict = ninjo_tags_dict
        self.tiff.ninjo_tags = ninjo_tags

    def __enter__(self):
        return self.tiff

    def __exit__(self, type_, value, traceback):
        self.tiff.close()

def _read_directories(self):
    """Iterate over directories in a tiff file.

    :Parameters:
    `self` : libtiff.TIFF instance.
    """
    yield self
    while not self.LastDirectory():
        self.ReadDirectory()
        yield self
    self.SetDirectory(0)

#------------------------------------------------------------------------------- 
#
# Read Ninjo products config file.
#
#-------------------------------------------------------------------------------
_g_products = {} # lazy read
def get_product_config(product_name=None, filename=None):
    def _readit(filename=filename):
        from ConfigParser import ConfigParser        

        def _find_a_config_file():
            name_ = 'ninjotiff_products.cfg'
            home_ = os.path.dirname(os.path.abspath(__file__))
            penv_ = os.environ.get('PPP_CONFIG_DIR', '')
            for fname_ in [os.path.join(x, name_) for x in (home_, penv_)]:
                if os.path.isfile(fname_):
                    return fname_
            raise ValueError("Could not find a Ninjo tiff config file")        

        def _eval(val):
            try:
                return eval(val)
            except:
                return str(val)

        filename = filename or _find_a_config_file()
        log.info("Using Ninjo config file: '%s'" % filename)

        cfg = ConfigParser()
        cfg.read(filename)
        products = {}
        for sec in cfg.sections():
            prd = {}
            for key, val in cfg.items(sec):
                prd[key] = _eval(val)
            products[sec] = prd
        return products

    global _g_products
    if not _g_products:
        _g_products = _readit()
    if product_name:
        return _g_products[product_name]
    return _g_products

#------------------------------------------------------------------------------- 
#
# Read tiff file.
#
#------------------------------------------------------------------------------- 
def info(filename):
    with _TIFF(filename) as self:
        for d in _read_directories(self):
            l = []
            for item in d.info().split('\n'):
                k, v = item.split(':', 1)
                if (k.endswith('OffSets') or 
                    k.endswith('ByteCounts') or
                    k == 'FileName' or
                    k == 'DataType'):
                    continue
                l.append(item)
            for tag in d.ninjo_tags:
                value = d.GetField(tag)
                name = d.ninjo_tags_dict[tag].field_name
                if value is None:
                    continue
                l.append('%s: %s' % (name, str(value)))
            yield '\n'.join(l)

def image_data(filename):
    with _TIFF(filename) as self:
        for d in _read_directories(self):
            yield d.read_tiles()    

def colortable(filename):
    with _TIFF(filename) as self:
        return self.GetField('ColorMap')


#------------------------------------------------------------------------------- 
#
# Write Ninjo Products
#
#------------------------------------------------------------------------------- 
def save(geo_image, filename, **kwargs):
    # MPOP compatible writer
    channels, fill_value = geo_image._finalize()
    area_def = geo_image.area
    area_name = area_def.proj_id.split('_')[1]
    time_slot = geo_image.time_slot
    ninjo_product_name = kwargs['ninjo_product_name']
    if geo_image.mode == 'RGB':
        if not fill_value:
            fill_value = (0, 0, 0)
        data = np.dstack((channels[0].filled(fill_value[0]),
                          channels[1].filled(fill_value[1]),
                          channels[2].filled(fill_value[2])))
        fill_value = fill_value[0]
    elif geo_image.mode == 'L':
        if not fill_value:
            fill_value = 0
        data = channels[0].filled(fill_value)
    else:
        raise ValueError("Don't known how til handle image mode '%s'" %
                         geo_image.mode)
        
    write(data, filename, 
          area_def=area_def,
          time_slot=time_slot,
          ninjo_product_name=ninjo_product_name,
          fill_value=fill_value)

def write(image_data, output_fn, **kwargs):
    # Generic writer
    area = kwargs.pop('area_def')
    time_stamp = kwargs.pop('time_slot')
    product_name = kwargs.pop('ninjo_product_name')
    fill_value = kwargs.get('fill_value', -1)
    cmap = kwargs.get('cmap', None)

    upper_left = area.get_lonlat(0, 0)
    lower_right = area.get_lonlat(area.shape[0], area.shape[1])
    scale = abs(lower_right[0] - upper_left[0])/area.shape[1],\
        abs(upper_left[1] - lower_right[1])/area.shape[0]

    if len(image_data.shape) == 3:
        shape = (area.y_size, area.x_size, 3)
        write_rgb = True
        log.info("Will generate RGB product '%s'" % product_name)
    else:
        shape = (area.y_size, area.x_size)
        write_rgb = False
        log.info("Will generate product '%s'" % product_name)

    if image_data.shape != shape:
        raise ValueError, "Raster shape %s does not correspond to expected shape %s" % (
            str(image_data.shape), str(shape))

    options = get_product_config(product_name)
    options['meridian_west'] = upper_left[0]
    options['meridian_east'] = lower_right[0]
    options['pixel_xres'] = scale[0]
    options['pixel_yres'] = scale[1]
    options['origin_lon'] = upper_left[0]
    options['origin_lat'] = upper_left[1]
    options['projection'] = area.proj_id.split('_')[-1]
    options['image_dt'] = time_stamp
    options['min_gray_val'] = image_data.min()
    options['max_gray_val'] = image_data.max()

    options['transparent_pix'] = fill_value
    options['cmap'] = cmap

    _write(image_data, output_fn, write_rgb=write_rgb, **options)
    
#------------------------------------------------------------------------------- 
#
# Write tiff file.
#
#------------------------------------------------------------------------------- 
def _write(image_data, output_fn, write_rgb=False, **kwargs):
    """Proudly Found Elsewhere (PFE) https://github.com/davidh-ssec/polar2grid
    by David Hoese.

    Create a NinJo compatible TIFF file with the tags used
    by the DWD's version of NinJo.  Also stores the image as tiles on disk
    and creates a multi-resolution/pyramid/overview set of images
    (deresolution: 2,4,8,16).

    :Parameters:
        image_data : 2D numpy array
            Satellite image data to be put into the NinJo compatible tiff
        filename : str
            The name of the TIFF file to be created

    :Keywords:
        cmap : tuple/list of 3 lists of uint16's
            Individual RGB arrays describing the color value for the
            corresponding data value.  For example, image data with a data
            type of unsigned 8-bit integers have 256 possible values (0-255).
            So each list in cmap will have 256 values ranging from 0 to
            65535 (2**16 - 1). (default linear B&W colormap)
        sat_id : int
            DWD NinJo Satellite ID number
        chan_id : int
            DWD NinJo Satellite Channel ID number
        data_source : str
            String describing where the data came from (SSEC, EUMCAST)
        tile_width : int
            Width of tiles on disk (default 512)
        tile_length : int
            Length of tiles on disk (default 512)
        data_cat : str
            NinJo specific data category
                - data_cat[0] = P (polar) or G (geostat)
                - data_cat[1] = O (original) or P (product)
                - data_cat[2:4] = RN or RB or RA or RN or AN (Raster, Bufr, ASCII, NIL)

            Example: 'PORN' or 'GORN' or 'GPRN' or 'PPRN'
        pixel_xres : float
            Nadir view pixel resolution in degrees longitude
        pixel_yres : float
            Nadir view pixel resolution in degrees latitude
        origin_lat : float
            Top left corner latitude
        origin_lon : float
            Top left corner longitude
        image_dt : datetime object
            Python datetime object describing the date and time of the image
            data provided in UTC
        projection : str
            NinJo compatible projection name (NPOL,PLAT,etc.)
        meridian_west : float
            Western image border (default 0.0)
        meridian_east : float
            Eastern image border (default 0.0)
        radius_a : float
            Large/equatorial radius of the earth (default <not written>)
        radius_b : float
            Small/polar radius of the earth (default <not written>)
        ref_lat1 : float
            Reference latitude 1 (default <not written>)
        ref_lat2 : float
            Reference latitude 2 (default <not written>)
        central_meridian : float
            Central Meridian (default <not written>)
        physic_value : str
            Physical value type. Examples:
                - Temperature = 'T'
                - Albedo = 'ALBEDO'
        physic_unit : str
            Physical value units. Examples:
                - 'CELSIUS'
                - '%'
        min_gray_val : int
            Minimum gray value (default 0)
        max_gray_val : int
            Maximum gray value (default 255)
        gradient : float
            Gradient/Slope
        axis_intercept : float
            Axis Intercept
        altitude : float
            Altitude of the data provided (default 0.0)
        is_atmo_corrected : bool
            Is the data atmosphere corrected? (True/1 for yes) (default False/0)
        is_calibrated : bool
            Is the data calibrated? (True/1 for yes) (default False/0)
        is_normalized : bool
            Is the data normalized (True/1 for yes) (default False/0)
        description : str
            Description string to be placed in the output TIFF (optional)
        transparent_pix : int
            Transparent pixel value (default -1)
    :Raises:
        KeyError :
            if required keyword is not provided
    """
    def _raise_value_error(text):
        log.error(text)
        raise ValueError(text)
    
    def _default_colormap():
         # Basic B&W colormap
        return [[ x*256 for x in range(256) ]]*3

    def _eval_or_none(key, eval_func):
        try:
            return eval_func(kwargs[key])
        except KeyError:
            return None

    log.info("Creating output file '%s'" % (output_fn,))
    tiff = TIFF.open(output_fn, "w")

    # Extract keyword arguments
    cmap = kwargs.pop("cmap", None)
    sat_id = int(kwargs.pop("sat_id"))
    chan_id = int(kwargs.pop("chan_id"))
    data_source = str(kwargs.pop("data_source"))
    tile_width = int(kwargs.pop("tile_width", 512))
    tile_length = int(kwargs.pop("tile_length", 512))
    data_cat = str(kwargs.pop("data_cat"))
    pixel_xres = float(kwargs.pop("pixel_xres"))
    pixel_yres = float(kwargs.pop("pixel_yres"))
    origin_lat = float(kwargs.pop("origin_lat"))
    origin_lon = float(kwargs.pop("origin_lon"))
    image_dt = kwargs.pop("image_dt")
    projection = str(kwargs.pop("projection"))
    meridian_west = float(kwargs.pop("meridian_west", 0.0))
    meridian_east = float(kwargs.pop("meridian_east", 0.0))
    radius_a = _eval_or_none("radius_a", float)
    radius_b = _eval_or_none("radius_b", float)
    ref_lat1 = _eval_or_none("ref_lat1", float)
    ref_lat2 = _eval_or_none("ref_lat2", float)
    central_meridian = _eval_or_none("central_meridian", float)
    min_gray_val = int(kwargs.pop("min_gray_val", 0))
    max_gray_val = int(kwargs.pop("max_gray_val", 255))
    altitude = float(kwargs.pop("altitude", 0.0))
    is_blac_corrected = int(bool(kwargs.pop("is_blac_corrected", 0)))
    is_atmo_corrected = int(bool(kwargs.pop("is_atmo_corrected", 0)))
    is_calibrated = int(bool(kwargs.pop("is_calibrated", 0)))
    is_normalized = int(bool(kwargs.pop("is_normalized", 0)))
    description = _eval_or_none("description", str)

    physic_value = str(kwargs.pop("physic_value"))
    physic_unit = str(kwargs.pop("physic_unit"))
    gradient = float(kwargs.pop("gradient"))
    axis_intercept = float(kwargs.pop("axis_intercept"))

    transparent_pix = int(kwargs.pop("transparent_pix", -1))

    # Keyword checks / verification
    if not cmap:
        cmap = _default_colormap()
    if len(cmap) != 3:
        _raise_value_error("Colormap (cmap) must be a list of 3 lists (RGB), not %d" %
                           len(cmap))

    if len(data_cat) != 4:
        _raise_value_error("NinJo data type must be 4 characters")
    if data_cat[0] not in ["P", "G"]:
        _raise_value_error("NinJo data type's first character must be 'P' or 'G' not '%s'" % 
                           data_cat[0])
    if data_cat[1] not in ["O", "P"]:
        _raise_value_error("NinJo data type's second character must be 'O' or 'P' not '%s'" %
                           data_cat[1])
    if data_cat[2:4] not in ["RN","RB","RA","BN","AN"]:
        _raise_value_error("NinJo data type's last 2 characters must be one of %s not '%s'" %
                           ("['RN','RB','RA','BN','AN']", data_cat[2:4]))

    if description is not None and len(description) >= 1000:
        log.error("NinJo description must be less than 1000 characters")
        raise ValueError("NinJo description must be less than 1000 characters")

    file_dt = datetime.utcnow()
    file_epoch = calendar.timegm(file_dt.timetuple())
    image_epoch = calendar.timegm(image_dt.timetuple())

    def _write_oneres(image_data, pixel_xres, pixel_yres):
        log.info("Writing tag data for a resolution %dx%d" % image_data.shape[:2])

        # Write Tag Data
        
        # Built ins
        tiff.SetField("ImageWidth", image_data.shape[1])
        tiff.SetField("ImageLength", image_data.shape[0])
        tiff.SetField("BitsPerSample", 8)
        tiff.SetField("Compression", libtiff.COMPRESSION_DEFLATE)
        if write_rgb:
            tiff.SetField("Photometric", libtiff.PHOTOMETRIC_RGB)
            tiff.SetField("SamplesPerPixel", 3)
        else:
            tiff.SetField("Photometric", libtiff.PHOTOMETRIC_PALETTE)
            tiff.SetField("SamplesPerPixel", 1)
            tiff.SetField("ColorMap", cmap)
        tiff.SetField("Orientation", libtiff.ORIENTATION_TOPLEFT)
        tiff.SetField("SMinSampleValue", 0)
        tiff.SetField("SMaxsampleValue", 255)
        tiff.SetField("PlanarConfig", libtiff.PLANARCONFIG_CONTIG)
        tiff.SetField("TileWidth", tile_width)
        tiff.SetField("TileLength", tile_length)
        tiff.SetField("SampleFormat", libtiff.SAMPLEFORMAT_UINT)

        # NinJo specific tags
        if description is not None:
            tiff.SetField("Description", description)

        if MODEL_PIXEL_SCALE_COUNT == 3:
            tiff.SetField("ModelPixelScale", [pixel_xres, pixel_yres, 0.0])
        else:
            tiff.SetField("ModelPixelScale", [pixel_xres, pixel_yres])
        tiff.SetField("ModelTiePoint", [0.0,  0.0, 0.0, origin_lon, origin_lat, 0.0])
        tiff.SetField("Magic", "NINJO")
        tiff.SetField("SatelliteNameID", sat_id)
        tiff.SetField("DateID", image_epoch)
        tiff.SetField("CreationDateID", file_epoch)
        tiff.SetField("ChannelID", chan_id)
        tiff.SetField("HeaderVersion", 2)
        tiff.SetField("FileName", output_fn)
        tiff.SetField("DataType", data_cat)
        tiff.SetField("SatelliteNumber", "\x00") # Hardcoded to 0
        if write_rgb:
            tiff.SetField("ColorDepth", 24)
        else:
            tiff.SetField("ColorDepth", 8)
        tiff.SetField("DataSource", data_source)
        tiff.SetField("XMinimum", 1)
        tiff.SetField("XMaximum", image_data.shape[1])
        tiff.SetField("YMinimum", 1)
        tiff.SetField("YMaximum", image_data.shape[0])
        tiff.SetField("Projection", projection)
        tiff.SetField("MeridianWest", meridian_west)
        tiff.SetField("MeridianEast", meridian_east)
        if radius_a is not None:
            tiff.SetField("EarthRadiusLarge", float(radius_a))
        if radius_b is not None:
            tiff.SetField("EarthRadiusSmall", float(radius_b))
        #tiff.SetField("GeodeticDate", "\x00") # ---?
        if ref_lat1 is not None:
            tiff.SetField("ReferenceLatitude1", ref_lat1)
        if ref_lat2 is not None:
            tiff.SetField("ReferenceLatitude2", ref_lat2)
        if central_meridian is not None:
            tiff.SetField("CentralMeridian", central_meridian)
        tiff.SetField("PhysicValue", physic_value) 
        tiff.SetField("PhysicUnit", physic_unit)
        tiff.SetField("MinGrayValue", min_gray_val)
        tiff.SetField("MaxGrayValue", max_gray_val)
        tiff.SetField("Gradient", gradient)
        tiff.SetField("AxisIntercept", axis_intercept)
        tiff.SetField("Altitude", altitude)
        tiff.SetField("IsBlackLineCorrection", is_blac_corrected)
        tiff.SetField("IsAtmosphereCorrected", is_atmo_corrected)
        tiff.SetField("IsCalibrated", is_calibrated)
        tiff.SetField("IsNormalized", is_normalized)

        tiff.SetField("TransparentPixel", transparent_pix)

        # Write Base Data Image
        tiff.write_tiles(image_data)
        tiff.WriteDirectory()

    # Write multi-resolution overviews (or not)
    tiff.SetDirectory(0)
    _write_oneres(image_data, pixel_xres, pixel_yres)
    for i, scale in enumerate((2, 4, 8, 16)):
        shape  = (image_data.shape[0]/scale,
                  image_data.shape[1]/scale)
        if shape[0] > tile_width and shape[1] > tile_length:
            tiff.SetDirectory(i+1)
            _write_oneres(image_data[::scale,::scale], pixel_xres*scale, pixel_yres*scale)
    tiff.close()

    log.info("Successfully created a NinJo tiff file: '%s'" % (output_fn,))

if __name__ == '__main__':
    import sys
    try:
        filename = sys.argv[1]
    except IndexError:
        print >> sys.stderr, "usage: python ninjotiff.py <ninjotiff-filename>"
        sys.exit(2)
    
    for inf in info(filename):
        print inf, '\n'
