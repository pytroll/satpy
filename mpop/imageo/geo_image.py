#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2011, 2012.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Module for geographic images.
"""
import os

import Image as pil
import numpy as np

import mpop.imageo.image
from mpop import CONFIG_PATH
from mpop.imageo.logger import LOG
from mpop.utils import ensure_dir

class GeoImage(mpop.imageo.image.Image):
    """This class defines geographic images. As such, it contains not only data
    of the different *channels* of the image, but also the area on which it is
    defined (*area* parameter) and *time_slot* of the snapshot.
    
    The channels are considered to contain floating point values in the range
    [0.0,1.0]. In order to normalize the input data, the *crange* parameter
    defines the original range of the data. The conversion to the classical
    [0,255] range and byte type is done automagically when saving the image to
    file.

    See also :class:`image.Image` for more information.
    """

    def __init__(self, channels, area, time_slot, 
                 mode = "L", crange = None, fill_value = None, palette = None):
        self.area = area
        self.time_slot = time_slot
        self.tags = {}
        self.gdal_options = {}

        super(GeoImage, self).__init__(channels, mode, crange,
                                      fill_value, palette)

    def save(self, filename, compression=6,
             tags=None, gdal_options=None,
             fformat=None, blocksize=256, **kwargs):
        """Save the image to the given *filename*. If the extension is "tif",
        the image is saved to geotiff_ format, in which case the *compression*
        level can be given ([0, 9], 0 meaning off). See also
        :meth:`image.Image.save`, :meth:`image.Image.double_save`, and
        :meth:`image.Image.secure_save`.  The *tags* argument is a dict of tags
        to include in the image (as metadata), and the *gdal_options* holds
        options for the gdal saving driver. A *blocksize* other than 0 will
        result in a tiled image (if possible), with tiles of size equal to
        *blocksize*.
        

        .. _geotiff: http://trac.osgeo.org/geotiff/
        """
        file_tuple = os.path.splitext(filename)
        fformat = fformat or file_tuple[1][1:]

        if fformat.lower() in ('tif', 'tiff'):
            self.geotiff_save(filename, compression, tags,
                              gdal_options, blocksize, **kwargs)
        else:
            super(GeoImage, self).save(filename, compression, format=fformat, **kwargs)

    def _gdal_write_channels(self, dst_ds, channels, opacity, fill_value):
        """Write *channels* in a gdal raster structure *dts_ds*, using
        *opacity* as alpha value for valid data, and *fill_value*.
        """
        if fill_value is not None:
            for i in range(len(channels)):
                chn = channels[i].filled(fill_value[i])
                dst_ds.GetRasterBand(i + 1).WriteArray(chn)
        else:
            mask = np.zeros(channels[0].shape, dtype=np.uint8)
            i = 0
            for i in range(len(channels)):
                dst_ds.GetRasterBand(i + 1).WriteArray(channels[i].filled(i))
                mask |= np.ma.getmaskarray(channels[i]) 
            
            try:
                mask |= np.ma.getmaskarray(opacity)
            except AttributeError:
                pass
            
            alpha = np.where(mask, 0, opacity).astype(np.uint8)
            dst_ds.GetRasterBand(i + 2).WriteArray(alpha)

    def geotiff_save(self, filename, compression=6,
                     tags=None, gdal_options=None,
                     blocksize=0, geotransform=None,
                     spatialref=None, floating_point=False):
        """Save the image to the given *filename* in geotiff_ format, with the
        *compression* level in [0, 9]. 0 means not compressed. The *tags*
        argument is a dict of tags to include in the image (as metadata).  By
        default it uses the 'area' instance to generate geotransform and
        spatialref information, this can be overwritten by the arguments
        *geotransform* and *spatialref*. *floating_point* allows the saving of
        'L' mode images in floating point format if set to True.
        
        .. _geotiff: http://trac.osgeo.org/geotiff/
        """
        from osgeo import gdal, osr
        
        raster = gdal.GetDriverByName("GTiff")

        if floating_point:
            if self.mode != "L":
                raise ValueError("Image must be in 'L' mode for floating point"
                                 " geotif saving")
            channels = [self.channels[0].astype(np.float64)]
            fill_value = self.fill_value or 0
            gformat = gdal.GDT_Float64
        else:
            channels, fill_value = self._finalize()
            gformat = gdal.GDT_Byte

        LOG.debug("Saving to GeoTiff.")

        if tags is not None:
            self.tags.update(tags)
        if gdal_options is not None:
            self.gdal_options.update(gdal_options)

        g_opts = ["=".join(i) for i in self.gdal_options.items()]

        if compression != 0:
            g_opts.append("COMPRESS=DEFLATE")
            g_opts.append("ZLEVEL=" + str(compression))

        if blocksize != 0:
            g_opts.append("TILED=YES")
            g_opts.append("BLOCKXSIZE=" + str(blocksize))
            g_opts.append("BLOCKYSIZE=" + str(blocksize))
            

        if(self.mode == "L"):
            ensure_dir(filename)
            if fill_value is not None:
                dst_ds = raster.Create(filename, 
                                       self.width,
                                       self.height, 
                                       1, 
                                       gformat,
                                       g_opts)
            else:
                g_opts.append("ALPHA=YES")
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       2, 
                                       gformat,
                                       g_opts)
            self._gdal_write_channels(dst_ds, channels, 255, fill_value)
        elif(self.mode == "LA"):
            ensure_dir(filename)
            g_opts.append("ALPHA=YES")
            dst_ds = raster.Create(filename, 
                                   self.width, 
                                   self.height, 
                                   2, 
                                   gformat,
                                   g_opts)
            self._gdal_write_channels(dst_ds,
                                      channels[:-1], channels[1],
                                      fill_value)
        elif(self.mode == "RGB"):
            ensure_dir(filename)
            if fill_value is not None:
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       3, 
                                       gformat,
                                       g_opts)
            else:
                g_opts.append("ALPHA=YES")
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       4, 
                                       gformat,
                                       g_opts)

            self._gdal_write_channels(dst_ds, channels, 255, fill_value)

        elif(self.mode == "RGBA"):
            ensure_dir(filename)
            g_opts.append("ALPHA=YES")
            dst_ds = raster.Create(filename, 
                                   self.width, 
                                   self.height, 
                                   4, 
                                   gformat,
                                   g_opts)

            self._gdal_write_channels(dst_ds, channels[:-1], channels[3], fill_value)
        else:
            raise NotImplementedError("Saving to GeoTIFF using image mode"
                                      " %s is not implemented."%self.mode)


                
        # Create raster GeoTransform based on upper left corner and pixel
        # resolution ... if not overwritten by argument geotranform.

        if geotransform:
            dst_ds.SetGeoTransform(geotransform)
            if spatialref:
                if not isinstance(spatialref, str):
                    spatialref = spatialref.ExportToWkt()
                dst_ds.SetProjection(spatialref)
        else:
            try:
                from pyresample import utils
                from mpop.projector import get_area_def
            
                area = get_area_def(self.area)
            except (utils.AreaNotFound, AttributeError):
                area = self.area


            try:
                adfgeotransform = [area.area_extent[0], area.pixel_size_x, 0,
                                   area.area_extent[3], 0, -area.pixel_size_y]
                dst_ds.SetGeoTransform(adfgeotransform)
                srs = osr.SpatialReference()
                srs.ImportFromProj4(area.proj4_string)
                srs.SetProjCS(area.proj_id)
                try:
                    srs.SetWellKnownGeogCS(area.proj_dict['ellps'])
                except KeyError:
                    pass
                srs = srs.ExportToWkt()
                dst_ds.SetProjection(srs)
            except AttributeError:
                LOG.exception("Could not load geographic data, invalid area")

        self.tags.update({'TIFFTAG_DATETIME':
                          self.time_slot.strftime("%Y:%m:%d %H:%M:%S")})

        dst_ds.SetMetadata(self.tags, '')
        
        # Close the dataset
        
        dst_ds = None


    def add_overlay(self, color=(0, 0, 0), width=0.5, resolution=None):
        """Add coastline and political borders to image, using *color* (tuple
        of integers between 0 and 255).
        Warning: Loses the masks !
        
        *resolution* is chosen automatically if None (default), otherwise it should be one of:
        +-----+-------------------------+---------+
        | 'f' | Full resolution         | 0.04 km | 
        | 'h' | High resolution         | 0.2 km  |
        | 'i' | Intermediate resolution | 1.0 km  |
        | 'l' | Low resolution          | 5.0 km  |
        | 'c' | Crude resolution        | 25  km  |
        +-----+-------------------------+---------+
        """


        
        img = self.pil_image()

        import ConfigParser
        conf = ConfigParser.ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, "mpop.cfg"))

        coast_dir = conf.get('shapes', 'dir')

        LOG.debug("Getting area for overlay: " + str(self.area))

        if self.area is None:
            raise ValueError("Area of image is None, can't add overlay.")

        from mpop.projector import get_area_def
        if isinstance(self.area, str):
            self.area = get_area_def(self.area) 
        LOG.info("Add coastlines and political borders to image.")
        LOG.debug("Area = " + str(self.area))

        if resolution is None:
        
            x_resolution = ((self.area.area_extent[2] -
                             self.area.area_extent[0]) /
                            self.area.x_size)
            y_resolution = ((self.area.area_extent[3] -
                             self.area.area_extent[1]) /
                            self.area.y_size)
            res = min(x_resolution, y_resolution)

            if res > 25000:
                resolution = "c"
            elif res > 5000:
                resolution = "l"
            elif res > 1000:
                resolution = "i"
            elif res > 200:
                resolution = "h"
            else:
                resolution = "f"

            LOG.debug("Automagically choose resolution " + resolution)
        
        from pycoast import ContourWriterAGG
        cw_ = ContourWriterAGG(coast_dir)
        cw_.add_coastlines(img, self.area, outline=color,
                           resolution=resolution, width=width)
        cw_.add_borders(img, self.area, outline=color,
                        resolution=resolution, width=width)

        arr = np.array(img)

        for idx in range(len(self.channels)):
            self.channels[idx] = np.ma.array(arr[:, :, idx] / 255.0)

