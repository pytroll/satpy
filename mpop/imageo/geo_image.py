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
    defined (*area_id* parameter) and *time_slot* of the snapshot.
    
    The channels are considered to contain floating point values in the range
    [0.0,1.0]. In order to normalize the input data, the *crange* parameter
    defines the original range of the data. The conversion to the classical
    [0,255] range and byte type is done automagically when saving the image to
    file.

    See also :class:`image.Image` for more information.
    """

    def __init__(self, channels, area_id, time_slot, 
                 mode = "L", crange = None, fill_value = None, palette = None):
        self.area_id = area_id
        self.time_slot = time_slot
        self.tags = {}
        self.gdal_options = {}

        super(GeoImage, self).__init__(channels, mode, crange,
                                      fill_value, palette)

    def save(self, filename, compression=6,
             tags=None, gdal_options=None,
             fformat=None, blocksize=256):
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
                              gdal_options, blocksize)
        else:
            super(GeoImage, self).save(filename, compression, format=fformat)

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
                     blocksize=0, geotransform=None, spatialref=None):
        """Save the image to the given *filename* in geotiff_ format, with the
        *compression* level in [0, 9]. 0 means not compressed. The *tags*
        argument is a dict of tags to include in the image (as metadata).  By
        default it uses the 'area' instance to generate geotransform and
        spatialref information, this can be overwritten by the arguments
        *geotransform* and *spatialref*.
        
        .. _geotiff: http://trac.osgeo.org/geotiff/
        """
        from osgeo import gdal, osr
        
        raster = gdal.GetDriverByName("GTiff")
                    
        channels, fill_value = self._finalize()

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
                                       gdal.GDT_Byte,
                                       g_opts)
            else:
                g_opts.append("ALPHA=YES")
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       2, 
                                       gdal.GDT_Byte,
                                       g_opts)
            self._gdal_write_channels(dst_ds, channels, 255, fill_value)
        elif(self.mode == "LA"):
            ensure_dir(filename)
            g_opts.append("ALPHA=YES")
            dst_ds = raster.Create(filename, 
                                   self.width, 
                                   self.height, 
                                   2, 
                                   gdal.GDT_Byte,
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
                                       gdal.GDT_Byte,
                                       g_opts)
            else:
                g_opts.append("ALPHA=YES")
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       4, 
                                       gdal.GDT_Byte,
                                       g_opts)

            self._gdal_write_channels(dst_ds, channels, 255, fill_value)

        elif(self.mode == "RGBA"):
            ensure_dir(filename)
            g_opts.append("ALPHA=YES")
            dst_ds = raster.Create(filename, 
                                   self.width, 
                                   self.height, 
                                   4, 
                                   gdal.GDT_Byte,
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
            
                area = get_area_def(self.area_id)
            except (utils.AreaNotFound, AttributeError):
                area = self.area_id


            try:
                adfgeotransform = [area.area_extent[0], area.pixel_size_x, 0,
                                   area.area_extent[3], 0, -area.pixel_size_y]
                dst_ds.SetGeoTransform(adfgeotransform)
                srs = osr.SpatialReference()
                srs.SetProjCS(area.proj_id)            
                srs.ImportFromProj4(area.proj4_string)
                srs = srs.ExportToWkt()
                dst_ds.SetProjection(srs)
            except AttributeError:
                LOG.exception("Could not load geographic data, invalid area")

        self.tags.update({'TIFFTAG_DATETIME':
                          self.time_slot.strftime("%Y:%m:%d %H:%M:%S")})

        dst_ds.SetMetadata(self.tags, '')
        
        # Close the dataset
        
        dst_ds = None


    def add_overlay(self, color = (0, 0, 0)):
        """Add coastline and political borders to image, using *color*.
        """
        import warnings
        warnings.warn(
            """The GeoImage.add_overlay method is deprecated and should not be
            used anymore. To add coastlines, borders and rivers to your images,
            use pycoast instead:
            http://pycoast.googlecode.com
            """,
            DeprecationWarning)

        
        import acpgimage
        import _acpgpilext
        import pps_array2image

        self.convert("RGB")

        import ConfigParser
        conf = ConfigParser.ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, "geo_image.cfg"))

        coast_dir = CONFIG_PATH
        coast_file = os.path.join(coast_dir, conf.get('coasts', 'coast_file'))

        arr = np.zeros(self.channels[0].shape, np.uint8)

        LOG.debug("Adding overlay: " + str(self.area_id))
        if not isinstance(self.area_id, str):
            area_id = self.area_id.area_id
        else:
            area_id = self.area_id
        LOG.info("Add coastlines and political borders to image. "
                 "Area = %s"%(area_id))
        rimg = acpgimage.image(area_id)
        rimg.info["nodata"] = 255
        rimg.data = arr
        area_overlayfile = ("%s/coastlines_%s.asc"
                            %(coast_dir, area_id))
        LOG.info("Read overlay. Try find something prepared on the area...")
        try:
            overlay = _acpgpilext.read_overlay(area_overlayfile)
            LOG.info("Got overlay for area: %s."%area_overlayfile)
        except IOError:
            LOG.info("Didn't find an area specific overlay."
                     " Have to read world-map...")
            overlay = _acpgpilext.read_overlay(coast_file)
        LOG.info("Add overlay.")
        overlay_image = pps_array2image.add_overlay(rimg,
                                                    overlay,
                                                    pil.fromarray(arr),
                                                    color = 1)

        val = np.ma.asarray(overlay_image)

        self.channels[0] = np.ma.where(val == 1, color[0], self.channels[0])
        self.channels[0].mask = np.where(val == 1,
                                         False,
                                         np.ma.getmaskarray(self.channels[0]))

        self.channels[1] = np.ma.where(val == 1, color[1], self.channels[1])
        self.channels[1].mask = np.where(val == 1,
                                         False,
                                         np.ma.getmaskarray(self.channels[1]))

        self.channels[2] = np.ma.where(val == 1, color[2], self.channels[2])
        self.channels[2].mask = np.where(val == 1,
                                         False,
                                         np.ma.getmaskarray(self.channels[2]))

