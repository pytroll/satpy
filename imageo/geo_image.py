#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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

import imageo.image
from pyresample import utils
from pp.utils import ensure_dir
import gdal
import osr
from imageo.logger import LOG

from imageo import CONFIG_PATH


class GeoImage(imageo.image.Image):
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
    area_id = None
    time_slot = None

    def __init__(self, channels, area_id, time_slot, 
                 mode = "L", crange = None, fill_value = None, palette = None):
        self.area_id = area_id
        self.time_slot = time_slot
        super(GeoImage, self).__init__(channels, mode, crange,
                                      fill_value, palette)

    def save(self, filename, compression=6, tags={}, gdal_options=[]):
        """Save the image to the given *filename*. If the extension is "tif",
        the image is saved to geotiff_ format, in which case the *compression*
        level can be given ([0, 9], 0 meaning off). See also
        :meth:`image.Image.save`, :meth:`image.Image.double_save`, and
        :meth:`image.Image.secure_save`.  The *tags* argument is a dict of tags
        to include in the image (as metadata).
        

        .. _geotiff: http://trac.osgeo.org/geotiff/
        """
        file_tuple = os.path.splitext(filename)

        if(file_tuple[1] == ".tif"):
            self._geotiff_save(filename, compression, tags, gdal_options)
        else:
            super(GeoImage, self).save(filename, compression)

    def _gdal_write_channels(self, dst_ds, channels, opacity, fill_value):
        """Write *channels* in a gdal raster structure *dts_ds*, using
        *opacity* as alpha value for valid data, and *fill_value*.
        """
        if fill_value is not None:
            for i in range(len(channels)):
                chn = channels[i].filled(fill_value[i])
                dst_ds.GetRasterBand(i + 1).WriteArray(chn)
        else:
            mask = np.zeros_like(channels[0])
            i = 0
            for i in range(len(channels)):
                dst_ds.GetRasterBand(i + 1).WriteArray(channels[i].filled(i))
                mask |= np.ma.getmaskarray(channels[i]) 
            
            alpha = np.where(mask, 0, opacity)
            if self.mode.endswith("A"):
                dst_ds.GetRasterBand(i + 1).WriteArray(alpha)
            else:
                dst_ds.GetRasterBand(i + 2).WriteArray(alpha)

    def _geotiff_save(self, filename, compression=6, tags={}, gdal_options=[]):
        """Save the image to the given *filename* in geotiff_ format, with the
        *compression* level in [0, 9]. 0 means not compressed. The *tags*
        argument is a dict of tags to include in the image (as metadata).
        
        .. _geotiff: http://trac.osgeo.org/geotiff/
        """
        raster = gdal.GetDriverByName("GTiff")
                    
        channels, fill_value = self._finalize()

        LOG.debug("Saving to GeoTiff.")

        #options = ["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
        #options = []

        #if compression != 0:
        #    options.append("COMPRESS=DEFLATE")
        #    options.append("ZLEVEL=" + str(compression))

        if(self.mode == "L"):
            ensure_dir(filename)
            if fill_value is not None:
                dst_ds = raster.Create(filename, 
                                       self.width,
                                       self.height, 
                                       1, 
                                       gdal.GDT_Byte,
                                       gdal_options)
            else:
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       2, 
                                       gdal.GDT_Byte,
                                       gdal_options)
            self._gdal_write_channels(dst_ds, channels, 255, fill_value)
        elif(self.mode == "RGB"):
            ensure_dir(filename)
            if fill_value is not None:
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       3, 
                                       gdal.GDT_Byte,
                                       gdal_options)
            else:
                dst_ds = raster.Create(filename, 
                                       self.width, 
                                       self.height, 
                                       4, 
                                       gdal.GDT_Byte,
                                       gdal_options)

            self._gdal_write_channels(dst_ds, channels, 255, fill_value)

        elif(self.mode == "RGBA"):
            ensure_dir(filename)
            dst_ds = raster.Create(filename, 
                                   self.width, 
                                   self.height, 
                                   4, 
                                   gdal.GDT_Byte,
                                   gdal_options)

            self._gdal_write_channels(dst_ds, channels, channels[3], fill_value)
        else:
            raise NotImplementedError("Saving to GeoTIFF using image mode"
                                      " %s is not implemented."%self.mode)


                
        # Create raster GeoTransform based on upper left corner and pixel
        # resolution

        area_file = os.path.join(CONFIG_PATH, "areas.def")
        area = utils.parse_area_file(area_file, self.area_id)[0]

        adfgeotransform = [area.area_extent[0], area.pixel_size_x, 0,
                           area.area_extent[3], 0, -area.pixel_size_y]

        dst_ds.SetGeoTransform(adfgeotransform)
        srs = osr.SpatialReference()
        srs.SetProjCS(area.proj_id)

        srs.ImportFromProj4(area.proj4_string)
        srs.SetWellKnownGeogCS('WGS84')
        dst_ds.SetProjection(srs.ExportToWkt())

        tags.update({'TIFFTAG_DATETIME':
                     self.time_slot.strftime("%Y:%m:%d %H:%M:%S")})

        dst_ds.SetMetadata(tags, '')
        
        # Close the dataset
        
        dst_ds = None


    def add_overlay(self, color = (0, 0, 0)):
        """Add coastline and political borders to image, using *color*.
        """
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
        
        LOG.info("Add coastlines and political borders to image. "
                 "Area = %s"%(self.area_id))
        rimg = acpgimage.image(self.area_id)
        rimg.info["nodata"] = 255
        rimg.data = arr
        area_overlayfile = ("%s/coastlines_%s.asc"
                            %(coast_dir, self.area_id))
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

