#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>
#   Stefano Cerino <s.cerino@vitrociset.it>
#   Katja Hungershofer <katja.Hungershoefer@dwd.de>

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

import numpy as np

try:
    from trollimage.image import Image, UnknownImageFormat
except ImportError:
    from mpop.imageo.image import Image, UnknownImageFormat


from mpop import CONFIG_PATH
import logging
from mpop.utils import ensure_dir

logger = logging.getLogger(__name__)

class GeoImage(Image):
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
                 mode="L", crange=None, fill_value=None, palette=None):
        self.area = area
        self.time_slot = time_slot
        self.tags = {}
        self.gdal_options = {}

        Image.__init__(self, channels, mode, crange,
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

        If the specified format *fformat* is not know to MPOP (and PIL), we
        will try to import module *fformat* and call the method `fformat.save`.


        .. _geotiff: http://trac.osgeo.org/geotiff/
        """
        file_tuple = os.path.splitext(filename)
        fformat = fformat or file_tuple[1][1:]

        if fformat.lower() in ('tif', 'tiff'):
            return self.geotiff_save(filename, compression, tags,
                                     gdal_options, blocksize, **kwargs)
        try:
            # Let image.pil_save it ?
            Image.save(self, filename, compression, fformat=fformat)
        except UnknownImageFormat:
            # No ... last resort, try to import an external module.
            logger.info("Importing image writer module '%s'" % fformat)
            try:
                saver = __import__(fformat, globals(), locals(), ['save'])
            except ImportError:
                raise  UnknownImageFormat(
                    "Unknown image format '%s'" % fformat)
            saver.save(self, filename, **kwargs)

    def _gdal_write_channels(self, dst_ds, channels, opacity, fill_value):
        """Write *channels* in a gdal raster structure *dts_ds*, using
        *opacity* as alpha value for valid data, and *fill_value*.
        """
        if fill_value is not None:
            for i, chan in enumerate(channels):
                chn = chan.filled(fill_value[i])
                dst_ds.GetRasterBand(i + 1).WriteArray(chn)
        else:
            mask = np.zeros(channels[0].shape, dtype=np.bool)
            i = 0
            for i, chan in enumerate(channels):
                dst_ds.GetRasterBand(i + 1).WriteArray(chan.filled(0))
                mask |= np.ma.getmaskarray(chan)
            try:
                mask |= np.ma.getmaskarray(opacity)
            except AttributeError:
                pass

            alpha = np.where(mask, 0, opacity).astype(chan.dtype)
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

        tags = tags or {}

        if floating_point:
            if self.mode != "L":
                raise ValueError("Image must be in 'L' mode for floating point"
                                 " geotif saving")
            if self.fill_value is None:
                logger.warning("Image with floats cannot be transparent, "
                               "so setting fill_value to 0")
                self.fill_value = 0
            channels = [self.channels[0].astype(np.float64)]
            fill_value = self.fill_value or [0]
            gformat = gdal.GDT_Float64
            opacity = 0
        else:
            nbits = int(tags.get("NBITS", "8"))
            if nbits > 16:
                dtype = np.uint32
                gformat = gdal.GDT_UInt32
            elif nbits > 8:
                dtype = np.uint16
                gformat = gdal.GDT_UInt16
            else:
                dtype = np.uint8
                gformat = gdal.GDT_Byte
            opacity = np.iinfo(dtype).max
            channels, fill_value = self._finalize(dtype)

        logger.debug("Saving to GeoTiff.")

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
            self._gdal_write_channels(dst_ds, channels,
                                      opacity, fill_value)
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

            self._gdal_write_channels(dst_ds, channels,
                                      opacity, fill_value)

        elif(self.mode == "RGBA"):
            ensure_dir(filename)
            g_opts.append("ALPHA=YES")
            dst_ds = raster.Create(filename,
                                   self.width,
                                   self.height,
                                   4,
                                   gformat,
                                   g_opts)

            self._gdal_write_channels(dst_ds,
                                      channels[:-1], channels[3],
                                      fill_value)
        else:
            raise NotImplementedError("Saving to GeoTIFF using image mode"
                                      " %s is not implemented."%self.mode)



        # Create raster GeoTransform based on upper left corner and pixel
        # resolution ... if not overwritten by argument geotransform.

        if geotransform:
            dst_ds.SetGeoTransform(geotransform)
            if spatialref:
                if not isinstance(spatialref, str):
                    spatialref = spatialref.ExportToWkt()
                dst_ds.SetProjection(spatialref)
        else:
            from pyresample import utils
            from mpop.projector import get_area_def
            try:
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
                try:
                    # Check for epsg code.
                    srs.SetAuthority('PROJCS', 'EPSG',
                                     int(area.proj_dict['init'].
                                         split('epsg:')[1]))
                except (KeyError, IndexError):
                    pass
                srs = srs.ExportToWkt()
                dst_ds.SetProjection(srs)
            except AttributeError:
                logger.exception("Could not load geographic data, invalid area")

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

        logger.debug("Getting area for overlay: " + str(self.area))

        if self.area is None:
            raise ValueError("Area of image is None, can't add overlay.")

        from mpop.projector import get_area_def
        if isinstance(self.area, str):
            self.area = get_area_def(self.area)
        logger.info("Add coastlines and political borders to image.")
        logger.debug("Area = " + str(self.area))

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

            logger.debug("Automagically choose resolution " + resolution)

        from pycoast import ContourWriterAGG
        cw_ = ContourWriterAGG(coast_dir)
        cw_.add_coastlines(img, self.area, outline=color,
                           resolution=resolution, width=width)
        cw_.add_borders(img, self.area, outline=color,
                        resolution=resolution, width=width)

        arr = np.array(img)

        if len(self.channels) == 1:
            self.channels[0] = np.ma.array(arr[:, :] / 255.0)
        else:
            for idx in range(len(self.channels)):
                self.channels[idx] = np.ma.array(arr[:, :, idx] / 255.0)


    def add_overlay_config(self, config_file):
        """Add overlay to image parsing a configuration file.
           
        """


        import ConfigParser
        conf = ConfigParser.ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, "mpop.cfg"))

        coast_dir = conf.get('shapes', 'dir')

        logger.debug("Getting area for overlay: " + str(self.area.area_id))

        try:
            import aggdraw
            from pycoast import ContourWriterAGG
            cw_ = ContourWriterAGG(coast_dir)
        except ImportError:
            logger.warning("AGGdraw lib not installed...width and opacity properties are not available for overlays.")
            from pycoast import ContourWriter
            cw_ = ContourWriter(coast_dir)
            

        logger.debug("Getting area for overlay: " + str(self.area))

        if self.area is None:
            raise ValueError("Area of image is None, can't add overlay.")

        if self.mode != "RGB":
            self.convert("RGB")

        img = self.pil_image()


        from mpop.projector import get_area_def
        if isinstance(self.area, str):
            self.area = get_area_def(self.area)
        logger.info("Add overlays to image.")
        logger.debug("Area = " + str(self.area.area_id))

        foreground=cw_.add_overlay_from_config(config_file, self.area)
        img.paste(foreground,mask=foreground.split()[-1])

        arr = np.array(img)

        if len(self.channels) == 1:
            self.channels[0] = np.ma.array(arr[:, :] / 255.0)
        else:
            for idx in range(len(self.channels)):
                self.channels[idx] = np.ma.array(arr[:, :, idx] / 255.0)

