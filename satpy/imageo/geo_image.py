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

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Module for geographic images.
"""
import os

import numpy as np

try:
    from trollimage.image import Image, UnknownImageFormat
except ImportError:
    from satpy.imageo.image import Image, UnknownImageFormat

from satpy.config import CONFIG_PATH
import logging
from satpy.utils import ensure_dir

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

    def __init__(self, channels, area, start_time, copy=True,
                 mode="L", crange=None, fill_value=None, palette=None, **kwargs):
        self.area = area
        # FIXME: Should we be concerned with start time and end time?
        self.time_slot = start_time
        self.tags = {}
        self.gdal_options = {}

        Image.__init__(self,
                       channels=channels,
                       mode=mode,
                       color_range=crange,
                       fill_value=fill_value,
                       palette=palette,
                       copy=copy)

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

        If the specified format *fformat* is not know to satpy (and PIL), we
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
                raise UnknownImageFormat(
                    "Unknown image format '%s'" % fformat)
            saver.save(self, filename, **kwargs)

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
        conf.read(os.path.join(CONFIG_PATH, "satpy.cfg"))

        coast_dir = conf.get('shapes', 'dir')

        logger.debug("Getting area for overlay: " + str(self.area))

        if self.area is None:
            raise ValueError("Area of image is None, can't add overlay.")

        from satpy.projector import get_area_def

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
        conf.read(os.path.join(CONFIG_PATH, "satpy.cfg"))

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

        from satpy.projector import get_area_def

        if isinstance(self.area, str):
            self.area = get_area_def(self.area)
        logger.info("Add overlays to image.")
        logger.debug("Area = " + str(self.area.area_id))

        foreground = cw_.add_overlay_from_config(config_file, self.area)
        img.paste(foreground, mask=foreground.split()[-1])

        arr = np.array(img)

        if len(self.channels) == 1:
            self.channels[0] = np.ma.array(arr[:, :] / 255.0)
        else:
            for idx in range(len(self.channels)):
                self.channels[idx] = np.ma.array(arr[:, :, idx] / 255.0)
