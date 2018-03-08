#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""GeoTIFF writer objects for creating GeoTIFF files from `Dataset` objects.

"""

import logging

import dask
import numpy as np
from osgeo import gdal, osr

from satpy.utils import ensure_dir
from satpy.writers import ImageWriter

LOG = logging.getLogger(__name__)


class GeoTIFFWriter(ImageWriter):

    """Writer to save GeoTIFF images.

    Basic example from Scene:

        scn.save_datasets(writer='geotiff')

    Un-enhanced float geotiff with NaN for fill values:

        scn.save_datasets(writer='geotiff', floating_point=True,
                          enhancement_config=False, fill_value=np.nan)

    """

    GDAL_OPTIONS = ("tfw",
                    "rpb",
                    "rpctxt",
                    "interleave",
                    "tiled",
                    "blockxsize",
                    "blockysize",
                    "nbits",
                    "compress",
                    "num_threads",
                    "predictor",
                    "discard_lsb",
                    "sparse_ok",
                    "jpeg_quality",
                    "jpegtablesmode",
                    "zlevel",
                    "photometric",
                    "alpha",
                    "profile",
                    "bigtiff",
                    "pixeltype",
                    "copy_src_overviews", )

    def __init__(self, floating_point=False, tags=None, **kwargs):
        ImageWriter.__init__(self,
                             default_config_filename="writers/geotiff.yaml",
                             **kwargs)

        self.floating_point = bool(self.info.get(
            "floating_point", None) if floating_point is None else
            floating_point)
        self.tags = self.info.get("tags",
                                  None) if tags is None else tags
        if self.tags is None:
            self.tags = {}
        elif not isinstance(self.tags, dict):
            # if it's coming from a config file
            self.tags = dict(tuple(x.split("=")) for x in self.tags.split(","))

        # GDAL specific settings
        self.gdal_options = {}
        for k in self.GDAL_OPTIONS:
            if k in kwargs or k in self.info:
                self.gdal_options[k] = kwargs.get(k, self.info[k])

    def _gdal_write_datasets(self, dst_ds, datasets, opacity):
        """Write *datasets* in a gdal raster structure *dts_ds*, using
        *opacity* as alpha value for valid data, and *fill_value*.
        """
        def _write_array(bnd, chn):
            bnd.WriteArray(chn.values)

        # queue up data writes so we don't waste computation time
        delayed = []
        for i, band in enumerate(datasets['bands']):
            chn = datasets.sel(bands=band)
            bnd = dst_ds.GetRasterBand(i + 1)
            bnd.SetNoDataValue(0)
            delay = dask.delayed(_write_array)(bnd, chn)
            delayed.append(delay)
        dask.compute(*delayed)

    def _create_file(self, filename, img, gformat, g_opts, opacity,
                     datasets, mode):
        raster = gdal.GetDriverByName("GTiff")

        if mode == "L":
            dst_ds = raster.Create(filename, img.width, img.height, 1,
                                   gformat, g_opts)
            self._gdal_write_datasets(dst_ds, datasets, opacity)
        elif mode == "LA":
            g_opts.append("ALPHA=YES")
            dst_ds = raster.Create(filename, img.width, img.height, 2, gformat,
                                   g_opts)
            self._gdal_write_datasets(dst_ds, datasets, datasets)
        elif mode == "RGB":
            dst_ds = raster.Create(filename, img.width, img.height, 3,
                                   gformat, g_opts)
            self._gdal_write_datasets(dst_ds, datasets, datasets)

        elif mode == "RGBA":
            g_opts.append("ALPHA=YES")
            dst_ds = raster.Create(filename, img.width, img.height, 4, gformat,
                                   g_opts)

            self._gdal_write_datasets(dst_ds, datasets, datasets)
        else:
            raise NotImplementedError(
                "Saving to GeoTIFF using image mode %s is not implemented." %
                mode)

        # Create raster GeoTransform based on upper left corner and pixel
        # resolution ... if not overwritten by argument geotransform.
        if "area" not in img.data.attrs:
            LOG.warning("No 'area' metadata found in image")
        else:
            area = img.data.attrs["area"]
            try:
                geotransform = [area.area_extent[0], area.pixel_size_x, 0,
                                area.area_extent[3], 0, -area.pixel_size_y]
                dst_ds.SetGeoTransform(geotransform)
                srs = osr.SpatialReference()

                srs.ImportFromProj4(area.proj4_string)
                srs.SetProjCS(area.proj_id)
                try:
                    srs.SetWellKnownGeogCS(area.proj_dict['ellps'])
                except KeyError:
                    pass
                try:
                    # Check for epsg code.
                    srs.ImportFromEPSG(int(
                        area.proj_dict['init'].lower().split('epsg:')[1]))
                except (KeyError, IndexError):
                    pass
                srs = srs.ExportToWkt()
                dst_ds.SetProjection(srs)
            except AttributeError:
                LOG.warning(
                    "Can't save geographic information to geotiff, unsupported area type")

        tags = self.tags.copy()
        if "start_time" in img.data.attrs:
            tags.update({'TIFFTAG_DATETIME': img.data.attrs["start_time"].strftime(
                "%Y:%m:%d %H:%M:%S")})

        dst_ds.SetMetadata(tags, '')

    def save_image(self, img, filename=None, floating_point=False,
                   compute=True, **kwargs):
        """Save the image to the given *filename* in geotiff_ format.
        `floating_point` allows the saving of
        'L' mode images in floating point format if set to True.

        .. _geotiff: http://trac.osgeo.org/geotiff/
        """
        filename = filename or self.get_filename(**img.data.attrs)

        # Update global GDAL options with these specific ones
        gdal_options = self.gdal_options.copy()
        for k in kwargs.keys():
            if k in self.GDAL_OPTIONS:
                gdal_options[k] = kwargs[k]

        floating_point = floating_point if floating_point is not None else self.floating_point

        if "alpha" in kwargs:
            raise ValueError(
                "Keyword 'alpha' is automatically set and should not be specified")
        if floating_point:
            if img.mode != "L":
                raise ValueError(
                    "Image must be in 'L' mode for floating point geotiff saving")
            raise NotImplementedError('Floating point saving not yet implemented.')
            # if img.fill_value is None:
            #     LOG.warning(
            #         "Image with floats cannot be transparent, so setting fill_value to 0")
            #     fill_value = 0
            # datasets = [img.channels[0].astype(np.float64)]
            # fill_value = img.fill_value or [0]
            # gformat = gdal.GDT_Float64
            # opacity = 0
        else:
            nbits = int(gdal_options.get("nbits", "8"))
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
            datasets, mode = img._finalize(dtype=dtype)

        LOG.debug("Saving to GeoTiff: %s", filename)

        g_opts = ["{0}={1}".format(k.upper(), str(v))
                  for k, v in gdal_options.items()]

        ensure_dir(filename)
        delayed = dask.delayed(self._create_file)(filename, img, gformat,
                                                  g_opts, opacity, datasets,
                                                  mode)
        if compute:
            return delayed.compute()
        else:
            return delayed
