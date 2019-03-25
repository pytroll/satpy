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

from satpy.utils import ensure_dir
from satpy.writers import ImageWriter

try:
    import rasterio
    gdal = osr = None
except ImportError as r_exc:
    try:
        # fallback to legacy gdal writer
        from osgeo import gdal, osr
        rasterio = None
    except ImportError:
        # raise the original rasterio exception
        raise r_exc

LOG = logging.getLogger(__name__)


class GeoTIFFWriter(ImageWriter):
    """Writer to save GeoTIFF images.

    Basic example from Scene:

        scn.save_datasets(writer='geotiff')

    Un-enhanced float geotiff with NaN for fill values:

        scn.save_datasets(writer='geotiff', dtype=np.float32, enhance=False)

    For performance tips on creating geotiffs quickly and making them smaller
    see the :doc:`faq`.

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

    def __init__(self, dtype=None, tags=None, **kwargs):
        super(GeoTIFFWriter, self).__init__(default_config_filename="writers/geotiff.yaml", **kwargs)
        self.dtype = self.info.get("dtype") if dtype is None else dtype
        self.tags = self.info.get("tags", None) if tags is None else tags
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

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs, kwargs = super(GeoTIFFWriter, cls).separate_init_kwargs(
            kwargs)
        for kw in ['dtype', 'tags']:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)

        return init_kwargs, kwargs

    def _gdal_write_datasets(self, dst_ds, datasets):
        """Write datasets in a gdal raster structure dts_ds"""
        for i, band in enumerate(datasets['bands']):
            chn = datasets.sel(bands=band)
            bnd = dst_ds.GetRasterBand(i + 1)
            bnd.SetNoDataValue(0)
            bnd.WriteArray(chn.values)

    def _gdal_write_geo(self, dst_ds, area):
        try:
            geotransform = [area.area_extent[0], area.pixel_size_x, 0,
                            area.area_extent[3], 0, -area.pixel_size_y]
            dst_ds.SetGeoTransform(geotransform)
            srs = osr.SpatialReference()

            srs.ImportFromProj4(area.proj_str)
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

    def _create_file(self, filename, img, gformat, g_opts, datasets, mode):
        num_bands = len(mode)
        if mode[-1] == 'A':
            g_opts.append("ALPHA=YES")

        def _delayed_create(create_opts, datasets, area, start_time, tags):
            raster = gdal.GetDriverByName("GTiff")
            dst_ds = raster.Create(*create_opts)
            self._gdal_write_datasets(dst_ds, datasets)

            # Create raster GeoTransform based on upper left corner and pixel
            # resolution ... if not overwritten by argument geotransform.
            if area is None:
                LOG.warning("No 'area' metadata found in image")
            else:
                self._gdal_write_geo(dst_ds, area)

            if start_time is not None:
                tags.update({'TIFFTAG_DATETIME': start_time.strftime(
                    "%Y:%m:%d %H:%M:%S")})

            dst_ds.SetMetadata(tags, '')

        create_opts = (filename, img.width, img.height, num_bands, gformat, g_opts)
        delayed = dask.delayed(_delayed_create)(
            create_opts, datasets, img.data.attrs.get('area'),
            img.data.attrs.get('start_time'),
            self.tags.copy())
        return delayed

    def save_image(self, img, filename=None, dtype=None, fill_value=None,
                   floating_point=None, compute=True, **kwargs):
        """Save the image to the given ``filename`` in geotiff_ format.

        Note for faster output and reduced memory usage the ``rasterio``
        library must be installed. This writer currently falls back to
        using ``gdal`` directly, but that will be deprecated in the future.

        Args:
            img (xarray.DataArray): Data to save to geotiff.
            filename (str): Filename to save the image to. Defaults to
                ``filename`` passed during writer creation. Unlike the
                creation ``filename`` keyword argument, this filename does not
                get formatted with data attributes.
            dtype (numpy.dtype): Numpy data type to save the image as.
                Defaults to 8-bit unsigned integer (``np.uint8``). If the
                ``dtype`` argument is provided during writer creation then
                that will be used as the default.
            fill_value (int or float): Value to use where data values are
                NaN/null. If this is specified in the writer configuration
                file that value will be used as the default.
            floating_point (bool): Deprecated. Use ``dtype=np.float64``
                instead.
            compute (bool): Compute dask arrays and save the image
                immediately. If ``False`` then the return value can be passed
                to :func:`~satpy.writers.compute_writer_results` to do the
                computation. This is useful when multiple images may share
                input calculations where dask can benefit from not repeating
                them multiple times. Defaults to ``True`` in the writer by
                itself, but is typically passed as ``False`` by callers where
                calculations can be combined.

        .. _geotiff: http://trac.osgeo.org/geotiff/

        """
        filename = filename or self.get_filename(**img.data.attrs)

        # Update global GDAL options with these specific ones
        gdal_options = self.gdal_options.copy()
        for k in kwargs.keys():
            if k in self.GDAL_OPTIONS:
                gdal_options[k] = kwargs[k]
        if fill_value is None:
            # fall back to fill_value from configuration file
            fill_value = self.info.get('fill_value')

        if floating_point is not None:
            import warnings
            warnings.warn("'floating_point' is deprecated, use"
                          "'dtype=np.float64' instead.",
                          DeprecationWarning)
            dtype = np.float64
        dtype = dtype if dtype is not None else self.dtype
        if dtype is None:
            dtype = np.uint8

        if "alpha" in kwargs:
            raise ValueError(
                "Keyword 'alpha' is automatically set based on 'fill_value' "
                "and should not be specified")
        if np.issubdtype(dtype, np.floating):
            if img.mode != "L":
                raise ValueError("Image must be in 'L' mode for floating "
                                 "point geotiff saving")
            if fill_value is None:
                LOG.debug("Alpha band not supported for float geotiffs, "
                          "setting fill value to 'NaN'")
                fill_value = np.nan

        try:
            import rasterio  # noqa
            # we can use the faster rasterio-based save
            return img.save(filename, fformat='tif', fill_value=fill_value,
                            dtype=dtype, compute=compute, **gdal_options)
        except ImportError:
            LOG.warning("Using legacy/slower geotiff save method, install "
                        "'rasterio' for faster saving.")
            warnings.warn("Using legacy/slower geotiff save method with 'gdal'."
                          "This will be deprecated in the future. Install "
                          "'rasterio' for faster saving and future "
                          "compatibility.", PendingDeprecationWarning)

            # Map numpy data types to GDAL data types
            NP2GDAL = {
                np.float32: gdal.GDT_Float32,
                np.float64: gdal.GDT_Float64,
                np.uint8: gdal.GDT_Byte,
                np.uint16: gdal.GDT_UInt16,
                np.uint32: gdal.GDT_UInt32,
                np.int16: gdal.GDT_Int16,
                np.int32: gdal.GDT_Int32,
                np.complex64: gdal.GDT_CFloat32,
                np.complex128: gdal.GDT_CFloat64,
            }

            # force to numpy dtype object
            dtype = np.dtype(dtype)
            gformat = NP2GDAL[dtype.type]

            gdal_options['nbits'] = int(gdal_options.get('nbits',
                                                         dtype.itemsize * 8))
            datasets, mode = img._finalize(fill_value=fill_value, dtype=dtype)
            LOG.debug("Saving to GeoTiff: %s", filename)
            g_opts = ["{0}={1}".format(k.upper(), str(v))
                      for k, v in gdal_options.items()]

            ensure_dir(filename)
            delayed = self._create_file(filename, img, gformat, g_opts,
                                        datasets, mode)
            if compute:
                return delayed.compute()
            return delayed
