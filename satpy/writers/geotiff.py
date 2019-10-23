#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""GeoTIFF writer objects for creating GeoTIFF files from `Dataset` objects.

"""

import logging
import numpy as np
from satpy.writers import ImageWriter
# make sure we have rasterio even though we don't use it until trollimage
# saves the image
import rasterio  # noqa

LOG = logging.getLogger(__name__)


class GeoTIFFWriter(ImageWriter):
    """Writer to save GeoTIFF images.

    Basic example from Scene:

        >>> scn.save_datasets(writer='geotiff')

    Un-enhanced float geotiff with NaN for fill values:

        >>> scn.save_datasets(writer='geotiff', dtype=np.float32, enhance=False)

    To add custom metadata use `tags`:

        >>> scn.save_dataset(dataset_name, writer='geotiff',
        ...                  tags={'offset': 291.8, 'scale': -0.35})

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
                    "copy_src_overviews",)

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

    def save_image(self, img, filename=None, dtype=None, fill_value=None,
                   compute=True, keep_palette=False, cmap=None, **kwargs):
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
            compute (bool): Compute dask arrays and save the image
                immediately. If ``False`` then the return value can be passed
                to :func:`~satpy.writers.compute_writer_results` to do the
                computation. This is useful when multiple images may share
                input calculations where dask can benefit from not repeating
                them multiple times. Defaults to ``True`` in the writer by
                itself, but is typically passed as ``False`` by callers where
                calculations can be combined.
            keep_palette (bool): Save palette/color table to geotiff.
                To be used with images that were palettized with the
                "palettize" enhancement. Setting this to ``True`` will cause
                the colormap of the image to be written as a "color table" in
                the output geotiff and the image data values will represent
                the index values in to that color table. By default, this will
                use the colormap used in the "palettize" operation.
                See the ``cmap`` option for other options. This option defaults
                to ``False`` and palettized images will be converted to RGB/A.
            cmap (trollimage.colormap.Colormap or None): Colormap to save
                as a color table in the output geotiff. See ``keep_palette``
                for more information. Defaults to the palette of the provided
                ``img`` object. The colormap's range should be set to match
                the index range of the palette
                (ex. `cmap.set_range(0, len(colors))`).
            tags (dict): Extra metadata to store in geotiff.

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
        if keep_palette and cmap is None and img.palette is not None:
            from satpy.enhancements import create_colormap
            cmap = create_colormap({'colors': img.palette})
            cmap.set_range(0, len(img.palette) - 1)

        tags = kwargs.get('tags', {})
        tags.update(self.tags)
        return img.save(filename, fformat='tif', fill_value=fill_value,
                        dtype=dtype, compute=compute,
                        keep_palette=keep_palette, cmap=cmap,
                        tags=tags,
                        **gdal_options)
