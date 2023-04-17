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
"""GeoTIFF writer objects for creating GeoTIFF files from `DataArray` objects."""
from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np

# make sure we have rasterio even though we don't use it until trollimage
# saves the image
import rasterio  # noqa
from trollimage.colormap import Colormap
from trollimage.xrimage import XRImage

from satpy._compat import DTypeLike
from satpy.writers import ImageWriter

LOG = logging.getLogger(__name__)


class GeoTIFFWriter(ImageWriter):
    """Writer to save GeoTIFF images.

    Basic example from Scene:

        >>> scn.save_datasets(writer='geotiff')

    By default the writer will use the :class:`~satpy.writers.Enhancer` class to
    linear stretch the data (see :doc:`../enhancements`).
    To get Un-enhanced images ``enhance=False`` can be specified which will
    write a geotiff with the data type of the dataset. The fill value defaults
    to the the datasets ``"_FillValue"`` attribute if not ``None`` and no value is
    passed to ``fill_value`` for integer data. In case of float data if ``fill_value``
    is not passed NaN will be used. If a geotiff with a
    certain datatype is desired for example 32 bit floating point geotiffs:

        >>> scn.save_datasets(writer='geotiff', dtype=np.float32, enhance=False)

    To add custom metadata use `tags`:

        >>> scn.save_dataset(dataset_name, writer='geotiff',
        ...                  tags={'offset': 291.8, 'scale': -0.35})

    Images are tiled by default. To create striped TIFF files ``tiled=False`` can be specified:

        >>> scn.save_datasets(writer='geotiff', tiled=False)

    For performance tips on creating geotiffs quickly and making them smaller
    see the :ref:`faq`.

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
                    "copy_src_overviews",
                    # COG driver options (different from GTiff above)
                    "blocksize",
                    "resampling",
                    "quality",
                    "level",
                    "overview_resampling",
                    "warp_resampling",
                    "overview_compress",
                    "overview_quality",
                    "overview_predictor",
                    "tiling_scheme",
                    "zoom_level_strategy",
                    "target_srs",
                    "res",
                    "extent",
                    "aligned_levels",
                    "add_alpha",
                    )

    def __init__(self, dtype=None, tags=None, **kwargs):
        """Init the writer."""
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
        """Separate the init keyword args."""
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs, kwargs = super(GeoTIFFWriter, cls).separate_init_kwargs(
            kwargs)
        for kw in ['dtype', 'tags']:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)

        return init_kwargs, kwargs

    def save_image(
            self,
            img: XRImage,
            filename: Optional[str] = None,
            compute: bool = True,
            dtype: Optional[DTypeLike] = None,
            fill_value: Optional[Union[int, float]] = None,
            keep_palette: bool = False,
            cmap: Optional[Colormap] = None,
            tags: Optional[dict[str, Any]] = None,
            overviews: Optional[list[int]] = None,
            overviews_minsize: int = 256,
            overviews_resampling: Optional[str] = None,
            include_scale_offset: bool = False,
            scale_offset_tags: Optional[tuple[str, str]] = None,
            colormap_tag: Optional[str] = None,
            driver: Optional[str] = None,
            tiled: bool = True,
            **kwargs
    ):
        """Save the image to the given ``filename`` in geotiff_ format.

        Note this writer requires the ``rasterio`` library to be installed.

        Args:
            img (xarray.DataArray): Data to save to geotiff.
            filename (str): Filename to save the image to. Defaults to
                ``filename`` passed during writer creation. Unlike the
                creation ``filename`` keyword argument, this filename does not
                get formatted with data attributes.
            compute (bool): Compute dask arrays and save the image
                immediately. If ``False`` then the return value can be passed
                to :func:`~satpy.writers.compute_writer_results` to do the
                computation. This is useful when multiple images may share
                input calculations where dask can benefit from not repeating
                them multiple times. Defaults to ``True`` in the writer by
                itself, but is typically passed as ``False`` by callers where
                calculations can be combined.
            dtype (DTypeLike): Numpy data type to save the image as.
                Defaults to 8-bit unsigned integer (``np.uint8``) or the data
                type of the data to be saved if ``enhance=False``. If the
                ``dtype`` argument is provided during writer creation then
                that will be used as the default.
            fill_value (float or int): Value to use where data values are
                NaN/null. If this is specified in the writer configuration
                file that value will be used as the default.
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
            overviews (list): The reduction factors of the overviews to include
                in the image, eg::

                    scn.save_datasets(overviews=[2, 4, 8, 16])

                If provided as an empty list, then levels will be
                computed as powers of two until the last level has less
                pixels than `overviews_minsize`.
                Default is to not add overviews.
            overviews_minsize (int): Minimum number of pixels for the smallest
                overview size generated when `overviews` is auto-generated.
                Defaults to 256.
            overviews_resampling (str): Resampling method
                to use when generating overviews. This must be the name of an
                enum value from :class:`rasterio.enums.Resampling` and
                only takes effect if the `overviews` keyword argument is
                provided. Common values include `nearest` (default),
                `bilinear`, `average`, and many others. See the rasterio
                documentation for more information.
            scale_offset_tags (Tuple[str, str]): If set, include inclusion of
                scale and offset in the GeoTIFF headers in the GDALMetaData
                tag.  The value of this argument should be a keyword argument
                ``(scale_label, offset_label)``, for example, ``("scale",
                "offset")``, indicating the labels to be used.
            colormap_tag (Optional[str]): If set and the image being saved was
                colorized or palettized then a comma-separated version of the
                colormap is saved to a custom geotiff tag with the provided
                name. See :meth:`trollimage.colormap.Colormap.to_csv` for more
                information.
            driver (Optional[str]): Name of GDAL driver to use to save the
                geotiff. If not specified or None (default) the "GTiff" driver
                is used. Another common option is "COG" for Cloud Optimized
                GeoTIFF. See GDAL documentation for more information.
            tiled (bool): For performance this defaults to ``True``.
                Pass ``False`` to created striped TIFF files.
            include_scale_offset (deprecated, bool): Deprecated.
                Use ``scale_offset_tags=("scale", "offset")`` to include scale
                and offset tags.

        .. _geotiff: http://trac.osgeo.org/geotiff/

        """
        filename = filename or self.get_filename(**img.data.attrs)

        gdal_options = self._get_gdal_options(kwargs)
        if fill_value is None:
            # fall back to fill_value from configuration file
            fill_value = self.info.get('fill_value')

        dtype = dtype if dtype is not None else self.dtype
        if dtype is None and self.enhancer is not False:
            dtype = np.uint8
        elif dtype is None:
            dtype = img.data.dtype.type

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

        if tags is None:
            tags = {}
        tags.update(self.tags)

        return img.save(filename, fformat='tif', driver=driver,
                        fill_value=fill_value,
                        dtype=dtype, compute=compute,
                        keep_palette=keep_palette, cmap=cmap,
                        tags=tags, include_scale_offset_tags=include_scale_offset,
                        scale_offset_tags=scale_offset_tags,
                        colormap_tag=colormap_tag,
                        overviews=overviews,
                        overviews_resampling=overviews_resampling,
                        overviews_minsize=overviews_minsize,
                        tiled=tiled,
                        **gdal_options)

    def _get_gdal_options(self, kwargs):
        # Update global GDAL options with these specific ones
        gdal_options = self.gdal_options.copy()
        for k in kwargs:
            if k in self.GDAL_OPTIONS:
                gdal_options[k] = kwargs[k]
        return gdal_options
