# Copyright (c) 2015-2025 Satpy developers
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

"""Compositors using lookup tables."""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr

from .core import CompositeBase, GenericCompositor

LOG = logging.getLogger(__name__)


class CategoricalDataCompositor(CompositeBase):
    """Compositor used to recategorize categorical data using a look-up-table.

    Each value in the data array will be recategorized to a new category defined in
    the look-up-table using the original value as an index for that look-up-table.

    Example:
        data = [[1, 3, 2], [4, 2, 0]]
        lut = [10, 20, 30, 40, 50]
        res = [[20, 40, 30], [50, 30, 10]]
    """

    def __init__(self, name, lut=None, **kwargs):  # noqa: D417
        """Get look-up-table used to recategorize data.

        Args:
            lut (list): a list of new categories. The lenght must be greater than the
                        maximum value in the data array that should be recategorized.
        """
        self.lut = np.array(lut)
        super(CategoricalDataCompositor, self).__init__(name, **kwargs)

    def _update_attrs(self, new_attrs):
        """Modify name and add LUT."""
        new_attrs["name"] = self.attrs["name"]
        new_attrs["composite_lut"] = list(self.lut)

    @staticmethod
    def _getitem(block, lut):
        return lut[block]

    def __call__(self, projectables, **kwargs):
        """Recategorize the data."""
        if len(projectables) != 1:
            raise ValueError("Can't have more than one dataset for a categorical data composite")

        data = projectables[0].astype(int)
        res = data.data.map_blocks(
            self._getitem,
            self.lut,
            dtype=self.lut.dtype,
            meta=np.ndarray((), dtype=self.lut.dtype),
        )

        new_attrs = data.attrs.copy()
        self._update_attrs(new_attrs)

        return xr.DataArray(res, dims=data.dims, attrs=new_attrs, coords=data.coords)


class ColormapCompositor(GenericCompositor):
    """A compositor that uses colormaps.

    .. warning::

        Deprecated since Satpy 0.39.

    This compositor is deprecated.  To apply a colormap, use a
    :class:`satpy.composites.core.SingleBandCompositor` composite with a
    :func:`~satpy.enhancements.colormap.colorize` or
    :func:`~satpy.enhancements.colormap.palettize` enhancement instead.
    For example, to make a ``cloud_top_height`` composite based on a dataset
    ``ctth_alti`` palettized by ``ctth_alti_pal``, the composite would be::

      cloud_top_height:
        compositor: !!python/name:satpy.composites.core.SingleBandCompositor
        prerequisites:
        - ctth_alti
        tandard_name: cloud_top_height

    and the enhancement::

      cloud_top_height:
        standard_name: cloud_top_height
        operations:
        - name: palettize
          method: !!python/name:satpy.enhancements.colormap.palettize
          kwargs:
            palettes:
              - dataset: ctth_alti_pal
                color_scale: 255
                min_value: 0
                max_value: 255
    """

    @staticmethod
    def build_colormap(palette, dtype, info):
        """Create the colormap from the `raw_palette` and the valid_range.

        Colormaps come in different forms, but they are all supposed to have
        color values between 0 and 255. The following cases are considered:

        - Palettes comprised of only a list of colors. If *dtype* is uint8,
          the values of the colormap are the enumeration of the colors.
          Otherwise, the colormap values will be spread evenly from the min
          to the max of the valid_range provided in `info`.
        - Palettes that have a palette_meanings attribute. The palette meanings
          will be used as values of the colormap.

        """
        from trollimage.colormap import Colormap

        squeezed_palette = np.asanyarray(palette).squeeze() / 255.0
        cmap = Colormap.from_array_with_metadata(
                palette,
                dtype,
                color_scale=255,
                valid_range=info.get("valid_range"),
                scale_factor=info.get("scale_factor", 1),
                add_offset=info.get("add_offset", 0))

        return cmap, squeezed_palette

    def __call__(self, projectables, **info):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" %
                             (len(projectables), ))
        data, palette = projectables

        colormap, palette = self.build_colormap(palette, data.dtype, data.attrs)

        channels = self._apply_colormap(colormap, data, palette)
        return self._create_composite_from_channels(channels, data)

    def _create_composite_from_channels(self, channels, template):
        mask = self._get_mask_from_data(template)
        channels = [self._create_masked_dataarray_like(channel, template, mask) for channel in channels]
        res = super(ColormapCompositor, self).__call__(channels, **template.attrs)
        res.attrs["_FillValue"] = np.nan
        return res

    @staticmethod
    def _get_mask_from_data(data):
        fill_value = data.attrs.get("_FillValue", np.nan)
        if np.isnan(fill_value):
            mask = data.notnull()
        else:
            mask = data != data.attrs["_FillValue"]
        return mask

    @staticmethod
    def _create_masked_dataarray_like(array, template, mask):
        return xr.DataArray(array.reshape(template.shape),
                            dims=template.dims, coords=template.coords,
                            attrs=template.attrs).where(mask)


class ColorizeCompositor(ColormapCompositor):
    """A compositor colorizing the data, interpolating the palette colors when needed.

    .. warning::

        Deprecated since Satpy 0.39.  See the :class:`ColormapCompositor`
        docstring for documentation on the alternative.
    """

    @staticmethod
    def _apply_colormap(colormap, data, palette):
        del palette
        return colormap.colorize(data.data.squeeze())


class PaletteCompositor(ColormapCompositor):
    """A compositor colorizing the data, not interpolating the palette colors.

    .. warning::

        Deprecated since Satpy 0.39.  See the :class:`ColormapCompositor`
        docstring for documentation on the alternative.
    """

    @staticmethod
    def _apply_colormap(colormap, data, palette):
        channels, colors = colormap.palettize(data.data.squeeze())
        channels = channels.map_blocks(_insert_palette_colors, palette, dtype=palette.dtype,
                                       meta=np.ndarray((), dtype=palette.dtype),
                                       new_axis=2, chunks=list(channels.chunks) + [palette.shape[1]])
        return [channels[:, :, i] for i in range(channels.shape[2])]


def _insert_palette_colors(channels, palette):
    channels = palette[channels]
    return channels
