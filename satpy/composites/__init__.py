# Copyright (c) 2015-2023 Satpy developers
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
"""Base classes for composite objects."""
from __future__ import annotations

import logging
import os
import warnings

import dask.array as da
import numpy as np
import xarray as xr
from trollimage.colormap import Colormap

import satpy
from satpy.aux_download import DataDownloadMixin
from satpy.dataset import DataID, combine_metadata
from satpy.dataset.dataid import minimal_default_keys_config
from satpy.utils import unify_chunks
from satpy.writers import get_enhanced_image

LOG = logging.getLogger(__name__)

NEGLIGIBLE_COORDS = ['time']
"""Keywords identifying non-dimensional coordinates to be ignored during composite generation."""

MASKING_COMPOSITOR_METHODS = ['less', 'less_equal', 'equal', 'greater_equal',
                              'greater', 'not_equal', 'isnan', 'isfinite',
                              'isneginf', 'isposinf']


class IncompatibleAreas(Exception):
    """Error raised upon compositing things of different shapes."""


class IncompatibleTimes(Exception):
    """Error raised upon compositing things from different times."""


def check_times(projectables):
    """Check that *projectables* have compatible times."""
    times = []
    for proj in projectables:
        try:
            if proj['time'].size and proj['time'][0] != 0:
                times.append(proj['time'][0].values)
            else:
                break  # right?
        except KeyError:
            # the datasets don't have times
            break
        except IndexError:
            # time is a scalar
            if proj['time'].values != 0:
                times.append(proj['time'].values)
            else:
                break
    else:
        # Is there a more gracious way to handle this ?
        if np.max(times) - np.min(times) > np.timedelta64(1, 's'):
            raise IncompatibleTimes
        mid_time = (np.max(times) - np.min(times)) / 2 + np.min(times)
        return mid_time


def sub_arrays(proj1, proj2):
    """Substract two DataArrays and combine their attrs."""
    attrs = combine_metadata(proj1.attrs, proj2.attrs)
    if (attrs.get('area') is None
            and proj1.attrs.get('area') is not None
            and proj2.attrs.get('area') is not None):
        raise IncompatibleAreas
    res = proj1 - proj2
    res.attrs = attrs
    return res


class CompositeBase:
    """Base class for all compositors and modifiers.

    A compositor in Satpy is a class that takes in zero or more input
    DataArrays and produces a new DataArray with its own identifier (name).
    The result of a compositor is typically a brand new "product" that
    represents something different than the inputs that went into the
    operation.

    See the :class:`~satpy.composites.ModifierBase` class for information
    on the similar concept of "modifiers".

    """

    def __init__(self, name, prerequisites=None, optional_prerequisites=None, **kwargs):
        """Initialise the compositor."""
        # Required info
        kwargs["name"] = name
        kwargs["prerequisites"] = prerequisites or []
        kwargs["optional_prerequisites"] = optional_prerequisites or []
        self.attrs = kwargs

    @property
    def id(self):
        """Return the DataID of the object."""
        try:
            return self.attrs['_satpy_id']
        except KeyError:
            id_keys = self.attrs.get('_satpy_id_keys', minimal_default_keys_config)
            return DataID(id_keys, **self.attrs)

    def __call__(self, datasets, optional_datasets=None, **info):
        """Generate a composite."""
        raise NotImplementedError()

    def __str__(self):
        """Stringify the object."""
        from pprint import pformat
        return pformat(self.attrs)

    def __repr__(self):
        """Represent the object."""
        from pprint import pformat
        return pformat(self.attrs)

    def apply_modifier_info(self, origin, destination):
        """Apply the modifier info from *origin* to *destination*."""
        o = getattr(origin, 'attrs', origin)
        d = getattr(destination, 'attrs', destination)

        try:
            dataset_keys = self.attrs['_satpy_id'].id_keys.keys()
        except KeyError:
            dataset_keys = ['name', 'modifiers']
        for k in dataset_keys:
            if k == 'modifiers' and k in self.attrs:
                d[k] = self.attrs[k]
            elif d.get(k) is None:
                if self.attrs.get(k) is not None:
                    d[k] = self.attrs[k]
                elif o.get(k) is not None:
                    d[k] = o[k]

    def match_data_arrays(self, data_arrays):
        """Match data arrays so that they can be used together in a composite.

        For the purpose of this method, "can be used together" means:

        - All arrays should have the same dimensions.
        - Either all arrays should have an area, or none should.
        - If all have an area, the areas should be all the same.

        In addition, negligible non-dimensional coordinates are dropped (see
        :meth:`drop_coordinates`) and dask chunks are unified (see
        :func:`satpy.utils.unify_chunks`).

        Args:
            data_arrays (List[arrays]): Arrays to be checked

        Returns:
            data_arrays (List[arrays]):
                Arrays with negligible non-dimensional coordinates removed.

        Raises:
            :class:`IncompatibleAreas`:
                If dimension or areas do not match.
            :class:`ValueError`:
                If some, but not all data arrays lack an area attribute.
        """
        self.check_geolocation(data_arrays)
        new_arrays = self.drop_coordinates(data_arrays)
        new_arrays = list(unify_chunks(*new_arrays))
        return new_arrays

    def drop_coordinates(self, data_arrays):
        """Drop negligible non-dimensional coordinates.

        Drops negligible coordinates if they do not correspond to any
        dimension.  Negligible coordinates are defined in the
        :attr:`NEGLIGIBLE_COORDS` module attribute.

        Args:
            data_arrays (List[arrays]): Arrays to be checked
        """
        new_arrays = []
        for ds in data_arrays:
            drop = [coord for coord in ds.coords
                    if coord not in ds.dims and
                    any([neglible in coord for neglible in NEGLIGIBLE_COORDS])]
            if drop:
                new_arrays.append(ds.drop(drop))
            else:
                new_arrays.append(ds)

        return new_arrays

    def check_geolocation(self, data_arrays):
        """Check that the geolocations of the *data_arrays* are compatible.

        For the purpose of this method, "compatible" means:

        - All arrays should have the same dimensions.
        - Either all arrays should have an area, or none should.
        - If all have an area, the areas should be all the same.

        Args:
            data_arrays (List[arrays]): Arrays to be checked

        Raises:
            :class:`IncompatibleAreas`:
                If dimension or areas do not match.
            :class:`ValueError`:
                If some, but not all data arrays lack an area attribute.
        """
        if len(data_arrays) == 1:
            return

        if 'x' in data_arrays[0].dims and \
                not all(x.sizes['x'] == data_arrays[0].sizes['x']
                        for x in data_arrays[1:]):
            raise IncompatibleAreas("X dimension has different sizes")
        if 'y' in data_arrays[0].dims and \
                not all(x.sizes['y'] == data_arrays[0].sizes['y']
                        for x in data_arrays[1:]):
            raise IncompatibleAreas("Y dimension has different sizes")

        areas = [ds.attrs.get('area') for ds in data_arrays]
        if all(a is None for a in areas):
            return
        if any(a is None for a in areas):
            raise ValueError("Missing 'area' attribute")

        if not all(areas[0] == x for x in areas[1:]):
            LOG.debug("Not all areas are the same in "
                      "'{}'".format(self.attrs['name']))
            raise IncompatibleAreas("Areas are different")


class DifferenceCompositor(CompositeBase):
    """Make the difference of two data arrays."""

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']
        info.update(self.attrs)  # attrs from YAML/__init__
        info.update(attrs)  # overwriting of DataID properties

        proj = projectables[0] - projectables[1]
        proj.attrs = info
        return proj


class RatioCompositor(CompositeBase):
    """Make the ratio of two data arrays."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']

        proj = projectables[0] / projectables[1]
        proj.attrs = info
        return proj


class SumCompositor(CompositeBase):
    """Make the sum of two data arrays."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        info = combine_metadata(*projectables)
        info['name'] = self.attrs['name']

        proj = projectables[0] + projectables[1]
        proj.attrs = info
        return proj


class SingleBandCompositor(CompositeBase):
    """Basic single-band composite builder.

    This preserves all the attributes of the dataset it is derived from.
    """

    @staticmethod
    def _update_missing_metadata(existing_attrs, new_attrs):
        for key, val in new_attrs.items():
            if key not in existing_attrs and val is not None:
                existing_attrs[key] = val

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Build the composite."""
        if len(projectables) != 1:
            raise ValueError("Can't have more than one band in a single-band composite")

        data = projectables[0]
        new_attrs = data.attrs.copy()
        self._update_missing_metadata(new_attrs, attrs)
        resolution = new_attrs.get('resolution', None)
        new_attrs.update(self.attrs)
        if resolution is not None:
            new_attrs['resolution'] = resolution

        return xr.DataArray(data=data.data, attrs=new_attrs,
                            dims=data.dims, coords=data.coords)


class CategoricalDataCompositor(CompositeBase):
    """Compositor used to recategorize categorical data using a look-up-table.

    Each value in the data array will be recategorized to a new category defined in
    the look-up-table using the original value as an index for that look-up-table.

    Example:
        data = [[1, 3, 2], [4, 2, 0]]
        lut = [10, 20, 30, 40, 50]
        res = [[20, 40, 30], [50, 30, 10]]
    """

    def __init__(self, name, lut=None, **kwargs):
        """Get look-up-table used to recategorize data.

        Args:
            lut (list): a list of new categories. The lenght must be greater than the
                        maximum value in the data array that should be recategorized.
        """
        self.lut = np.array(lut)
        super(CategoricalDataCompositor, self).__init__(name, **kwargs)

    def _update_attrs(self, new_attrs):
        """Modify name and add LUT."""
        new_attrs['name'] = self.attrs['name']
        new_attrs['composite_lut'] = list(self.lut)

    @staticmethod
    def _getitem(block, lut):
        return lut[block]

    def __call__(self, projectables, **kwargs):
        """Recategorize the data."""
        if len(projectables) != 1:
            raise ValueError("Can't have more than one dataset for a categorical data composite")

        data = projectables[0].astype(int)
        res = data.data.map_blocks(self._getitem, self.lut, dtype=self.lut.dtype)

        new_attrs = data.attrs.copy()
        self._update_attrs(new_attrs)

        return xr.DataArray(res, dims=data.dims, attrs=new_attrs, coords=data.coords)


class GenericCompositor(CompositeBase):
    """Basic colored composite builder."""

    modes = {1: 'L', 2: 'LA', 3: 'RGB', 4: 'RGBA'}

    def __init__(self, name, common_channel_mask=True, **kwargs):
        """Collect custom configuration values.

        Args:
            common_channel_mask (bool): If True, mask all the channels with
                a mask that combines all the invalid areas of the given data.

        """
        self.common_channel_mask = common_channel_mask
        super(GenericCompositor, self).__init__(name, **kwargs)

    @classmethod
    def infer_mode(cls, data_arr):
        """Guess at the mode for a particular DataArray."""
        if 'mode' in data_arr.attrs:
            return data_arr.attrs['mode']
        if 'bands' not in data_arr.dims:
            return cls.modes[1]
        if 'bands' in data_arr.coords and isinstance(data_arr.coords['bands'][0].item(), str):
            return ''.join(data_arr.coords['bands'].values)
        return cls.modes[data_arr.sizes['bands']]

    def _concat_datasets(self, projectables, mode):
        try:
            data = xr.concat(projectables, 'bands', coords='minimal')
            data['bands'] = list(mode)
        except ValueError as e:
            LOG.debug("Original exception for incompatible areas: {}".format(str(e)))
            raise IncompatibleAreas

        return data

    def _get_sensors(self, projectables):
        sensor = set()
        for projectable in projectables:
            current_sensor = projectable.attrs.get("sensor", None)
            if current_sensor:
                if isinstance(current_sensor, (str, bytes)):
                    sensor.add(current_sensor)
                else:
                    sensor |= current_sensor
        if len(sensor) == 0:
            sensor = None
        elif len(sensor) == 1:
            sensor = list(sensor)[0]
        return sensor

    def __call__(self, projectables, nonprojectables=None, **attrs):
        """Build the composite."""
        if 'deprecation_warning' in self.attrs:
            warnings.warn(self.attrs['deprecation_warning'], UserWarning)
            self.attrs.pop('deprecation_warning', None)
        num = len(projectables)
        mode = attrs.get('mode')
        if mode is None:
            # num may not be in `self.modes` so only check if we need to
            mode = self.modes[num]
        if len(projectables) > 1:
            projectables = self.match_data_arrays(projectables)
            data = self._concat_datasets(projectables, mode)
            # Skip masking if user wants it or a specific alpha channel is given.
            if self.common_channel_mask and mode[-1] != 'A':
                data = data.where(data.notnull().all(dim='bands'))
        else:
            data = projectables[0]

        # if inputs have a time coordinate that may differ slightly between
        # themselves then find the mid time and use that as the single
        # time coordinate value
        if len(projectables) > 1:
            time = check_times(projectables)
            if time is not None and 'time' in data.dims:
                data['time'] = [time]

        new_attrs = combine_metadata(*projectables)
        # remove metadata that shouldn't make sense in a composite
        new_attrs["wavelength"] = None
        new_attrs.pop("units", None)
        new_attrs.pop('calibration', None)
        new_attrs.pop('modifiers', None)

        new_attrs.update({key: val
                          for (key, val) in attrs.items()
                          if val is not None})
        resolution = new_attrs.get('resolution', None)
        new_attrs.update(self.attrs)
        if resolution is not None:
            new_attrs['resolution'] = resolution
        new_attrs["sensor"] = self._get_sensors(projectables)
        new_attrs["mode"] = mode

        return xr.DataArray(data=data.data, attrs=new_attrs,
                            dims=data.dims, coords=data.coords)


class FillingCompositor(GenericCompositor):
    """Make a regular RGB, filling the RGB bands with the first provided dataset's values."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        projectables[1] = projectables[1].fillna(projectables[0])
        projectables[2] = projectables[2].fillna(projectables[0])
        projectables[3] = projectables[3].fillna(projectables[0])
        return super(FillingCompositor, self).__call__(projectables[1:], **info)


class Filler(GenericCompositor):
    """Fix holes in projectable 1 with data from projectable 2."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        filled_projectable = projectables[0].fillna(projectables[1])
        return super(Filler, self).__call__([filled_projectable], **info)


class MultiFiller(SingleBandCompositor):
    """Fix holes in projectable 1 with data from the next projectables."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        filled_projectable = projectables[0]
        for next_projectable in projectables[1:]:
            filled_projectable = filled_projectable.fillna(next_projectable)
        if 'optional_datasets' in info.keys():
            for next_projectable in info['optional_datasets']:
                filled_projectable = filled_projectable.fillna(next_projectable)

        return super().__call__([filled_projectable], **info)


class RGBCompositor(GenericCompositor):
    """Make a composite from three color bands (deprecated)."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        warnings.warn("RGBCompositor is deprecated, use GenericCompositor instead.", DeprecationWarning)
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(projectables),))
        return super(RGBCompositor, self).__call__(projectables, **info)


class ColormapCompositor(GenericCompositor):
    """A compositor that uses colormaps.

    .. warning::

        Deprecated since Satpy 0.39.

    This compositor is deprecated.  To apply a colormap, use a
    :class:`SingleBandCompositor` composite with a
    :func:`~satpy.enhancements.colorize` or
    :func:`~satpy.enhancements.palettize` enhancement instead.
    For example, to make a ``cloud_top_height`` composite based on a dataset
    ``ctth_alti`` palettized by ``ctth_alti_pal``, the composite would be::

      cloud_top_height:
        compositor: !!python/name:satpy.composites.SingleBandCompositor
        prerequisites:
        - ctth_alti
        tandard_name: cloud_top_height

    and the enhancement::

      cloud_top_height:
        standard_name: cloud_top_height
        operations:
        - name: palettize
          method: !!python/name:satpy.enhancements.palettize
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
        res.attrs['_FillValue'] = np.nan
        return res

    @staticmethod
    def _get_mask_from_data(data):
        fill_value = data.attrs.get('_FillValue', np.nan)
        if np.isnan(fill_value):
            mask = data.notnull()
        else:
            mask = data != data.attrs['_FillValue']
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
                                       new_axis=2, chunks=list(channels.chunks) + [palette.shape[1]])
        return [channels[:, :, i] for i in range(channels.shape[2])]


def _insert_palette_colors(channels, palette):
    channels = palette[channels]
    return channels


class DayNightCompositor(GenericCompositor):
    """A compositor that blends day data with night data.

    Using the `day_night` flag it is also possible to provide only a day product
    or only a night product and mask out (make transparent) the opposite portion
    of the image (night or day). See the documentation below for more details.
    """

    def __init__(self, name, lim_low=85., lim_high=88., day_night="day_night", include_alpha=True, **kwargs):
        """Collect custom configuration values.

        Args:
            lim_low (float): lower limit of Sun zenith angle for the
                             blending of the given channels
            lim_high (float): upper limit of Sun zenith angle for the
                             blending of the given channels
            day_night (string): "day_night" means both day and night portions will be kept
                                "day_only" means only day portion will be kept
                                "night_only" means only night portion will be kept
            include_alpha (bool): This only affects the "day only" or "night only" result.
                                  True means an alpha band will be added to the output image for transparency.
                                  False means the output is a single-band image with undesired pixels being masked out
                                  (replaced with NaNs).

        """
        self.lim_low = lim_low
        self.lim_high = lim_high
        self.day_night = day_night
        self.include_alpha = include_alpha
        super(DayNightCompositor, self).__init__(name, **kwargs)

    def __call__(self, projectables, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        # At least one composite is requested.
        foreground_data = projectables[0]

        lim_low = np.cos(np.deg2rad(self.lim_low))
        lim_high = np.cos(np.deg2rad(self.lim_high))
        try:
            coszen = np.cos(np.deg2rad(projectables[2 if self.day_night == "day_night" else 1]))
        except IndexError:
            from satpy.modifiers.angles import get_cos_sza
            LOG.debug("Computing sun zenith angles.")
            # Get chunking that matches the data
            coszen = get_cos_sza(foreground_data)
        # Calculate blending weights
        coszen -= np.min((lim_high, lim_low))
        coszen /= np.abs(lim_low - lim_high)
        coszen = coszen.clip(0, 1)

        # Apply enhancements
        foreground_data = enhance2dataset(foreground_data)

        if "only" in self.day_night:
            # Only one portion (day or night) is selected. One composite is requested.
            # Add alpha band to single L/RGB composite to make the masked-out portion transparent when needed
            # L -> LA
            # RGB -> RGBA
            if self.include_alpha:
                foreground_data = add_alpha_bands(foreground_data)

            # Use coszen to determine the undesired pixels and replace the coszen of these pixels with NaN.
            # They will be passed to subsequent calculation and therefore make sure those pixels being masked-out.
            if "day" in self.day_night:
                coszen = da.where(coszen != 0, coszen, np.nan).compute()
            else:
                coszen = da.where(coszen != 1, coszen, np.nan).compute()

            # No need to replace missing channel data with zeros
            # Get metadata
            attrs = foreground_data.attrs.copy()

            # Determine the composite position
            day_data = foreground_data if "day" in self.day_night else 0
            night_data = foreground_data if "night" in self.day_night else 0

        else:
            # Both day and night portions are selected. Two composites are requested. Get the second one merged.
            background_data = projectables[1]

            # Apply enhancements
            background_data = enhance2dataset(background_data)

            # Adjust bands so that they match
            # L/RGB -> RGB/RGB
            # LA/RGB -> RGBA/RGBA
            # RGB/RGBA -> RGBA/RGBA
            foreground_data = add_bands(foreground_data, background_data['bands'])
            background_data = add_bands(background_data, foreground_data['bands'])

            # Replace missing channel data with zeros
            foreground_data = zero_missing_data(foreground_data, background_data)
            background_data = zero_missing_data(background_data, foreground_data)

            # Get merged metadata
            attrs = combine_metadata(foreground_data, background_data)

            # Determine the composite position
            day_data = foreground_data
            night_data = background_data

        # Blend the two images together
        day_portion = coszen * day_data
        night_portion = (1 - coszen) * night_data
        data = night_portion + day_portion
        data.attrs = attrs

        # Split to separate bands so the mode is correct
        data = [data.sel(bands=b) for b in data['bands']]

        return super(DayNightCompositor, self).__call__(data, **kwargs)


def add_alpha_bands(data):
    """Only used for DayNightCompositor.

    Add an alpha band to L or RGB composite as prerequisites for the following band matching
    to make the masked-out area transparent.
    """
    if 'A' not in data['bands'].data:
        new_data = [data.sel(bands=band) for band in data['bands'].data]
        # Create alpha band based on a copy of the first "real" band
        alpha = new_data[0].copy()
        alpha.data = da.ones((data.sizes['y'],
                              data.sizes['x']),
                             chunks=new_data[0].chunks)
        # Rename band to indicate it's alpha
        alpha['bands'] = 'A'
        new_data.append(alpha)
        new_data = xr.concat(new_data, dim='bands')
        new_data.attrs['mode'] = data.attrs['mode'] + 'A'
        data = new_data
    return data


def enhance2dataset(dset, convert_p=False):
    """Return the enhancement dataset *dset* as an array.

    If `convert_p` is True, enhancements generating a P mode will be converted to RGB or RGBA.
    """
    attrs = dset.attrs
    data = _get_data_from_enhanced_image(dset, convert_p)
    data.attrs = attrs
    # remove 'mode' if it is specified since it may have been updated
    data.attrs.pop('mode', None)
    # update mode since it may have changed (colorized/palettize)
    data.attrs['mode'] = GenericCompositor.infer_mode(data)
    return data


def _get_data_from_enhanced_image(dset, convert_p):
    img = get_enhanced_image(dset)
    if convert_p and img.mode == 'P':
        img = _apply_palette_to_image(img)
    if img.mode != 'P':
        data = img.data.clip(0.0, 1.0)
    else:
        data = img.data
    return data


def _apply_palette_to_image(img):
    if len(img.palette[0]) == 3:
        img = img.convert('RGB')
    elif len(img.palette[0]) == 4:
        img = img.convert('RGBA')
    return img


def add_bands(data, bands):
    """Add bands so that they match *bands*."""
    # Add R, G and B bands, remove L band
    bands = bands.compute()
    if 'P' in data['bands'].data or 'P' in bands.data:
        raise NotImplementedError('Cannot mix datasets of mode P with other datasets at the moment.')
    if 'L' in data['bands'].data and 'R' in bands.data:
        lum = data.sel(bands='L')
        # Keep 'A' if it was present
        if 'A' in data['bands']:
            alpha = data.sel(bands='A')
            new_data = (lum, lum, lum, alpha)
            new_bands = ['R', 'G', 'B', 'A']
            mode = 'RGBA'
        else:
            new_data = (lum, lum, lum)
            new_bands = ['R', 'G', 'B']
            mode = 'RGB'
        data = xr.concat(new_data, dim='bands', coords={'bands': new_bands})
        data['bands'] = new_bands
        data.attrs['mode'] = mode
    # Add alpha band
    if 'A' not in data['bands'].data and 'A' in bands.data:
        new_data = [data.sel(bands=band) for band in data['bands'].data]
        # Create alpha band based on a copy of the first "real" band
        alpha = new_data[0].copy()
        alpha.data = da.ones((data.sizes['y'],
                              data.sizes['x']),
                             chunks=new_data[0].chunks)
        # Rename band to indicate it's alpha
        alpha['bands'] = 'A'
        new_data.append(alpha)
        new_data = xr.concat(new_data, dim='bands')
        new_data.attrs['mode'] = data.attrs['mode'] + 'A'
        data = new_data
    return data


def zero_missing_data(data1, data2):
    """Replace NaN values with zeros in data1 if the data is valid in data2."""
    nans = np.logical_and(np.isnan(data1), np.logical_not(np.isnan(data2)))
    return data1.where(~nans, 0)


class RealisticColors(GenericCompositor):
    """Create a realistic colours composite for SEVIRI."""

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        vis06 = projectables[0]
        vis08 = projectables[1]
        hrv = projectables[2]

        try:
            ch3 = 3 * hrv - vis06 - vis08
            ch3.attrs = hrv.attrs
        except ValueError:
            raise IncompatibleAreas

        ndvi = (vis08 - vis06) / (vis08 + vis06)
        ndvi = np.where(ndvi < 0, 0, ndvi)

        ch1 = ndvi * vis06 + (1 - ndvi) * vis08
        ch1.attrs = vis06.attrs
        ch2 = ndvi * vis08 + (1 - ndvi) * vis06
        ch2.attrs = vis08.attrs

        res = super(RealisticColors, self).__call__((ch1, ch2, ch3),
                                                    *args, **kwargs)
        return res


class CloudCompositor(GenericCompositor):
    """Detect clouds based on thresholding and use it as a mask for compositing."""

    def __init__(self, name, transition_min=258.15, transition_max=298.15,
                 transition_gamma=3.0, **kwargs):
        """Collect custom configuration values.

        Args:
            transition_min (float): Values below or equal to this are
                                    clouds -> opaque white
            transition_max (float): Values above this are
                                    cloud free -> transparent
            transition_gamma (float): Gamma correction to apply at the end

        """
        self.transition_min = transition_min
        self.transition_max = transition_max
        self.transition_gamma = transition_gamma
        super(CloudCompositor, self).__init__(name, **kwargs)

    def __call__(self, projectables, **kwargs):
        """Generate the composite."""
        data = projectables[0]

        # Default to rough IR thresholds
        # Values below or equal to this are clouds -> opaque white
        tr_min = self.transition_min
        # Values above this are cloud free -> transparent
        tr_max = self.transition_max
        # Gamma correction
        gamma = self.transition_gamma

        slope = 1 / (tr_min - tr_max)
        offset = 1 - slope * tr_min

        alpha = data.where(data > tr_min, 1.)
        alpha = alpha.where(data <= tr_max, 0.)
        alpha = alpha.where((data <= tr_min) | (data > tr_max), slope * data + offset)

        # gamma adjustment
        alpha **= gamma
        res = super(CloudCompositor, self).__call__((data, alpha), **kwargs)
        return res


class RatioSharpenedRGB(GenericCompositor):
    """Sharpen RGB bands with ratio of a high resolution band to a lower resolution version.

    Any pixels where the ratio is computed to be negative or infinity, it is
    reset to 1. Additionally, the ratio is limited to 1.5 on the high end to
    avoid high changes due to small discrepancies in instrument detector
    footprint. Note that the input data to this compositor must already be
    resampled so all data arrays are the same shape.

    Example::

        R_lo -  1000m resolution - shape=(2000, 2000)
        G - 1000m resolution - shape=(2000, 2000)
        B - 1000m resolution - shape=(2000, 2000)
        R_hi -  500m resolution - shape=(4000, 4000)

        ratio = R_hi / R_lo
        new_R = R_hi
        new_G = G * ratio
        new_B = B * ratio

    """

    def __init__(self, *args, **kwargs):
        """Instanciate the ration sharpener."""
        self.high_resolution_color = kwargs.pop("high_resolution_band", "red")
        if self.high_resolution_color not in ['red', 'green', 'blue', None]:
            raise ValueError("RatioSharpenedRGB.high_resolution_band must "
                             "be one of ['red', 'green', 'blue', None]. Not "
                             "'{}'".format(self.high_resolution_color))
        super(RatioSharpenedRGB, self).__init__(*args, **kwargs)

    def __call__(self, datasets, optional_datasets=None, **info):
        """Sharpen low resolution datasets by multiplying by the ratio of ``high_res / low_res``.

        The resulting RGB has the units attribute removed.
        """
        if len(datasets) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(datasets), ))
        if not all(x.shape == datasets[0].shape for x in datasets[1:]) or \
                (optional_datasets and
                 optional_datasets[0].shape != datasets[0].shape):
            raise IncompatibleAreas('RatioSharpening requires datasets of '
                                    'the same size. Must resample first.')

        optional_datasets = tuple() if optional_datasets is None else optional_datasets
        datasets = self.match_data_arrays(datasets + optional_datasets)
        red, green, blue, new_attrs = self._get_and_sharpen_rgb_data_arrays_and_meta(datasets, optional_datasets)
        combined_info = self._combined_sharpened_info(info, new_attrs)
        res = super(RatioSharpenedRGB, self).__call__((red, green, blue,), **combined_info)
        res.attrs.pop("units", None)
        return res

    def _get_and_sharpen_rgb_data_arrays_and_meta(self, datasets, optional_datasets):
        new_attrs = {}
        low_res_red = datasets[0]
        low_res_green = datasets[1]
        low_res_blue = datasets[2]
        if optional_datasets and self.high_resolution_color is not None:
            LOG.debug("Sharpening image with high resolution {} band".format(self.high_resolution_color))
            high_res = datasets[3]
            if 'rows_per_scan' in high_res.attrs:
                new_attrs.setdefault('rows_per_scan', high_res.attrs['rows_per_scan'])
            new_attrs.setdefault('resolution', high_res.attrs['resolution'])
            low_res_colors = ['red', 'green', 'blue']
            low_resolution_index = low_res_colors.index(self.high_resolution_color)
        else:
            LOG.debug("No sharpening band specified for ratio sharpening")
            high_res = None
            low_resolution_index = 0

        if high_res is not None:
            low_res = (low_res_red, low_res_green, low_res_blue)[low_resolution_index]
            ratio = da.map_blocks(
                _get_sharpening_ratio,
                high_res.data,
                low_res.data,
                meta=np.array((), dtype=high_res.dtype),
                dtype=high_res.dtype,
                chunks=high_res.chunks,
            )
            with xr.set_options(keep_attrs=True):
                low_res_red = high_res if low_resolution_index == 0 else low_res_red * ratio
                low_res_green = high_res if low_resolution_index == 1 else low_res_green * ratio
                low_res_blue = high_res if low_resolution_index == 2 else low_res_blue * ratio
        return low_res_red, low_res_green, low_res_blue, new_attrs

    def _combined_sharpened_info(self, info, new_attrs):
        combined_info = {}
        combined_info.update(info)
        combined_info.update(new_attrs)
        # Update that information with configured information (including name)
        combined_info.update(self.attrs)
        # Force certain pieces of metadata that we *know* to be true
        combined_info.setdefault("standard_name", "true_color")
        return combined_info


def _get_sharpening_ratio(high_res, low_res):
    ratio = high_res / low_res
    # make ratio a no-op (multiply by 1) where the ratio is NaN, infinity,
    # or it is negative.
    ratio[~np.isfinite(ratio) | (ratio < 0)] = 1.0
    # we don't need ridiculously high ratios, they just make bright pixels
    np.clip(ratio, 0, 1.5, out=ratio)
    return ratio


def _mean4(data, offset=(0, 0), block_id=None):
    rows, cols = data.shape
    # we assume that the chunks except the first ones are aligned
    if block_id[0] == 0:
        row_offset = offset[0] % 2
    else:
        row_offset = 0
    if block_id[1] == 0:
        col_offset = offset[1] % 2
    else:
        col_offset = 0
    row_after = (row_offset + rows) % 2
    col_after = (col_offset + cols) % 2
    pad = ((row_offset, row_after), (col_offset, col_after))

    rows2 = rows + row_offset + row_after
    cols2 = cols + col_offset + col_after

    av_data = np.pad(data, pad, 'edge')
    new_shape = (int(rows2 / 2.), 2, int(cols2 / 2.), 2)
    with np.errstate(invalid='ignore'):
        data_mean = np.nanmean(av_data.reshape(new_shape), axis=(1, 3))
    data_mean = np.repeat(np.repeat(data_mean, 2, axis=0), 2, axis=1)
    data_mean = data_mean[row_offset:row_offset + rows, col_offset:col_offset + cols]
    return data_mean


class SelfSharpenedRGB(RatioSharpenedRGB):
    """Sharpen RGB with ratio of a band with a strided-version of itself.

    Example::

        R -  500m resolution - shape=(4000, 4000)
        G - 1000m resolution - shape=(2000, 2000)
        B - 1000m resolution - shape=(2000, 2000)

        ratio = R / four_element_average(R)
        new_R = R
        new_G = G * ratio
        new_B = B * ratio

    """

    @staticmethod
    def four_element_average_dask(d):
        """Average every 4 elements (2x2) in a 2D array."""
        try:
            offset = d.attrs['area'].crop_offset
        except (KeyError, AttributeError):
            offset = (0, 0)

        res = d.data.map_blocks(_mean4, offset=offset, dtype=d.dtype)
        return xr.DataArray(res, attrs=d.attrs, dims=d.dims, coords=d.coords)

    def __call__(self, datasets, optional_datasets=None, **attrs):
        """Generate the composite."""
        colors = ['red', 'green', 'blue']
        if self.high_resolution_color not in colors:
            raise ValueError("SelfSharpenedRGB requires at least one high resolution band, not "
                             "'{}'".format(self.high_resolution_color))

        high_res = datasets[colors.index(self.high_resolution_color)]
        high_mean = self.four_element_average_dask(high_res)
        red = high_mean if self.high_resolution_color == 'red' else datasets[0]
        green = high_mean if self.high_resolution_color == 'green' else datasets[1]
        blue = high_mean if self.high_resolution_color == 'blue' else datasets[2]
        return super(SelfSharpenedRGB, self).__call__((red, green, blue), optional_datasets=(high_res,), **attrs)


class LuminanceSharpeningCompositor(GenericCompositor):
    """Create a high resolution composite by sharpening a low resolution using high resolution luminance.

    This is done by converting to YCbCr colorspace, replacing Y, and convertin back to RGB.
    """

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        from trollimage.image import rgb2ycbcr, ycbcr2rgb
        projectables = self.match_data_arrays(projectables)
        luminance = projectables[0].copy()
        luminance /= 100.
        # Limit between min(luminance) ... 1.0
        luminance = da.where(luminance > 1., 1., luminance)

        # Get the enhanced version of the composite to be sharpened
        rgb_img = enhance2dataset(projectables[1])

        # This all will be eventually replaced with trollimage convert() method
        # ycbcr_img = rgb_img.convert('YCbCr')
        # ycbcr_img.data[0, :, :] = luminance
        # rgb_img = ycbcr_img.convert('RGB')

        # Replace luminance of the IR composite
        y__, cb_, cr_ = rgb2ycbcr(rgb_img.data[0, :, :],
                                  rgb_img.data[1, :, :],
                                  rgb_img.data[2, :, :])

        r__, g__, b__ = ycbcr2rgb(luminance, cb_, cr_)
        y_size, x_size = r__.shape
        r__ = da.reshape(r__, (1, y_size, x_size))
        g__ = da.reshape(g__, (1, y_size, x_size))
        b__ = da.reshape(b__, (1, y_size, x_size))

        rgb_img.data = da.vstack((r__, g__, b__))
        return super(LuminanceSharpeningCompositor, self).__call__(rgb_img, *args, **kwargs)


class SandwichCompositor(GenericCompositor):
    """Make a sandwich product."""

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        luminance = projectables[0]
        luminance = luminance / 100.
        # Limit between min(luminance) ... 1.0
        luminance = luminance.clip(max=1.)

        # Get the enhanced version of the RGB composite to be sharpened
        rgb_img = enhance2dataset(projectables[1])
        # Ignore alpha band when applying luminance
        rgb_img = rgb_img.where(rgb_img.bands == 'A', rgb_img * luminance)
        return super(SandwichCompositor, self).__call__(rgb_img, *args, **kwargs)


# TODO: Turn this into a weighted RGB compositor
class NaturalEnh(GenericCompositor):
    """Enhanced version of natural color composite by Simon Proud.

    Args:
        ch16_w (float): weight for red channel (1.6 um). Default: 1.3
        ch08_w (float): weight for green channel (0.8 um). Default: 2.5
        ch06_w (float): weight for blue channel (0.6 um). Default: 2.2

    """

    def __init__(self, name, ch16_w=1.3, ch08_w=2.5, ch06_w=2.2,
                 *args, **kwargs):
        """Initialize the class."""
        self.ch06_w = ch06_w
        self.ch08_w = ch08_w
        self.ch16_w = ch16_w
        super(NaturalEnh, self).__init__(name, *args, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        ch16 = projectables[0]
        ch08 = projectables[1]
        ch06 = projectables[2]

        ch1 = self.ch16_w * ch16 + self.ch08_w * ch08 + self.ch06_w * ch06
        ch1.attrs = ch16.attrs
        ch2 = ch08
        ch3 = ch06

        return super(NaturalEnh, self).__call__((ch1, ch2, ch3),
                                                *args, **kwargs)


class StaticImageCompositor(GenericCompositor, DataDownloadMixin):
    """A compositor that loads a static image from disk.

    Environment variables in the filename are automatically expanded.

    """

    def __init__(self, name, filename=None, url=None, known_hash=None, area=None,
                 **kwargs):
        """Collect custom configuration values.

        Args:
            filename (str): Name to use when storing and referring to the file
                in the ``data_dir`` cache. If ``url`` is provided (preferred),
                then this is used as the filename in the cache and will be
                appended to ``<data_dir>/composites/<class_name>/``. If
                ``url`` is provided and ``filename`` is not then the
                ``filename`` will be guessed from the ``url``.
                If ``url`` is not provided, then it is assumed ``filename``
                refers to a local file. If the ``filename`` does not come with
                an absolute path, ``data_dir`` will be used as the directory path.
                Environment variables are expanded.
            url (str): URL to remote file. When the composite is created the
                file will be downloaded and cached in Satpy's ``data_dir``.
                Environment variables are expanded.
            known_hash (str or None): Hash of the remote file used to verify
                a successful download. If not provided then the download will
                not be verified. See :func:`satpy.aux_download.register_file`
                for more information.
            area (str): Name of area definition for the image.  Optional
                for images with built-in area definitions (geotiff).

        Use cases:
            1. url + no filename:
               Satpy determines the filename based on the filename in the URL,
               then downloads the URL, and saves it to <data_dir>/<filename>.
               If the file already exists and known_hash is also provided, then the pooch
               library compares the hash of the file to the known_hash. If it does not
               match, then the URL is re-downloaded. If it matches then no download.
            2. url + relative filename:
               Same as case 1 but filename is already provided so download goes to
               <data_dir>/<filename>. Same hashing behavior. This does not check for an
               absolute path.
            3. No url + absolute filename:
               No download, filename is passed directly to generic_image reader. No hashing
               is done.
            4. No url + relative filename:
               Check if <data_dir>/<filename> exists. If it does then make filename an
               absolute path. If it doesn't, then keep it as is and let the exception at
               the bottom of the method get raised.
        """
        filename, url = self._get_cache_filename_and_url(filename, url)
        self._cache_filename = filename
        self._url = url
        self._known_hash = known_hash
        self.area = None
        if area is not None:
            from satpy.resample import get_area_def
            self.area = get_area_def(area)

        super(StaticImageCompositor, self).__init__(name, **kwargs)
        cache_keys = self.register_data_files([])
        self._cache_key = cache_keys[0]

    @staticmethod
    def _check_relative_filename(filename):
        data_dir = satpy.config.get('data_dir')
        path = os.path.join(data_dir, filename)

        return path if os.path.exists(path) else filename

    def _get_cache_filename_and_url(self, filename, url):
        if filename:
            filename = os.path.expanduser(os.path.expandvars(filename))

            if not os.path.isabs(filename) and not url:
                filename = self._check_relative_filename(filename)

        if url:
            url = os.path.expandvars(url)
            if not filename:
                filename = os.path.basename(url)
        elif not filename or not os.path.isabs(filename):
            raise ValueError("StaticImageCompositor needs a remote 'url', "
                             "or absolute path to 'filename', "
                             "or an existing 'filename' relative to Satpy's 'data_dir'.")

        return filename, url

    def register_data_files(self, data_files):
        """Tell Satpy about files we may want to download."""
        if os.path.isabs(self._cache_filename):
            return [None]
        return super().register_data_files([{
            'url': self._url,
            'known_hash': self._known_hash,
            'filename': self._cache_filename,
        }])

    def _retrieve_data_file(self):
        from satpy.aux_download import retrieve
        if os.path.isabs(self._cache_filename):
            return self._cache_filename
        return retrieve(self._cache_key)

    def __call__(self, *args, **kwargs):
        """Call the compositor."""
        from satpy import Scene
        local_file = self._retrieve_data_file()
        scn = Scene(reader='generic_image', filenames=[local_file])
        scn.load(['image'])
        img = scn['image']
        # use compositor parameters as extra metadata
        # most important: set 'name' of the image
        img.attrs.update(self.attrs)
        # Check for proper area definition.  Non-georeferenced images
        # do not have `area` in the attributes
        if 'area' not in img.attrs:
            if self.area is None:
                raise AttributeError("Area definition needs to be configured")
            img.attrs['area'] = self.area
        img.attrs['sensor'] = None
        img.attrs['mode'] = ''.join(img.bands.data)
        img.attrs.pop('modifiers', None)
        img.attrs.pop('calibration', None)
        # Add start time if not present in the filename
        if 'start_time' not in img.attrs or not img.attrs['start_time']:
            import datetime as dt
            img.attrs['start_time'] = dt.datetime.utcnow()
        if 'end_time' not in img.attrs or not img.attrs['end_time']:
            import datetime as dt
            img.attrs['end_time'] = dt.datetime.utcnow()

        return img


class BackgroundCompositor(GenericCompositor):
    """A compositor that overlays one composite on top of another."""

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        projectables = self.match_data_arrays(projectables)
        # Get enhanced datasets
        foreground = enhance2dataset(projectables[0], convert_p=True)
        background = enhance2dataset(projectables[1], convert_p=True)
        # Adjust bands so that they match
        # L/RGB -> RGB/RGB
        # LA/RGB -> RGBA/RGBA
        # RGB/RGBA -> RGBA/RGBA
        foreground = add_bands(foreground, background['bands'])
        background = add_bands(background, foreground['bands'])

        attrs = self._combine_metadata_with_mode_and_sensor(foreground, background)
        data = self._get_merged_image_data(foreground, background)
        res = super(BackgroundCompositor, self).__call__(data, **kwargs)
        res.attrs.update(attrs)
        return res

    def _combine_metadata_with_mode_and_sensor(self,
                                               foreground: xr.DataArray,
                                               background: xr.DataArray
                                               ) -> dict:
        # Get merged metadata
        attrs = combine_metadata(foreground, background)
        # 'mode' is no longer valid after we've remove the 'A'
        # let the base class __call__ determine mode
        attrs.pop("mode", None)
        if attrs.get('sensor') is None:
            # sensor can be a set
            attrs['sensor'] = self._get_sensors([foreground, background])
        return attrs

    @staticmethod
    def _get_merged_image_data(foreground: xr.DataArray,
                               background: xr.DataArray
                               ) -> list[xr.DataArray]:
        if 'A' in foreground.attrs['mode']:
            # Use alpha channel as weight and blend the two composites
            alpha = foreground.sel(bands='A')
            data = []
            # NOTE: there's no alpha band in the output image, it will
            # be added by the data writer
            for band in foreground.mode[:-1]:
                fg_band = foreground.sel(bands=band)
                bg_band = background.sel(bands=band)
                chan = (fg_band * alpha + bg_band * (1 - alpha))
                chan = xr.where(chan.isnull(), bg_band, chan)
                data.append(chan)
        else:
            data_arr = xr.where(foreground.isnull(), background, foreground)
            # Split to separate bands so the mode is correct
            data = [data_arr.sel(bands=b) for b in data_arr['bands']]

        return data


class MaskingCompositor(GenericCompositor):
    """A compositor that masks e.g. IR 10.8 channel data using cloud products from NWC SAF."""

    _supported_modes = {"LA", "RGBA"}

    def __init__(self, name, transparency=None, conditions=None, mode="LA",
                 **kwargs):
        """Collect custom configuration values.

        Kwargs:
            transparency (dict): transparency for each cloud type as
                                 key-value pairs in a dictionary.
                                 Will be converted to `conditions`.
                                 DEPRECATED.
            conditions (list): list of three items determining the masking
                               settings.
            mode (str, optional): Image mode to return.  For single-band input,
                                  this shall be "LA" (default) or "RGBA".  For
                                  multi-band input, this argument is ignored
                                  as the result is always RGBA.

        Each condition in *conditions* consists of three items:

        - `method`: Numpy method name.  The following are supported
            operations: `less`, `less_equal`, `equal`, `greater_equal`,
            `greater`, `not_equal`, `isnan`, `isfinite`, `isinf`,
            `isneginf`, or `isposinf`.
        - `value`: threshold value of the *mask* applied with the
            operator.  Can be a string, in which case the corresponding
            value will be determined from `flag_meanings` and
            `flag_values` attributes of the mask.
            NOTE: the `value` should not be given to 'is*` methods.
        - `transparency`: transparency from interval [0 ... 100] used
            for the method/threshold. Value of 100 is fully transparent.

        Example::

          >>> conditions = [{'method': 'greater_equal', 'value': 0,
                             'transparency': 100},
                            {'method': 'greater_equal', 'value': 1,
                             'transparency': 80},
                            {'method': 'greater_equal', 'value': 2,
                             'transparency': 0},
                            {'method': 'isnan',
                             'transparency': 100}]
          >>> compositor = MaskingCompositor("masking compositor",
                                             transparency=transparency)
          >>> result = compositor([data, mask])


        This will set transparency of `data` based on the values in
        the `mask` dataset.  Locations where `mask` has values of `0`
        will be fully transparent, locations with `1` will be
        semi-transparent and locations with `2` will be fully visible
        in the resulting image.  In the end all `NaN` areas in the mask are
        set to full transparency.  All the unlisted locations will be
        visible.

        The transparency is implemented by adding an alpha layer to
        the composite.  The locations with transparency of `100` will
        be set to NaN in the data.  If the input `data` contains an
        alpha channel, it will be discarded.

        """
        if transparency:
            LOG.warning("Using 'transparency' is deprecated in "
                        "MaskingCompositor, use 'conditions' instead.")
            self.conditions = []
            for key, transp in transparency.items():
                self.conditions.append({'method': 'equal',
                                        'value': key,
                                        'transparency': transp})
            LOG.info("Converted 'transparency' to 'conditions': %s",
                     str(self.conditions))
        else:
            self.conditions = conditions
        if self.conditions is None:
            raise ValueError("Masking conditions not defined.")
        if mode not in self._supported_modes:
            raise ValueError(f"Invalid mode {mode!s}.  Supported modes: " +
                             ", ".join(self._supported_modes))
        self.mode = mode

        super(MaskingCompositor, self).__init__(name, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        """Call the compositor."""
        if len(projectables) != 2:
            raise ValueError("Expected 2 datasets, got %d" % (len(projectables),))
        projectables = self.match_data_arrays(projectables)
        data_in = projectables[0]
        mask_in = projectables[1]

        alpha_attrs = data_in.attrs.copy()
        data = self._select_data_bands(data_in)

        alpha = self._get_alpha_bands(data, mask_in, alpha_attrs)
        data.append(alpha)

        res = super(MaskingCompositor, self).__call__(data, **kwargs)
        return res

    def _get_mask(self, method, value, mask_data):
        """Get mask array from *mask_data* using *method* and threshold *value*.

        The *method* is the name of a numpy function.

        """
        if method not in MASKING_COMPOSITOR_METHODS:
            raise AttributeError("Unsupported Numpy method %s, use one of %s",
                                 method, str(MASKING_COMPOSITOR_METHODS))

        func = getattr(np, method)

        if value is None:
            return func(mask_data)
        return func(mask_data, value)

    def _set_data_nans(self, data, mask, attrs):
        """Set *data* to nans where *mask* is True.

        The attributes *attrs** will be written to each band in *data*.

        """
        for i, dat in enumerate(data):
            data[i] = xr.where(mask, np.nan, dat)
            data[i].attrs = attrs

        return data

    def _select_data_bands(self, data_in):
        """Select data to be composited from input data.

        From input data, select the bands that need to have masking applied.
        """
        if 'bands' in data_in.dims:
            return [data_in.sel(bands=b) for b in data_in['bands'] if b != 'A']
        if self.mode == "RGBA":
            return [data_in, data_in, data_in]
        return [data_in]

    def _get_alpha_bands(self, data, mask_in, alpha_attrs):
        """Get alpha bands.

        From input data, masks, and attributes, get alpha band.
        """
        # Create alpha band
        mask_data = mask_in.data
        alpha = da.ones((data[0].sizes['y'],
                         data[0].sizes['x']),
                        chunks=data[0].chunks)

        for condition in self.conditions:
            method = condition['method']
            value = condition.get('value', None)
            if isinstance(value, str):
                value = _get_flag_value(mask_in, value)
            transparency = condition['transparency']
            mask = self._get_mask(method, value, mask_data)

            if transparency == 100.0:
                data = self._set_data_nans(data, mask, alpha_attrs)
            alpha_val = 1. - transparency / 100.
            alpha = da.where(mask, alpha_val, alpha)

        return xr.DataArray(data=alpha, attrs=alpha_attrs,
                            dims=data[0].dims, coords=data[0].coords)


def _get_flag_value(mask, val):
    """Get a numerical value of the named flag.

    This function assumes the naming used in product generated with
    NWC SAF GEO/PPS softwares.

    """
    flag_meanings = mask.attrs['flag_meanings']
    flag_values = mask.attrs['flag_values']
    if isinstance(flag_meanings, str):
        flag_meanings = flag_meanings.split()

    index = flag_meanings.index(val)

    return flag_values[index]


class LongitudeMaskingCompositor(SingleBandCompositor):
    """Masks areas outside defined longitudes."""

    def __init__(self, name, lon_min=None, lon_max=None, **kwargs):
        """Collect custom configuration values.

        Args:
            lon_min (float): lower longitude limit
            lon_max (float): upper longitude limit
        """
        self.lon_min = lon_min
        self.lon_max = lon_max
        if self.lon_min is None and self.lon_max is None:
            raise ValueError("Masking conditions not defined. \
                At least lon_min or lon_max has to be specified.")
        if not self.lon_min:
            self.lon_min = -180.
        if not self.lon_max:
            self.lon_max = 180.
        super().__init__(name, **kwargs)

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        projectable = projectables[0]
        lons, lats = projectable.attrs["area"].get_lonlats()

        if self.lon_max > self.lon_min:
            lon_min_max = np.logical_and(lons >= self.lon_min, lons <= self.lon_max)
        else:
            lon_min_max = np.logical_or(lons >= self.lon_min, lons <= self.lon_max)

        masked_projectable = projectable.where(lon_min_max)
        return super().__call__([masked_projectable], **info)
