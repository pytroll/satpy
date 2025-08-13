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

"""Core functionality of composites."""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Sequence

import dask.array as da
import numpy as np
import xarray as xr

from satpy.dataset import DataID, combine_metadata
from satpy.dataset.dataid import minimal_default_keys_config
from satpy.utils import unify_chunks

LOG = logging.getLogger(__name__)

NEGLIGIBLE_COORDS = ["time"]
"""Keywords identifying non-dimensional coordinates to be ignored during composite generation."""

TIME_COMPATIBILITY_TOLERANCE = np.timedelta64(1, "s")


class IncompatibleAreas(Exception):
    """Error raised upon compositing things of different shapes."""


class IncompatibleTimes(Exception):
    """Error raised upon compositing things from different times."""


class CompositeBase:
    """Base class for all compositors and modifiers.

    A compositor in Satpy is a class that takes in zero or more input
    DataArrays and produces a new DataArray with its own identifier (name).
    The result of a compositor is typically a brand new "product" that
    represents something different than the inputs that went into the
    operation.

    See the :class:`~satpy.modifiers.base.ModifierBase` class for information
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
    def id(self):  # noqa: A003
        """Return the DataID of the object."""
        try:
            return self.attrs["_satpy_id"]
        except KeyError:
            id_keys = self.attrs.get("_satpy_id_keys", minimal_default_keys_config)
            return DataID(id_keys, **self.attrs)

    def __call__(
            self,
            datasets: Sequence[xr.DataArray],
            optional_datasets: Optional[Sequence[xr.DataArray]] = None,
            **info
    ) -> xr.DataArray:
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
        try:
            dataset_keys = self.attrs["_satpy_id"].id_keys.keys()
        except KeyError:
            dataset_keys = ["name", "modifiers"]

        self._collect_modifier_info(origin, destination, dataset_keys)

    def _collect_modifier_info(self, origin, destination, dataset_keys):
        o = getattr(origin, "attrs", origin)
        d = getattr(destination, "attrs", destination)

        for k in dataset_keys:
            if self._is_existing_modifier(k):
                d[k] = self.attrs[k]
            elif d.get(k) is None:
                self._add_missing_modifier(k, d, o)

    def _is_existing_modifier(self, k):
        return (k == "modifiers") and (k in self.attrs)

    def _add_missing_modifier(self, key, destination, origin):
        if self.attrs.get(key) is not None:
            destination[key] = self.attrs[key]
        elif origin.get(key) is not None:
            destination[key] = origin[key]

    def match_data_arrays(self, data_arrays: Sequence[xr.DataArray]) -> list[xr.DataArray]:
        """Match data arrays so that they can be used together in a composite.

        For the purpose of this method, "can be used together" means:

        - All arrays should have the same dimensions.
        - Either all arrays should have an area, or none should.
        - If all have an area, the areas should be all the same.

        In addition, negligible non-dimensional coordinates are dropped (see
        :meth:`drop_coordinates`) and dask chunks are unified (see
        :func:`satpy.utils.unify_chunks`).

        Args:
            data_arrays: Arrays to be checked

        Returns:
            Arrays with negligible non-dimensional coordinates removed.

        Raises:
            :class:`IncompatibleAreas`:
                If dimension or areas do not match.
            :class:`ValueError`:
                If some, but not all data arrays lack an area attribute.
        """
        self.check_geolocation(data_arrays)
        new_arrays = self.drop_coordinates(data_arrays)
        new_arrays = self.align_geo_coordinates(new_arrays)
        new_arrays = list(unify_chunks(*new_arrays))
        return new_arrays

    def check_geolocation(self, data_arrays: Sequence[xr.DataArray]) -> None:
        """Check that the geolocations of the *data_arrays* are compatible.

        For the purpose of this method, "compatible" means:

        - All arrays should have the same dimensions.
        - Either all arrays should have an area, or none should.
        - If all have an area, the areas should be all the same.

        Args:
            data_arrays: Arrays to be checked

        Raises:
            :class:`IncompatibleAreas`:
                If dimension or areas do not match.
            :class:`ValueError`:
                If some, but not all data arrays lack an area attribute.
        """
        if len(data_arrays) == 1:
            return

        self._check_dimension_size(data_arrays, "x")
        self._check_dimension_size(data_arrays, "y")

        areas = [ds.attrs.get("area") for ds in data_arrays]
        if all(a is None for a in areas):
            return

        self._check_areas_are_valid(areas)

    @staticmethod
    def _check_dimension_size(data_arrays, coordinate):
        if coordinate in data_arrays[0].dims and \
           not all(x.sizes[coordinate] == data_arrays[0].sizes[coordinate]
                   for x in data_arrays[1:]):
            coordinate = coordinate.upper()
            raise IncompatibleAreas(f"{coordinate} dimension has different sizes")

    def _check_areas_are_valid(self, areas):
        if any(a is None for a in areas):
            raise ValueError("Missing 'area' attribute")

        if not all(areas[0] == x for x in areas[1:]):
            LOG.debug("Not all areas are the same in "
                      "'{}'".format(self.attrs["name"]))
            raise IncompatibleAreas("Areas are different")

    @staticmethod
    def drop_coordinates(data_arrays: Sequence[xr.DataArray]) -> list[xr.DataArray]:
        """Drop negligible non-dimensional coordinates.

        Drops negligible coordinates if they do not correspond to any
        dimension.  Negligible coordinates are defined in the
        :attr:`NEGLIGIBLE_COORDS` module attribute.

        Args:
            data_arrays: Arrays to be checked
        """
        new_arrays = []
        for ds in data_arrays:
            drop = [coord for coord in ds.coords
                    if coord not in ds.dims and
                    any([neglible in coord for neglible in NEGLIGIBLE_COORDS])]
            if drop:
                new_arrays.append(ds.drop_vars(drop))
            else:
                new_arrays.append(ds)

        return new_arrays

    @staticmethod
    def align_geo_coordinates(data_arrays: Sequence[xr.DataArray]) -> list[xr.DataArray]:
        """Align DataArrays along geolocation coordinates.

        See :func:`~xarray.align` for more information. This function uses
        the "override" join method to essentially ignore differences between
        coordinates. The :meth:`check_geolocation` should be called before
        this to ensure that geolocation coordinates and "area" are compatible.
        The :meth:`drop_coordinates` method should be called before this to
        ensure that coordinates that are considered "negligible" when computing
        composites do not affect alignment.

        """
        non_geo_coords = tuple(
            coord_name for data_arr in data_arrays
            for coord_name in data_arr.coords if coord_name not in ("x", "y"))
        return list(xr.align(*data_arrays, join="override", exclude=non_geo_coords))


def enhance2dataset(dset, convert_p=False):
    """Return the enhancement dataset *dset* as an array.

    If `convert_p` is True, enhancements generating a P mode will be converted to RGB or RGBA.
    """
    attrs = dset.attrs
    data = _get_data_from_enhanced_image(dset, convert_p)
    data.attrs = attrs
    # remove 'mode' if it is specified since it may have been updated
    data.attrs.pop("mode", None)
    # update mode since it may have changed (colorized/palettize)
    data.attrs["mode"] = GenericCompositor.infer_mode(data)
    return data


def _get_data_from_enhanced_image(dset, convert_p):
    from satpy.enhancements.enhancer import get_enhanced_image

    img = get_enhanced_image(dset)
    if convert_p and img.mode == "P":
        img = _apply_palette_to_image(img)
    if img.mode != "P":
        data = img.data.clip(0.0, 1.0)
    else:
        data = img.data
    return data


def _apply_palette_to_image(img):
    if len(img.palette[0]) == 3:
        img = img.convert("RGB")
    elif len(img.palette[0]) == 4:
        img = img.convert("RGBA")
    return img


def add_bands(data, bands):
    """Add bands so that they match *bands*."""
    # Add R, G and B bands, remove L band
    bands = bands.compute()
    data = _check_mode_p(data, bands)
    data = _check_mode_l(data, bands)
    # Add alpha band
    data = _check_alpha_band(data, bands)
    return data


def _check_mode_p(data, bands):
    if "P" in data["bands"].data or "P" in bands.data:
        raise NotImplementedError("Cannot mix datasets of mode P with other datasets at the moment.")
    return data


def _check_mode_l(data, bands):
    if "L" in data["bands"].data and "R" in bands.data:
        lum = data.sel(bands="L")
        # Keep 'A' if it was present
        if "A" in data["bands"]:
            alpha = data.sel(bands="A")
            new_data = (lum, lum, lum, alpha)
            new_bands = ["R", "G", "B", "A"]
            mode = "RGBA"
        else:
            new_data = (lum, lum, lum)
            new_bands = ["R", "G", "B"]
            mode = "RGB"
        data = xr.concat(new_data, dim="bands", coords={"bands": new_bands})
        data["bands"] = new_bands
        data.attrs["mode"] = mode
    return data


def _check_alpha_band(data, bands):
    if "A" not in data["bands"].data and "A" in bands.data:
        new_data = [data.sel(bands=band) for band in data["bands"].data]
        # Create alpha band based on a copy of the first "real" band
        alpha = new_data[0].copy()
        alpha.data = da.ones((data.sizes["y"],
                              data.sizes["x"]),
                             dtype=new_data[0].dtype,
                             chunks=new_data[0].chunks)
        # Rename band to indicate it's alpha
        alpha["bands"] = "A"
        new_data.append(alpha)
        new_data = xr.concat(new_data, dim="bands")
        new_data.attrs["mode"] = data.attrs["mode"] + "A"
        data = new_data
    return data


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
        resolution = new_attrs.get("resolution", None)
        new_attrs.update(self.attrs)
        if resolution is not None:
            new_attrs["resolution"] = resolution

        return xr.DataArray(data=data.data, attrs=new_attrs,
                            dims=data.dims, coords=data.coords)


class GenericCompositor(CompositeBase):
    """Basic colored composite builder."""

    modes = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}

    def __init__(self, name, common_channel_mask=True, **kwargs):  # noqa: D417
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
        if "mode" in data_arr.attrs:
            return data_arr.attrs["mode"]
        if "bands" not in data_arr.dims:
            return cls.modes[1]
        if "bands" in data_arr.coords and isinstance(data_arr.coords["bands"][0].item(), str):
            return "".join(data_arr.coords["bands"].values)
        return cls.modes[data_arr.sizes["bands"]]

    def _concat_datasets(self, projectables, mode):
        try:
            data = xr.concat(projectables, "bands", coords="minimal")
            data["bands"] = list(mode)
        except ValueError as e:
            LOG.debug("Original exception for incompatible areas: {}".format(str(e)))
            raise IncompatibleAreas("Areas do not match.")

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

    def __call__(
            self,
            datasets: Sequence[xr.DataArray],
            optional_datasets: Optional[Sequence[xr.DataArray]] = None,
            **attrs
    ) -> xr.DataArray:
        """Build the composite."""
        if "deprecation_warning" in self.attrs:
            warnings.warn(
                self.attrs["deprecation_warning"],
                UserWarning,
                stacklevel=2
            )
            self.attrs.pop("deprecation_warning", None)

        mode = self._get_mode(attrs, len(datasets))

        if len(datasets) > 1:
            datasets, data = self._check_datasets_and_data(datasets, mode)
        else:
            data = datasets[0]

        new_attrs = self._get_updated_attrs(datasets, attrs, mode)

        return xr.DataArray(data=data.data, attrs=new_attrs,
                            dims=data.dims, coords=data.coords)

    def _get_mode(self, attrs, num):
        mode = attrs.get("mode")
        if mode is None:
            # num may not be in `self.modes` so only check if we need to
            mode = self.modes[num]
        return mode

    def _check_datasets_and_data(self, datasets, mode):
        datasets = self.match_data_arrays(datasets)
        data = self._concat_datasets(datasets, mode)
        # Skip masking if user wants it or a specific alpha channel is given.
        if self.common_channel_mask and mode[-1] != "A":
            data = data.where(data.notnull().all(dim="bands"))
        # if inputs have a time coordinate that may differ slightly between
        # themselves then find the mid time and use that as the single
        # time coordinate value
        time = check_times(datasets)
        if time is not None and "time" in data.dims:
            data["time"] = [time]

        return datasets, data

    def _get_updated_attrs(self, datasets, attrs, mode):
        new_attrs = combine_metadata(*datasets)
        # remove metadata that shouldn't make sense in a composite
        new_attrs["wavelength"] = None
        new_attrs.pop("units", None)
        new_attrs.pop("calibration", None)
        new_attrs.pop("modifiers", None)

        new_attrs.update({key: val
                          for (key, val) in attrs.items()
                          if val is not None})
        resolution = new_attrs.get("resolution", None)
        new_attrs.update(self.attrs)
        if resolution is not None:
            new_attrs["resolution"] = resolution
        new_attrs["sensor"] = self._get_sensors(datasets)
        new_attrs["mode"] = mode

        return new_attrs


def check_times(projectables):
    """Check that *projectables* have compatible times."""
    times = []
    for proj in projectables:
        status = _collect_time_from_proj(times, proj)
        if not status:
            break
    else:
        return _get_average_time(times)


def _collect_time_from_proj(times, proj):
    status = False
    try:
        if proj["time"].size and proj["time"][0] != 0:
            times.append(proj["time"][0].values)
            status = True
    except KeyError:
        # the datasets don't have times
        pass
    except IndexError:
        # time is a scalar
        if proj["time"].values != 0:
            times.append(proj["time"].values)
            status = True
    return status


def _get_average_time(times):
    # Is there a more gracious way to handle this ?
    if np.max(times) - np.min(times) > TIME_COMPATIBILITY_TOLERANCE:
        raise IncompatibleTimes("Times do not match.")
    return (np.max(times) - np.min(times)) / 2 + np.min(times)


class RGBCompositor(GenericCompositor):
    """Make a composite from three color bands (deprecated)."""

    def __call__(self, projectables, nonprojectables=None, **info):
        """Generate the composite."""
        warnings.warn(
            "RGBCompositor is deprecated, use GenericCompositor instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if len(projectables) != 3:
            raise ValueError("Expected 3 datasets, got %d" % (len(projectables),))
        return super(RGBCompositor, self).__call__(projectables, **info)
