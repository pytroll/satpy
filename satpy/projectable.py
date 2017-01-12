#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Projectable objects.
"""

import numpy as np
import six


class InfoObject(object):

    def __init__(self, **attributes):
        self.info = attributes


def combine_info(*info_objects):
    """Combine the metadata of two or more Datasets

    Args:
        *info_objects: InfoObject or dict objects to combine

    Returns:
        the combined metadata
    """
    shared_keys = None
    info_dicts = []
    # grab all of the dictionary objects provided and make a set of the shared
    # keys
    for info_object in info_objects:
        if isinstance(info_object, dict):
            info_dict = info_object
        elif hasattr(info_object, "info"):
            info_dict = info_object.info
        else:
            continue
        info_dicts.append(info_dict)

        if shared_keys is None:
            shared_keys = set(info_dict.keys())
        else:
            shared_keys &= set(info_dict.keys())

    # combine all of the dictionaries
    shared_info = {}
    for k in shared_keys:
        values = [nfo[k] for nfo in info_dicts]
        any_arrays = any([isinstance(val, np.ndarray) for val in values])
        if any_arrays:
            if all(np.all(val == values[0]) for val in values[1:]):
                shared_info[k] = values[0]
        elif all(val == values[0] for val in values[1:]):
            shared_info[k] = values[0]

    return shared_info


def copy_info(func):
    """Decorator function for combining the infos of two Datasets

    Args:
        func: the function to decorate

    Returns:
        the decorated function
    """

    def wrapper(self, other, *args, **kwargs):
        res = func(self, other, *args, **kwargs)
        res.info = combine_info(self, other)
        return res
    return wrapper


def copy_info1(func):
    """Decorator for copying the info of a Dataset

    Args:
        func: the function to decorate

    Returns:
        the decorated function
    """

    def wrapper(self, *args, **kwargs):
        res = func(self, *args, **kwargs)
        res.info = self.info.copy()
        return res
    return wrapper


class Dataset(np.ma.MaskedArray):
    _array_kwargs = ["mask", "dtype", "copy", "subok",
                     "ndmin", "keep_mask", "hard_mask", "shrink"]
    _shared_kwargs = ["fill_value"]

    def __new__(cls, data, **info):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        # pull out kwargs that are meant for the masked array
        array_kwargs = {k: info.pop(k) for k in cls._array_kwargs if k in info}
        array_kwargs.update({k: info[k]
                             for k in cls._shared_kwargs if k in info})
        obj = np.ma.MaskedArray(data, **array_kwargs).view(cls)
        # add the new attribute to the created instance
        obj.info = getattr(data, "info", {})
        obj.info.update(info)
        # Finally, we must return the newly created object:
        return obj

    def _update_info(self, obj):
        """Update the metadata from another object

        Args:
            obj: another dataset
        """
        self.info = combine_info(self, obj)

    def _update_from(self, obj):
        """Copies some attributes of obj to self.
        """
        super(Dataset, self)._update_from(obj)
        if self.info is None:
            self.info = {}
        self._update_info(obj)

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(self, 'info', {})
        super(Dataset, self).__array_finalize__(obj)
        self._update_info(obj)

    @copy_info
    def __add__(self, other):
        return super(Dataset, self).__add__(other)

    @copy_info
    def __sub__(self, other):
        return super(Dataset, self).__sub__(other)

    @copy_info
    def __mul__(self, other):
        return super(Dataset, self).__mul__(other)

    @copy_info
    def __floordiv__(self, other):
        return super(Dataset, self).__floordiv__(other)

    @copy_info
    def __mod__(self, other):
        return super(Dataset, self).__mod__(other)

    def __divmod__(self, other):
        res = super(Dataset, self).__divmod__(other)
        new_info = combine_info(self, other)
        res[0].info = new_info
        res[1].info = new_info
        return res

    @copy_info
    def __pow__(self, power):
        return super(Dataset, self).__pow__(power)

    @copy_info
    def __lshift__(self, other):
        return super(Dataset, self).__lshift__(other)

    @copy_info
    def __rshift__(self, other):
        return super(Dataset, self).__rshift__(other)

    @copy_info
    def __and__(self, other):
        return super(Dataset, self).__and__(other)

    @copy_info
    def __xor__(self, other):
        return super(Dataset, self).__xor__(other)

    @copy_info
    def __or__(self, other):
        return super(Dataset, self).__or__(other)

    @copy_info
    def __div__(self, other):
        return super(Dataset, self).__div__(other)

    @copy_info
    def __truediv__(self, other):
        return super(Dataset, self).__truediv__(other)

    @copy_info
    def __radd__(self, other):
        return super(Dataset, self).__radd__(other)

    @copy_info
    def __rsub__(self, other):
        return super(Dataset, self).__rsub__(other)

    @copy_info
    def __rmul__(self, other):
        return super(Dataset, self).__rmul__(other)

    @copy_info
    def __rfloordiv__(self, other):
        return super(Dataset, self).__rfloordiv__(other)

    @copy_info
    def __rmod__(self, other):
        return super(Dataset, self).__rmod__(other)

    def __rdivmod__(self, other):
        res = super(Dataset, self).__rdivmod__(other)
        new_info = combine_info(self, other)
        res[0].info = new_info
        res[1].info = new_info
        return res

    @copy_info
    def __rpow__(self, power):
        return super(Dataset, self).__rpow__(power)

    @copy_info
    def __rlshift__(self, other):
        return super(Dataset, self).__rlshift__(other)

    @copy_info
    def __rrshift__(self, other):
        return super(Dataset, self).__rrshift__(other)

    @copy_info
    def __rand__(self, other):
        return super(Dataset, self).__rand__(other)

    @copy_info
    def __rxor__(self, other):
        return super(Dataset, self).__rxor__(other)

    @copy_info
    def __ror__(self, other):
        return super(Dataset, self).__ror__(other)

    @copy_info
    def __rdiv__(self, other):
        return super(Dataset, self).__rdiv__(other)

    @copy_info
    def __rtruediv__(self, other):
        return super(Dataset, self).__rtruediv__(other)

    @copy_info
    def __iadd__(self, other):
        return super(Dataset, self).__iadd__(other)

    @copy_info
    def __isub__(self, other):
        return super(Dataset, self).__isub__(other)

    @copy_info
    def __imul__(self, other):
        return super(Dataset, self).__imul__(other)

    @copy_info
    def __ifloordiv__(self, other):
        return super(Dataset, self).__ifloordiv__(other)

    @copy_info
    def __imod__(self, other):
        return super(Dataset, self).__imod__(other)

    @copy_info
    def __ipow__(self, power):
        return super(Dataset, self).__ipow__(power)

    @copy_info
    def __ilshift__(self, other):
        return super(Dataset, self).__ilshift__(other)

    @copy_info
    def __irshift__(self, other):
        return super(Dataset, self).__irshift__(other)

    @copy_info
    def __iand__(self, other):
        return super(Dataset, self).__iand__(other)

    @copy_info
    def __ixor__(self, other):
        return super(Dataset, self).__ixor__(other)

    @copy_info
    def __ior__(self, other):
        return super(Dataset, self).__ior__(other)

    @copy_info
    def __idiv__(self, other):
        return super(Dataset, self).__idiv__(other)

    @copy_info
    def __itruediv__(self, other):
        return super(Dataset, self).__itruediv__(other)

    @copy_info1
    def __neg__(self):
        return super(Dataset, self).__neg__()

    @copy_info1
    def __pos__(self):
        return super(Dataset, self).__pos__()

    @copy_info1
    def __abs__(self):
        return super(Dataset, self).__abs__()

    @copy_info1
    def __invert__(self):
        return super(Dataset, self).__invert__()

    def copy(self):
        """Copy self. The metadata is just a shallow copy.

        Returns:
            A copy of self.
        """
        res = np.ma.MaskedArray.copy(self)
        res.info = self.info.copy()
        return res

    def is_loaded(self):
        """Check if data is loaded in the Dataset

        Returns:
            A boolean
        """
        return self.size > 0

    def __str__(self):
        return self._str_info() + "\n" + np.ma.MaskedArray.__str__(self)

    def _str_info(self):
        res = list()
        try:
            res.append(self.info["name"] + ": ")

            if "sensor" in self.info:
                res[0] = str(self.info["sensor"]) + "/" + res[0]
        except KeyError:
            pass

        for key in sorted(self.info.keys()):
            if key == "wavelength_range":
                res.append("{0}: {1} μm".format(key, self.info[key]))
            elif key == "resolution":
                res.append("{0}: {1} m".format(key, self.info[key]))
            elif key == "area":
                res.append("{0}: {1}".format(key, self.info[key].name))
            elif key in ["name", "sensor"]:
                continue
            else:
                res.append("{0}: {1}".format(key, self.info[key]))

        if self.size > 0:
            try:
                res.append("shape: {0}".format(self.shape))
            except AttributeError:
                pass
        else:
            res.append("not loaded")

        return "\n\t".join(res)


# the generic projectable dataset class


class Projectable(Dataset):

    def __new__(self, data=None, name="undefined", **info):
        return Dataset.__new__(self, data, name=name, **info)

    def resample(self, destination_area, **kwargs):
        """Resample the current projectable and return the resampled one.

        Args:
            destination_area: The destination onto which to project the data, either a full blown area definition or
            a string corresponding to the name of the area as defined in the area file.
            **kwargs: The extra parameters to pass to the resampling functions.

        Returns:
            A resampled projectable, with updated .info["area"] field
        """
        # avoid circular imports, this is just a convenience function anyway
        from satpy.resample import resample, get_area_def
        # call the projection stuff here
        source_area = self.info["area"]

        if isinstance(source_area, (str, six.text_type)):
            source_area = get_area_def(source_area)
        if isinstance(destination_area, (str, six.text_type)):
            destination_area = get_area_def(destination_area)

        if self.ndim == 3:
            data = np.rollaxis(self, 0, 3)
        else:
            data = self
        new_data = resample(source_area, data, destination_area, **kwargs)

        if new_data.ndim == 3:
            new_data = np.rollaxis(new_data, 2)

        # FIXME: is this necessary with the ndarray subclass ?
        res = Projectable(new_data, **self.info)
        res.info["area"] = destination_area
        return res
