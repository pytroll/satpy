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
from mpop.resample import resample, get_area_def
import six

class InfoObject(object):
    def __init__(self, **attributes):
        self.info = attributes


class Dataset(np.ma.MaskedArray):

    def __new__(cls, data, **info):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.ma.MaskedArray(data).view(cls)
        # add the new attribute to the created instance
        obj.info = getattr(data, "info", {})
        obj.info.update(info)
        # Finally, we must return the newly created object:
        return obj

    def _update_from(self, obj):
        """Copies some attributes of obj to self.
        """
        np.ma.MaskedArray._update_from(self, obj)
        if self.info is None:
            self.info = {}
        self.info.update(getattr(obj, "info", {}))

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)
        np.ma.MaskedArray.__array_finalize__(self, obj)

    def copy(self):
        res = np.ma.MaskedArray.copy(self)
        res.info = self.info.copy()
        return res


    def is_loaded(self):
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

