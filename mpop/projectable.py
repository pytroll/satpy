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
from mpop.imageo.geo_image import GeoImage
from mpop.resample import resample_kd_tree_nearest

class InfoObject(object):
    def __init__(self, **attributes):
        self.info = attributes


class Dataset(InfoObject):
    def __init__(self, data, **attributes):
        InfoObject.__init__(self, **attributes)
        self.data = data

    def __str__(self):
        return str(self.data) + "\n" + str(self.info)

    def __repr__(self):
        return repr(self.data) + "\n" + repr(self.info)

    def copy(self, copy_data=True):
        if copy_data:
            data = self.data.copy()
        else:
            data = self.data
        return Dataset(data, **self.info)

    def __pow__(self, other, modulo=None):
        if isinstance(other, Dataset):
            return self.__class__(data=pow(self.data, other.data, modulo))
        else:
            return self.__class__(data=pow(self.data, other, modulo))

    def __mul__(self, other):
        if isinstance(other, Dataset):
            return self.__class__(data=self.data * other.data)
        else:
            return self.__class__(data=self.data * other)

    def __rmul__(self, other):
        return self.__class__(data=self.data * other)

    def __sub__(self, other):
        if isinstance(other, Dataset):
            return self.__class__(data=self.data - other.data)
        else:
            return self.__class__(data=self.data - other)

    def __rsub__(self, other):
        return self.__class__(data=other - self.data)

    def __add__(self, other):
        if isinstance(other, Dataset):
            return self.__class__(data=self.data + other.data)
        else:
            return self.__class__(data=self.data + other)

    def __radd__(self, other):
        return self.__class__(data=self.data + other)

    def __div__(self, other):
        if isinstance(other, Dataset):
            return self.__class__(data=self.data / other.data)
        else:
            return self.__class__(data=self.data / other)

    def __rdiv__(self, other):
        return self.__class__(data=other / self.data)


    def __neg__(self):
        return self.__class__(data=-self.data)

    def __abs__(self):
        return self.__class__(data=abs(self.data))


# the generic projectable dataset class


class Projectable(Dataset):
    def __init__(self, data=None, uid="undefined", **info):
        Dataset.__init__(self, data, uid=uid, **info)

    def resample(self, destination_area, **kwargs):
        # call the projection stuff here
        source_area = self.info["area"]
        if self.data.ndim == 3:
            data = np.rollaxis(self.data, 0, 3)
        else:
            data = self.data
        from pyresample import kd_tree
        #new_data = kd_tree.resample_nearest(source_area, data, destination_area, **kwargs)
        #res = resample(self, destination_area, **kwargs)
        new_data = resample_kd_tree_nearest(source_area, data, destination_area, **kwargs)


        if new_data.ndim == 3:
            new_data = np.rollaxis(new_data, 2)

        res = Projectable(new_data, **self.info)
        res.info["area"] = destination_area
        return res


    def is_loaded(self):
        return self.data is not None

    def show(self, filename=None, stretch="crude", **kwargs):
        """Display the channel as an image.
        """
        if not self.is_loaded():
            raise ValueError("Channel not loaded, cannot display.")

        img = self.to_image(stretch=stretch, **kwargs)
        if filename is not None:
            img.save(filename)
        else:
            img.show()

    def to_image(self, copy=True, **kwargs):
        info = self.info.copy()
        info.update(kwargs)
        if self.data.ndim == 2:
            return GeoImage([self.data],
                            copy=copy,
                            **info)
        elif self.data.ndim == 3:
            return GeoImage([band for band in self.data],
                            copy=copy,
                            **info)
        else:
            raise ValueError("Don't know how to convert array with ndim %d to image"%self.data.ndim)


    def __str__(self):
        res = list()
        res.append(self.info["uid"])

        if "sensor" in self.info:
            res[0] = self.info["sensor"] + "/" + res[0]

        if "wavelength_range" in self.info:
            res.append("{0} μm".format(self.info["wavelength_range"]))
        if "resolution" in self.info:
            res.append("{0} m".format(self.info["resolution"]))
        for key in self.info:
            if key not in ["sensor", "wavelength_range", "resolution", "uid"]:
                res.append(str(self.info[key]))
        if self.data is not None:
            try:
                res.append("{0}".format(self.data.shape))
            except AttributeError:
                pass
        else:
            res.append("not loaded")

        return ", ".join(res)
