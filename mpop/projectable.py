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

    def __sub__(self, other):
        if isinstance(other, Dataset):
            return self.__class__(data=self.data - other.data)
        else:
            return self.__class__(data=self.data - other)

    def __add__(self, other):
        if isinstance(other, Dataset):
            return self.__class__(data=self.data + other.data)
        else:
            return self.__class__(data=self.data + other)

    def __div__(self, other):
        if isinstance(other, Dataset):
            return self.__class__(data=self.data / other.data)
        else:
            return self.__class__(data=self.data / other)

    def __neg__(self):
        return self.__class__(data=-self.data)

    def __abs__(self):
        return self.__class__(data=abs(self.data))

# the generic projectable dataset class


class Projectable(Dataset):

    def __init__(self, data=None, uid="undefined", **info):
        Dataset.__init__(self, data, uid=uid, **info)

    def project(self, destination_area):
        # call the projection stuff here
        pass

    def is_loaded(self):
        return self.data is not None

    def show(self, filename=None):
        """Display the channel as an image.
        """
        if not self.is_loaded():
            raise ValueError("Channel not loaded, cannot display.")

        from PIL import Image as pil

        data = ((self.data - self.data.min()) * 255.0 /
                (self.data.max() - self.data.min()))
        if isinstance(data, np.ma.core.MaskedArray):
            img = pil.fromarray(np.array(data.filled(0), np.uint8))
        else:
            img = pil.fromarray(np.array(data, np.uint8))
        if filename is not None:
            img.save(filename)
        else:
            img.show()

    def to_image(self, copy=True, **kwargs):
        kwargs.update(self.info["image_config"])
        print self.data.shape
        if self.data.ndim == 2:
            return GeoImage([self.data],
                            self.info["area"],
                            self.info["time_slot"],
                            copy=copy,
                            **kwargs)
        elif self.data.ndim == 3:
            return GeoImage([band for band in self.data],
                            self.info["area"],
                            self.info["time_slot"],
                            copy=copy,
                            **kwargs)



    def __str__(self):
        res = []
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

