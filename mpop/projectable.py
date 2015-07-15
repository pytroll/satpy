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
from mpop.resample import resample
from trollimage.image import Image
from mpop.writers import Enhancer
import os

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

    def __str__(self):
        return np.ma.MaskedArray.__str__(self)

    def __repr__(self):
        return np.ma.MaskedArray.__repr__(self) + "\n" + repr(self.info)

    def is_loaded(self):
        return self.size > 0

    def show(self, **kwargs):
        """Display the channel as an image.
        """
        if not self.is_loaded():
            raise ValueError("Dataset not loaded, cannot display.")

        img = self.get_enhanced_image(**kwargs)
        img.show()

    def _determine_mode(self):
        if "mode" in self.info:
            return self.info["mode"]

        if self.ndim == 2:
            return "L"
        elif self.shape[0] == 2:
            return "LA"
        elif self.shape[0] == 3:
            return "RGB"
        elif self.shape[0] == 4:
            return "RGBA"
        else:
            raise RuntimeError("Can't determine 'mode' of dataset: %s" % (self.info.get("name", None),))


    def get_enhanced_image(self, enhancer=None, fill_value=None, ppp_config_dir=None, enhancement_config_file=None):
        mode = self._determine_mode()

        if ppp_config_dir is None:
            ppp_config_dir = os.environ["PPP_CONFIG_DIR"]

        if enhancer is None:
            enhancer = Enhancer(ppp_config_dir, enhancement_config_file)

        if enhancer.enhancement_tree is None:
            raise RuntimeError("No enhancement configuration files found or specified, can not automatically enhance dataset")

        if self.info.get("sensor", None):
            enhancer.add_sensor_enhancements(self.info["sensor"])

        # Create an image for enhancement
        img = self.to_image(mode=mode, fill_value=fill_value)
        enhancer.apply(img, **self.info)

        img.info.update(self.info)

        return img



    def to_image(self, copy=True, **kwargs):
        # Only add keywords if they are present
        if "mode" in self.info:
            kwargs.setdefault("mode", self.info["mode"])
        if "fill_value" in self.info:
            kwargs.setdefault("fill_value", self.info["fill_value"])
        if "palette" in self.info:
            kwargs.setdefault("palette", self.info["palette"])

        if self.ndim == 2:
            return Image([self],
                          copy=copy,
                          **kwargs)
        elif self.ndim == 3:
            return Image([band for band in self],
                          copy=copy,
                          **kwargs)
        else:
            raise ValueError("Don't know how to convert array with ndim %d to image" % self.ndim)



# the generic projectable dataset class


class Projectable(Dataset):

    def __new__(self, data=None, name="undefined", **info):
        return Dataset.__new__(self, data, name=name, **info)

    def resample(self, destination_area, **kwargs):
        # call the projection stuff here
        source_area = self.info["area"]
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

    def __str__(self):
        res = list()
        res.append(self.info["name"] + ": ")

        if "sensor" in self.info:
            res[0] = str(self.info["sensor"]) + "/" + res[0]

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
