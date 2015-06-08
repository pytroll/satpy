#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""Scene objects to hold satellite data.
"""

import numbers
import ConfigParser
import os
import trollsift
import glob
import fnmatch
import weakref
import numpy as np
import imp
import mpop.satin
from mpop.utils import debug_on
debug_on()


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

# the generic projectable dataset class


class Projectable(Dataset):

    def __init__(self, uid, data=None, **info):
        Dataset.__init__(self, data, uid=uid, **info)

    def project(self, destination_area):
        # call the projection stuff here
        pass

    def is_loaded(self):
        return self.data is not None


class Channel(Projectable):

    def show(self):
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
        img.show()

    def __str__(self):

        res = ["{0}/{1}".format(self.info["instrument"], self.info["uid"])]

        if "wavelength_range" in self.info:
            res.append("{0} μm".format(self.info["wavelength_range"]))
        if "resolution" in self.info:
            res.append("{0} m".format(self.info["resolution"]))
        for key in self.info:
            if key not in ["wavelength_range", "resolution", "uid"]:
                res.append(str(self.info[key]))
        if self.data is not None:
            try:
                res.append("{0}".format(self.data.shape))
            except AttributeError:
                pass
        else:
            res.append("not loaded")

        return ", ".join(res)


def get_custom_composites(name):
    """Get the home made methods for building composites for a given satellite
    or instrument *name*.
    """
    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, "mpop.cfg"))
    try:
        module_name = conf.get("composites", "module")
    except (NoSectionError, NoOptionError):
        return []

    try:
        name = name.replace("/", "")
        module = __import__(module_name, globals(), locals(), [name])
    except ImportError:
        return []

    try:
        return getattr(module, name)
    except AttributeError:
        return []


class ProductHolder(object):

    def __init__(self, scene):
        self.scene = scene

    def product(self, func):
        def inner(*args, **kwargs):
            return func(self.scene, *args, **kwargs)

        inner.prerequisites = func.prerequisites
        return inner

    def add(self, func):
        return setattr(self, func.__name__,
                       self.product(func))


class Scene(InfoObject):

    def __init__(self, filenames=None, **info):
        InfoObject.__init__(self, **info)
        self.projectables = []
        if filenames is not None:
            self.open(*filenames)
            self.filenames = filenames
        # self.product = ProductHolder(weakref.proxy(self))
        self.products = {}

    def add_product(self, name, obj):
        self.products[name] = obj

    def _read_config(self, cfg_file):
        conf = ConfigParser.RawConfigParser()

        if not os.path.exists(cfg_file):
            raise IOError("No such file: " + cfg_file)

        conf.read(cfg_file)

        self.info["platform_name"] = conf.get("platform", "platform_name")
        self.info["sensors"] = conf.get(
            "platform", "sensors").split(",")

        instruments = self.info["sensors"]
        for instrument in instruments:
            for section in sorted(conf.sections()):
                if not section.startswith(instrument):
                    continue
                elif section.startswith(instrument + "-level"):
                    self.info.setdefault("inputs", []).append(
                        dict(conf.items(section)))
                else:
                    wl = [float(elt)
                          for elt in conf.get(section, "frequency").split(",")]
                    res = conf.getint(section, "resolution")
                    uid = conf.get(section, "name")
                    new_chn = Channel(wavelength_range=wl,
                                      resolution=res,
                                      uid=uid,
                                      instrument=instrument)
                    self.projectables.append(new_chn)
            # for method in get_custom_composites(instrument):
            #    self.add_method_to_instance(method)

    def __str__(self):
        return "\n".join((str(prj) for prj in self.projectables))

    def __iter__(self):
        return self.projectables.__iter__()

    def __getitem__(self, key):
        # get by wavelength
        if isinstance(key, numbers.Number):
            channels = [chn for chn in self.projectables
                        if("wavelength_range" in chn.info and
                           chn.info["wavelength_range"][0] <= key and
                           chn.info["wavelength_range"][2] >= key)]
            channels = sorted(channels,
                              lambda ch1, ch2:
                              cmp(abs(ch1.info["wavelength_range"][1] - key),
                                  abs(ch2.info["wavelength_range"][1] - key)))

            for chn in channels:
                # FIXME: is this reasonable ?
                if not chn.info.get("uid", "").startswith("_"):
                    return chn
            raise KeyError("Can't find any projectable at %gum" % key)
        # get by name
        else:
            for chn in self.projectables:
                if chn.info.get("uid", None) == key:
                    return chn
        raise KeyError("No channel corresponding to " + str(key) + ".")

    def __contains__(self, uid):
        for prj in self.projectables:
            if prj.uid == uid:
                return True
        return False

    def open(self, *files):
        filename = files[0]
        for config_file in glob.glob(os.path.join(os.environ.get("PPP_CONFIG_DIR", "."),
                                                  "*.cfg")):
            conf = ConfigParser.RawConfigParser()
            conf.read(config_file)
            if "platform" in conf.sections():
                instruments = conf.get("platform", "sensors").split(",")
                for instrument in instruments:
                    for section in sorted(conf.sections()):
                        if section.startswith(instrument + "-level"):
                            try:
                                pattern = trollsift.globify(
                                    conf.get(section, "pattern"))
                                if fnmatch.fnmatch(os.path.basename(filename),
                                                   os.path.basename(pattern)):
                                    self._read_config(config_file)
                                    return
                            except ConfigParser.NoOptionError:
                                pass
        raise IOError("Don't know how to open that file")

    def read(self, *projectable_names, **kwargs):
        # get reader
        for input_config in self.info["inputs"]:
            try:
                pattern = trollsift.globify(input_config["pattern"])
            except KeyError:
                continue

            if fnmatch.fnmatch(os.path.basename(self.filenames[0]),
                               os.path.basename(pattern)):
                reader = input_config["format"]
                break
        reader_module, reading_element = reader.rsplit(".", 1)
        reader = "mpop.satin." + reader_module

        try:
            # Look for builtin reader
            imp.find_module(reader_module, mpop.satin.__path__)
        except ImportError:
            # Look for custom reader
            loader = __import__(reader_module, globals(),
                                locals(), [reading_element])
        else:
            loader = __import__(reader, globals(),
                                locals(), [reading_element])

        loader = getattr(loader, reading_element)
        reader_instance = loader(self)
        setattr(self, loader.pformat + "_reader", reader_instance)

        reader_instance.load(
            self, set(projectable_names), filename=self.filenames)

    def project(self, destination, channels, **kwargs):
        new_scene = self.__class__(**self.info)
        for proj in set(self.projectables) & set(channels):
            new_scene.projectables.append(proj.project(destination))


class CompositeBase(object):

    def __init__(self, **kwargs):
        self.prerequisites = []

    def __call__(self, scene):
        raise NotImplementedError()


class VIIRSFog(CompositeBase):

    def __call__(self, scene):
        return scene["i05"] - scene["i04"]


class VIIRSTrueColor(CompositeBase):

    def __init__(self, name="true_color"):
        CompositeBase.__init__(self)
        self.name = name
        self.prerequisites = ["m01", "m03", "m04"]

    def __call__(self, scene):
        return Projectable(uid="true_color",
                           data=np.dstack(
                               (scene["m04"], scene["m03"], scene["m01"])),
                           **scene["m04"].info)


import unittest


class TestScene(unittest.TestCase):

    def test_config_reader(self):
        "Check config reading"
        scn = Scene()
        scn._read_config(
            "/home/a001673/usr/src/newconfig/Suomi-NPP.cfg")
        self.assertTrue("DNB" in scn)

    def test_channel_get(self):
        scn = Scene()
        scn._read_config(
            "/home/a001673/usr/src/newconfig/Suomi-NPP.cfg")
        self.assertEqual(scn[0.67], scn["M05"])

    def test_metadata(self):
        scn = Scene()
        scn._read_config(
            "/home/a001673/usr/src/newconfig/Suomi-NPP.cfg")
        self.assertEqual(scn.info["platform_name"], "Suomi-NPP")

    def test_open(self):
        scn = Scene()
        scn.open(
            "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/04/20/SDR/SVM02_npp_d20150420_t0536333_e0537575_b18015_c20150420054512262557_cspp_dev.h5")

        self.assertEqual(scn.info["platform_name"], "Suomi-NPP")

        self.assertRaises(IOError, scn.open, "bla")


class TestProjectable(unittest.TestCase):
    pass

if __name__ == '__main__':
    scn = Scene()
    scn._read_config("/home/a001673/usr/src/pytroll-config/etc/Suomi-NPP.cfg")

    myfiles = ["/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/04/20/SDR/SVM02_npp_d20150420_t0536333_e0537575_b18015_c20150420054512262557_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/04/20/SDR/GMTCO_npp_d20150420_t0536333_e0537575_b18015_c20150420054511332482_cspp_dev.h5"]

    scn = Scene(filenames=myfiles)

    scn.add_product("fog", VIIRSFog())
    scn.add_product("true_color", VIIRSTrueColor())

    scn.read("fog", "I01", "M02", "true_color")

    scn["M02"].show()

    # unittest.main()

    #########
    #
    # this part can be put in a user-owned file

    # def nice_composite(self, some_param=None):
    #     # do something here
    #     return self

    # nice_composite.prerequisites = ["i05", "dnb", "fog"]

    # scn.add_product(nice_composite)

    # def fog(self):
    #     return self["i05"] - self["i04"]

    # fog.prerequisites = ["i05", "i04"]

    # scn.add_product(fog)

    # # end of this part
    # #
    # ##########

    # # nice composite uses fog
    # scn.load("nice_composite", area="europe")

    # scn.products.nice_composite
