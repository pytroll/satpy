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


class InfoObject(object):

    def __init__(self, **attributes):
        self.info = attributes

    def __getattr__(self, name):
        try:
            return self.info[name]
        except KeyError:
            raise AttributeError

    def get(self, key, default=None):
        return self.info.get(key, default)


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
        self.callback = None

    def __call__(self, *args, **kwargs):
        try:
            self.callback(*args, **kwargs)
        except TypeError:
            raise TypeError("No callback defined")

    def project(self, destination_area):
        # call the projection stuff here
        pass


class Channel(Projectable):

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

    def __init__(self, **info):
        self.projectables = []

        InfoObject.__init__(self, **info)
        self.product = ProductHolder(weakref.proxy(self))

    # Black magic to add methods on the fly
    def add_method_to_instance(self, func):
        """Add a method to the instance.
        """
        return setattr(self, func.__name__,
                       types.MethodType(func, self.__class__))

    def add_product(self, func):
        return setattr(self, func.__name__,
                       types.MethodType(func, self.__class__))

    def _read_config(self, cfg_file):
        conf = ConfigParser.RawConfigParser()

        if not os.path.exists(cfg_file):
            raise IOError("No such file: " + cfg_file)

        conf.read(cfg_file)

        self.info["platform_name"] = conf.get("satellite", "satname")

        instruments = conf.get("satellite", "instruments").split(",")
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

    def __getitem__(self, key):
        # get by wavelength
        if isinstance(key, numbers.Number):
            channels = [chn for chn in self.projectables
                        if("wavelength_range" in chn.info and
                           chn.wavelength_range[0] <= key and
                           chn.wavelength_range[2] >= key)]
            channels = sorted(channels,
                              lambda ch1, ch2:
                              cmp(abs(ch1.wavelength_range[1] - key),
                                  abs(ch2.wavelength_range[1] - key)))

            for chn in channels:
                # FIXME: is this reasonable ?
                if not chn.info.get("uid", "").startswith("_"):
                    return chn
            raise KeyError("Can't find any projectable at %gum" % key)
        # get by name
        else:
            for chn in self.projectables:
                if chn.get("uid", None) == key:
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
            if "satellite" in conf.sections():
                instruments = conf.get("satellite", "instruments").split(",")
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

    def load(self, *channels, **kwargs):
        # get reader
        reader(channels, **kwargs)

    def project(self, destination, channels, **kwargs):
        new_scene = self.__class__(**self.info)
        for proj in set(self.projectables) & set(channels):
            new_scene.projectables.append(proj.project(destination))


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

    myfile = "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/04/20/SDR/SVM02_npp_d20150420_t0536333_e0537575_b18015_c20150420054512262557_cspp_dev.h5"

    scn = Scene()

    scn.open(myfile)

    unittest.main()

    #########
    #
    # this part can be put in a user-owned file

    def nice_composite(self, some_param=None):
        # do something here
        return self

    nice_composite.prerequisites = ["i05", "dnb", "fog"]

    scn.add_product(nice_composite)

    def fog(self):
        return self["i05"] - self["i04"]

    fog.prerequisites = ["i05", "i04"]

    scn.add_product(fog)

    # end of this part
    #
    ##########

    # nice composite uses fog
    scn.load("nice_composite", area="europe")

    scn.products.nice_composite
