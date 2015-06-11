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
from mpop.imageo.geo_image import GeoImage
from mpop.utils import debug_on
debug_on()
from mpop.projectable import Projectable, InfoObject

import logging

logger = logging.getLogger(__name__)

class IncompatibleAreas(StandardError):
    pass

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


class Scene(InfoObject):

    def __init__(self, filenames=None, **info):
        InfoObject.__init__(self, **info)
        self.projectables = []
        if filenames is not None:
            self.open(*filenames)
            self.filenames = filenames
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
                    new_chn = Projectable(wavelength_range=wl,
                                          resolution=res,
                                          uid=uid,
                                          sensor=instrument,
                                          platform_name=self.info["platform_name"]
                                          )
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
        self.info["wishlist"] = projectable_names
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

        # compute the depencies to load from file
        pnames = set(projectable_names)
        needed_bands = None
        rerun = True
        while rerun:
            rerun = False
            needed_bands = set()
            for band in pnames:
                if band in self.products:
                    needed_bands |= set(self.products[band].prerequisites)
                    rerun = True
                else:
                    needed_bands.add(band)
            pnames = needed_bands

        reader_instance.load(
            self, set(pnames), filename=self.filenames)

    def compute(self, *requirements):
        if not requirements:
            requirements = self.info["wishlist"]
        for requirement in requirements:
            if requirement not in self.products:
                continue
            if requirement in [p.info["uid"] for p in self.projectables]:
                continue
            self.compute(*self.products[requirement].prerequisites)
            try:
                self.projectables.append(self.products[requirement](scn))
            except IncompatibleAreas:
                for projectable in self.projectables:
                    if projectable.info["uid"] in self.products[requirement].prerequisites:
                        projectable.info["keep"] = True

    def unload(self):
        to_del = [projectable for projectable in self.projectables
                  if projectable.info["uid"] not in self.info["wishlist"] and
                      not projectable.info.get("keep", False)]
        for projectable in to_del:
            self.projectables.remove(projectable)

    def load(self, *wishlist, **kwargs):
        self.read(*wishlist, **kwargs)
        if kwargs.get("compute", True):
            self.compute()
        if kwargs.get("unload", True):
            self.unload()

    def resample(self, destination, channels=None, **kwargs):
        """Resample the projectables and return a new scene.
        """
        new_scn = Scene()
        new_scn.info = self.info.copy()
        for projectable in self.projectables:
            logger.debug("Resampling %s", projectable.info["uid"])
            if channels and not projectable.info["uid"] in channels:
                continue
            new_scn.projectables.append(projectable.resample(destination, **kwargs))
        return new_scn

    def images(self):
        for projectable in self.projectables:
            if projectable.info["uid"] in self.info["wishlist"]:
                yield projectable.to_image()


class CompositeBase(InfoObject):

    def __init__(self, **kwargs):
        InfoObject.__init__(self, **kwargs)
        self.prerequisites = []

    def __call__(self, scene):
        raise NotImplementedError()


class VIIRSFog(CompositeBase):

    def __init__(self, uid="fog", **kwargs):
        CompositeBase.__init__(self, **kwargs)
        self.uid = uid
        self.prerequisites = ["I04", "I05"]

    def __call__(self, scene):
        fog = scene["I05"] - scene["I04"]
        fog.info["area"] = scene["I05"].info["area"]
        fog.info["uid"] = self.uid
        return fog


class VIIRSTrueColor(CompositeBase):

    def __init__(self, uid="true_color", image_config=None, **kwargs):
        default_image_config={"mode": "RGB",
                              "stretch": "log"}
        if image_config is not None:
            default_image_config.update(image_config)

        CompositeBase.__init__(self, **kwargs)
        self.uid = uid
        self.prerequisites = ["M02", "M04", "M05"]
        self.info["image_config"] = default_image_config

    def __call__(self, scene):
        # raise IncompatibleAreas
        return Projectable(uid=self.uid,
                           data=np.concatenate(
                               ([scene["M05"].data], [scene["M04"].data], [scene["M02"].data]), axis=0),
                           area=scene["M05"].info["area"],
                           time_slot=scene.info["start_time"],
                           **self.info)



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

    myfiles = ["/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/04/20/SDR/SVM16_npp_d20150420_t0536333_e0537575_b18015_c20150420054512738521_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/04/20/SDR/GMTCO_npp_d20150420_t0536333_e0537575_b18015_c20150420054511332482_cspp_dev.h5"]

    myfiles = ["/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVI01_npp_d20150311_t1125112_e1126354_b17451_c20150311113328862761_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVI02_npp_d20150311_t1125112_e1126354_b17451_c20150311113328951540_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVI03_npp_d20150311_t1125112_e1126354_b17451_c20150311113329042562_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVI04_npp_d20150311_t1125112_e1126354_b17451_c20150311113329143755_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVI05_npp_d20150311_t1125112_e1126354_b17451_c20150311113329234947_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM01_npp_d20150311_t1125112_e1126354_b17451_c20150311113329326838_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM02_npp_d20150311_t1125112_e1126354_b17451_c20150311113329360063_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM03_npp_d20150311_t1125112_e1126354_b17451_c20150311113329390738_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM04_npp_d20150311_t1125112_e1126354_b17451_c20150311113329427332_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM05_npp_d20150311_t1125112_e1126354_b17451_c20150311113329464787_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM06_npp_d20150311_t1125112_e1126354_b17451_c20150311113329503232_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM07_npp_d20150311_t1125112_e1126354_b17451_c20150311113330249624_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM08_npp_d20150311_t1125112_e1126354_b17451_c20150311113329572000_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM09_npp_d20150311_t1125112_e1126354_b17451_c20150311113329602050_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM10_npp_d20150311_t1125112_e1126354_b17451_c20150311113329632503_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM11_npp_d20150311_t1125112_e1126354_b17451_c20150311113329662488_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM12_npp_d20150311_t1125112_e1126354_b17451_c20150311113329692444_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM13_npp_d20150311_t1125112_e1126354_b17451_c20150311113329722069_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM14_npp_d20150311_t1125112_e1126354_b17451_c20150311113329767340_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM15_npp_d20150311_t1125112_e1126354_b17451_c20150311113329796873_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVM16_npp_d20150311_t1125112_e1126354_b17451_c20150311113329826626_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/GDNBO_npp_d20150311_t1125112_e1126354_b17451_c20150311113327046285_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/GITCO_npp_d20150311_t1125112_e1126354_b17451_c20150311113327852159_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/GMTCO_npp_d20150311_t1125112_e1126354_b17451_c20150311113328505792_cspp_dev.h5",
               "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/03/11/SDR/SVDNB_npp_d20150311_t1125112_e1126354_b17451_c20150311113326791425_cspp_dev.h5",
               ]

    scn = Scene(filenames=myfiles)

    scn.add_product("fog", VIIRSFog())
    scn.add_product("true_color", VIIRSTrueColor())

    scn.load("fog", "I01", "M16", "true_color")

    #img = scn["true_color"].to_image()
    #img.show()

    from mpop.projector import get_area_def
    eurol = get_area_def("eurol")
    newscn = scn.resample(eurol, radius_of_influence=2000)

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
