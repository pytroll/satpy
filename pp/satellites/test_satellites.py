#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Unit tests for the module :mod:`pp.satellites`.
"""
import ConfigParser
import random
import unittest

import pp.instruments.visir
import pp.satellites


INSTRUMENTS = ()

def random_string(length,
                  choices="abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    return "".join([random.choice(choices)
                    for dummy_itr in range(length)])


def patch_configparser():
    """Patch to fake ConfigParser.
    """
    class FakeConfigParser:
        """Dummy ConfigParser class.
        """
        def __init__(self, *args, **kwargs):
            pass
        
        def sections(self, *args, **kwargs):
            """Dummy sections method.
            """
            self = self
            del args, kwargs
            sections = []
            for i in INSTRUMENTS:
                for j in range(int(random.random() * 10 + 1)):
                    sections += [i + str(j)]
            return sections
        
        def read(self, *args, **kwargs):
            """Dummy read method
            """
            del args, kwargs
            self = self

        def get(self, *args, **kwargs):
            """Dummy get method
            """
            del kwargs
            self = self
            if args[1] == "instruments":
                return str(INSTRUMENTS)
            if args[1] == "name":
                return "'" + random_string(3) + "'"
            if args[1] == "resolution":
                return str(int(random.random() * 50000 + 1))
            if args[1] == "frequency":
                return str(random.random())
            
        
    ConfigParser.OldConfigParser = ConfigParser.ConfigParser
    ConfigParser.ConfigParser = FakeConfigParser

def unpatch_configparser():
    """Unpatch fake ConfigParser.
    """
    ConfigParser.ConfigParser = ConfigParser.OldConfigParser
    delattr(ConfigParser, "OldConfigParser")

def patch_scene():
    """Patch the :mod:`pp.instruments.visir` module to avoid using it in these
    tests.
    """
    class FakeChannel(object):
        """FakeChannel class.
        """
        def __init__(self, val):
            self.data = val

        def check_range(self, *args):
            """Dummy check_range function.
            """
            del args
            return self.data

    
    class FakeSatscene(object):
        """Fake SatelliteInstrumentScene.
        """
        __version__ = "fake"
        def __init__(self):
            self.channels = None
            self.area = None
            self.time_slot = None
            self.error = []
        
        def check_channels(self, *args):
            """Dummy check_channels function.
            """
            self.channels = args

        def __getitem__(self, key):
            if key in self.error:
                raise KeyError()
            return FakeChannel(key)
    pp.instruments.visir.OldVisirScene = pp.instruments.visir.VisirScene
    pp.instruments.visir.VisirScene = FakeSatscene
    reload(pp.satellites)
    

def unpatch_scene():
    """Unpatch the :mod:`pp.scene` module.
    """
    pp.instruments.visir.VisirScene = pp.instruments.visir.OldVisirScene
    delattr(pp.instruments.visir, "OldVisirScene")
    reload(pp)
    reload(pp.instruments)
    reload(pp.instruments.visir)
    reload(pp.satellites)


class TestSatellites(unittest.TestCase):
    """Test the satellites base functions.
    """

    def setUp(self):
        """Patch stuff.
        """
        patch_configparser()
        patch_scene()
        
    def test_buildinstrument(self):
        """Test the :func:`pp.satellites.build_instrument` function.
        """
        name = random_string(10)
        ch_list = [random_string(10), random_string(12)]
        inst = pp.satellites.build_instrument(name, ch_list)

        # Test that the patches are applied
        self.assertEquals(inst.__version__, "fake")

        
        self.assertEquals(inst.channel_list, ch_list)
        self.assertEquals(inst.instrument_name, name)
        self.assertEquals(inst.mro()[1], pp.instruments.visir.VisirScene)

    def test_build_satellite_class(self):
        """Test the :func:`pp.satellites.build_satellite_class` function.
        """
        global INSTRUMENTS
        inst = random_string(10)
        INSTRUMENTS = ("avhrr", inst)
        satname = random_string(10)
        satnumber = random_string(10)
        satvar = random_string(10)
        classes = pp.satellites.build_satellite_class(satname,
                                                      satnumber,
                                                      satvar)
        self.assertEquals(len(classes), len(INSTRUMENTS))
        for i in classes:
            self.assertEquals(i.satname, satname)
            self.assertEquals(i.number, satnumber)
            self.assertEquals(i.variant, satvar)
            if i.instrument_name == "avhrr":
                self.assertEquals(i.mro()[1], pp.instruments.avhrr.AvhrrScene)
            else:
                self.assertEquals(i.mro()[1].__name__,
                                  inst.capitalize() +
                                  "Scene")
            
    def test_get_satellite_class(self):
        """Test the :func:`pp.satellites.get_satellite_class` function.
        """
        klass = pp.satellites.get_satellite_class("meteosat", "09")
        self.assertEquals(klass, pp.satellites.meteosat09.Meteosat09SeviriScene)


        global INSTRUMENTS

        inst = random_string(10)
        INSTRUMENTS = ("avhrr", inst)
        satname = random_string(11)
        satnumber = random_string(10)
        satvar = random_string(10)
        klass = pp.satellites.get_satellite_class(satname, satnumber, satvar)
        for i in klass:
            self.assertTrue(i.mro()[0].__name__.startswith(
                satvar.capitalize() + satname.capitalize() +
                satnumber.capitalize()))


        INSTRUMENTS = (inst,)
        satname = random_string(11)
        satnumber = random_string(10)
        satvar = random_string(10)
        klass = pp.satellites.get_satellite_class(satname, satnumber, satvar)
        pklass = klass.mro()[0]
        self.assertTrue(pklass.__name__.startswith(
            satvar.capitalize() + satname.capitalize() +
            satnumber.capitalize()))

    def tearDown(self):
        """Unpatch stuff.
        """
        unpatch_configparser()
        unpatch_scene()

if __name__ == '__main__':
    unittest.main()
