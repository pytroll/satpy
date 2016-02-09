#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Unit tests for the module :mod:`pp.satellites`.
"""
import ConfigParser
import random
import unittest

import satpy.instruments.visir
import satpy.satellites

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
            if args[1] == "module":
                return random_string(8)
            
    ConfigParser.OldConfigParser = ConfigParser.ConfigParser
    ConfigParser.ConfigParser = FakeConfigParser

def unpatch_configparser():
    """Unpatch fake ConfigParser.
    """
    ConfigParser.ConfigParser = ConfigParser.OldConfigParser
    delattr(ConfigParser, "OldConfigParser")

def patch_scene():
    """Patch the :mod:`satpy.instruments.visir` module to avoid using it in these
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
    satpy.instruments.visir.OldVisirCompositer = satpy.instruments.visir.VisirCompositer
    satpy.instruments.visir.VisirCompositer = FakeSatscene
    reload(satpy.satellites)
    

def unpatch_scene():
    """Unpatch the :mod:`satpy.scene` module.
    """
    satpy.instruments.visir.VisirCompositer = satpy.instruments.visir.OldVisirCompositer
    delattr(satpy.instruments.visir, "OldVisirCompositer")
    reload(satpy)
    reload(satpy.instruments)
    reload(satpy.instruments.visir)
    reload(satpy.satellites)


class TestSatellites(unittest.TestCase):
    """Test the satellites base functions.
    """

    def setUp(self):
        """Patch stuff.
        """
        patch_configparser()
        patch_scene()
        
    def test_buildinstrument(self):
        """Test the :func:`satpy.satellites.build_instrument` function.
        """
        name = random_string(10)
        #ch_list = [random_string(10), random_string(12)]
        inst = satpy.satellites.build_instrument_compositer(name)

        # Test that the patches are applied
        self.assertEquals(inst.__version__, "fake")

        
        #self.assertEquals(inst.channel_list, ch_list)
        self.assertEquals(inst.instrument_name, name)
        self.assertEquals(inst.mro()[1], satpy.instruments.visir.VisirCompositer)

    def test_build_satellite_class(self):
        """Test the :func:`satpy.satellites.build_satellite_class` function.
        """
        global INSTRUMENTS
        inst = random_string(10)
        INSTRUMENTS = (inst, )
        satname = random_string(10)
        satnumber = random_string(10)
        satvar = random_string(10)
        myclass = satpy.satellites.build_sat_instr_compositer((satname,
                                                               satnumber,
                                                               satvar),
                                                              inst)
        #self.assertEquals(myclass.satname, satname)
        #self.assertEquals(myclass.number, satnumber)
        #self.assertEquals(myclass.variant, satvar)
        self.assertEquals(myclass.mro()[1].__name__,
                          inst.capitalize() +
                          "Compositer")
            
    def test_get_satellite_class(self):
        """Test the :func:`satpy.satellites.get_satellite_class` function.
        """
        global INSTRUMENTS

        inst = random_string(10)
        INSTRUMENTS = ("avhrr", inst)
        satname = random_string(11)
        satnumber = random_string(10)
        satvar = random_string(10)
        klass = satpy.satellites.get_sat_instr_compositer((satname,
                                                           satnumber,
                                                           satvar),
                                                          inst)
        self.assertTrue(klass.mro()[0].__name__.startswith(
            satvar.capitalize() + satname.capitalize() +
            satnumber.capitalize()))


        INSTRUMENTS = (inst,)
        satname = random_string(11)
        satnumber = random_string(10)
        satvar = random_string(10)
        klass = satpy.satellites.get_sat_instr_compositer((satname,
                                                           satnumber,
                                                           satvar),
                                                          inst)
        pklass = klass.mro()[0]
        self.assertTrue(pklass.__name__.startswith(
            satvar.capitalize() + satname.capitalize() +
            satnumber.capitalize()))

    def tearDown(self):
        """Unpatch stuff.
        """
        unpatch_configparser()
        unpatch_scene()

def suite():
    """The test suite for test_satellites.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSatellites))
    
    return mysuite
