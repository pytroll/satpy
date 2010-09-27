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

"""Test module for mipp plugin.
"""

import unittest
import ConfigParser
import pp.scene
import datetime
import xrit.sat
import string
import satin.mipp
import random
import datetime
import numpy as np


def random_string(length, choices = string.letters):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    return "".join([random.choice(choices)
                    for i in range(length)])

channels = [random_string(3)
            for i in range(int(random.random() * 40))]
instrument_name = random_string(10)


def patch_configparser():
    """Patch to fake ConfigParser.
    """
    class FakeConfigParser:
        """Dummy ConfigParser class.
        """
        def __init__(self, *args, **kwargs):
            pass
        
        def read(self, *args, **kwargs):
            """Dummy read method
            """
            del args, kwargs
            self = self

        def get(self, *args, **kwargs):
            """Dummy get method
            """
            del args, kwargs
            self = self
            return DUMMY_STRING

        def sections(self):
            self = self
            return [instrument_name + "-" + str(i)
                    for i, chn in enumerate(channels)]
        
        
        def items(self, arg):
            self = self
            try:
                chn_nb = arg[arg.find("-") + 1:]
                return [("name", "'" + channels[int(chn_nb)] + "'"),
                        ("size", str((int(random.random() * 1000),
                                      int(random.random() * 1000)))),
                        ("resolution", str(int(random.random() * 1000)))]
            except ValueError:
                return []
        
    ConfigParser._ConfigParser = ConfigParser.ConfigParser
    ConfigParser.ConfigParser = FakeConfigParser

def unpatch_configparser():
    """Unpatch fake ConfigParser.
    """
    ConfigParser.ConfigParser = ConfigParser._ConfigParser
    delattr(ConfigParser, "_ConfigParser")


def patch_satellite():
    """Patch the SatelliteInstrumentScene.
    """
    class FakeChannel:
        """Dummy Channel
        """
        def __init__(self, data):
            self.info = {}
            self.data = data
    
    class FakeSatelliteInstrumentScene:
        """Dummy SatelliteInstrumentScene.
        """

        def __init__(self, *args, **kwargs):
            self.fullname = random_string(10)
            self.satname = random_string(10)
            self.number = random_string(2)
            self.instrument_name = instrument_name
            self.channels_to_load = [channels[int(random.random() *
                                                  len(channels))]
                                     for i in range(int(random.random() *
                                                        len(channels)))]
            self.time_slot = (datetime.timedelta(seconds=int(random.random() *
                                                             9999999999)) +
                              datetime.datetime(1970, 1, 1))
            self.info = {}
            self.area_def = None
            self.area_id = ""
            self.area = None
            self.channels = {}

        def __getitem__(self, key):
            return self.channels[key]
        
        def __setitem__(self, key, data):
            self.channels[key] = FakeChannel(data)
    pp.scene._SatelliteInstrumentScene = pp.scene.SatelliteInstrumentScene
    pp.scene.SatelliteInstrumentScene = FakeSatelliteInstrumentScene

def unpatch_satellite():
    """Unpatch the SatelliteInstrumentScene.
    """
    pp.scene.SatelliteInstrumentScene = pp.scene._SatelliteInstrumentScene
    delattr(pp.scene, "_SatelliteInstrumentScene")


def patch_mipp():
    """Patch the SatelliteInstrumentScene.
    """
    
    class FakeMetadata:
        def __init__(self, *args, **kwargs):
            self.calibration_unit = random_string(1)
            self.proj4_params = "proj=uie a=4646"
            self.pixel_size = (random.random() * 5642,
                               random.random() * 5642)
    class FakeSlicer(object):
        """Fake slicer for mipp.
        """
        def __getitem__(self, key):
            return FakeMetadata(), np.random.standard_normal((3,3))
    
    def fake_load(*args, **kwargs):
        """Fake satellite loading function.
        """
        del args, kwargs
        return FakeSlicer()
    
    xrit.sat._load = xrit.sat.load
    xrit.sat.load = fake_load

def unpatch_mipp():
    """Unpatch the SatelliteInstrumentScene.
    """
    xrit.sat.load = xrit.sat._load
    delattr(xrit.sat, "_load")


class TestMipp(unittest.TestCase):
    """Class for testing the mipp loader.
    """

    def setUp(self):
        """Patch foreign modules.
        """
        patch_configparser()
        patch_satellite()
        patch_mipp()
        
    def test_load(self):
        """Test the loading function.
        """
        satscene = pp.scene.SatelliteInstrumentScene()
        satin.mipp.load(satscene)
        print satscene.area_def
        for chn in channels:
            if chn in satscene.channels_to_load:
                self.assertEquals(satscene.channels[chn].data.shape, (3,3))
            else:
                self.assertFalse(chn in satscene.channels)
        
    def tearDown(self):
        """Unpatch foreign modules.
        """
        unpatch_configparser()
        unpatch_satellite()
        unpatch_mipp()
if __name__ == '__main__':
    unittest.main()
