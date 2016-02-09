#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014.

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

"""Test module for mipp plugin.
"""
import ConfigParser
import datetime
import random
import unittest

import numpy as np
import mipp.xrit.sat

import satpy.satin.mipp_xrit
import satpy.scene
from satpy.satellites import GeostationaryFactory


def random_string(length,
                  choices="abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    return "".join([random.choice(choices)
                    for dummy_itr in range(length)])

CHANNELS = [random_string(3)
            for dummy_j in range(int(random.random() * 40))]
INSTRUMENT_NAME = random_string(10)

DUMMY_STRING = random_string(10)

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
            del kwargs
            self = self
            sec = args[0]
            if args[1] in ["name"]:
                num = int(sec[len(INSTRUMENT_NAME) + 1:])
                return "'"+CHANNELS[num]+"'"
            elif args[1] in ["resolution"]:
                return str(random.randint(1,10000))
            elif args[1] in ["frequency"]:
                return str((random.random(),
                            random.random()+1,
                            random.random()+2))
            elif args[1] == "format":
                return "mipp"
            else:
                return DUMMY_STRING

        def sections(self):
            """Dummy sections function.
            """
            self = self
            secs = [INSTRUMENT_NAME+"-level2"] + [INSTRUMENT_NAME + "-" + str(j)
                    for j, dummy in enumerate(CHANNELS)]
            return secs
        
        
        def items(self, arg):
            """Dummy items function.
            """
            self = self
            try:
                chn_nb = arg[arg.find("-") + 1:]
                return [("name", "'" + CHANNELS[int(chn_nb)] + "'"),
                        ("size", str((int(random.random() * 1000),
                                      int(random.random() * 1000)))),
                        ("resolution", str(int(random.random() * 1000)))]
            except ValueError:
                return []
        
    ConfigParser.OldConfigParser = ConfigParser.ConfigParser
    ConfigParser.ConfigParser = FakeConfigParser

def unpatch_configparser():
    """Unpatch fake ConfigParser.
    """
    ConfigParser.ConfigParser = ConfigParser.OldConfigParser
    delattr(ConfigParser, "OldConfigParser")


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
            del args, kwargs
            self.fullname = random_string(10)
            self.satname = random_string(10)
            self.number = random_string(2)
            self.instrument_name = INSTRUMENT_NAME
            self.channels_to_load = [CHANNELS[int(random.random() *
                                                  len(CHANNELS))]
                                     for dummy_i in range(int(random.random() *
                                                              len(CHANNELS)))]
            self.time_slot = (datetime.timedelta(seconds=int(random.random() *
                                                             9999999999)) +
                              datetime.datetime(1970, 1, 1))
            self.info = {}
            self.area_def = None
            self.area_id = ""
            self.area = None
            self.channels = {}

        def add_to_history(self, *args):
            pass

        def __getitem__(self, key):
            return self.channels[key]
        
        def __setitem__(self, key, data):
            self.channels[key] = FakeChannel(data)
    satpy.scene.OldSatelliteInstrumentScene = satpy.scene.SatelliteInstrumentScene
    satpy.scene.SatelliteInstrumentScene = FakeSatelliteInstrumentScene

def unpatch_satellite():
    """Unpatch the SatelliteInstrumentScene.
    """
    satpy.scene.SatelliteInstrumentScene = satpy.scene.OldSatelliteInstrumentScene
    delattr(satpy.scene, "OldSatelliteInstrumentScene")


def patch_mipp():
    """Patch the SatelliteInstrumentScene.
    """
    
    class FakeMetadata:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.calibration_unit = random_string(1)
            self.proj4_params = "proj=geos h=45684"
            self.pixel_size = (random.random() * 5642,
                               random.random() * 5642)
            self.area_extent = (random.random() * 5642000,
                                random.random() * 5642000,
                                random.random() * 5642000,
                                random.random() * 5642000)
    class FakeSlicer(object):
        """Fake slicer for mipp.
        """
        def __getitem__(self, key):
            return FakeMetadata(), np.random.standard_normal((3, 3))
        def __call__(self, *args):
            return FakeMetadata(), np.random.standard_normal((3, 3))
    
    def fake_load(*args, **kwargs):
        """Fake satellite loading function.
        """
        del args, kwargs
        return FakeSlicer()
    
    mipp.xrit.sat.old_load = mipp.xrit.sat.load
    mipp.xrit.sat.load = fake_load

def unpatch_mipp():
    """Unpatch the SatelliteInstrumentScene.
    """
    mipp.xrit.sat.load = mipp.xrit.sat.old_load
    delattr(mipp.xrit.sat, "old_load")


class TestMipp(unittest.TestCase):
    """Class for testing the mipp loader.
    """

    def setUp(self):
        """Patch foreign modules.
        """
        patch_configparser()
        patch_satellite()
        patch_mipp()
        
    # def test_load(self):
    #     """Test the loading function.
    #     """
    #     channels = ["VIS006", 'VIS008', 'IR_016', 'IR_039', 'WV_062', 'WV_073',
    #                 'IR_087', 'IR_097', 'IR_108', 'IR_120', 'IR_134', 'HRV']
    #     satscene = GeostationaryFactory.create_scene("meteosat", "09", INSTRUMENT_NAME, None)
    #     channels_to_load = [CHANNELS[random.randint(0, len(CHANNELS)-1)]]
    #     satscene.load(channels_to_load)
    #     for chn in CHANNELS:
    #         if chn in satscene.channels_to_load:
    #             self.assertEquals(satscene.channels[chn].data.shape, (3, 3))
        
    def tearDown(self):
        """Unpatch foreign modules.
        """
        unpatch_configparser()
        unpatch_satellite()
        unpatch_mipp()

def suite():
    """The test suite for test_mipp.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMipp))
    
    return mysuite
