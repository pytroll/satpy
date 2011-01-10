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

"""Integration testing of
 - :mod:`mpop.scene`
 - :mod:`mpop.channel`
 - :mod:`mpop.projector`
"""
import ConfigParser
import random
import unittest

import numpy as np
from pyresample import geometry, utils, kd_tree, image

import mpop.scene


class FakeAreaDefinition:
    """Fake AreaDefinition.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.shape = None
        self.area_id = args[0]
        self.x_size = random.random()
        self.y_size = random.random()
        self.proj_id = random_string(20)
        self.proj_dict = random_string(20)
        self.area_extent = random_string(20)

class FakeSwathDefinition:
    """Fake SwathDefinition.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.shape = None
        self.area_id = random_string(20)


class FakeImageContainer:
    """Fake ImageContainer
    """
    def __init__(self, data, *args, **kwargs):
        del args, kwargs
        self.data = data

    def get_array_from_linesample(self, *args):
        """Fake method.
        """
        del args
        return self.data + 1

def patch_geometry():
    """Patching the geometry module
    """
    geometry.OldAreaDefinition = geometry.AreaDefinition
    geometry.AreaDefinition = FakeAreaDefinition
    geometry.OldSwathDefinition = geometry.SwathDefinition
    geometry.SwathDefinition = FakeSwathDefinition

def unpatch_geometry():
    """Unpatching the geometry module.
    """
    geometry.AreaDefinition = geometry.OldAreaDefinition
    delattr(geometry, "OldAreaDefinition")
    geometry.SwathDefinition = geometry.OldSwathDefinition
    delattr(geometry, "OldSwathDefinition")


def patch_kd_tree():
    """Patching the kd_tree module.
    """
    def fake_get_neighbour_info(*args, **kwargs):
        """Fake function.
        """
        del args, kwargs
        return (np.random.standard_normal((3, 1)),
                np.random.standard_normal((3, 1)),
                np.random.standard_normal((3, 1)),
                np.random.standard_normal((3, 1)))

    def fake_gsfni(typ, area, data, *args, **kwargs):
        """Fake function.
        """
        del typ, area, args, kwargs
        return data - 1

    kd_tree.old_get_neighbour_info = kd_tree.get_neighbour_info
    kd_tree.get_neighbour_info = fake_get_neighbour_info
    kd_tree.old_gsfni = kd_tree.get_sample_from_neighbour_info
    kd_tree.get_sample_from_neighbour_info = fake_gsfni
    
def unpatch_kd_tree():
    """Unpatching the kd_tree module.
    """

    kd_tree.get_neighbour_info = kd_tree.old_get_neighbour_info
    delattr(kd_tree, "old_get_neighbour_info")
    kd_tree.get_sample_from_neighbour_info = kd_tree.old_gsfni
    delattr(kd_tree, "old_gsfni")
    

def patch_utils():
    """Patching the utils module.
    """

    def fake_parse_area_file(filename, area):
        """Fake function.
        """
        del filename
        if area == "raise" or not isinstance(area, str):
            raise utils.AreaNotFound("This area is not to be found")
        else:
            return [geometry.AreaDefinition(area)]
        
    def fake_gqla(*args):
        """Fake function.
        """
        del args
        return (np.random.standard_normal((3, 1)),
                np.random.standard_normal((3, 1)))

    utils.old_parse_area_file = utils.parse_area_file
    utils.parse_area_file = fake_parse_area_file
    utils.old_generate_quick_linesample_arrays = \
                       utils.generate_quick_linesample_arrays
    utils.generate_quick_linesample_arrays = \
                       fake_gqla
    
def unpatch_utils():
    """Unpatching the utils module.
    """
    utils.parse_area_file = utils.old_parse_area_file
    delattr(utils, "old_parse_area_file")
    utils.generate_quick_linesample_arrays = \
          utils.old_generate_quick_linesample_arrays
    delattr(utils, "old_generate_quick_linesample_arrays")
    

def patch_image():
    """Patching the pyresample.image module.
    """
    image.OldImageContainer = image.ImageContainer
    image.ImageContainer = FakeImageContainer

def unpatch_image():
    """Unpatching the pyresample.image module.
    """
    image.ImageContainer = image.OldImageContainer
    delattr(image, "OldImageContainer")

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
            return "test_plugin"
        
        def sections(self):
            """Dummy sections method
            """
            raise ConfigParser.NoSectionError("Dummy sections.")

    ConfigParser.OldConfigParser = ConfigParser.ConfigParser
    ConfigParser.ConfigParser = FakeConfigParser

def unpatch_configparser():
    """Unpatch fake ConfigParser.
    """
    ConfigParser.ConfigParser = ConfigParser.OldConfigParser
    delattr(ConfigParser, "OldConfigParser")


def random_string(length,
                  choices="abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    return "".join([random.choice(choices)
                    for dummy_itr in range(length)])

class TestPPCore(unittest.TestCase):
    """Class for testing the core of mpop.
    """

    def setUp(self):
        """Apply patches.
        """

        self.scene = mpop.scene.SatelliteInstrumentScene()
        patch_geometry()
        patch_utils()
        patch_kd_tree()
        patch_image()
        patch_configparser()

    def test_channel_list_syntax(self):
        """Test syntax for channel list
        """
        channels = [["00_7", (0.5, 0.7, 0.9), 2500],
                    ["06_4", (5.7, 6.4, 7.1), 5000],
                    ["11_5", (10.5, 11.5, 12.5), 5000]]
        
        class Satscene(mpop.scene.SatelliteInstrumentScene):
            """Adding a channel list.
            """
            instrument_name = random_string(8)
            channel_list = channels

        self.scene = Satscene()
        for i, chn in enumerate(self.scene.channels):
            self.assertTrue(isinstance(chn, mpop.channel.Channel))
            self.assertEquals(chn.name, channels[i][0])
            self.assertEquals(chn.wavelength_range, channels[i][1])
            self.assertEquals(chn.resolution, channels[i][2])

    def test_project(self):
        """Test project
        """
        channels = [["00_7", (0.5, 0.7, 0.9), 2500],
                    ["06_4", (5.7, 6.4, 7.1), 5000],
                    ["11_5", (10.5, 11.5, 12.5), 5000]]
        
        class Satscene(mpop.scene.SatelliteInstrumentScene):
            """Adding a channel list.
            """
            instrument_name = random_string(8)
            channel_list = channels

        area = random_string(8)
        self.scene = Satscene(area=area)
        area2 = random_string(8)

        new_scene = self.scene.project(area2)
        self.assertEquals(new_scene.area.area_id, area2)

        for chn in new_scene.channels:
            self.assertEquals(chn.area, area2)
        
        

    def tearDown(self):
        """Remove patches.
        """
        unpatch_geometry()
        unpatch_utils()
        unpatch_kd_tree()
        unpatch_image()
        unpatch_configparser()


if __name__ == '__main__':
    unittest.main()
