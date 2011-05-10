#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Test module for mpop.projector.
"""
import ConfigParser
import unittest

import numpy as np
from pyresample import geometry, utils, kd_tree, image

from mpop.projector import Projector


class FakeAreaDefinition:
    """Fake AreaDefinition.
    """
    def __init__(self, *args):
        self.args = args
        self.shape = None
        self.area_id = random_string(20)

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
            return "abc"

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


class TestProjector(unittest.TestCase):
    """Class for testing the Projector class.
    """

    proj = None

    def setUp(self):
        """Apply patches
        """
        patch_geometry()
        patch_utils()
        patch_kd_tree()
        patch_image()
        patch_configparser()
        
    def test_init(self):
        """Creation of coverage.
        """

        # in case of wrong number of arguments
        
        self.assertRaises(TypeError, Projector)
        self.assertRaises(TypeError, Projector, random_string(20))


        # in case of string arguments

        in_area_id = random_string(20)
        out_area_id = random_string(20)
        self.proj = Projector(in_area_id, out_area_id)
        self.assertTrue(isinstance(self.proj.in_area, geometry.AreaDefinition))
        self.assertEquals(self.proj.in_area.args[0], in_area_id)
        self.assertEquals(self.proj.out_area.args[0], out_area_id)

        
        # in case of undefined areas
        
        self.assertRaises(utils.AreaNotFound,
                          Projector,
                          "raise",
                          random_string(20))
        self.assertRaises(utils.AreaNotFound,
                          Projector,
                          random_string(20),
                          "raise")

        # in case of geometry objects as input

        in_area = geometry.AreaDefinition()
        self.proj = Projector(in_area, out_area_id)
        self.assertEquals(self.proj.in_area, in_area)

        in_area = geometry.SwathDefinition()
        self.proj = Projector(in_area, out_area_id)
        self.assertEquals(self.proj.in_area, in_area)

        out_area = geometry.AreaDefinition()
        self.proj = Projector(in_area_id, out_area)
        self.assertEquals(self.proj.out_area, out_area)

        # in case of lon/lat is input
        
        self.proj = Projector("raise", out_area_id, ([1, 2, 3], [1, 2, 3]))
        self.assertTrue(isinstance(self.proj.in_area, geometry.SwathDefinition))


        # in case of wrong mode

        self.assertRaises(ValueError,
                          Projector,
                          random_string(20),
                          random_string(20),
                          mode=random_string(20))

        # quick mode cache
        self.proj = Projector(in_area_id, out_area_id, mode="quick")
        cache = getattr(self.proj, "_cache")
        self.assertTrue(cache['row_idx'] is not None)
        self.assertTrue(cache['col_idx'] is not None)

        # nearest mode cache

        self.proj = Projector(in_area_id, out_area_id, mode="nearest")
        cache = getattr(self.proj, "_cache")
        self.assertTrue(cache['valid_index'] is not None)
        self.assertTrue(cache['valid_output_index'] is not None)
        self.assertTrue(cache['index_array'] is not None)


    def test_project_array(self):
        """Test the project_array function.
        """
        in_area_id = random_string(20)
        out_area_id = random_string(20)

        # test computation skip if equal areas
        in_area = geometry.AreaDefinition()
        self.proj = Projector(in_area, in_area)
        self.assertEquals(self.proj.in_area, self.proj.out_area)
        data = np.random.standard_normal((3, 1))
        self.assertTrue(np.all(data == self.proj.project_array(data)))

        # test quick
        self.proj = Projector(in_area_id, out_area_id, mode="quick")
        self.assertTrue(np.allclose(data, self.proj.project_array(data) - 1))
        
        # test nearest
        self.proj = Projector(in_area_id, out_area_id, mode="nearest")
        self.assertTrue(np.allclose(data.ravel(),
                                    self.proj.project_array(data) + 1))
        


    def tearDown(self):
        """Unpatch things.
        """
        unpatch_utils()
        unpatch_geometry()
        unpatch_kd_tree()
        unpatch_image()
        unpatch_configparser()
        
def random_string(length,
                  choices="abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    """Generates a random string with elements from *set* of the specified
    *length*.
    """
    import random
    return "".join([random.choice(choices)
                    for dummy in range(length)])

if __name__ == '__main__':
    unittest.main()
