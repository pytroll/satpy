#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2014.

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

"""Module for testing the :mod:`satpy.instruments.visir` module.
"""
import random
import unittest

import numpy as np

import satpy.instruments.visir
import satpy.scene
from satpy.imageo import geo_image


def patch_scene():
    """Patch the :mod:`satpy.scene` module to avoid using it in these tests.
    """
    class FakeChannel(object):
        """FakeChannel class.
        """
        def __init__(self, val):
            self.data = val
            self.area = None
            
        def check_range(self):
            """Dummy check_range function.
            """
            return self.data

    
    class FakeSatscene(object):
        """Fake SatelliteInstrumentScene.
        """
        __version__ = "fake"
        def __init__(self):
            self.channels = None
            self.area = None
            self.time_slot = None
        
        def check_channels(self, *args):
            """Dummy check_channels function.
            """
            self.channels = args

        def __getitem__(self, key):
            return FakeChannel(key)
    satpy.scene.OldSatelliteInstrumentScene = satpy.scene.SatelliteInstrumentScene
    satpy.scene.SatelliteInstrumentScene = FakeSatscene
    reload(satpy)
    reload(satpy.instruments)
    reload(satpy.instruments.visir)

def unpatch_scene():
    """Unpatch the :mod:`satpy.scene` module.
    """
    satpy.scene.SatelliteInstrumentScene = satpy.scene.OldSatelliteInstrumentScene
    delattr(satpy.scene, "OldSatelliteInstrumentScene")
    reload(satpy)
    reload(satpy.instruments)
    reload(satpy.instruments.visir)

def patch_geo_image():
    """Patch the :mod:`imageo.geo_image` module to avoid using it in these
    tests.
    """
    class FakeGeoImage:
        """FakeGeoImage class.
        """
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def enhance(self, **kwargs):
            """Dummy enhance function.
            """
            self.kwargs.update(kwargs)

        def clip(self):
            """Dummy clip function.
            """
            pass

    geo_image.OldGeoImage = geo_image.GeoImage
    geo_image.GeoImage = FakeGeoImage

def unpatch_geo_image():
    """Unpatch the :mod:`imageo.geo_image` module.
    """
    geo_image.GeoImage = geo_image.OldGeoImage
    delattr(geo_image, "OldGeoImage")

class TestComposites(unittest.TestCase):
    """Class for testing the composites.
    """

    def setUp(self):
        """Setup stuff.
        """
        patch_geo_image()
        patch_scene()
        self.scene = satpy.instruments.visir.VisirCompositer(satpy.scene.SatelliteInstrumentScene())
        

    def test_channel_image(self):
        """Test channel_image.
        """
        chn = random.random()
        img = self.scene.channel_image(chn)
        self.assertEquals(chn, img.args[0])
        self.assertEquals(img.kwargs["stretch"], "crude")
        self.assertEquals(img.kwargs["mode"], "L")
        self.assertEquals(img.kwargs["fill_value"], 0)
        self.assertTrue("crange" not in img.kwargs)
        
    def test_overview(self):
        """Test overview.
        """
        img = self.scene.overview()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], (0.635, 0.85, -10.8))
        self.assertEquals(img.kwargs["stretch"], "crude")
        self.assertEquals(img.kwargs["gamma"], 1.6)
        self.assertTrue("crange" not in img.kwargs)
        #self.assertEquals(self.scene.overview.prerequisites,
        #                  set([0.635, 0.85, 10.8]))

    def test_airmass(self):
        """Test airmass.
        """
        img = self.scene.airmass()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertTrue(np.allclose(np.array(img.args[0]),
                                    np.array((-0.6, -1.1, 6.7))))
        self.assertEquals(img.kwargs["crange"], ((-25, 0),
                                                 (-40, 5),
                                                 (243, 208)))
        self.assertTrue("gamma" not in img.kwargs)
        self.assertTrue("stretch" not in img.kwargs)
        #self.assertEquals(self.scene.airmass.prerequisites, set([6.7, 7.3,
        #                                                         9.7, 10.8]))

    
    # def test_vis06(self):
    #     """Test vis06.
    #     """
    #     img = self.scene.vis06()
    #     self.assertEquals(0.6, img.args[0])
    #     self.assertEquals(img.kwargs["mode"], "L")
    #     self.assertEquals(img.kwargs["fill_value"], 0)
    #     self.assertEquals(img.kwargs["stretch"], "crude")
    #     self.assertTrue("gamma" not in img.kwargs)
    #     self.assertTrue("crange" not in img.kwargs)
    #     #self.assertEquals(self.scene.vis06.prerequisites,
    #     #                  set([0.635]))

    # def test_ir108(self):
    #     """Test ir108.
    #     """
    #     img = self.scene.ir108()
    #     self.assertEquals(10.8, img.args[0])
    #     self.assertEquals(img.kwargs["mode"], "L")
    #     self.assertEquals(img.kwargs["fill_value"], 0)
    #     self.assertEquals(img.kwargs["crange"], (-70 + 273.15, 57.5 + 273.15))
    #     self.assertEquals(img.kwargs["inverse"], True)
    #     self.assertTrue("gamma" not in img.kwargs)
    #     self.assertTrue("stretch" not in img.kwargs)
    #     #self.assertEquals(self.scene.ir108.prerequisites,
    #     #                  set([10.8]))

    # def test_wv_high(self):
    #     """Test wv_high.
    #     """
    #     img = self.scene.wv_high()
    #     self.assertEquals(6.7, img.args[0])
    #     self.assertEquals(img.kwargs["mode"], "L")
    #     self.assertEquals(img.kwargs["fill_value"], 0)
    #     self.assertEquals(img.kwargs["stretch"], "linear")
    #     self.assertEquals(img.kwargs["inverse"], True)
    #     self.assertTrue("gamma" not in img.kwargs)
    #     self.assertTrue("crange" not in img.kwargs)
    #     #self.assertEquals(self.scene.wv_high.prerequisites,
    #     #                  set([6.7]))

    # def test_wv_low(self):
    #     """Test wv_low.
    #     """
    #     img = self.scene.wv_low()
    #     self.assertEquals(7.3, img.args[0])
    #     self.assertEquals(img.kwargs["mode"], "L")
    #     self.assertEquals(img.kwargs["fill_value"], 0)
    #     self.assertEquals(img.kwargs["stretch"], "linear")
    #     self.assertEquals(img.kwargs["inverse"], True)
    #     self.assertTrue("gamma" not in img.kwargs)
    #     self.assertTrue("crange" not in img.kwargs)
    #     #self.assertEquals(self.scene.wv_low.prerequisites,
    #     #                  set([7.3]))

    def test_natural(self):
        """Test natural.
        """
        img = self.scene.natural()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], (1.63, 0.85, 0.635))
        self.assertEquals(img.kwargs["crange"], ((0, 90),
                                                 (0, 90),
                                                 (0, 90)))
        self.assertEquals(img.kwargs["gamma"], 1.8)
        self.assertTrue("stretch" not in img.kwargs)
        #self.assertEquals(self.scene.natural.prerequisites,
        #                  set([0.635, 0.85, 1.63]))
        

    # def test_green_snow(self):
    #     """Test green_snow.
    #     """
    #     img = self.scene.green_snow()
    #     self.assertEquals(img.kwargs["mode"], "RGB")
    #     self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
    #     self.assertEquals(img.args[0], (1.63, 0.85, -10.8))
    #     self.assertEquals(img.kwargs["stretch"], "crude")
    #     self.assertEquals(img.kwargs["gamma"], 1.6)
    #     self.assertTrue("crange" not in img.kwargs)
    #     #self.assertEquals(self.scene.green_snow.prerequisites,
    #     #                  set([1.63, 0.85, 10.8]))

    # def test_red_snow(self):
    #     """Test red_snow.
    #     """
    #     img = self.scene.red_snow()
    #     self.assertEquals(img.kwargs["mode"], "RGB")
    #     self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
    #     self.assertEquals(img.args[0], (0.635, 1.63, -10.8))
    #     self.assertEquals(img.kwargs["stretch"], "crude")
    #     self.assertTrue("crange" not in img.kwargs)
    #     self.assertTrue("gamma" not in img.kwargs)
    #     #self.assertEquals(self.scene.red_snow.prerequisites,
    #     #                  set([1.63, 0.635, 10.8]))
    

    def test_convection(self):
        """Test convection.
        """
        img = self.scene.convection()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0],(6.7 - 7.3, 3.75 - 10.8, 1.63 - 0.635))
        self.assertEquals(img.kwargs["crange"], ((-30, 0),
                                                 (0, 55),
                                                 (-70, 20)))
        self.assertTrue("gamma" not in img.kwargs)
        self.assertTrue("stretch" not in img.kwargs)
        #self.assertEquals(self.scene.convection.prerequisites,
        #                  set([0.635, 1.63, 3.75, 6.7, 7.3, 10.8]))

    def test_dust(self):
        """Test dust.
        """
        img = self.scene.dust()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], (12.0 - 10.8, 10.8 - 8.7, 10.8))
        self.assertEquals(img.kwargs["crange"], ((-4, 2),
                                                 (0, 15),
                                                 (261, 289)))
        self.assertEquals(img.kwargs["gamma"], (1.0, 2.5, 1.0))
        self.assertTrue("stretch" not in img.kwargs)
        #self.assertEquals(self.scene.dust.prerequisites,
        #                  set([8.7, 10.8, 12.0]))

    def test_ash(self):
        """Test ash.
        """
        img = self.scene.ash()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], (12.0 - 10.8, 10.8 - 8.7, 10.8))
        self.assertEquals(img.kwargs["crange"], ((-4, 2),
                                                 (-4, 5),
                                                 (243, 303)))
        self.assertTrue("gamma" not in img.kwargs)
        self.assertTrue("stretch" not in img.kwargs)
        #self.assertEquals(self.scene.ash.prerequisites,
        #                  set([8.7, 10.8, 12.0]))


    def test_fog(self):
        """Test fog.
        """
        img = self.scene.fog()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], (12.0 - 10.8, 10.8 - 8.7, 10.8))
        self.assertEquals(img.kwargs["crange"], ((-4, 2),
                                                 (0, 6),
                                                 (243, 283)))
        self.assertEquals(img.kwargs["gamma"], (1.0, 2.0, 1.0))
        self.assertTrue("stretch" not in img.kwargs)
        #self.assertEquals(self.scene.fog.prerequisites,
        #                  set([8.7, 10.8, 12.0]))

    def test_night_fog(self):
        """Test night_fog.
        """
        img = self.scene.night_fog()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], (12.0 - 10.8, 10.8 - 3.75, 10.8))
        self.assertEquals(img.kwargs["crange"], ((-4, 2),
                                                 (0, 6),
                                                 (243, 293)))
        self.assertEquals(img.kwargs["gamma"], (1.0, 2.0, 1.0))
        self.assertTrue("stretch" not in img.kwargs)
        #self.assertEquals(self.scene.night_fog.prerequisites,
        #                  set([3.75, 10.8, 12.0]))

    def test_cloud_top(self):
        """Test cloud_top.
        """
        img = self.scene.cloudtop()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], (-3.75, -10.8, -12.0))
        self.assertTrue("crange" not in img.kwargs)
        self.assertTrue("gamma" not in img.kwargs)
        self.assertEquals(img.kwargs["stretch"], (0.005, 0.005))
        #self.assertEquals(self.scene.cloudtop.prerequisites,
        #                  set([3.75, 10.8, 12.0]))
       
 

    def tearDown(self):
        unpatch_scene()
        unpatch_geo_image()


def suite():
    """The test suite for test_visir.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestComposites))
    
    return mysuite
