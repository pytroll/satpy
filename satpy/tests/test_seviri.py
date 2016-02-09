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

"""Module for testing the :mod:`satpy.instruments.seviri` module.
"""
import random
import unittest

import numpy as np

import satpy.instruments.seviri
from satpy.imageo import geo_image


def patch_scene_mask():
    """Patch the :mod:`satpy.scene` module to avoid using it in these tests.
    """
    class FakeChannel(object):
        """FakeChannel class.
        """
        def __init__(self, val):
            del val
            self.data = np.ma.array((random.random(),))

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
            self.channels = {}
            self.area = None
            self.time_slot = None
            self.error = []
        def check_channels(self, *args):
            """Dummy check_channels function.
            """
            for chn in args:
                if chn in self.error:
                    raise RuntimeError()

        def __getitem__(self, key):
            if key not in self.channels:
                self.channels[key] = FakeChannel(key)
            return self.channels[key]
        
    satpy.instruments.visir.OldVisirCompositer = satpy.instruments.visir.VisirCompositer
    satpy.instruments.visir.VisirCompositer = FakeSatscene
    reload(satpy)
    reload(satpy.instruments)
    reload(satpy.instruments.seviri)

def patch_scene():
    """Patch the :mod:`satpy.scene` module to avoid using it in these tests.
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
            self._data_holder = self
            
        def check_channels(self, *args):
            """Dummy check_channels function.
            """
            self.channels = args

        def __contains__(self, point):
            return True
            

        def __getitem__(self, key):
            if key == "_IR39Corr":
                return FakeChannel(3.75)
            elif key == "HRV":
                return FakeChannel(0.7)
            return FakeChannel(key)

    satpy.instruments.visir.OldVisirCompositer = satpy.instruments.visir.VisirCompositer
    satpy.instruments.visir.VisirCompositer = FakeSatscene
    reload(satpy)
    reload(satpy.instruments)
    reload(satpy.instruments.seviri)

def unpatch_scene():
    """Unpatch the :mod:`satpy.scene` module.
    """
    satpy.instruments.visir.VisirCompositer = satpy.instruments.visir.OldVisirCompositer
    delattr(satpy.instruments.visir, "OldVisirCompositer")
    reload(satpy)
    reload(satpy.instruments)
    reload(satpy.instruments.visir)
    reload(satpy.instruments.seviri)

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
            self.lum = None
            self.channels = [self]
            
        def enhance(self, **kwargs):
            """Dummy enhance function.
            """
            self.kwargs.update(kwargs)

        def replace_luminance(self, lum):
            """Dummy remplace_luminance.
            """
            self.lum = lum
        

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
        self.scene = satpy.instruments.seviri.SeviriCompositer()


    # def test_cloudtop(self):
    #     """Test cloudtop.
    #     """
    #     img = self.scene.cloudtop()
    #     self.assertEquals(img.kwargs["mode"], "RGB")
    #     self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
    #     self.assertEquals(img.args[0], (-3.75, -10.8, -12.0))
    #     self.assertEquals(img.kwargs["stretch"], (0.005, 0.005))
    #     self.assertTrue("crange" not in img.kwargs)
    #     self.assertTrue("gamma" not in img.kwargs)

    # def test_night_fog(self):
    #     """Test night_fog.
    #     """
    #     img = self.scene.night_fog()
    #     self.assertEquals(img.kwargs["mode"], "RGB")
    #     self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
    #     self.assertEquals(img.args[0], (12.0 - 10.8, 10.8 - 3.75, 10.8))
    #     self.assertEquals(img.kwargs["crange"], ((-4, 2),
    #                                              (0, 6),
    #                                              (243, 293)))
    #     self.assertEquals(img.kwargs["gamma"], (1.0, 2.0, 1.0))
    #     self.assertTrue("stretch" not in img.kwargs)

#     def test_hr_overview(self):
#         """Test hr_overview.
#         """
#         img = self.scene.hr_overview()
#         self.assertEquals(img.kwargs["mode"], "RGB")
#         self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
#         self.assertEquals(img.args[0], (0.635, 0.85, -10.8))
#         self.assertEquals(img.kwargs["stretch"], "crude")
#         self.assertEquals(list(img.kwargs["gamma"]), list((1.6, 1.6, 1.1)))
#         self.assertTrue("crange" not in img.kwargs)

#         self.assertEquals(img.lum.kwargs["mode"], "L")
#         self.assertEquals(img.lum.kwargs["crange"], (0, 100))
#         self.assertEquals(img.lum.kwargs["gamma"], 2.0)
#         self.assertTrue("stretch" not in img.lum.kwargs)
#         self.assertTrue("fill_value" not in img.lum.kwargs)

#     def test_hr_visual(self):
#         """Test hr_visual.
#         """
#         img = self.scene.hr_visual()
#         self.assertEquals(img.kwargs["mode"], "L")
#         self.assertEquals(img.kwargs["fill_value"], 0)
#         self.assertEquals(img.args[0], 0.7)
#         self.assertEquals(img.kwargs["stretch"], "crude")
#         self.assertTrue("crange" not in img.kwargs)
#         self.assertTrue("gamma" not in img.kwargs)
        
        

    def tearDown(self):
        unpatch_scene()
        unpatch_geo_image()


class TestCo2Corr(unittest.TestCase):
    """Class for testing the composites.
    """

    def setUp(self):
        """Setup stuff.
        """
        patch_geo_image()
        patch_scene_mask()
        self.scene = satpy.instruments.seviri.SeviriCompositer()


    def test_co2corr(self):
        """Test CO2 correction.
        """
        res = self.scene.co2corr()
        bt039 = self.scene[3.9].data
        bt108 = self.scene[10.8].data
        bt134 = self.scene[13.4].data
        
        dt_co2 = (bt108-bt134)/4.0
        rcorr = bt108 ** 4 - (bt108-dt_co2) ** 4
        
        
        t4_co2corr = bt039 ** 4 + rcorr
        if t4_co2corr < 0.0:
            t4_co2corr = 0
        solution = t4_co2corr ** 0.25
        self.assertEquals(res, solution)

        self.scene.error = [3.75]
        res = self.scene.co2corr()

        self.assertTrue(res is None)


    def tearDown(self):
        unpatch_scene()
        unpatch_geo_image()


def suite():
    """The test suite for test_seviri.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestComposites))
    mysuite.addTest(loader.loadTestsFromTestCase(TestCo2Corr))
    
    return mysuite
