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

"""Module for testing the :mod:`mpop.instruments.avhrr` module.
"""
import unittest

import mpop.instruments.avhrr
import mpop.instruments.visir
from mpop.imageo import geo_image


def patch_scene():
    """Patch the :mod:`mpop.scene` module to avoid using it in these tests.
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
    mpop.instruments.visir.OldVisirScene = mpop.instruments.visir.VisirScene
    mpop.instruments.visir.VisirScene = FakeSatscene
    reload(mpop)
    reload(mpop.instruments)
    reload(mpop.instruments.avhrr)

def unpatch_scene():
    """Unpatch the :mod:`mpop.scene` module.
    """
    mpop.instruments.visir.VisirScene = mpop.instruments.visir.OldVisirScene
    delattr(mpop.instruments.visir, "OldVisirScene")
    reload(mpop)
    reload(mpop.instruments)
    reload(mpop.instruments.visir)
    reload(mpop.instruments.avhrr)

def patch_geo_image():
    """Patch the :mod:`imageo.geo_image` module to avoid using it in these
    tests.
    """
    class FakeGeoImage:
        """FakeGeoImage class.
        """
        def __init__(self, *args, **kwargs):
            self.args = list(args)
            self.kwargs = kwargs

        def enhance(self, **kwargs):
            """Dummy enhance function.
            """
            self.kwargs.update(kwargs)

        def merge(self, img):
            """Dummy merge function.
            """
            self.args[0] = set(self.args[0]) | set(img.args[0])

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
        self.scene = mpop.instruments.avhrr.AvhrrScene()

    def test_cloudtop(self):
        """Test overview.
        """
        img = self.scene.cloudtop()
        self.assertEquals(img.kwargs["mode"], "RGB")
        self.assertEquals(img.kwargs["fill_value"], (0, 0, 0))
        self.assertEquals(img.args[0], set((1.63, -3.75, -10.8, -12.0)))
        self.assertEquals(img.kwargs["stretch"], (0.005, 0.005))
        self.assertTrue("crange" not in img.kwargs)
        self.assertTrue("gamma" not in img.kwargs)
        #self.assertEquals(self.scene.overview.prerequisites,
        #                  set([10.8, 12.0]))

        self.scene.error = (3.75, 1.63)
        img = self.scene.cloudtop()
        self.assertTrue(img is None)


    def tearDown(self):
        unpatch_scene()
        unpatch_geo_image()


if __name__ == '__main__':
    unittest.main()
