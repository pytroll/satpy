#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2012, 2014.

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
import random
import unittest

from mock import MagicMock
import sys
sys.modules['pyresample'] = MagicMock()

import mpop.scene

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
            self.assertEquals(chn.wavelength_range, list(channels[i][1]))
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
        self.assertEquals(new_scene.area_id, area2)

        for chn in new_scene.channels:
            print chn.area
            self.assertEquals(chn.area, area2)
        
        

def suite():
    """The test suite for test_pp_core.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestPPCore))
    
    return mysuite
