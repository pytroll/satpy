#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018

# Author(s):

#   Panu Lahtinen <panu.lahtinen@fmi.fi>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test compositors."""
import unittest
import datetime as dt

import numpy as np

from satpy.dataset import Dataset
from satpy.composites import DayNightCompositor
from satpy.resample import get_area_def

AREA = get_area_def("scan")


class TestDayNightCompositor(unittest.TestCase):

    """
    Test DayNightCompositor
    """

    shp = (3, AREA.shape[0], AREA.shape[1])
    # Create day and night data. Adjust half of the image so that the
    # linear stretching doesn't remove all the data
    day_data = Dataset(100 * np.ones(shp, dtype=np.uint8),
                       start_time=dt.datetime(2018, 1, 23, 8, 0),
                       area=AREA)
    day_data[:, int(AREA.shape[0] / 2):, :] += 10
    night_data = Dataset(np.zeros(shp, dtype=np.uint8))
    night_data[:, :, int(AREA.shape[0] / 2):] += 10

    # Add day/night division to
    day_sun_zen = 80.
    night_sun_zen = 100.
    sun_zen = night_sun_zen * np.ones(AREA.shape)
    sun_zen[:, int(AREA.shape[1] / 2):] = day_sun_zen

    def test_with_sunzen(self):
        comp = DayNightCompositor('foo')
        img = comp([self.day_data.copy(), self.night_data.copy(),
                    self.sun_zen.copy()])
        # With this test data, only bottom-right corner should be ones
        for i in range(3):
            bottom_right = img[i,
                               int(AREA.shape[0] / 2):,
                               int(AREA.shape[1] / 2):]
        self.assertTrue(np.all(bottom_right == 1))
        num_ones = (img == 1).sum()
        self.assertEqual(num_ones, 3 * bottom_right.size)
        # All the other quadrants should be zeros
        num_zeros = (img == 0).sum()
        self.assertTrue(num_ones + num_zeros == img.size)

    def test_without_sunzen(self):
        comp = DayNightCompositor('foo')
        img = comp([self.day_data.copy(), self.night_data.copy()])
        for i in range(3):
            # With this test data, top-left corner should be zeros
            self.assertTrue(np.all(img[i,
                                       :int(AREA.shape[0] / 2),
                                       :int(AREA.shape[1] / 2)] == 0))
            # And bottom-right corner should be ones
            self.assertTrue(np.all(img[i,
                                       int(AREA.shape[0] / 2):,
                                       int(AREA.shape[1] / 2):] == 1))


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestDayNightCompositor))

    return my_suite
