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
try:
    from unittest import mock
except ImportError:
    import mock

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
    day_data = Dataset(255 * np.ones(shp, dtype=np.uint8),
                       start_time=dt.datetime(2018, 1, 23, 8, 0),
                       area=AREA)
    night_data = Dataset(np.zeros(shp, dtype=np.uint8))

    # Add day/night division to
    day_sun_zen = 80.
    night_sun_zen = 100.
    sun_zen = night_sun_zen * np.ones(AREA.shape)
    sun_zen[:, int(AREA.shape[1] / 2):] = day_sun_zen

    def test_with_coszen(self):
        comp = DayNightCompositor('foo')
        img = comp([self.day_data, self.night_data, self.sun_zen])
        # Do something to test that the left half of the image is
        # zeros and right half is ones.  There should be 131072 of
        # both values.

    def test_without_coszen(self):
        comp = DayNightCompositor('foo')
        img = comp([self.day_data, self.night_data])
        # Do something to test that zenith angles are computed and
        # that in the northern part the image is zeros / southern part
        # is full of ones.


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestDayNightCompositor))

    return my_suite
