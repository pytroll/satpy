#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2015-2018 Satpy Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Test objects and functions in the dataset module.
"""

import sys
from datetime import datetime

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestDatasetID(unittest.TestCase):
    """Test DatasetID object creation and other methods."""

    def test_basic_init(self):
        """Test basic ways of creating a DatasetID."""
        from satpy.dataset import DatasetID
        DatasetID(name="a")
        DatasetID(name="a", wavelength=0.86)
        DatasetID(name="a", resolution=1000)
        DatasetID(name="a", calibration='radiance')
        DatasetID(name="a", wavelength=0.86, resolution=250,
                  calibration='radiance')
        DatasetID(name="a", wavelength=0.86, resolution=250,
                       calibration='radiance', modifiers=('sunz_corrected',))
        DatasetID(wavelength=0.86)

    def test_init_bad_modifiers(self):
        """Test that modifiers are a tuple."""
        from satpy.dataset import DatasetID
        self.assertRaises(TypeError, DatasetID, name="a", modifiers="str")

    def test_compare_no_wl(self):
        """Compare fully qualified wavelength ID to no wavelength ID."""
        from satpy.dataset import DatasetID
        d1 = DatasetID(name="a", wavelength=(0.1, 0.2, 0.3))
        d2 = DatasetID(name="a", wavelength=None)

        # this happens when sorting IDs during dependency checks
        self.assertFalse(d1 < d2)
        self.assertTrue(d2 < d1)


class TestCombineMetadata(unittest.TestCase):
    """Test how metadata is combined."""

    def test_average_datetimes(self):
        """Test the average_datetimes helper function."""
        from satpy.dataset import average_datetimes
        dts = (
            datetime(2018, 2, 1, 11, 58, 0),
            datetime(2018, 2, 1, 11, 59, 0),
            datetime(2018, 2, 1, 12, 0, 0),
            datetime(2018, 2, 1, 12, 1, 0),
            datetime(2018, 2, 1, 12, 2, 0),
        )
        ret = average_datetimes(dts)
        self.assertEqual(dts[2], ret)

    def test_combine_times(self):
        """Test the combine_metadata with times."""
        from satpy.dataset import combine_metadata
        dts = (
            {'start_time': datetime(2018, 2, 1, 11, 58, 0)},
            {'start_time': datetime(2018, 2, 1, 11, 59, 0)},
            {'start_time': datetime(2018, 2, 1, 12, 0, 0)},
            {'start_time': datetime(2018, 2, 1, 12, 1, 0)},
            {'start_time': datetime(2018, 2, 1, 12, 2, 0)},
        )
        ret = combine_metadata(*dts)
        self.assertEqual(dts[2]['start_time'], ret['start_time'])
        ret = combine_metadata(*dts, average_times=False)
        # times are not equal so don't include it in the final result
        self.assertNotIn('start_time', ret)


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestDatasetID))
    my_suite.addTest(loader.loadTestsFromTestCase(TestCombineMetadata))

    return my_suite
