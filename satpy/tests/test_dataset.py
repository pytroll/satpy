#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Test objects and functions in the dataset module."""

from datetime import datetime

import unittest
import pytest


class TestDatasetID(unittest.TestCase):
    """Test DatasetID object creation and other methods."""

    def test_make_dsid_class(self):
        """Test making a new DatasetID class."""
        from satpy.dataset import make_dsid_class, WavelengthRange, ModifierTuple
        types = {'wavelength': WavelengthRange,
                 'modifiers': ModifierTuple}
        klass = make_dsid_class(types, name='', wavelength=None, modifiers=ModifierTuple())
        dsid = klass('hej', (1., 2., 3.))
        klass2 = make_dsid_class(name='', polarization='')
        dsid2 = klass2('hej')
        assert(dsid == dsid2)

        klass3 = make_dsid_class(types, name='', wavelength=None, view='nadir')
        dsid3 = klass3('hej', 2.)
        assert(dsid == dsid3)
        assert(hash(dsid))

    def test_basic_init(self):
        """Test basic ways of creating a DatasetID."""
        from satpy.dataset import make_dsid_class, WavelengthRange, ModifierTuple
        types = {'wavelength': WavelengthRange,
                 'modifiers': ModifierTuple}
        DatasetID = make_dsid_class(types,
                                    name='', wavelength=None, resolution=None,
                                    calibration=None, modifiers=ModifierTuple())

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
        from satpy.dataset import make_dsid_class, WavelengthRange, ModifierTuple
        types = {'wavelength': WavelengthRange,
                 'modifiers': ModifierTuple}
        DatasetID = make_dsid_class(types, name='', wavelength=None, modifiers=ModifierTuple())
        self.assertRaises(TypeError, DatasetID, name="a", modifiers="str")

    def test_compare_no_wl(self):
        """Compare fully qualified wavelength ID to no wavelength ID."""
        from satpy.dataset import make_dsid_class, WavelengthRange, ModifierTuple
        types = {'wavelength': WavelengthRange,
                 'modifiers': ModifierTuple}
        DatasetID = make_dsid_class(types, name='', wavelength=None, modifiers=ModifierTuple())
        d1 = DatasetID(name="a", wavelength=(0.1, 0.2, 0.3))
        d2 = DatasetID(name="a", wavelength=None)

        # this happens when sorting IDs during dependency checks
        self.assertFalse(d1 < d2)
        self.assertTrue(d2 < d1)

    def test_bad_calibration(self):
        """Test that asking for a bad calibration fails."""
        from satpy.tests.utils import DatasetID
        with pytest.raises(ValueError):
            DatasetID(name='C05', calibration='_bad_')


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
