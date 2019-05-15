#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""test file handler baseclass.
"""

import unittest

try:
    from unittest import mock
except ImportError:
    import mock

import numpy as np

from satpy.readers.file_handlers import BaseFileHandler


class TestBaseFileHandler(unittest.TestCase):
    """Test the BaseFileHandler."""

    def setUp(self):
        """Setup the test."""
        self._old_set = BaseFileHandler.__abstractmethods__
        BaseFileHandler._abstractmethods__ = set()
        self.fh = BaseFileHandler(
            'filename', {'filename_info': 'bla'}, 'filetype_info')

    def test_combine_times(self):
        """Combine times."""
        info1 = {'start_time': 1}
        info2 = {'start_time': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'start_time': 1}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'start_time': 1}
        self.assertDictEqual(res, exp)

        info1 = {'end_time': 1}
        info2 = {'end_time': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'end_time': 2}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'end_time': 2}
        self.assertDictEqual(res, exp)

    def test_combine_orbits(self):
        """Combine orbits."""
        info1 = {'start_orbit': 1}
        info2 = {'start_orbit': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'start_orbit': 1}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'start_orbit': 1}
        self.assertDictEqual(res, exp)

        info1 = {'end_orbit': 1}
        info2 = {'end_orbit': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'end_orbit': 2}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'end_orbit': 2}
        self.assertDictEqual(res, exp)

    @mock.patch('satpy.readers.file_handlers.SwathDefinition')
    def test_combine_area(self, sdef):
        """Combine area."""
        area1 = mock.MagicMock()
        area1.lons = np.arange(5)
        area1.lats = np.arange(5)
        area1.name = 'area1'

        area2 = mock.MagicMock()
        area2.lons = np.arange(5)
        area2.lats = np.arange(5)
        area2.name = 'area2'

        info1 = {'area': area1}
        info2 = {'area': area2}

        self.fh.combine_info([info1, info2])
        self.assertTupleEqual(sdef.call_args[1]['lons'].shape, (2, 5))
        self.assertTupleEqual(sdef.call_args[1]['lats'].shape, (2, 5))
        self.assertEqual(sdef.return_value.name, 'area1_area2')

    def tearDown(self):
        """Tear down the test."""
        BaseFileHandler.__abstractmethods__ = self._old_set


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestBaseFileHandler))

    return my_suite
