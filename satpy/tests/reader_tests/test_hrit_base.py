# -*- coding: utf-8 -*-

# Copyright (c) 2017 Martin Raspaud

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
"""The hrit base reader tests package.
"""

import sys
from datetime import datetime

import numpy as np

from satpy.readers.hrit_base import HRITFileHandler, dec10216

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestDec10216(unittest.TestCase):
    """Test the dec10216 function."""

    def test_dec10216(self):
        res = dec10216(np.array([255, 255, 255, 255, 255], dtype=np.uint8))
        exp = (np.ones((4, )) * 1023).astype(np.uint16)
        self.assertTrue(np.all(res == exp))
        res = dec10216(np.array([1, 1, 1, 1, 1], dtype=np.uint8))
        exp = np.array([4,  16,  64, 257], dtype=np.uint16)
        self.assertTrue(np.all(res == exp))


class TestHRITFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.hrit_base.np.fromfile')
    def setUp(self, fromfile):
        """Setup the hrit file handler for testing."""
        m = mock.mock_open()
        fromfile.return_value = np.array([(1, 2)], dtype=[('total_header_length', int),
                                                          ('hdr_id', int)])

        with mock.patch('satpy.readers.hrit_base.open', m, create=True) as newopen:
            newopen.return_value.__enter__.return_value.tell.return_value = 1
            self.reader = HRITFileHandler('filename',
                                          {'platform_shortname': 'MSG3',
                                           'start_time': datetime(2016, 3, 3, 0, 0)},
                                          {'filetype': 'info'},
                                          [mock.MagicMock(), mock.MagicMock(),
                                           mock.MagicMock()])

    def test_get_xy_from_linecol(self):
        """Test get_xy_from_linecol."""
        x__, y__ = self.reader.get_xy_from_linecol(0, 0, (10, 10), (5, 5))
        self.assertEqual(-131072, x__)
        self.assertEqual(-131072, y__)
        x__, y__ = self.reader.get_xy_from_linecol(10, 10, (10, 10), (5, 5))
        self.assertEqual(0, x__)
        self.assertEqual(0, y__)
        x__, y__ = self.reader.get_xy_from_linecol(20, 20, (10, 10), (5, 5))
        self.assertEqual(131072, x__)
        self.assertEqual(131072, y__)

    def test_get_area_extent(self):
        res = self.reader.get_area_extent((20, 20), (10, 10), (5, 5), 33)
        exp = (-71717.44995740513, -71717.44995740513,
               79266.655216079365, 79266.655216079365)
        self.assertTupleEqual(res, exp)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestDec10216))
    mysuite.addTest(loader.loadTestsFromTestCase(TestHRITFileHandler))
    return mysuite
