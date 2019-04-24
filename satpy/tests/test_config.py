#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy Developers
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
"""Test objects and functions in the satpy.config module.
"""

import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest
try:
    from unittest import mock
except ImportError:
    import mock


class TestCheckSatpy(unittest.TestCase):
    """Test the 'check_satpy' function."""

    def test_basic_check_satpy(self):
        """Test 'check_satpy' basic functionality."""
        from satpy.config import check_satpy
        check_satpy()

    def test_specific_check_satpy(self):
        """Test 'check_satpy' with specific features provided."""
        from satpy.config import check_satpy
        with mock.patch('satpy.config.print') as print_mock:
            check_satpy(readers=['viirs_sdr'], extras=('cartopy', '__fake'))
            checked_fake = False
            for call in print_mock.mock_calls:
                if len(call[1]) > 0 and '__fake' in call[1][0]:
                    self.assertNotIn('ok', call[1][1])
                    checked_fake = True
            self.assertTrue(checked_fake, "Did not find __fake module "
                                          "mentioned in checks")


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestCheckSatpy))

    return my_suite
