#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""The enhancements tests package.
"""

import sys

from satpy.tests.enhancement_tests import test_enhancements, test_viirs

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def suite():
    """Test suite for all enhancement tests"""
    mysuite = unittest.TestSuite()
    mysuite.addTests(test_enhancements.suite())
    mysuite.addTests(test_viirs.suite())

    return mysuite
