#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014 Martin Raspaud

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
"""The tests package.
"""

import sys
import logging

from satpy.tests import test_readers  # test_viirs_sdr,
from satpy.tests import (test_dataset, test_helper_functions, test_resample,
                         test_scene, test_writers, test_yaml_reader)

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def suite():
    """The global test suite.
    """
    logging.basicConfig(level=logging.DEBUG)

    mysuite = unittest.TestSuite()
    mysuite.addTests(test_scene.suite())
    # mysuite.addTests(test_viirs_sdr.suite())
    mysuite.addTests(test_dataset.suite())
    mysuite.addTests(test_writers.suite())
    mysuite.addTests(test_readers.suite())
    mysuite.addTests(test_resample.suite())
    mysuite.addTests(test_yaml_reader.suite())
    mysuite.addTests(test_helper_functions.suite())

    return mysuite


def load_tests(loader, tests, pattern):
    return suite()
