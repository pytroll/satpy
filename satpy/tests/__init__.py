#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014, 2017 Martin Raspaud

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

import logging
import sys

from satpy.tests import (reader_tests, test_dataset, test_file_handlers,
                         test_readers, test_resample, test_demo,
                         test_scene, test_utils, test_writers,
                         test_yaml_reader, writer_tests,
                         test_enhancements, compositor_tests, test_multiscene)


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
    mysuite.addTests(test_dataset.suite())
    mysuite.addTests(test_writers.suite())
    mysuite.addTests(test_readers.suite())
    mysuite.addTests(test_resample.suite())
    mysuite.addTests(test_demo.suite())
    mysuite.addTests(test_yaml_reader.suite())
    mysuite.addTests(reader_tests.suite())
    mysuite.addTests(writer_tests.suite())
    mysuite.addTests(test_file_handlers.suite())
    mysuite.addTests(test_utils.suite())
    mysuite.addTests(test_enhancements.suite())
    mysuite.addTests(compositor_tests.suite())
    mysuite.addTests(test_multiscene.suite())

    return mysuite


def load_tests(loader, tests, pattern):
    return suite()
