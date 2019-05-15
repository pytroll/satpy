#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 David Hoese
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
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
"""Tests for the NinJoTIFF writer.
"""
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestNinjoTIFFWriter(unittest.TestCase):
    def test_init(self):
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        NinjoTIFFWriter()


def suite():
    """The test suite for this writer's tests."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNinjoTIFFWriter))
    return mysuite
