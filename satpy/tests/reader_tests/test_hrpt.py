#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2021 Satpy developers
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
"""Tests for the hrpt reader."""

import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from satpy.readers.hrpt import dtype


class TestHRPTReading(unittest.TestCase):
    """Test case for reading hrpt data."""

    def setUp(self) -> None:
        """Set up the test case."""
        np.ones(10, dtype=dtype)
        with NamedTemporaryFile(mode='w+', suffix='.hmf', delete=False) as hrpt_file:
            self.filename = hrpt_file.name

    def test_reading(self):
        """Test that fails."""
        self.assertEqual(True, False)
