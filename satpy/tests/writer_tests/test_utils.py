#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Tests for writer utilities"""

import unittest
import satpy.writers.utils as wutils


class WriterUtilsTest(unittest.TestCase):
    def test_flatten_dict(self):
        d = {'a': 1, 'b': {'c': 1, 'd': {'e': 1, 'f': {'g': [1, 2]}}}}
        expected = {'a': 1,
                    'b_c': 1,
                    'b_d_e': 1,
                    'b_d_f_g': [1, 2]}
        self.assertDictEqual(wutils.flatten_dict(d), expected)
