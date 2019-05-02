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
"""Test CREFL rayleigh correction functions.
"""
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestCreflUtils(unittest.TestCase):
    """Test crefl_utils."""

    def test_get_atm_variables_abi(self):
        import numpy as np
        from satpy.composites.crefl_utils import get_atm_variables_abi
        sphalb, rhoray, TtotraytH2O, tOG = get_atm_variables_abi(0.17690244, 6.123234e-17, 530.61332168, 405.,
                                                                 21.71342113, 77.14385758, 56.214566960,
                                                                 0.0043149700000000004, 0.0037296,
                                                                 0.014107995000000002, 0.052349)
        self.assertLess(abs(np.array(sphalb) - 0.045213532544630494), 1e-10)
        self.assertLess(abs(rhoray - 2.2030281148621356), 1e-10)
        self.assertLess(abs(TtotraytH2O - 0.30309880915889087), 1e-10)
        self.assertLess(abs(tOG - 0.5969089524560548), 1e-10)


def suite():
    """The test suite for test_crefl_utils."""

    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCreflUtils))
    return mysuite
