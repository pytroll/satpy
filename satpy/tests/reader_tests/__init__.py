#!/usr/bin/env python
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
"""The reader tests package.
"""

import sys

from satpy.tests.reader_tests import (test_abi_l1b, test_hrit_base,
                                      test_viirs_sdr, test_viirs_l1b,
                                      test_native_msg, test_msg_base,
                                      test_hdf5_utils, test_netcdf_utils,
                                      test_hdf4_utils,
                                      test_acspo, test_amsr2_l1b,
                                      test_omps_edr, test_nucaps, test_geocat)

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def suite():
    """Test suite for all reader tests"""
    mysuite = unittest.TestSuite()
    mysuite.addTests(test_abi_l1b.suite())
    mysuite.addTests(test_viirs_sdr.suite())
    mysuite.addTests(test_viirs_l1b.suite())
    mysuite.addTests(test_hrit_base.suite())
    mysuite.addTests(test_native_msg.suite())
    mysuite.addTests(test_msg_base.suite())
    mysuite.addTests(test_hdf4_utils.suite())
    mysuite.addTests(test_hdf5_utils.suite())
    mysuite.addTests(test_netcdf_utils.suite())
    mysuite.addTests(test_acspo.suite())
    mysuite.addTests(test_amsr2_l1b.suite())
    mysuite.addTests(test_omps_edr.suite())
    mysuite.addTests(test_nucaps.suite())
    mysuite.addTests(test_geocat.suite())

    return mysuite
