#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2017, 2018 Martin Raspaud

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
                                      test_hdf4_utils, test_utils,
                                      test_acspo, test_amsr2_l1b,
                                      test_omps_edr, test_nucaps, test_geocat,
                                      test_seviri_calibration, test_clavrx,
                                      test_grib, test_hrit_goes, test_ahi_hsd,
                                      test_iasi_l2, test_generic_image,
                                      test_scmi, test_hrit_jma, test_nc_goes,
                                      test_nc_slstr, test_nc_olci,
                                      test_viirs_flood, test_nc_nwcsaf,
                                      test_hrit_msg)


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
    mysuite.addTests(test_utils.suite())
    mysuite.addTests(test_acspo.suite())
    mysuite.addTests(test_amsr2_l1b.suite())
    mysuite.addTests(test_omps_edr.suite())
    mysuite.addTests(test_nucaps.suite())
    mysuite.addTests(test_geocat.suite())
    mysuite.addTests(test_nc_olci.suite())
    mysuite.addTests(test_seviri_calibration.suite())
    mysuite.addTests(test_clavrx.suite())
    mysuite.addTests(test_grib.suite())
    mysuite.addTests(test_hrit_goes.suite())
    mysuite.addTests(test_ahi_hsd.suite())
    mysuite.addTests(test_iasi_l2.suite())
    mysuite.addTests(test_generic_image.suite())
    mysuite.addTests(test_scmi.suite())
    mysuite.addTests(test_viirs_flood.suite())
    mysuite.addTests(test_hrit_jma.suite())
    mysuite.addTests(test_nc_goes.suite())
    mysuite.addTests(test_nc_slstr.suite())
    mysuite.addTests(test_nc_nwcsaf.suite())
    mysuite.addTests(test_hrit_msg.suite())

    return mysuite
