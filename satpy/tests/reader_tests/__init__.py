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
                                      test_viirs_sdr, test_viirs_l1b, test_virr_l1b,
                                      test_seviri_l1b_native, test_seviri_base,
                                      test_hdf5_utils, test_netcdf_utils,
                                      test_hdf4_utils, test_utils,
                                      test_acspo, test_amsr2_l1b,
                                      test_omps_edr, test_nucaps, test_geocat,
                                      test_seviri_l1b_calibration, test_clavrx,
                                      test_grib, test_goes_imager_hrit, test_ahi_hsd,
                                      test_iasi_l2, test_generic_image,
                                      test_scmi, test_ahi_hrit, test_goes_imager_nc,
                                      test_nc_slstr, test_olci_nc,
                                      test_viirs_edr_flood, test_nwcsaf_nc,
                                      test_seviri_l1b_hrit, test_sar_c_safe,
                                      test_modis_l1b, test_viirs_edr_active_fires,
                                      test_safe_sar_l2_ocn, test_electrol_hrit)


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
    mysuite.addTests(test_virr_l1b.suite())
    mysuite.addTests(test_hrit_base.suite())
    mysuite.addTests(test_seviri_l1b_native.suite())
    mysuite.addTests(test_seviri_base.suite())
    mysuite.addTests(test_hdf4_utils.suite())
    mysuite.addTests(test_hdf5_utils.suite())
    mysuite.addTests(test_netcdf_utils.suite())
    mysuite.addTests(test_utils.suite())
    mysuite.addTests(test_acspo.suite())
    mysuite.addTests(test_amsr2_l1b.suite())
    mysuite.addTests(test_omps_edr.suite())
    mysuite.addTests(test_nucaps.suite())
    mysuite.addTests(test_geocat.suite())
    mysuite.addTests(test_olci_nc.suite())
    mysuite.addTests(test_seviri_l1b_calibration.suite())
    mysuite.addTests(test_clavrx.suite())
    mysuite.addTests(test_grib.suite())
    mysuite.addTests(test_goes_imager_hrit.suite())
    mysuite.addTests(test_ahi_hsd.suite())
    mysuite.addTests(test_iasi_l2.suite())
    mysuite.addTests(test_generic_image.suite())
    mysuite.addTests(test_scmi.suite())
    mysuite.addTests(test_viirs_edr_flood.suite())
    mysuite.addTests(test_ahi_hrit.suite())
    mysuite.addTests(test_goes_imager_nc.suite())
    mysuite.addTests(test_nc_slstr.suite())
    mysuite.addTests(test_nwcsaf_nc.suite())
    mysuite.addTests(test_seviri_l1b_hrit.suite())
    mysuite.addTests(test_sar_c_safe.suite())
    mysuite.addTests(test_modis_l1b.suite())
    mysuite.addTests(test_viirs_edr_active_fires.suite())
    mysuite.addTests(test_safe_sar_l2_ocn.suite())
    mysuite.addTests(test_electrol_hrit.suite())

    return mysuite
