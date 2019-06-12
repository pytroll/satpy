#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""The writer tests package.
"""

import sys

from satpy.tests.writer_tests import (test_cf, test_geotiff,
                                      test_simple_image,
                                      test_scmi, test_mitiff,
                                      test_utils)
# FIXME: pyninjotiff is not xarray/dask friendly
from satpy.tests.writer_tests import test_ninjotiff  # noqa

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


def suite():
    """Test suite for all writer tests"""
    mysuite = unittest.TestSuite()
    mysuite.addTests(test_cf.suite())
    mysuite.addTests(test_geotiff.suite())
    # mysuite.addTests(test_ninjotiff.suite())
    mysuite.addTests(test_simple_image.suite())
    mysuite.addTests(test_scmi.suite())
    mysuite.addTests(test_mitiff.suite())
    mysuite.addTests(test_utils.suite())
    return mysuite
