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
"""Test objects and functions in the satpy.config module.
"""

import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest
try:
    from unittest import mock
except ImportError:
    import mock


class TestCheckSatpy(unittest.TestCase):
    """Test the 'check_satpy' function."""

    def test_basic_check_satpy(self):
        """Test 'check_satpy' basic functionality."""
        from satpy.config import check_satpy
        check_satpy()

    def test_specific_check_satpy(self):
        """Test 'check_satpy' with specific features provided."""
        from satpy.config import check_satpy
        with mock.patch('satpy.config.print') as print_mock:
            check_satpy(readers=['viirs_sdr'], extras=('cartopy', '__fake'))
            checked_fake = False
            for call in print_mock.mock_calls:
                if len(call[1]) > 0 and '__fake' in call[1][0]:
                    self.assertNotIn('ok', call[1][1])
                    checked_fake = True
            self.assertTrue(checked_fake, "Did not find __fake module "
                                          "mentioned in checks")


class TestBuiltinAreas(unittest.TestCase):
    """Test that the builtin areas are all valid."""

    def test_areas_pyproj(self):
        """Test all areas have valid projections with pyproj."""
        import pyproj
        from pyresample import parse_area_file
        from satpy.resample import get_area_file

        all_areas = parse_area_file(get_area_file())
        for area_obj in all_areas:
            if getattr(area_obj, 'optimize_projection', False):
                # the PROJ.4 is known to not be valid on this DynamicAreaDef
                continue
            proj_dict = area_obj.proj_dict
            _ = pyproj.Proj(proj_dict)

    def test_areas_rasterio(self):
        """Test all areas have valid projections with rasterio."""
        try:
            from rasterio.crs import CRS
        except ImportError:
            return unittest.skip("Missing rasterio dependency")
        if not hasattr(CRS, 'from_dict'):
            return unittest.skip("RasterIO 1.0+ required")

        from pyresample import parse_area_file
        from satpy.resample import get_area_file
        all_areas = parse_area_file(get_area_file())
        for area_obj in all_areas:
            if getattr(area_obj, 'optimize_projection', False):
                # the PROJ.4 is known to not be valid on this DynamicAreaDef
                continue
            proj_dict = area_obj.proj_dict
            _ = CRS.from_dict(proj_dict)


def suite():
    """The test suite for test_config."""
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestCheckSatpy))
    my_suite.addTest(loader.loadTestsFromTestCase(TestBuiltinAreas))

    return my_suite
