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
"""Test objects and functions in the satpy.config module."""

import os
import unittest
from unittest import mock


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
        from pyresample.geometry import SwathDefinition
        from satpy.resample import get_area_file
        import numpy as np
        import xarray as xr

        lons = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lats = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lons = xr.DataArray(lons)
        lats = xr.DataArray(lats)
        swath_def = SwathDefinition(lons, lats)
        all_areas = parse_area_file(get_area_file())
        for area_obj in all_areas:
            if hasattr(area_obj, 'freeze'):
                try:
                    area_obj = area_obj.freeze(lonslats=swath_def)
                except RuntimeError:
                    # we didn't provide enough info to freeze, hard to guess
                    # in a generic test so just skip this area
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
        from pyresample.geometry import SwathDefinition
        from satpy.resample import get_area_file
        import numpy as np
        import xarray as xr

        lons = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lats = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lons = xr.DataArray(lons)
        lats = xr.DataArray(lats)
        swath_def = SwathDefinition(lons, lats)
        all_areas = parse_area_file(get_area_file())
        for area_obj in all_areas:
            if hasattr(area_obj, 'freeze'):
                try:
                    area_obj = area_obj.freeze(lonslats=swath_def)
                except RuntimeError:
                    # we didn't provide enough info to freeze, hard to guess
                    # in a generic test so just skip this area
                    continue
            proj_dict = area_obj.proj_dict
            if proj_dict.get('proj') in ('ob_tran', 'nsper') and \
                    'wktext' not in proj_dict:
                # FIXME: rasterio doesn't understand ob_tran unless +wktext
                # See: https://github.com/pyproj4/pyproj/issues/357
                # pyproj 2.0+ seems to drop wktext from PROJ dict
                continue
            _ = CRS.from_dict(proj_dict)


class TestPluginsConfigs(unittest.TestCase):
    """Test that plugins are working."""

    @mock.patch('satpy.config.pkg_resources.iter_entry_points')
    def test_get_plugin_configs(self, iter_entry_points):
        """Check that the plugin configs are looked for."""
        import pkg_resources
        ep = pkg_resources.EntryPoint.parse('example_composites = satpy_cpe')
        ep.dist = pkg_resources.Distribution.from_filename('satpy_cpe-0.0.0-py3.8.egg')
        ep.dist.module_path = os.path.join(os.path.sep + 'bla', 'bla')
        iter_entry_points.return_value = [ep]

        from satpy.config import get_entry_points_config_dirs
        dirs = get_entry_points_config_dirs('satpy.composites')
        self.assertListEqual(dirs, [os.path.join(ep.dist.module_path, 'satpy_cpe', 'etc')])
