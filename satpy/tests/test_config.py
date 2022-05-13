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
import sys
import unittest
from unittest import mock

import pytest


class TestBuiltinAreas(unittest.TestCase):
    """Test that the builtin areas are all valid."""

    def test_areas_pyproj(self):
        """Test all areas have valid projections with pyproj."""
        import numpy as np
        import pyproj
        import xarray as xr
        from pyresample import parse_area_file
        from pyresample.geometry import SwathDefinition

        from satpy.resample import get_area_file

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

        import numpy as np
        import xarray as xr
        from pyresample import parse_area_file
        from pyresample.geometry import SwathDefinition

        from satpy.resample import get_area_file

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

    @mock.patch('satpy._config.pkg_resources.iter_entry_points')
    def test_get_plugin_configs(self, iter_entry_points):
        """Check that the plugin configs are looked for."""
        import pkg_resources
        ep = pkg_resources.EntryPoint.parse('example_composites = satpy_cpe')
        ep.dist = pkg_resources.Distribution.from_filename('satpy_cpe-0.0.0-py3.8.egg')
        ep.dist.module_path = os.path.join(os.path.sep + 'bla', 'bla')
        iter_entry_points.return_value = [ep]

        import satpy
        from satpy._config import get_entry_points_config_dirs

        # don't let user env vars affect results
        with satpy.config.set(config_path=[]):
            dirs = get_entry_points_config_dirs('satpy.composites')
            self.assertListEqual(dirs, [os.path.join(ep.dist.module_path, 'satpy_cpe', 'etc')])


class TestConfigObject:
    """Test basic functionality of the central config object."""

    def test_custom_config_file(self):
        """Test adding a custom configuration file using SATPY_CONFIG."""
        import tempfile
        from importlib import reload

        import yaml

        import satpy
        my_config_dict = {
            'cache_dir': "/path/to/cache",
        }
        try:
            with tempfile.NamedTemporaryFile(mode='w+t', suffix='.yaml', delete=False) as tfile:
                yaml.dump(my_config_dict, tfile)
                tfile.close()
                with mock.patch.dict('os.environ', {'SATPY_CONFIG': tfile.name}):
                    reload(satpy._config)
                    reload(satpy)
                    assert satpy.config.get('cache_dir') == '/path/to/cache'
        finally:
            os.remove(tfile.name)

    def test_deprecated_env_vars(self):
        """Test that deprecated variables are mapped to new config."""
        from importlib import reload

        import satpy
        old_vars = {
            'PPP_CONFIG_DIR': '/my/ppp/config/dir',
            'SATPY_ANCPATH': '/my/ancpath',
        }

        with mock.patch.dict('os.environ', old_vars):
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('data_dir') == '/my/ancpath'
            assert satpy.config.get('config_path') == ['/my/ppp/config/dir']

    def test_config_path_multiple(self):
        """Test that multiple config paths are accepted."""
        from importlib import reload

        import satpy
        exp_paths, env_paths = _os_specific_multipaths()
        old_vars = {
            'SATPY_CONFIG_PATH': env_paths,
        }

        with mock.patch.dict('os.environ', old_vars):
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('config_path') == exp_paths

    def test_config_path_multiple_load(self):
        """Test that config paths from subprocesses load properly.

        Satpy modifies the config path environment variable when it is imported.
        If Satpy is imported again from a subprocess then it should be able to parse this
        modified variable.
        """
        from importlib import reload

        import satpy
        exp_paths, env_paths = _os_specific_multipaths()
        old_vars = {
            'SATPY_CONFIG_PATH': env_paths,
        }

        with mock.patch.dict('os.environ', old_vars):
            # these reloads will update env variable "SATPY_CONFIG_PATH"
            reload(satpy._config)
            reload(satpy)

            # load the updated env variable and parse it again.
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('config_path') == exp_paths

    def test_bad_str_config_path(self):
        """Test that a str config path isn't allowed."""
        from importlib import reload

        import satpy
        old_vars = {
            'SATPY_CONFIG_PATH': '/my/configs1',
        }

        # single path from env var still works
        with mock.patch.dict('os.environ', old_vars):
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('config_path') == ['/my/configs1']

        # strings are not allowed, lists are
        with satpy.config.set(config_path='/single/string/paths/are/bad'):
            pytest.raises(ValueError, satpy._config.get_config_path_safe)


def _os_specific_multipaths():
    exp_paths = ['/my/configs1', '/my/configs2', '/my/configs3']
    if sys.platform.startswith("win"):
        exp_paths = ["C:" + p for p in exp_paths]
    path_str = os.pathsep.join(exp_paths)
    return exp_paths, path_str
