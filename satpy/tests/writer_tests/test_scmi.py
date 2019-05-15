#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017-2018 Satpy Developers
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
"""Tests for the SCMI writer
"""
import os
import sys
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import dask.array as da

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestSCMIWriter(unittest.TestCase):
    """Test basic functionality of SCMI writer."""

    def setUp(self):
        """Create temporary directory to save files to."""
        import tempfile
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_init(self):
        """Test basic init method of writer."""
        from satpy.writers.scmi import SCMIWriter
        SCMIWriter(base_dir=self.base_dir)

    def test_basic_numbered_1_tile(self):
        """Test creating a single numbered tile."""
        from satpy.writers.scmi import SCMIWriter
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. '
                              '+lat_0=25 +lat_1=25 +units=m +no_defs'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        now = datetime(2018, 1, 1, 12, 0, 0)
        ds = DataArray(
            da.from_array(np.linspace(0., 1., 20000, dtype=np.float32).reshape((200, 100)), chunks=50),
            attrs=dict(
                name='test_ds',
                platform_name='PLAT',
                sensor='SENSOR',
                units='1',
                area=area_def,
                start_time=now,
                end_time=now + timedelta(minutes=20))
        )
        w.save_datasets([ds], sector_id='TEST', source_name='TESTS')
        all_files = glob(os.path.join(self.base_dir, 'TESTS_AII*.nc'))
        self.assertEqual(len(all_files), 1)
        self.assertEqual(os.path.basename(all_files[0]), 'TESTS_AII_PLAT_SENSOR_test_ds_TEST_T001_20180101_1200.nc')

    def test_basic_numbered_tiles(self):
        """Test creating a multiple numbered tiles."""
        from satpy.writers.scmi import SCMIWriter
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. '
                              '+lat_0=25 +lat_1=25 +units=m +no_defs'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        now = datetime(2018, 1, 1, 12, 0, 0)
        ds = DataArray(
            da.from_array(np.linspace(0., 1., 20000, dtype=np.float32).reshape((200, 100)), chunks=50),
            attrs=dict(
                name='test_ds',
                platform_name='PLAT',
                sensor='SENSOR',
                units='1',
                area=area_def,
                start_time=now,
                end_time=now + timedelta(minutes=20))
        )
        w.save_datasets([ds], sector_id='TEST', source_name="TESTS", tile_count=(3, 3))
        all_files = glob(os.path.join(self.base_dir, 'TESTS_AII*.nc'))
        self.assertEqual(len(all_files), 9)

    def test_basic_lettered_tiles(self):
        """Test creating a lettered grid."""
        from satpy.writers.scmi import SCMIWriter
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. '
                              '+lat_0=25 +lat_1=25 +units=m +no_defs'),
            1000,
            2000,
            (-1000000., -1500000., 1000000., 1500000.),
        )
        now = datetime(2018, 1, 1, 12, 0, 0)
        ds = DataArray(
            da.from_array(np.linspace(0., 1., 2000000, dtype=np.float32).reshape((2000, 1000)), chunks=500),
            attrs=dict(
                name='test_ds',
                platform_name='PLAT',
                sensor='SENSOR',
                units='1',
                area=area_def,
                start_time=now,
                end_time=now + timedelta(minutes=20))
        )
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        all_files = glob(os.path.join(self.base_dir, 'TESTS_AII*.nc'))
        self.assertEqual(len(all_files), 16)

    def test_lettered_tiles_no_fit(self):
        """Test creating a lettered grid with no data."""
        from satpy.writers.scmi import SCMIWriter
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. '
                              '+lat_0=25 +lat_1=25 +units=m +no_defs'),
            1000,
            2000,
            (4000000., 5000000., 5000000., 6000000.),
        )
        now = datetime(2018, 1, 1, 12, 0, 0)
        ds = DataArray(
            da.from_array(np.linspace(0., 1., 2000000, dtype=np.float32).reshape((2000, 1000)), chunks=500),
            attrs=dict(
                name='test_ds',
                platform_name='PLAT',
                sensor='SENSOR',
                units='1',
                area=area_def,
                start_time=now,
                end_time=now + timedelta(minutes=20))
        )
        w.save_datasets([ds], sector_id='LCC', source_name="TESTS", tile_count=(3, 3), lettered_grid=True)
        # No files created
        all_files = glob(os.path.join(self.base_dir, 'TESTS_AII*.nc'))
        self.assertEqual(len(all_files), 0)

    def test_lettered_tiles_bad_filename(self):
        """Test creating a lettered grid with a bad filename."""
        from satpy.writers.scmi import SCMIWriter
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True, filename="{Bad Key}.nc")
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. '
                              '+lat_0=25 +lat_1=25 +units=m +no_defs'),
            1000,
            2000,
            (-1000000., -1500000., 1000000., 1500000.),
        )
        now = datetime(2018, 1, 1, 12, 0, 0)
        ds = DataArray(
            da.from_array(np.linspace(0., 1., 2000000, dtype=np.float32).reshape((2000, 1000)), chunks=500),
            attrs=dict(
                name='test_ds',
                platform_name='PLAT',
                sensor='SENSOR',
                units='1',
                area=area_def,
                start_time=now,
                end_time=now + timedelta(minutes=20))
        )
        self.assertRaises(KeyError, w.save_datasets,
                          [ds],
                          sector_id='LCC',
                          source_name='TESTS',
                          tile_count=(3, 3),
                          lettered_grid=True)

    def test_basic_numbered_tiles_rgb(self):
        """Test creating a multiple numbered tiles with RGB."""
        from satpy.writers.scmi import SCMIWriter
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. '
                              '+lat_0=25 +lat_1=25 +units=m +no_defs'),
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        now = datetime(2018, 1, 1, 12, 0, 0)
        ds = DataArray(
            da.from_array(np.linspace(0., 1., 60000, dtype=np.float32).reshape((3, 200, 100)), chunks=50),
            dims=('bands', 'y', 'x'),
            coords={'bands': ['R', 'G', 'B']},
            attrs=dict(
                name='test_ds',
                platform_name='PLAT',
                sensor='SENSOR',
                units='1',
                area=area_def,
                start_time=now,
                end_time=now + timedelta(minutes=20))
        )
        w.save_datasets([ds], sector_id='TEST', source_name="TESTS", tile_count=(3, 3))
        all_files = glob(os.path.join(self.base_dir, 'TESTS_AII*test_ds_R*.nc'))
        self.assertEqual(len(all_files), 9)
        all_files = glob(os.path.join(self.base_dir, 'TESTS_AII*test_ds_G*.nc'))
        self.assertEqual(len(all_files), 9)
        all_files = glob(os.path.join(self.base_dir, 'TESTS_AII*test_ds_B*.nc'))
        self.assertEqual(len(all_files), 9)


def suite():
    """The test suite for this writer's tests."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSCMIWriter))
    return mysuite
