#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 David Hoese
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
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
from datetime import datetime, timedelta

import numpy as np

try:
    from unittest import mock
except ImportError:
    import mock

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestSCMIWriter(unittest.TestCase):
    """Test basic functionality of SCMI writer"""

    def setUp(self):
        """Create temporary directory to save files to"""
        import tempfile
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test"""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_init(self):
        """Test basic init method of writer"""
        from satpy.writers.scmi import SCMIWriter
        w = SCMIWriter(base_dir=self.base_dir)

    def test_basic_numbered_1_tile(self):
        """Test creating a single numbered tile"""
        from satpy.writers.scmi import SCMIWriter
        from satpy import Dataset
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict=proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs'),
            x_size=100,
            y_size=200,
            area_extent=(-1000., -1500., 1000., 1500.),
        )
        now = datetime.utcnow()
        ds = Dataset(
            np.linspace(0., 1., 20000, dtype=np.float32).reshape((200, 100)),
            name='test_ds',
            platform='PLAT',
            sensor='SENSOR',
            units='1',
            area=area_def,
            start_time=now,
            end_time=now + timedelta(minutes=20),
        )
        fn = w.save_dataset(ds, sector_id='TEST', source_name="TESTS")
        self.assertTrue(os.path.isfile(fn))

    def test_basic_numbered_tiles(self):
        """Test creating a multiple numbered tiles"""
        from satpy.writers.scmi import SCMIWriter
        from satpy import Dataset
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict=proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs'),
            x_size=100,
            y_size=200,
            area_extent=(-1000., -1500., 1000., 1500.),
        )
        now = datetime.utcnow()
        ds = Dataset(
            np.linspace(0., 1., 20000, dtype=np.float32).reshape((200, 100)),
            name='test_ds',
            platform='PLAT',
            sensor='SENSOR',
            units='1',
            area=area_def,
            start_time=now,
            end_time=now + timedelta(minutes=20),
        )
        fn = w.save_dataset(ds,
                            sector_id='TEST',
                            source_name="TESTS",
                            tile_count=(3, 3))
        # `fn` is currently the last file created
        self.assertTrue(os.path.isfile(fn))
        self.assertIn('T009', fn)

    def test_basic_lettered_tiles(self):
        """Test creating a lettered grid"""
        from satpy.writers.scmi import SCMIWriter
        from satpy import Dataset
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        w = SCMIWriter(base_dir=self.base_dir, compress=True)
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict=proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs'),
            x_size=1000,
            y_size=2000,
            area_extent=(-1000000., -1500000., 1000000., 1500000.),
        )
        now = datetime.utcnow()
        ds = Dataset(
            np.linspace(0., 1., 2000000, dtype=np.float32).reshape((2000, 1000)),
            name='test_ds',
            platform='PLAT',
            sensor='SENSOR',
            units='1',
            area=area_def,
            start_time=now,
            end_time=now + timedelta(minutes=20),
        )
        fn = w.save_dataset(ds,
                            sector_id='LCC',
                            source_name="TESTS",
                            tile_count=(3, 3),
                            lettered_grid=True)
        # `fn` is currently the last file created
        print(fn)
        self.assertTrue(os.path.isfile(fn))


import logging
logging.basicConfig(level=logging.DEBUG)
def suite():
    """The test suite for this writer's tests.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSCMIWriter))
    return mysuite
