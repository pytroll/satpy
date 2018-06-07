#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Trygve Aspenes
#
# Author(s):
#
#   Trygve Aspenes <trygveas@met.no>
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
"""Tests for the mitiff writer.
Based on the test for geotiff writer
"""
import sys

import logging

logger = logging.getLogger()
logger.level = logging.DEBUG
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestMITIFFWriter(unittest.TestCase):
    """Test the MITIFF Writer class."""

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

    def _get_test_dataset(self, bands=3):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict=proj4_str_to_dict('+proj=stere +datum=WGS84 +ellps=WGS84 +lon_0=0. +lat_0=90 +lat_ts=60 +units=km'),
            x_size=100,
            y_size=200,
            area_extent=(-1000., -1500., 1000., 1500.),
        )

        ds1 = xr.DataArray(
            da.zeros((bands, 100, 200), chunks=50),
            dims=('bands', 'y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow(),
                   'platform_name': "TEST_PLATFORM_NAME",
                   'sensor': 'TEST_SENSOR_NAME',
                   'area': area_def,
                   'prerequisites': ['1', '2', '3']}
        )
        return ds1

    def test_init(self):
        """Test creating the writer with no arguments."""
        from satpy.writers.mitiff import MITIFFWriter
        MITIFFWriter()

    def test_simple_write(self):
        """Test basic writer operation."""
        from satpy.writers.mitiff import MITIFFWriter
        dataset = self._get_test_dataset()
        w = MITIFFWriter(mitiff_dir=self.base_dir)
        w.save_dataset(dataset, writer='mitiff', mitiff_dir=self.base_dir)


def suite():
    """The test suite for this writer's tests.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMITIFFWriter))
    return mysuite

if __name__ == '__main__':
    unittest.main()
