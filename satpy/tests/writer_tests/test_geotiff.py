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
"""Tests for the geotiff writer.
"""
import sys
import numpy as np

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestGeoTIFFWriter(unittest.TestCase):
    """Test the GeoTIFF Writer class."""

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

    def _get_test_datasets(self):
        """Helper function to create a single test dataset."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow()}
        )
        return [ds1]

    def test_init(self):
        """Test creating the writer with no arguments."""
        from satpy.writers.geotiff import GeoTIFFWriter
        GeoTIFFWriter()

    def test_simple_write(self):
        """Test basic writer operation."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = self._get_test_datasets()
        w = GeoTIFFWriter(base_dir=self.base_dir)
        w.save_datasets(datasets)

    def test_simple_delayed_write(self):
        """Test writing can be delayed."""
        import dask.array as da
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = self._get_test_datasets()
        w = GeoTIFFWriter(base_dir=self.base_dir)
        # when we switch to rio_save on XRImage then this will be sources
        # and targets
        res = w.save_datasets(datasets, compute=False)
        # this will fail if rasterio isn't installed
        self.assertIsInstance(res, tuple)
        # two lists, sources and destinations
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], list)
        self.assertIsInstance(res[1], list)
        self.assertIsInstance(res[0][0], da.Array)
        da.store(res[0], res[1])
        for target in res[1]:
            if hasattr(target, 'close'):
                target.close()

    def test_float_write(self):
        """Test that geotiffs can be written as floats.

        NOTE: Does not actually check that the output is floats.

        """
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = self._get_test_datasets()
        w = GeoTIFFWriter(base_dir=self.base_dir,
                          enhancement_config=False,
                          dtype=np.float32)
        w.save_datasets(datasets)

    def test_fill_value_from_config(self):
        """Test fill_value coming from the writer config."""
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = self._get_test_datasets()
        w = GeoTIFFWriter(base_dir=self.base_dir)
        w.info['fill_value'] = 128
        with mock.patch('satpy.writers.XRImage.save') as save_method:
            save_method.return_value = None
            w.save_datasets(datasets, compute=False)
            self.assertEqual(save_method.call_args[1]['fill_value'], 128)


def suite():
    """The test suite for this writer's tests."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGeoTIFFWriter))
    return mysuite
