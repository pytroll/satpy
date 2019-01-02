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
"""Tests for the CF writer.
"""
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestPillowWriter(unittest.TestCase):

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

    @staticmethod
    def _get_test_datasets():
        """Create DataArray for testing."""
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
        """Test creating the default writer."""
        from satpy.writers.simple_image import PillowWriter
        PillowWriter()

    def test_simple_write(self):
        """Test writing datasets with default behavior."""
        from satpy.writers.simple_image import PillowWriter
        datasets = self._get_test_datasets()
        w = PillowWriter(base_dir=self.base_dir)
        w.save_datasets(datasets)

    def test_simple_delayed_write(self):
        """Test writing datasets with delayed computation."""
        from dask.delayed import Delayed
        from satpy.writers.simple_image import PillowWriter
        from satpy.writers import compute_writer_results
        datasets = self._get_test_datasets()
        w = PillowWriter(base_dir=self.base_dir)
        res = w.save_datasets(datasets, compute=False)
        for r__ in res:
            self.assertIsInstance(r__, Delayed)
            r__.compute()
        compute_writer_results(res)


def suite():
    """The test suite for this writer's tests."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestPillowWriter))
    return mysuite
