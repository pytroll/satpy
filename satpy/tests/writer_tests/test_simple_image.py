#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""Tests for the simple image writer."""
import unittest


class TestPillowWriter(unittest.TestCase):
    """Test Pillow/PIL writer."""

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
        from datetime import datetime

        import dask.array as da
        import xarray as xr
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

        from satpy.writers import compute_writer_results
        from satpy.writers.simple_image import PillowWriter
        datasets = self._get_test_datasets()
        w = PillowWriter(base_dir=self.base_dir)
        res = w.save_datasets(datasets, compute=False)
        for r__ in res:
            self.assertIsInstance(r__, Delayed)
            r__.compute()
        compute_writer_results(res)
