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
import os
import sys

import numpy as np

try:
    from unittest import mock
except ImportError:
    import mock

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest


class TestGeoTIFFWriter(unittest.TestCase):

    def _get_test_datasets(self):
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
        from satpy.writers.geotiff import GeoTIFFWriter
        w = GeoTIFFWriter()

    def test_simple_write(self):
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = self._get_test_datasets()
        w = GeoTIFFWriter()
        w.save_datasets(datasets)

    def test_simple_delayed_write(self):
        from dask.delayed import Delayed
        from satpy.writers.geotiff import GeoTIFFWriter
        datasets = self._get_test_datasets()
        w = GeoTIFFWriter()
        # when we switch to rio_save on XRImage then this will be sources
        # and targets
        res = w.save_datasets(datasets, compute=False)
        self.assertIsInstance(res, Delayed)
        res.compute()


def suite():
    """The test suite for this writer's tests.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGeoTIFFWriter))
    return mysuite
