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
"""Tests for the NinJoTIFF writer."""

import sys
import unittest
import xarray as xr
from dask.delayed import Delayed
import six
if six.PY3:
    from unittest import mock
else:
    import mock


class FakeImage:
    """Fake image."""

    def __init__(self, data, mode):
        """Init fake image."""
        self.data = data
        self.mode = mode

    def get_scaling_from_history(self):
        """Return dummy scale and offset."""
        return xr.DataArray(1), xr.DataArray(0)


modules = {'pyninjotiff': mock.Mock(),
           'pyninjotiff.ninjotiff': mock.Mock()}


@mock.patch.dict(sys.modules, modules)
class TestNinjoTIFFWriter(unittest.TestCase):
    """The ninjo tiff writer tests."""

    def test_init(self):
        """Test the init."""
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ninjo_tags = {40000: 'NINJO'}
        ntw = NinjoTIFFWriter(tags=ninjo_tags)
        self.assertDictEqual(ntw.tags, ninjo_tags)

    @mock.patch('satpy.writers.ninjotiff.ImageWriter.save_dataset')
    @mock.patch('satpy.writers.ninjotiff.convert_units')
    def test_dataset(self, uconv, iwsd):
        """Test saving a dataset."""
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ntw = NinjoTIFFWriter()
        dataset = xr.DataArray([1, 2, 3], attrs={'units': 'K'})
        ntw.save_dataset(dataset, physic_unit='CELSIUS')
        uconv.assert_called_once_with(dataset, 'K', 'CELSIUS')
        self.assertEqual(iwsd.call_count, 1)

    @mock.patch('satpy.writers.ninjotiff.NinjoTIFFWriter.save_dataset')
    @mock.patch('satpy.writers.ninjotiff.ImageWriter.save_image')
    def test_image(self, iwsi, save_dataset):
        """Test saving an image."""
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ntw = NinjoTIFFWriter()
        dataset = xr.DataArray([1, 2, 3], attrs={'units': 'K'})
        img = FakeImage(dataset, 'L')
        ret = ntw.save_image(img, filename='bla.tif', compute=False)
        self.assertIsInstance(ret, Delayed)


def suite():
    """Test suite for this writer's tests."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNinjoTIFFWriter))
    return mysuite
