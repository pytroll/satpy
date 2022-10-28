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
from unittest import mock

import numpy as np
import pytest
import xarray as xr


class FakeImage:
    """Fake image."""

    def __init__(self, data, mode):
        """Init fake image."""
        self.data = data
        self.mode = mode

    def get_scaling_from_history(self):
        """Return dummy scale and offset."""
        return xr.DataArray(1), xr.DataArray(0)


pyninjotiff_mock = mock.Mock()
pyninjotiff_mock.ninjotiff = mock.Mock()


@mock.patch.dict(sys.modules, {'pyninjotiff': pyninjotiff_mock, 'pyninjotiff.ninjotiff': pyninjotiff_mock.ninjotiff})
class TestNinjoTIFFWriter(unittest.TestCase):
    """The ninjo tiff writer tests."""

    @mock.patch('satpy.writers.ninjotiff.nt', pyninjotiff_mock.ninjotiff)
    def test_init(self):
        """Test the init."""
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ninjo_tags = {40000: 'NINJO'}
        ntw = NinjoTIFFWriter(tags=ninjo_tags)
        self.assertDictEqual(ntw.tags, ninjo_tags)

    @mock.patch('satpy.writers.ninjotiff.ImageWriter.save_dataset')
    @mock.patch('satpy.writers.ninjotiff.nt', pyninjotiff_mock.ninjotiff)
    def test_dataset(self, iwsd):
        """Test saving a dataset."""
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ntw = NinjoTIFFWriter()
        dataset = xr.DataArray([1, 2, 3], attrs={'units': 'K'})
        with mock.patch('satpy.writers.ninjotiff.convert_units') as uconv:
            ntw.save_dataset(dataset, physic_unit='CELSIUS')
            uconv.assert_called_once_with(dataset, 'K', 'CELSIUS')
        self.assertEqual(iwsd.call_count, 1)

    @mock.patch('satpy.writers.ninjotiff.ImageWriter.save_dataset')
    @mock.patch('satpy.writers.ninjotiff.nt', pyninjotiff_mock.ninjotiff)
    def test_dataset_skip_unit_conversion(self, iwsd):
        """Test saving a dataset without unit conversion."""
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ntw = NinjoTIFFWriter()
        dataset = xr.DataArray([1, 2, 3], attrs={'units': 'K'})
        with mock.patch('satpy.writers.ninjotiff.convert_units') as uconv:
            ntw.save_dataset(dataset, physic_unit='CELSIUS',
                             convert_temperature_units=False)
            uconv.assert_not_called()
        self.assertEqual(iwsd.call_count, 1)

    @mock.patch('satpy.writers.ninjotiff.NinjoTIFFWriter.save_dataset')
    @mock.patch('satpy.writers.ninjotiff.ImageWriter.save_image')
    @mock.patch('satpy.writers.ninjotiff.nt', pyninjotiff_mock.ninjotiff)
    def test_image(self, iwsi, save_dataset):
        """Test saving an image."""
        nt = pyninjotiff_mock.ninjotiff
        nt.reset_mock()
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ntw = NinjoTIFFWriter()
        dataset = xr.DataArray([1, 2, 3], attrs={'units': 'K'})
        img = FakeImage(dataset, 'L')
        ret = ntw.save_image(img, filename='bla.tif', compute=False)
        nt.save.assert_called()
        assert nt.save.mock_calls[0][2]['compute'] is False
        assert nt.save.mock_calls[0][2]['ch_min_measurement_unit'] < nt.save.mock_calls[0][2]['ch_max_measurement_unit']
        assert ret == nt.save.return_value

    def test_convert_units_self(self):
        """Test that unit conversion to themselves do nothing."""
        from satpy.writers.ninjotiff import convert_units

        from ..utils import make_fake_scene

        # ensure that converting from % to itself does not change the data
        sc = make_fake_scene(
                {"VIS006": np.arange(25, dtype="f4").reshape(5, 5)},
                common_attrs={"units": "%"})
        ds_in = sc["VIS006"]
        ds_out = convert_units(ds_in, "%", "%")
        np.testing.assert_array_equal(ds_in, ds_out)
        assert ds_in.attrs == ds_out.attrs

    def test_convert_units_temp(self):
        """Test that temperature unit conversions works as expected."""
        # test converting between °C and K
        from satpy.writers.ninjotiff import convert_units

        from ..utils import make_fake_scene
        sc = make_fake_scene(
                {"IR108": np.arange(25, dtype="f4").reshape(5, 5)},
                common_attrs={"units": "K"})
        ds_in_k = sc["IR108"]
        for out_unit in ("C", "CELSIUS"):
            ds_out_c = convert_units(ds_in_k, "K", out_unit)
            np.testing.assert_array_almost_equal(ds_in_k - 273.15, ds_out_c)
            assert ds_in_k.attrs != ds_out_c.attrs
            assert ds_out_c.attrs["units"] == out_unit
        # test that keys aren't lost
        assert ds_out_c.attrs.keys() - ds_in_k.attrs.keys() <= {"units"}
        assert ds_in_k.attrs.keys() <= ds_out_c.attrs.keys()

    def test_convert_units_other(self):
        """Test that other unit conversions are not implemented."""
        # test arbitrary different conversion
        from satpy.writers.ninjotiff import convert_units

        from ..utils import make_fake_scene
        sc = make_fake_scene(
                {"rain_rate": np.arange(25, dtype="f8").reshape(5, 5)},
                common_attrs={"units": "millimeter/hour"})

        ds_in = sc["rain_rate"]
        with pytest.raises(NotImplementedError):
            convert_units(ds_in, "millimeter/hour", "m/s")

    @mock.patch('satpy.writers.ninjotiff.NinjoTIFFWriter.save_dataset')
    @mock.patch('satpy.writers.ninjotiff.ImageWriter.save_image')
    @mock.patch('satpy.writers.ninjotiff.nt', pyninjotiff_mock.ninjotiff)
    def test_P_image_is_uint8(self, iwsi, save_dataset):
        """Test that a P-mode image is converted to uint8s."""
        nt = pyninjotiff_mock.ninjotiff
        nt.reset_mock()
        from satpy.writers.ninjotiff import NinjoTIFFWriter
        ntw = NinjoTIFFWriter()
        dataset = xr.DataArray([1, 2, 3]).astype(int)
        img = FakeImage(dataset, 'P')
        ntw.save_image(img, filename='bla.tif', compute=False)
        assert nt.save.mock_calls[0][1][0].data.dtype == np.uint8
