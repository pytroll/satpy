#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Module for testing the satpy.readers.nc_slstr module."""
import unittest
import unittest.mock as mock

import numpy as np
import pytest
import xarray as xr
from enum import Enum
from datetime import datetime
from satpy.dataset.dataid import WavelengthRange, ModifierTuple, DataID

local_id_keys_config = {'name': {
    'required': True,
},
    'wavelength': {
    'type': WavelengthRange,
},
    'resolution': None,
    'calibration': {
    'enum': [
        'reflectance',
        'brightness_temperature',
        'radiance',
        'counts'
    ]
},
    'stripe': {
        'enum': [
            'a',
            'b',
            'c',
            'i',
            'f',
        ]
    },
    'view': {
        'enum': [
            'nadir',
            'oblique',
        ]
    },
    'modifiers': {
    'required': True,
    'default': ModifierTuple(),
    'type': ModifierTuple,
},
}


class test_slstr_l1b_Base(unittest.TestCase):
    """Common setup for SLSTR_L1B tests."""

    @mock.patch('satpy.readers.slstr_l1b.xr')
    def setUp(self, xr_):
        """Create a fake dataset using the given radiance data."""
        self.base_data = np.array(([1., 2., 3.], [4., 5., 6.]))
        self.det_data = np.array(([0, 1, 1], [0, 1, 0]))
        self.start_time =  "2020-05-10T12:01:15.585Z"
        self.end_time = "2020-05-10T12:06:18.012Z"
        rad = xr.DataArray(
            self.base_data,
            dims=('columns', 'rows'),
            attrs={'scale_factor': 1.0, 'add_offset': 0.0,
                '_FillValue': -32768, 'units': 'mW.m-2.sr-1.nm-1',
            }
        )
        det = xr.DataArray(
            self.base_data,
            dims=('columns', 'rows'),
            attrs={'scale_factor': 1.0, 'add_offset': 0.0,
                '_FillValue': 255,
            }
        )
        self.fake_dataset = xr.Dataset(
            data_vars={
                'S5_radiance_an': rad,
                'S9_BT_ao': rad,
                'foo_radiance_an': rad,
                'S5_solar_irradiances': rad,
                'detector_an': det,
            },
            attrs={
                "start_time": self.start_time,
                "stop_time": self.end_time,
            },
        )


def make_dataid(**items):
    """Make a data id."""
    return DataID(local_id_keys_config, **items)


class testSLSTRReader(test_slstr_l1b_Base):
    """Test various nc_slstr file handlers."""

    @mock.patch('satpy.readers.slstr_l1b.xr')
    def test_instantiate(self, xr_):
        """Test initialization of file handlers."""
        from satpy.readers.slstr_l1b import NCSLSTR1B, NCSLSTRGeo, NCSLSTRAngles, NCSLSTRFlag

        xr_.open_dataset.return_value = self.fake_dataset

        good_start = datetime.strptime(self.start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
        good_end = datetime.strptime(self.end_time, '%Y-%m-%dT%H:%M:%S.%fZ')

        ds_id = make_dataid(name='foo', calibration='radiance', stripe='a', view='nadir')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'stripe': 'a', 'view': 'n'}
        test = NCSLSTR1B('somedir/S1_radiance_an.nc', filename_info, 'c')
        assert(test.view == 'nadir')
        assert(test.stripe == 'a')
        test.get_dataset(ds_id, dict(filename_info, **{'file_key': 'foo'}))
        self.assertEqual(test.start_time, good_start)
        self.assertEqual(test.end_time, good_end)
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'stripe': 'c', 'view': 'o'}
        test = NCSLSTR1B('somedir/S1_radiance_co.nc', filename_info, 'c')
        assert(test.view == 'oblique')
        assert(test.stripe == 'c')
        test.get_dataset(ds_id, dict(filename_info, **{'file_key': 'foo'}))
        self.assertEqual(test.start_time, good_start)
        self.assertEqual(test.end_time, good_end)
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'stripe': 'a', 'view': 'n'}
        test = NCSLSTRGeo('somedir/S1_radiance_an.nc', filename_info, 'c')
        test.get_dataset(ds_id, dict(filename_info, **{'file_key': 'foo'}))
        self.assertEqual(test.start_time, good_start)
        self.assertEqual(test.end_time, good_end)
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        test = NCSLSTRAngles('somedir/S1_radiance_an.nc', filename_info, 'c')
        # TODO: Make this test work
        #test.get_dataset(ds_id, filename_info)
        self.assertEqual(test.start_time, good_start)
        self.assertEqual(test.end_time, good_end)
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        test = NCSLSTRFlag('somedir/S1_radiance_an.nc', filename_info, 'c')
        assert(test.view == 'nadir')
        assert(test.stripe == 'a')
        self.assertEqual(test.start_time, good_start)
        self.assertEqual(test.end_time, good_end)
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()


class testSLSTRCalibration(test_slstr_l1b_Base):
    """Test the implementation of the calibration factors."""

    @mock.patch('satpy.readers.slstr_l1b.xr')
    def test_calibration(self, xr_):
        from satpy.readers.slstr_l1b import NCSLSTR1B, CHANCALIB_FACTORS
        xr_.open_dataset.return_value = self.fake_dataset

        ds_id = make_dataid(name='foo', calibration='radiance', stripe='a', view='nadir')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'stripe': 'a', 'view': 'n'}

        test = NCSLSTR1B('somedir/S1_radiance_co.nc', filename_info, 'c')
        # Check warning is raised if we don't have calibration
        with pytest.warns(UserWarning):
            test.get_dataset(ds_id, dict(filename_info, **{'file_key': 'foo'}))

        # Check user calibration is used correctly
        test = NCSLSTR1B('somedir/S1_radiance_co.nc', filename_info, 'c',
                         user_calibration={'foo_nadir': 0.4})
        data = test.get_dataset(ds_id, dict(filename_info, **{'file_key': 'foo'}))
        np.testing.assert_allclose(data.values, self.base_data*0.4)

        # Check internal calibration is used correctly
        ds_id = make_dataid(name='S5', calibration='radiance', stripe='a', view='nadir')
        filename_info = {'mission_id': 'S3A', 'dataset_name': 'S5', 'start_time': 0, 'end_time': 0,
                         'stripe': 'a', 'view': 'n'}
        test = NCSLSTR1B('somedir/S1_radiance_an.nc', filename_info, 'c')
        data = test.get_dataset(ds_id, dict(filename_info, **{'file_key': 'S5'}))
        np.testing.assert_allclose(data.values,
                                   self.base_data * CHANCALIB_FACTORS['S5_nadir'])

        # Test reflectance calibration
        ds_id = make_dataid(name='S5', calibration='reflectance', stripe='a', view='nadir')
        data = test.get_dataset(ds_id, dict(filename_info, **{'file_key': 'S5'}))
        print(data)
        print(ds_id)


