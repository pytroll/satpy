#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Satpy developers
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
"""The HRIT msg reader tests package."""

import unittest
from unittest import mock
from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from satpy.readers.seviri_l1b_nc import NCSEVIRIFileHandler
from satpy.tests.reader_tests.test_seviri_l1b_calibration import (
    TestFileHandlerCalibrationBase
)
from satpy.tests.utils import make_dataid


def new_read_file(instance):
    """Fake read file."""
    new_ds = xr.Dataset({'ch4': (['num_rows_vis_ir', 'num_columns_vis_ir'], np.random.random((2, 2))),
                         'planned_chan_processing': (["channels_dim"], np.ones(12, dtype=np.int8) * 2)},
                        coords={'num_rows_vis_ir': [1, 2], 'num_columns_vis_ir': [1, 2]})
    # dataset attrs
    attrs = {'comment': 'comment', 'long_name': 'long_name', 'nc_key': 'ch4', 'scale_factor': np.float64(1.0),
             'add_offset': np.float64(1.0), 'valid_min': np.float64(1.0), 'valid_max': np.float64(1.0)}
    new_ds['ch4'].attrs = attrs

    # global attrs
    new_ds.attrs['satellite_id'] = '324'

    instance.nc = new_ds

    instance.mda['projection_parameters'] = {'a': 1,
                                             'b': 1,
                                             'h': 35785831.00,
                                             'ssp_longitude': 0}


class TestNCSEVIRIFileHandler(unittest.TestCase):
    """Tester for the file handler."""

    def setUp(self):
        """Set up the test case."""
        start_time = datetime(2016, 3, 3, 0, 0)
        with mock.patch.object(NCSEVIRIFileHandler, '_read_file', new=new_read_file):
            self.reader = NCSEVIRIFileHandler(
                'filename',
                {'platform_shortname': 'MSG3',
                 'start_time': start_time,
                 'service': 'MSG'},
                {'filetype': 'info'})
            self.reader.deltaSt = start_time

    def test_get_dataset_remove_attrs(self):
        """Test getting the hrv dataset."""
        dataset_id = make_dataid(name='IR_039', calibration='counts')
        dataset_info = {'nc_key': 'ch4', 'units': 'units', 'wavelength': 'wavelength', 'standard_name': 'standard_name'}

        res = self.reader.get_dataset(dataset_id, dataset_info)

        strip_attrs = ["comment", "long_name", "nc_key", "scale_factor", "add_offset", "valid_min", "valid_max"]
        self.assertFalse(any([k in res.attrs.keys() for k in strip_attrs]))


class TestCalibration(TestFileHandlerCalibrationBase):
    """Unit tests for calibration."""

    @pytest.fixture(name='file_handler')
    def file_handler(self, counts):
        """Create a mocked file handler."""
        with mock.patch(
            'satpy.readers.seviri_l1b_nc.NCSEVIRIFileHandler.__init__',
            return_value=None
        ):
            # Create dataset and set calibration coefficients
            ds = xr.Dataset(
                {
                    'VIS006': counts.copy(),
                    'IR_108': counts.copy(),
                    'planned_chan_processing': self.radiance_types,
                },
                attrs={
                    'satellite_id': self.platform_id
                }
            )
            ds['VIS006'].attrs.update({
                'scale_factor': self.gains_nominal[0],
                'add_offset': self.offsets_nominal[0]
            })
            ds['IR_108'].attrs.update({
                'scale_factor': self.gains_nominal[8],
                'add_offset': self.offsets_nominal[8],
            })

            # Add some attributes so that the reader can strip them
            strip_attrs = {
                'comment': None,
                'long_name': None,
                'valid_min': None,
                'valid_max': None
            }
            for name in ['VIS006', 'IR_108']:
                ds[name].attrs.update(strip_attrs)

            # Create file handler
            fh = NCSEVIRIFileHandler()
            fh.nc = ds
            fh.deltaSt = self.scan_time
            fh.mda = {
                'projection_parameters': {
                    'ssp_longitude': 0,
                    'h': 12345
                }
            }
            return fh

    @pytest.mark.parametrize(
        ('channel', 'calibration', 'use_ext_coefs'),
        [
            # VIS channel, internal coefficients
            ('VIS006', 'counts', False),
            ('VIS006', 'radiance', False),
            ('VIS006', 'reflectance', False),
            # VIS channel, external coefficients
            ('VIS006', 'radiance', True),
            ('VIS006', 'reflectance', True),
            # IR channel, internal coefficients
            ('IR_108', 'counts', False),
            ('IR_108', 'radiance', False),
            ('IR_108', 'brightness_temperature', False),
            # IR channel, external coefficients
            ('IR_108', 'radiance', True),
            ('IR_108', 'brightness_temperature', True),
            # FUTURE: Enable once HRV reading has been fixed.
            # # HRV channel, internal coefficiens
            # ('HRV', 'counts', False),
            # ('HRV', 'radiance', False),
            # ('HRV', 'reflectance', False),
            # # HRV channel, external coefficients (mode should have no effect)
            # ('HRV', 'radiance', True),
            # ('HRV', 'reflectance', True),
        ]
    )
    def test_calibrate(
            self, file_handler, channel, calibration, use_ext_coefs
    ):
        """Test the calibration."""
        external_coefs = self.external_coefs if use_ext_coefs else {}
        expected = self._get_expected(
            channel=channel,
            calibration=calibration,
            calib_mode='NOMINAL',
            use_ext_coefs=use_ext_coefs
        )
        fh = file_handler
        fh.ext_calib_coefs = external_coefs

        dataset_info = {
            'nc_key': channel
        }
        dataset_id = make_dataid(name=channel, calibration=calibration)

        res = fh.get_dataset(dataset_id, dataset_info)

        # Flip dataset to achieve compatibility with other SEVIRI readers.
        # FUTURE: Remove if flipping has been disabled.
        res = res.isel(y=slice(None, None, -1))

        xr.testing.assert_allclose(res, expected)
