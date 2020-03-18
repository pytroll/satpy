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
import xarray as xr

from satpy.readers.seviri_l1b_nc import NCSEVIRIFileHandler


def new_read_file(instance):
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

    def setUp(self):
        with mock.patch.object(NCSEVIRIFileHandler, '_read_file', new=new_read_file):
            self.reader = NCSEVIRIFileHandler(
                'filename',
                {'platform_shortname': 'MSG3',
                 'start_time': datetime(2016, 3, 3, 0, 0),
                 'service': 'MSG'},
                {'filetype': 'info'})

    def test_get_dataset_remove_attrs(self):
        """Test getting the hrv dataset."""
        dataset_id = mock.MagicMock(calibration='counts')
        dataset_id.name = 'IR_039'
        dataset_info = {'nc_key': 'ch4', 'units': 'units', 'wavelength': 'wavelength', 'standard_name': 'standard_name'}

        res = self.reader.get_dataset(dataset_id, dataset_info)

        strip_attrs = ["comment", "long_name", "nc_key", "scale_factor", "add_offset", "valid_min", "valid_max"]
        self.assertFalse(any([k in res.attrs.keys() for k in strip_attrs]))
