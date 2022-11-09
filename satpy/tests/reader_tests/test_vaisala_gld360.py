#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Unittesting the Vaisala GLD360 reader."""

import unittest
from io import StringIO

import numpy as np

from satpy.readers.vaisala_gld360 import VaisalaGLD360TextFileHandler
from satpy.tests.utils import make_dataid


class TestVaisalaGLD360TextFileHandler(unittest.TestCase):
    """Test the VaisalaGLD360TextFileHandler."""

    def test_vaisala_gld360(self):
        """Test basic functionality for vaisala file handler."""
        expected_power = np.array([12.3, 13.2, -31.])
        expected_lat = np.array([30.5342, -0.5727, 12.1529])
        expected_lon = np.array([-90.1152, 104.0688, -10.8756])
        expected_time = np.array(['2017-06-20T00:00:00.007178000', '2017-06-20T00:00:00.020162000',
                                  '2017-06-20T00:00:00.023183000'], dtype='datetime64[ns]')

        filename = StringIO(
            u'2017-06-20 00:00:00.007178  30.5342  -90.1152    12.3 kA\n'
            '2017-06-20 00:00:00.020162  -0.5727  104.0688    13.2 kA\n'
            '2017-06-20 00:00:00.023183  12.1529  -10.8756   -31.0 kA'
        )
        filename_info = {}
        filetype_info = {}

        self.handler = VaisalaGLD360TextFileHandler(
            filename, filename_info, filetype_info
        )

        filename.close()

        # test power
        dataset_id = make_dataid(name='power')
        dataset_info = {'units': 'kA'}
        result = self.handler.get_dataset(dataset_id, dataset_info).values
        np.testing.assert_allclose(result, expected_power, rtol=1e-05)

        # test lat
        dataset_id = make_dataid(name='latitude')
        dataset_info = {}
        result = self.handler.get_dataset(dataset_id, dataset_info).values
        np.testing.assert_allclose(result, expected_lat, rtol=1e-05)

        # test lon
        dataset_id = make_dataid(name='longitude')
        dataset_info = {}
        result = self.handler.get_dataset(dataset_id, dataset_info).values
        np.testing.assert_allclose(result, expected_lon, rtol=1e-05)

        # test time
        dataset_id = make_dataid(name='time')
        dataset_info = {}
        result = self.handler.get_dataset(dataset_id, dataset_info).values
        np.testing.assert_array_equal(result, expected_time)
