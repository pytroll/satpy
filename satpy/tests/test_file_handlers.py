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
"""test file handler baseclass."""

import unittest
from datetime import datetime, timedelta
from unittest import mock

import numpy as np
import pytest

from satpy.readers.file_handlers import BaseFileHandler, open_dataset
from satpy.tests.utils import FakeFileHandler


def test_open_dataset():
    """Test xr.open_dataset wrapper."""
    fn = mock.MagicMock()
    str_file_path = "path/to/file.nc"
    with mock.patch('xarray.open_dataset') as xr_open:
        _ = open_dataset(fn, decode_cf=True, chunks=500)
        fn.open.assert_called_once_with()
        xr_open.assert_called_once_with(fn.open(), decode_cf=True, chunks=500)

        xr_open.reset_mock()
        _ = open_dataset(str_file_path, decode_cf=True, chunks=500)
        xr_open.assert_called_once_with(str_file_path, decode_cf=True, chunks=500)


class TestBaseFileHandler(unittest.TestCase):
    """Test the BaseFileHandler."""

    def setUp(self):
        """Set up the test."""
        self.fh = BaseFileHandler(
            'filename', {'filename_info': 'bla'}, 'filetype_info')

    def test_combine_times(self):
        """Combine times."""
        info1 = {'start_time': 1}
        info2 = {'start_time': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'start_time': 1}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'start_time': 1}
        self.assertDictEqual(res, exp)

        info1 = {'end_time': 1}
        info2 = {'end_time': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'end_time': 2}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'end_time': 2}
        self.assertDictEqual(res, exp)

    def test_combine_orbits(self):
        """Combine orbits."""
        info1 = {'start_orbit': 1}
        info2 = {'start_orbit': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'start_orbit': 1}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'start_orbit': 1}
        self.assertDictEqual(res, exp)

        info1 = {'end_orbit': 1}
        info2 = {'end_orbit': 2}
        res = self.fh.combine_info([info1, info2])
        exp = {'end_orbit': 2}
        self.assertDictEqual(res, exp)
        res = self.fh.combine_info([info2, info1])
        exp = {'end_orbit': 2}
        self.assertDictEqual(res, exp)

    @mock.patch('satpy.readers.file_handlers.SwathDefinition')
    def test_combine_area(self, sdef):
        """Combine area."""
        area1 = mock.MagicMock()
        area1.lons = np.arange(5)
        area1.lats = np.arange(5)
        area1.name = 'area1'

        area2 = mock.MagicMock()
        area2.lons = np.arange(5)
        area2.lats = np.arange(5)
        area2.name = 'area2'

        info1 = {'area': area1}
        info2 = {'area': area2}

        self.fh.combine_info([info1, info2])
        self.assertTupleEqual(sdef.call_args[1]['lons'].shape, (2, 5))
        self.assertTupleEqual(sdef.call_args[1]['lats'].shape, (2, 5))
        self.assertEqual(sdef.return_value.name, 'area1_area2')

    def test_combine_orbital_parameters(self):
        """Combine orbital parameters."""
        info1 = {'orbital_parameters': {'projection_longitude': 1,
                                        'projection_latitude': 1,
                                        'projection_altitude': 1,
                                        'satellite_nominal_longitude': 1,
                                        'satellite_nominal_latitude': 1,
                                        'satellite_actual_longitude': 1,
                                        'satellite_actual_latitude': 1,
                                        'satellite_actual_altitude': 1,
                                        'nadir_longitude': 1,
                                        'nadir_latitude': 1,
                                        'only_in_1': False}}
        info2 = {'orbital_parameters': {'projection_longitude': 2,
                                        'projection_latitude': 2,
                                        'projection_altitude': 2,
                                        'satellite_nominal_longitude': 2,
                                        'satellite_nominal_latitude': 2,
                                        'satellite_actual_longitude': 2,
                                        'satellite_actual_latitude': 2,
                                        'satellite_actual_altitude': 2,
                                        'nadir_longitude': 2,
                                        'nadir_latitude': 2,
                                        'only_in_2': True}}
        exp = {'orbital_parameters': {'projection_longitude': 1.5,
                                      'projection_latitude': 1.5,
                                      'projection_altitude': 1.5,
                                      'satellite_nominal_longitude': 1.5,
                                      'satellite_nominal_latitude': 1.5,
                                      'satellite_actual_longitude': 1.5,
                                      'satellite_actual_latitude': 1.5,
                                      'satellite_actual_altitude': 1.5,
                                      'nadir_longitude': 1.5,
                                      'nadir_latitude': 1.5,
                                      'only_in_1': False,
                                      'only_in_2': True}}
        res = self.fh.combine_info([info1, info2])
        self.assertDictEqual(res, exp)

        # Identity
        self.assertEqual(self.fh.combine_info([info1]), info1)

        # Empty
        self.fh.combine_info([{}])

    def test_combine_time_parameters(self):
        """Combine times in 'time_parameters."""
        time_params1 = {
            'nominal_start_time': datetime(2020, 1, 1, 12, 0, 0),
            'nominal_end_time': datetime(2020, 1, 1, 12, 2, 30),
            'observation_start_time': datetime(2020, 1, 1, 12, 0, 2, 23821),
            'observation_end_time': datetime(2020, 1, 1, 12, 2, 23, 12348),
        }
        time_params2 = {}
        time_shift = timedelta(seconds=1.5)
        for key, value in time_params1.items():
            time_params2[key] = value + time_shift
        res = self.fh.combine_info([
            {'time_parameters': time_params1},
            {'time_parameters': time_params2}
        ])
        res_time_params = res['time_parameters']
        assert res_time_params['nominal_start_time'] == datetime(2020, 1, 1, 12, 0, 0)
        assert res_time_params['nominal_end_time'] == datetime(2020, 1, 1, 12, 2, 31, 500000)
        assert res_time_params['observation_start_time'] == datetime(2020, 1, 1, 12, 0, 2, 23821)
        assert res_time_params['observation_end_time'] == datetime(2020, 1, 1, 12, 2, 24, 512348)

    def test_file_is_kept_intact(self):
        """Test that the file object passed (string, path, or other) is kept intact."""
        open_file = mock.MagicMock()
        bfh = BaseFileHandler(open_file, {'filename_info': 'bla'}, 'filetype_info')
        assert bfh.filename == open_file

        from pathlib import Path
        filename = Path('/bla/bla.nc')
        bfh = BaseFileHandler(filename, {'filename_info': 'bla'}, 'filetype_info')
        assert isinstance(bfh.filename, Path)


@pytest.mark.parametrize(
    ("file_type", "ds_file_type", "exp_result"),
    [
        ("fake1", "fake1", True),
        ("fake1", ["fake1"], True),
        ("fake1", ["fake1", "fake2"], True),
        ("fake1", ["fake2"], None),
        ("fake1", "fake2", None),
        ("fake1", "fake1_with_suffix", None),
    ]
)
def test_file_type_match(file_type, ds_file_type, exp_result):
    """Test that file type matching uses exactly equality."""
    fh = FakeFileHandler("some_file.txt", {}, {"file_type": file_type})
    assert fh.file_type_matches(ds_file_type) is exp_result
