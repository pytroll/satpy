#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# This file is part of Satpy.
#
# Satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Module for testing the satpy.readers.tropomi_l2 module."""

import os
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import xarray as xr

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler

DEFAULT_FILE_DTYPE = np.float32
DEFAULT_FILE_SHAPE = (9001, 18000)
DEFAULT_LAT = np.linspace(-90, 90, DEFAULT_FILE_SHAPE[0], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_LON = np.linspace(-180, 180, DEFAULT_FILE_SHAPE[1], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
file_content_units = dict()


class FakeNetCDF4FileHandlerMimic(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        from xarray import DataArray
        dt_s = filename_info.get('start_time', datetime(2019, 6, 19, 13, 0))
        dt_e = filename_info.get('end_time', datetime(2019, 6, 19, 13, 0))

        if filetype_info['file_type'] == 'mimicTPW2_comp':
            file_content = {
                '/attr/start_time': dt_s.strftime('%Y%m%d.%H%M%S'),
                '/attr/end_time': dt_e.strftime('%Y%m%d.%H%M%S'),
                '/attr/platform_shortname': 'aggregated microwave',
                '/attr/sensor': 'mimic',
            }
            file_content['latArr'] = DEFAULT_LAT
            file_content['latArr/shape'] = (DEFAULT_FILE_SHAPE[0],)
            file_content['latArr/attr/units'] = 'degress_north'

            file_content['lonArr'] = DEFAULT_LON
            file_content['lonArr/shape'] = (DEFAULT_FILE_SHAPE[1],)
            file_content['lonArr/attr/units'] = 'degrees_east'

            file_content['tpwGrid'] = DEFAULT_FILE_DATA
            file_content['tpwGrid/shape'] = DEFAULT_FILE_SHAPE
            file_content_units['tpwGrid'] = 'mm'

            file_content['/dimension/lat'] = DEFAULT_FILE_SHAPE[0]
            file_content['/dimension/lon'] = DEFAULT_FILE_SHAPE[1]

            # convert to xarrays
            for key, val in file_content.items():
                if key == 'lonArr' or key == 'latArr':
                    file_content[key] = DataArray(val)
                elif isinstance(val, np.ndarray):
                    if val.ndim > 1:
                        file_content[key] = DataArray(val, dims=('y', 'x'), attrs={"units": file_content_units[key]})
                    else:
                        file_content[key] = DataArray(val)
        else:
            msg = 'Wrong Test Reader for file_type {}'.format(filetype_info['file_type'])
            raise AssertionError(msg)

        return file_content


class TestMimicTPW2Reader(unittest.TestCase):
    """Test Mimic Reader."""

    yaml_file = "mimicTPW2_comp.yaml"

    def setUp(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.mimic_TPW2_nc import MimicTPW2FileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(MimicTPW2FileHandler, '__bases__', (FakeNetCDF4FileHandlerMimic,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic initialization of this reader."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'comp20190619.130000.nc',
        ])
        self.assertEqual(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_mimic(self):
        """Load Mimic data."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.mimic_TPW2_nc.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'comp20190619.130000.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(['tpwGrid'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.attrs['platform_shortname'], 'aggregated microwave')
            self.assertEqual(d.attrs['sensor'], 'mimic')
            self.assertIn('area', d.attrs)
            self.assertIn('units', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
