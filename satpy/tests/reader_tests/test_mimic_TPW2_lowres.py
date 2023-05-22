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

import itertools
import os
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import xarray as xr

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler

DEFAULT_FILE_DTYPE = np.float32
DEFAULT_FILE_SHAPE = (721, 1440)
DEFAULT_DATE = datetime(2019, 6, 19, 13, 0)
DEFAULT_LAT = np.linspace(-90, 90, DEFAULT_FILE_SHAPE[0], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_LON = np.linspace(-180, 180, DEFAULT_FILE_SHAPE[1], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_FILE_FLOAT_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                                    dtype=DEFAULT_FILE_DTYPE)
DEFAULT_FILE_DATE_DATA = np.clip(DEFAULT_FILE_FLOAT_DATA, 0, 1049)
DEFAULT_FILE_UBYTE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                                    dtype=np.ubyte)
float_variables = ['tpwGrid', 'tpwGridPrior', 'tpwGridSubseq', 'footGridPrior', 'footGridSubseq']
date_variables = ['timeAwayGridPrior', 'timeAwayGridSubseq']
ubyte_variables = ['satGridPrior', 'satGridSubseq']
file_content_attr = dict()


class FakeNetCDF4FileHandlerMimicLow(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content for lower resolution files."""
        dt_s = filename_info.get('start_time', DEFAULT_DATE)
        dt_e = filename_info.get('end_time', DEFAULT_DATE)

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

            file_content['/dimension/lat'] = DEFAULT_FILE_SHAPE[0]
            file_content['/dimension/lon'] = DEFAULT_FILE_SHAPE[1]

            for float_var in float_variables:
                file_content[float_var] = DEFAULT_FILE_FLOAT_DATA.reshape(DEFAULT_FILE_SHAPE)
                file_content['{}/shape'.format(float_var)] = DEFAULT_FILE_SHAPE
                file_content_attr[float_var] = {"units": "mm"}
            for date_var in date_variables:
                file_content[date_var] = DEFAULT_FILE_DATE_DATA.reshape(DEFAULT_FILE_SHAPE)
                file_content['{}/shape'.format(date_var)] = DEFAULT_FILE_SHAPE
                file_content_attr[date_var] = {"units": "minutes"}
            for ubyte_var in ubyte_variables:
                file_content[ubyte_var] = DEFAULT_FILE_UBYTE_DATA.reshape(DEFAULT_FILE_SHAPE)
                file_content['{}/shape'.format(ubyte_var)] = DEFAULT_FILE_SHAPE
                file_content_attr[ubyte_var] = {"source_key": "Key: 0: None, 1: NOAA-N, 2: NOAA-P, 3: Metop-A, \
                                                              4: Metop-B, 5: SNPP, 6: SSMI-17, 7: SSMI-18"}

            # convert to xarrays
            for key, val in file_content.items():
                if key == 'lonArr' or key == 'latArr':
                    file_content[key] = xr.DataArray(val)
                elif isinstance(val, np.ndarray):
                    if val.ndim > 1:
                        file_content[key] = xr.DataArray(val, dims=('y', 'x'), attrs=file_content_attr[key])
                    else:
                        file_content[key] = xr.DataArray(val)
            for key in itertools.chain(float_variables, ubyte_variables):
                file_content[key].attrs['_FillValue'] = -999.0
                file_content[key].attrs['name'] = key
                file_content[key].attrs['file_key'] = key
                file_content[key].attrs['file_type'] = self.filetype_info['file_type']
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
        self.p = mock.patch.object(MimicTPW2FileHandler, '__bases__', (FakeNetCDF4FileHandlerMimicLow,))
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

    def test_load_mimic_float(self):
        """Load TPW mimic float data."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.mimic_TPW2_nc.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'comp20190619.130000.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(float_variables)
        self.assertEqual(len(ds), len(float_variables))
        for d in ds.values():
            self.assertEqual(d.attrs['platform_shortname'], 'aggregated microwave')
            self.assertEqual(d.attrs['sensor'], 'mimic')
            self.assertEqual(d.attrs['units'], 'mm')
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_load_mimic_timedelta(self):
        """Load TPW mimic timedelta data (data latency variables)."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.mimic_TPW2_nc.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'comp20190619.130000.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(date_variables)
        self.assertEqual(len(ds), len(date_variables))
        for d in ds.values():
            self.assertEqual(d.attrs['platform_shortname'], 'aggregated microwave')
            self.assertEqual(d.attrs['sensor'], 'mimic')
            self.assertEqual(d.attrs['units'], 'minutes')
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertEqual(d.dtype, DEFAULT_FILE_DTYPE)

    def test_load_mimic_ubyte(self):
        """Load TPW mimic sensor grids."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.mimic_TPW2_nc.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'comp20190619.130000.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(ubyte_variables)
        self.assertEqual(len(ds), len(ubyte_variables))
        for d in ds.values():
            self.assertEqual(d.attrs['platform_shortname'], 'aggregated microwave')
            self.assertEqual(d.attrs['sensor'], 'mimic')
            self.assertIn('source_key', d.attrs)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertEqual(d.dtype, np.uint8)
