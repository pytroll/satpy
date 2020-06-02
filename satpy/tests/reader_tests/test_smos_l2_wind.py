#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
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
"""Module for testing the satpy.readers.smos_l2_wind module."""

import os
import unittest
import numpy as np
import xarray as xr
from unittest import mock
from datetime import datetime
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler


class FakeNetCDF4FileHandlerSMOSL2WIND(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        from xarray import DataArray
        dt_s = filename_info.get('start_time', datetime(2020, 4, 22, 12, 0, 0))
        dt_e = filename_info.get('end_time', datetime(2020, 4, 22, 12, 0, 0))

        if filetype_info['file_type'] == 'smos_l2_wind':
            file_content = {
                '/attr/time_coverage_start': dt_s.strftime('%Y-%m-%dT%H:%M:%S Z'),
                '/attr/time_coverage_end': dt_e.strftime('%Y-%m-%dT%H:%M:%S Z'),
                '/attr/platform_shortname': 'SM',
                '/attr/platform': 'SMOS',
                '/attr/instrument': 'MIRAS',
                '/attr/processing_level': 'L2',
                '/attr/geospatial_bounds_vertical_crs': 'EPSG:4623',
            }

            file_content['lat'] = np.arange(-90., 90.25, 0.25)
            file_content['lat/shape'] = (len(file_content['lat']),)
            file_content['lat'] = DataArray(file_content['lat'], dims=('lat'))
            file_content['lat'].attrs['_FillValue'] = -999.0

            file_content['lon'] = np.arange(0., 360., 0.25)
            file_content['lon/shape'] = (len(file_content['lon']),)
            file_content['lon'] = DataArray(file_content['lon'], dims=('lon'))
            file_content['lon'].attrs['_FillValue'] = -999.0

            file_content['wind_speed'] = np.ndarray(shape=(1,  # Time dimension
                                                           len(file_content['lat']),
                                                           len(file_content['lon'])))
            file_content['wind_speed/shape'] = (1,
                                                len(file_content['lat']),
                                                len(file_content['lon']))
            file_content['wind_speed'] = DataArray(file_content['wind_speed'], dims=('time', 'lat', 'lon'),
                                                   coords=[[1], file_content['lat'], file_content['lon']])
            file_content['wind_speed'].attrs['_FillValue'] = -999.0

        else:
            assert False

        return file_content


class TestSMOSL2WINDReader(unittest.TestCase):
    """Test SMOS L2 WINDReader"""
    yaml_file = "smos_l2_wind.yaml"

    def setUp(self):
        """Wrap NetCDF4 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.smos_l2_wind import SMOSL2WINDFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(SMOSL2WINDFileHandler, '__bases__', (FakeNetCDF4FileHandlerSMOSL2WIND,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic initialization of this reader."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SM_OPER_MIR_SCNFSW_20200420T021649_20200420T035013_110_001_7.nc',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_wind_speed(self):
        """Load wind_speed dataset"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.smos_l2_wind.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'SM_OPER_MIR_SCNFSW_20200420T021649_20200420T035013_110_001_7.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(['wind_speed'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.attrs['platform_shortname'], 'SM')
            self.assertEqual(d.attrs['sensor'], 'MIRAS')
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertIn('y', d.dims)
            self.assertIn('x', d.dims)
            self.assertEqual(d.shape, (719, 1440))
            self.assertEqual(d.y[0].data, -89.75)
            self.assertEqual(d.y[d.shape[0] - 1].data, 89.75)

    def test_load_lat(self):
        """Load lat dataset"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.smos_l2_wind.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'SM_OPER_MIR_SCNFSW_20200420T021649_20200420T035013_110_001_7.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(['lat'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertIn('y', d.dims)
            self.assertEqual(d.shape, (719,))
            self.assertEqual(d.data[0], -89.75)
            self.assertEqual(d.data[d.shape[0] - 1], 89.75)

    def test_load_lon(self):
        """Load lon dataset"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.smos_l2_wind.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'SM_OPER_MIR_SCNFSW_20200420T021649_20200420T035013_110_001_7.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(['lon'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertIn('x', d.dims)
            self.assertEqual(d.shape, (1440,))
            self.assertEqual(d.data[0], -180.0)
            self.assertEqual(d.data[d.shape[0] - 1], 179.75)

    def test_adjust_lon(self):
        """Load adjust longitude dataset"""
        from xarray import DataArray
        from satpy.readers.smos_l2_wind import SMOSL2WINDFileHandler
        smos_l2_wind_fh = SMOSL2WINDFileHandler('SM_OPER_MIR_SCNFSW_20200420T021649_20200420T035013_110_001_7.nc',
                                                {}, filetype_info={'file_type': 'smos_l2_wind'})
        data = DataArray(np.arange(0., 360., 0.25), dims=('lon'))
        adjusted = smos_l2_wind_fh._adjust_lon_coord(data)
        expected = DataArray(np.concatenate((np.arange(0, 180., 0.25),
                                             np.arange(-180.0, 0, 0.25))),
                             dims=('lon'))
        self.assertEqual(adjusted.data.tolist(), expected.data.tolist())

    def test_roll_dataset(self):
        """Load roll of dataset along the lon coordinate"""
        from xarray import DataArray
        from satpy.readers.smos_l2_wind import SMOSL2WINDFileHandler
        smos_l2_wind_fh = SMOSL2WINDFileHandler('SM_OPER_MIR_SCNFSW_20200420T021649_20200420T035013_110_001_7.nc',
                                                {}, filetype_info={'file_type': 'smos_l2_wind'})
        data = DataArray(np.arange(0., 360., 0.25), dims=('lon'))
        data = smos_l2_wind_fh._adjust_lon_coord(data)
        adjusted = smos_l2_wind_fh._roll_dataset_lon_coord(data)
        expected = np.arange(-180., 180., 0.25)
        self.assertEqual(adjusted.data.tolist(), expected.tolist())
