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
"""Module for testing the satpy.readers.geocat module."""

import os
import unittest
from unittest import mock

import numpy as np

from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import convert_file_content_to_data_array

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        file_content = {
            '/attr/Platform_Name': filename_info['platform_shortname'],
            '/attr/Element_Resolution': 2.,
            '/attr/Line_Resolution': 2.,
            '/attr/Subsatellite_Longitude': -70.2 if 'GOES' in filename_info['platform_shortname'] else 140.65,
            'pixel_longitude': DEFAULT_LON_DATA,
            'pixel_longitude/attr/scale_factor': 1.,
            'pixel_longitude/attr/add_offset': 0.,
            'pixel_longitude/shape': DEFAULT_FILE_SHAPE,
            'pixel_longitude/attr/_FillValue': np.nan,
            'pixel_latitude': DEFAULT_LAT_DATA,
            'pixel_latitude/attr/scale_factor': 1.,
            'pixel_latitude/attr/add_offset': 0.,
            'pixel_latitude/shape': DEFAULT_FILE_SHAPE,
            'pixel_latitude/attr/_FillValue': np.nan,
        }
        sensor = {
            'HIMAWARI-8': 'himawari8',
            'GOES-17': 'goesr',
            'GOES-16': 'goesr',
            'GOES-13': 'goes',
            'GOES-14': 'goes',
            'GOES-15': 'goes',
        }[filename_info['platform_shortname']]
        file_content['/attr/Sensor_Name'] = sensor

        if filename_info['platform_shortname'] == 'HIMAWARI-8':
            file_content['pixel_longitude'] = DEFAULT_LON_DATA + 130.

        file_content['variable1'] = DEFAULT_FILE_DATA.astype(np.float32)
        file_content['variable1/attr/_FillValue'] = -1
        file_content['variable1/attr/scale_factor'] = 1.
        file_content['variable1/attr/add_offset'] = 0.
        file_content['variable1/attr/units'] = '1'
        file_content['variable1/shape'] = DEFAULT_FILE_SHAPE

        # data with fill values
        file_content['variable2'] = np.ma.masked_array(
            DEFAULT_FILE_DATA.astype(np.float32),
            mask=np.zeros_like(DEFAULT_FILE_DATA))
        file_content['variable2'].mask[::5, ::5] = True
        file_content['variable2/attr/_FillValue'] = -1
        file_content['variable2/attr/scale_factor'] = 1.
        file_content['variable2/attr/add_offset'] = 0.
        file_content['variable2/attr/units'] = '1'
        file_content['variable2/shape'] = DEFAULT_FILE_SHAPE

        # category
        file_content['variable3'] = DEFAULT_FILE_DATA.astype(np.byte)
        file_content['variable3/attr/_FillValue'] = -128
        file_content['variable3/attr/flag_meanings'] = "clear water supercooled mixed ice unknown"
        file_content['variable3/attr/flag_values'] = [0, 1, 2, 3, 4, 5]
        file_content['variable3/attr/units'] = '1'
        file_content['variable3/shape'] = DEFAULT_FILE_SHAPE

        attrs = ('_FillValue', 'flag_meanings', 'flag_values', 'units')
        convert_file_content_to_data_array(
            file_content, attrs=attrs,
            dims=('z', 'lines', 'elements'))
        return file_content


class TestGEOCATReader(unittest.TestCase):
    """Test GEOCAT Reader."""

    yaml_file = "geocat.yaml"

    def setUp(self):
        """Wrap NetCDF4 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.geocat import GEOCATFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(GEOCATFileHandler, '__bases__', (FakeNetCDF4FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'geocatL2.GOES-13.2015143.234500.nc',
        ])
        self.assertEqual(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_all_old_goes(self):
        """Test loading all test datasets from old GOES files."""
        import xarray as xr

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.geocat.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'geocatL2.GOES-13.2015143.234500.nc',
            ])
            r.create_filehandlers(loadables)
        datasets = r.load(['variable1',
                           'variable2',
                           'variable3'])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            assert 'calibration' not in v.attrs
            self.assertEqual(v.attrs['units'], '1')
        self.assertIsNotNone(datasets['variable3'].attrs.get('flag_meanings'))

    def test_load_all_himawari8(self):
        """Test loading all test datasets from H8 NetCDF file."""
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.geocat.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'geocatL2.HIMAWARI-8.2017092.210730.R304.R20.nc',
            ])
            r.create_filehandlers(loadables)
        datasets = r.load(['variable1',
                           'variable2',
                           'variable3'])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            assert 'calibration' not in v.attrs
            self.assertEqual(v.attrs['units'], '1')
        self.assertIsNotNone(datasets['variable3'].attrs.get('flag_meanings'))
        self.assertIsInstance(datasets['variable1'].attrs['area'], AreaDefinition)

    def test_load_all_goes17_hdf4(self):
        """Test loading all test datasets from GOES-17 HDF4 file."""
        import xarray as xr
        from pyresample.geometry import AreaDefinition

        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.geocat.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'geocatL2.GOES-17.CONUS.2020041.163130.hdf',
            ])
            r.create_filehandlers(loadables)
        datasets = r.load(['variable1',
                           'variable2',
                           'variable3'])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            assert 'calibration' not in v.attrs
            self.assertEqual(v.attrs['units'], '1')
        self.assertIsNotNone(datasets['variable3'].attrs.get('flag_meanings'))
        self.assertIsInstance(datasets['variable1'].attrs['area'], AreaDefinition)
