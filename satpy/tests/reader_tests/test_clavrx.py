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
"""Module for testing the satpy.readers.clavrx module."""

import os
import numpy as np
import dask.array as da
import xarray as xr
from satpy.tests.reader_tests.test_hdf4_utils import FakeHDF4FileHandler
from pyresample.geometry import AreaDefinition

import unittest
from unittest import mock

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeHDF4FileHandlerPolar(FakeHDF4FileHandler):
    """Swap-in HDF4 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        file_content = {
            '/attr/platform': 'SNPP',
            '/attr/sensor': 'VIIRS',
        }

        file_content['longitude'] = xr.DataArray(
            da.from_array(DEFAULT_LON_DATA, chunks=4096),
            attrs={
                '_FillValue': np.nan,
                'scale_factor': 1.,
                'add_offset': 0.,
                'standard_name': 'longitude',
            })
        file_content['longitude/shape'] = DEFAULT_FILE_SHAPE

        file_content['latitude'] = xr.DataArray(
            da.from_array(DEFAULT_LAT_DATA, chunks=4096),
            attrs={
                '_FillValue': np.nan,
                'scale_factor': 1.,
                'add_offset': 0.,
                'standard_name': 'latitude',
            })
        file_content['latitude/shape'] = DEFAULT_FILE_SHAPE

        file_content['variable1'] = xr.DataArray(
            da.from_array(DEFAULT_FILE_DATA, chunks=4096).astype(np.float32),
            attrs={
                '_FillValue': -1,
                'scale_factor': 1.,
                'add_offset': 0.,
                'units': '1',
            })
        file_content['variable1/shape'] = DEFAULT_FILE_SHAPE

        # data with fill values
        file_content['variable2'] = xr.DataArray(
            da.from_array(DEFAULT_FILE_DATA, chunks=4096).astype(np.float32),
            attrs={
                '_FillValue': -1,
                'scale_factor': 1.,
                'add_offset': 0.,
                'units': '1',
            })
        file_content['variable2/shape'] = DEFAULT_FILE_SHAPE
        file_content['variable2'] = file_content['variable2'].where(
                                        file_content['variable2'] % 2 != 0)

        # category
        file_content['variable3'] = xr.DataArray(
            da.from_array(DEFAULT_FILE_DATA, chunks=4096).astype(np.byte),
            attrs={
                '_FillValue': -128,
                'flag_meanings': 'clear water supercooled mixed ice unknown',
                'flag_values': [0, 1, 2, 3, 4, 5],
                'units': '1',
            })
        file_content['variable3/shape'] = DEFAULT_FILE_SHAPE

        return file_content


class TestCLAVRXReaderPolar(unittest.TestCase):
    """Test CLAVR-X Reader with Polar files."""
    yaml_file = "clavrx.yaml"

    def setUp(self):
        """Wrap HDF4 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.clavrx import CLAVRXFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(CLAVRXFileHandler, '__bases__', (FakeHDF4FileHandlerPolar,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'clavrx_npp_d20170520_t2053581_e2055223_b28822.level2.hdf',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_available_datasets(self):
        """Test available_datasets with fake variables from YAML."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'clavrx_npp_d20170520_t2053581_e2055223_b28822.level2.hdf',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

        # mimic the YAML file being configured for more datasets
        fake_dataset_info = [
            (None, {'name': 'variable1', 'resolution': None, 'file_type': ['level2']}),
            (True, {'name': 'variable2', 'resolution': 742, 'file_type': ['level2']}),
            (True, {'name': 'variable2', 'resolution': 1, 'file_type': ['level2']}),
            (None, {'name': 'variable2', 'resolution': 1, 'file_type': ['level2']}),
            (None, {'name': '_fake1', 'file_type': ['level2']}),
            (None, {'name': 'variable1', 'file_type': ['level_fake']}),
            (True, {'name': 'variable3', 'file_type': ['level2']}),
        ]
        new_ds_infos = list(r.file_handlers['level2'][0].available_datasets(
            fake_dataset_info))
        self.assertEqual(len(new_ds_infos), 9)

        # we have this and can provide the resolution
        self.assertTrue(new_ds_infos[0][0])
        self.assertEqual(new_ds_infos[0][1]['resolution'], 742)  # hardcoded

        # we have this, but previous file handler said it knew about it
        # and it is producing the same resolution as what we have
        self.assertTrue(new_ds_infos[1][0])
        self.assertEqual(new_ds_infos[1][1]['resolution'], 742)

        # we have this, but don't want to change the resolution
        # because a previous handler said it has it
        self.assertTrue(new_ds_infos[2][0])
        self.assertEqual(new_ds_infos[2][1]['resolution'], 1)

        # even though the previous one was known we can still
        # produce it at our new resolution
        self.assertTrue(new_ds_infos[3][0])
        self.assertEqual(new_ds_infos[3][1]['resolution'], 742)

        # we have this and can update the resolution since
        # no one else has claimed it
        self.assertTrue(new_ds_infos[4][0])
        self.assertEqual(new_ds_infos[4][1]['resolution'], 742)

        # we don't have this variable, don't change it
        self.assertFalse(new_ds_infos[5][0])
        self.assertIsNone(new_ds_infos[5][1].get('resolution'))

        # we have this, but it isn't supposed to come from our file type
        self.assertIsNone(new_ds_infos[6][0])
        self.assertIsNone(new_ds_infos[6][1].get('resolution'))

        # we could have loaded this but some other file handler said it has this
        self.assertTrue(new_ds_infos[7][0])
        self.assertIsNone(new_ds_infos[7][1].get('resolution'))

        # we can add resolution to the previous dataset, so we do
        self.assertTrue(new_ds_infos[8][0])
        self.assertEqual(new_ds_infos[8][1]['resolution'], 742)

    def test_load_all(self):
        """Test loading all test datasets"""
        from satpy.readers import load_reader
        import xarray as xr
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.clavrx.SDS', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'clavrx_npp_d20170520_t2053581_e2055223_b28822.level2.hdf',
            ])
            r.create_filehandlers(loadables)
        datasets = r.load(['variable1',
                           'variable2',
                           'variable3'])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            self.assertIs(v.attrs['calibration'], None)
            self.assertEqual(v.attrs['units'], '1')
        self.assertIsNotNone(datasets['variable3'].attrs.get('flag_meanings'))


class FakeHDF4FileHandlerGeo(FakeHDF4FileHandler):
    """Swap-in HDF4 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        file_content = {
            '/attr/platform': 'HIM8',
            '/attr/sensor': 'AHI',
            # this is a Level 2 file that came from a L1B file
            '/attr/L1B': 'clavrx_H08_20180806_1800',
        }

        file_content['longitude'] = xr.DataArray(
            DEFAULT_LON_DATA,
            dims=('y', 'x'),
            attrs={
                '_FillValue': np.nan,
                'scale_factor': 1.,
                'add_offset': 0.,
                'standard_name': 'longitude',
            })
        file_content['longitude/shape'] = DEFAULT_FILE_SHAPE

        file_content['latitude'] = xr.DataArray(
            DEFAULT_LAT_DATA,
            dims=('y', 'x'),
            attrs={
                '_FillValue': np.nan,
                'scale_factor': 1.,
                'add_offset': 0.,
                'standard_name': 'latitude',
            })
        file_content['latitude/shape'] = DEFAULT_FILE_SHAPE

        file_content['variable1'] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.float32),
            dims=('y', 'x'),
            attrs={
                '_FillValue': -1,
                'scale_factor': 1.,
                'add_offset': 0.,
                'units': '1',
                'valid_range': (-32767, 32767),
            })
        file_content['variable1/shape'] = DEFAULT_FILE_SHAPE

        # data with fill values
        file_content['variable2'] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.float32),
            dims=('y', 'x'),
            attrs={
                '_FillValue': -1,
                'scale_factor': 1.,
                'add_offset': 0.,
                'units': '1',
            })
        file_content['variable2/shape'] = DEFAULT_FILE_SHAPE
        file_content['variable2'] = file_content['variable2'].where(
            file_content['variable2'] % 2 != 0)

        # category
        file_content['variable3'] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.byte),
            dims=('y', 'x'),
            attrs={
                '_FillValue': -128,
                'flag_meanings': 'clear water supercooled mixed ice unknown',
                'flag_values': [0, 1, 2, 3, 4, 5],
                'units': '1',
            })
        file_content['variable3/shape'] = DEFAULT_FILE_SHAPE

        return file_content


class TestCLAVRXReaderGeo(unittest.TestCase):
    """Test CLAVR-X Reader with Geo files."""
    yaml_file = "clavrx.yaml"

    def setUp(self):
        """Wrap HDF4 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.clavrx import CLAVRXFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(CLAVRXFileHandler, '__bases__', (FakeHDF4FileHandlerGeo,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'clavrx_H08_20180806_1800.level2.hdf',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_no_nav_donor(self):
        """Test exception raised when no donor file is available."""
        from satpy.readers import load_reader
        import xarray as xr
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.clavrx.SDS', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'clavrx_H08_20180806_1800.level2.hdf',
            ])
            r.create_filehandlers(loadables)
        self.assertRaises(IOError, r.load, ['variable1', 'variable2', 'variable3'])

    def test_load_all_old_donor(self):
        """Test loading all test datasets with old donor."""
        from satpy.readers import load_reader
        import xarray as xr
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.clavrx.SDS', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'clavrx_H08_20180806_1800.level2.hdf',
            ])
            r.create_filehandlers(loadables)
        with mock.patch('satpy.readers.clavrx.glob') as g, mock.patch('satpy.readers.clavrx.netCDF4.Dataset') as d:
            g.return_value = ['fake_donor.nc']
            x = np.linspace(-0.1518, 0.1518, 300)
            y = np.linspace(0.1518, -0.1518, 10)
            proj = mock.Mock(
                semi_major_axis=6378.137,
                semi_minor_axis=6356.7523142,
                perspective_point_height=35791,
                longitude_of_projection_origin=140.7,
                sweep_angle_axis='y',
            )
            d.return_value = fake_donor = mock.MagicMock(
                variables={'Projection': proj, 'x': x, 'y': y},
            )
            fake_donor.__getitem__.side_effect = lambda key: fake_donor.variables[key]
            datasets = r.load(['variable1', 'variable2', 'variable3'])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            self.assertIs(v.attrs['calibration'], None)
            self.assertEqual(v.attrs['units'], '1')
            self.assertIsInstance(v.attrs['area'], AreaDefinition)
        self.assertIsNotNone(datasets['variable3'].attrs.get('flag_meanings'))

    def test_load_all_new_donor(self):
        """Test loading all test datasets with new donor."""
        from satpy.readers import load_reader
        import xarray as xr
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.clavrx.SDS', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'clavrx_H08_20180806_1800.level2.hdf',
            ])
            r.create_filehandlers(loadables)
        with mock.patch('satpy.readers.clavrx.glob') as g, mock.patch('satpy.readers.clavrx.netCDF4.Dataset') as d:
            g.return_value = ['fake_donor.nc']
            x = np.linspace(-0.1518, 0.1518, 300)
            y = np.linspace(0.1518, -0.1518, 10)
            proj = mock.Mock(
                semi_major_axis=6378137,
                semi_minor_axis=6356752.3142,
                perspective_point_height=35791000,
                longitude_of_projection_origin=140.7,
                sweep_angle_axis='y',
            )
            d.return_value = fake_donor = mock.MagicMock(
                variables={'goes_imager_projection': proj, 'x': x, 'y': y},
            )
            fake_donor.__getitem__.side_effect = lambda key: fake_donor.variables[key]
            datasets = r.load(['variable1', 'variable2', 'variable3'])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            self.assertIs(v.attrs['calibration'], None)
            self.assertEqual(v.attrs['units'], '1')
            self.assertIsInstance(v.attrs['area'], AreaDefinition)
        self.assertIsNotNone(datasets['variable3'].attrs.get('flag_meanings'))
