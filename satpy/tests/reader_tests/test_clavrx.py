#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.clavrx module.
"""

import os
import sys
import numpy as np
import xarray as xr
from satpy.tests.reader_tests.test_hdf4_utils import FakeHDF4FileHandler
from pyresample.geometry import AreaDefinition

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

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
            DEFAULT_LON_DATA,
            attrs={
                '_FillValue': np.nan,
                'scale_factor': 1.,
                'add_offset': 0.,
                'standard_name': 'longitude',
            })
        file_content['longitude/shape'] = DEFAULT_FILE_SHAPE

        file_content['latitude'] = xr.DataArray(
            DEFAULT_LAT_DATA,
            attrs={
                '_FillValue': np.nan,
                'scale_factor': 1.,
                'add_offset': 0.,
                'standard_name': 'latitude',
            })
        file_content['latitude/shape'] = DEFAULT_FILE_SHAPE

        file_content['variable1'] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.float32),
            attrs={
                '_FillValue': -1,
                'scale_factor': 1.,
                'add_offset': 0.,
                'units': '1',
            })
        file_content['variable1/shape'] = DEFAULT_FILE_SHAPE

        # data with fill values
        file_content['variable2'] = xr.DataArray(
            DEFAULT_FILE_DATA.astype(np.float32),
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


def suite():
    """The test suite for test_viirs_l1b.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCLAVRXReaderPolar))
    mysuite.addTest(loader.loadTestsFromTestCase(TestCLAVRXReaderGeo))

    return mysuite
