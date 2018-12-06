#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.nucaps module.
"""

import os
import sys
import numpy as np
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


DEFAULT_FILE_DTYPE = np.float32
DEFAULT_FILE_SHAPE = (120,)
DEFAULT_PRES_FILE_SHAPE = (120, 100,)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_PRES_FILE_DATA = np.arange(DEFAULT_PRES_FILE_SHAPE[1], dtype=DEFAULT_FILE_DTYPE)
DEFAULT_PRES_FILE_DATA = np.repeat([DEFAULT_PRES_FILE_DATA], DEFAULT_PRES_FILE_SHAPE[0], axis=0)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[0]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[0]).astype(DEFAULT_FILE_DTYPE)
ALL_PRESSURE_LEVELS = [
    0.0161, 0.0384, 0.0769, 0.137, 0.2244, 0.3454, 0.5064, 0.714,
    0.9753, 1.2972, 1.6872, 2.1526, 2.7009, 3.3398, 4.077, 4.9204,
    5.8776, 6.9567, 8.1655, 9.5119, 11.0038, 12.6492, 14.4559, 16.4318,
    18.5847, 20.9224, 23.4526, 26.1829, 29.121, 32.2744, 35.6505,
    39.2566, 43.1001, 47.1882, 51.5278, 56.126, 60.9895, 66.1253,
    71.5398, 77.2396, 83.231, 89.5204, 96.1138, 103.017, 110.237,
    117.777, 125.646, 133.846, 142.385, 151.266, 160.496, 170.078,
    180.018, 190.32, 200.989, 212.028, 223.441, 235.234, 247.408,
    259.969, 272.919, 286.262, 300, 314.137, 328.675, 343.618, 358.966,
    374.724, 390.893, 407.474, 424.47, 441.882, 459.712, 477.961,
    496.63, 515.72, 535.232, 555.167, 575.525, 596.306, 617.511, 639.14,
    661.192, 683.667, 706.565, 729.886, 753.628, 777.79, 802.371,
    827.371, 852.788, 878.62, 904.866, 931.524, 958.591, 986.067,
    1013.95, 1042.23, 1070.92, 1100
]
ALL_PRESSURE_LEVELS = np.repeat([ALL_PRESSURE_LEVELS], DEFAULT_PRES_FILE_SHAPE[0], axis=0)


class FakeNetCDF4FileHandler2(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        file_content = {
            '/attr/time_coverage_start': filename_info['start_time'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            '/attr/time_coverage_end': filename_info['end_time'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            '/attr/start_orbit_number': 1,
            '/attr/end_orbit_number': 2,
            '/attr/platform_name': 'NPP',
            '/attr/instrument_name': 'CrIS, ATMS, VIIRS',
        }
        for k, units, standard_name in [
            ('Solar_Zenith', 'degrees', 'solar_zenith_angle'),
            ('Topography', 'meters', ''),
            ('Land_Fraction', '1', ''),
            ('Surface_Pressure', 'mb', ''),
            ('Skin_Temperature', 'Kelvin', 'surface_temperature'),
        ]:
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/units'] = units
            file_content[k + '/attr/valid_range'] = (0., 120.)
            file_content[k + '/attr/_FillValue'] = -9999.
            if standard_name:
                file_content[k + '/attr/standard_name'] = standard_name
        for k, units, standard_name in [
            ('Temperature', 'Kelvin', 'air_temperature'),
            ('H2O', '1', ''),
            ('H2O_MR', 'g/g', ''),
            ('O3', '1', ''),
            ('O3_MR', '1', ''),
            ('Liquid_H2O', '1', ''),
            ('Liquid_H2O_MR', 'g/g', 'cloud_liquid_water_mixing_ratio'),
            ('CO', '1', ''),
            ('CO_MR', '1', ''),
            ('CH4', '1', ''),
            ('CH4_MR', '1', ''),
            ('CO2', '1', ''),
            ('HNO3', '1', ''),
            ('HNO3_MR', '1', ''),
            ('N2O', '1', ''),
            ('N2O_MR', '1', ''),
            ('SO2', '1', ''),
            ('SO2_MR', '1', ''),
        ]:
            file_content[k] = DEFAULT_PRES_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_PRES_FILE_SHAPE
            file_content[k + '/attr/units'] = units
            file_content[k + '/attr/valid_range'] = (0., 120.)
            file_content[k + '/attr/_FillValue'] = -9999.
            if standard_name:
                file_content[k + '/attr/standard_name'] = standard_name
        k = 'Pressure'
        file_content[k] = ALL_PRESSURE_LEVELS
        file_content[k + '/shape'] = DEFAULT_PRES_FILE_SHAPE
        file_content[k + '/attr/units'] = 'mb'
        file_content[k + '/attr/valid_range'] = (0., 2000.)
        file_content[k + '/attr/_FillValue'] = -9999.

        k = 'Quality_Flag'
        file_content[k] = DEFAULT_FILE_DATA.astype(np.int32)
        file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
        file_content[k + '/attr/valid_range'] = (0, 31)
        file_content[k + '/attr/_FillValue'] = -9999.

        k = 'Longitude'
        file_content[k] = DEFAULT_LON_DATA
        file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
        file_content[k + '/attr/units'] = 'degrees_east'
        file_content[k + '/attr/valid_range'] = (-180., 180.)
        file_content[k + '/attr/standard_name'] = 'longitude'
        file_content[k + '/attr/_FillValue'] = -9999.

        k = 'Latitude'
        file_content[k] = DEFAULT_LAT_DATA
        file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
        file_content[k + '/attr/units'] = 'degrees_north'
        file_content[k + '/attr/valid_range'] = (-90., 90.)
        file_content[k + '/attr/standard_name'] = 'latitude'
        file_content[k + '/attr/_FillValue'] = -9999.

        # convert to xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                attrs = {}
                for a in ['_FillValue', 'flag_meanings', 'flag_values', 'units']:
                    if key + '/attr/' + a in file_content:
                        attrs[a] = file_content[key + '/attr/' + a]
                if val.ndim == 1:
                    file_content[key] = DataArray(val, dims=('number_of_FORs',), attrs=attrs)
                elif val.ndim > 1:
                    file_content[key] = DataArray(val, dims=('number_of_FORs', 'number_of_p_levels'), attrs=attrs)
                else:
                    file_content[key] = DataArray(val, attrs=attrs)

        return file_content


class TestNUCAPSReader(unittest.TestCase):
    """Test NUCAPS Reader"""
    yaml_file = "nucaps.yaml"

    def setUp(self):
        """Wrap NetCDF4 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.nucaps import NUCAPSFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(NUCAPSFileHandler, '__bases__', (FakeNetCDF4FileHandler2,))
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
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_nonpressure_based(self):
        """Test loading all channels that aren't based on pressure"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['Solar_Zenith',
                           'Topography',
                           'Land_Fraction',
                           'Surface_Pressure',
                           'Skin_Temperature',
                           'Quality_Flag',
                           ])
        self.assertEqual(len(datasets), 6)
        for v in datasets.values():
            # self.assertNotEqual(v.info['resolution'], 0)
            # self.assertEqual(v.info['units'], 'degrees')
            self.assertEqual(v.ndim, 1)
            self.assertEqual(v.attrs['sensor'], ['CrIS', 'ATMS', 'VIIRS'])

    def test_load_pressure_based(self):
        """Test loading all channels based on pressure"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['Temperature',
                           'H2O',
                           'H2O_MR',
                           'O3',
                           'O3_MR',
                           'Liquid_H2O',
                           'Liquid_H2O_MR',
                           'CO',
                           'CO_MR',
                           'CH4',
                           'CH4_MR',
                           'CO2',
                           'HNO3',
                           'HNO3_MR',
                           'N2O',
                           'N2O_MR',
                           'SO2',
                           'SO2_MR',
                           ])
        self.assertEqual(len(datasets), 18)
        for v in datasets.values():
            # self.assertNotEqual(v.info['resolution'], 0)
            self.assertEqual(v.ndim, 2)

    def test_load_individual_pressure_levels_true(self):
        """Test loading Temperature with individual pressure datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(r.pressure_dataset_names['Temperature'], pressure_levels=True)
        self.assertEqual(len(datasets), 100)
        for v in datasets.values():
            self.assertEqual(v.ndim, 1)

    def test_load_individual_pressure_levels_min_max(self):
        """Test loading individual Temperature with min/max level specified"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(r.pressure_dataset_names['Temperature'], pressure_levels=(100., 150.))
        self.assertEqual(len(datasets), 6)
        for v in datasets.values():
            self.assertEqual(v.ndim, 1)

    def test_load_individual_pressure_levels_single(self):
        """Test loading individual Temperature with specific levels"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(r.pressure_dataset_names['Temperature'], pressure_levels=(103.017,))
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.ndim, 1)

    def test_load_pressure_levels_true(self):
        """Test loading Temperature with all pressure levels"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['Temperature'], pressure_levels=True)
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.ndim, 2)
            self.assertTupleEqual(v.shape, DEFAULT_PRES_FILE_SHAPE)

    def test_load_pressure_levels_min_max(self):
        """Test loading Temperature with min/max level specified"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['Temperature'], pressure_levels=(100., 150.))
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.ndim, 2)
            self.assertTupleEqual(v.shape,
                                  (DEFAULT_PRES_FILE_SHAPE[0], 6))

    def test_load_pressure_levels_single(self):
        """Test loading a specific Temperature level"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['Temperature'], pressure_levels=(103.017,))
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.ndim, 2)
            self.assertTupleEqual(v.shape,
                                  (DEFAULT_PRES_FILE_SHAPE[0], 1))

    def test_load_pressure_levels_single_and_pressure_levels(self):
        """Test loading a specific Temperature level and pressure levels"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'NUCAPS-EDR_v1r0_npp_s201603011158009_e201603011158307_c201603011222270.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['Temperature', 'Pressure_Levels'], pressure_levels=(103.017,))
        self.assertEqual(len(datasets), 2)
        t_ds = datasets['Temperature']
        self.assertEqual(t_ds.ndim, 2)
        self.assertTupleEqual(t_ds.shape,
                              (DEFAULT_PRES_FILE_SHAPE[0], 1))
        pl_ds = datasets['Pressure_Levels']
        self.assertTupleEqual(pl_ds.shape, (1,))


def suite():
    """The test suite for test_nucaps.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNUCAPSReader))

    return mysuite
