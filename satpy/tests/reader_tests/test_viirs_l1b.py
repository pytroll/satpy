#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.viirs_l1b module.
"""

import os
import sys
from datetime import datetime, timedelta
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
    """Swap-in NetCDF4 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        dt = filename_info.get('start_time', datetime(2016, 1, 1, 12, 0, 0))
        file_type = filename[:5].lower()
        # num_lines = {
        #     'vl1bi': 3248 * 2,
        #     'vl1bm': 3248,
        #     'vl1bd': 3248,
        # }[file_type]
        # num_pixels = {
        #     'vl1bi': 6400,
        #     'vl1bm': 3200,
        #     'vl1bd': 4064,
        # }[file_type]
        # num_scans = 203
        # num_luts = 65536
        num_lines = DEFAULT_FILE_SHAPE[0]
        num_pixels = DEFAULT_FILE_SHAPE[1]
        num_scans = 5
        num_luts = DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1]
        file_content = {
            '/dimension/number_of_scans': num_scans,
            '/dimension/number_of_lines': num_lines,
            '/dimension/number_of_pixels': num_pixels,
            '/dimension/number_of_LUT_values': num_luts,
            '/attr/time_coverage_start': dt.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            '/attr/time_coverage_end': (dt + timedelta(minutes=6)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            '/attr/orbit_number': 26384,
            '/attr/instrument': 'viirs',
            '/attr/platform': 'Suomi-NPP',
        }
        if file_type.startswith('vgeo'):
            file_content['/attr/OrbitNumber'] = file_content.pop('/attr/orbit_number')
            file_content['geolocation_data/latitude'] = DEFAULT_LAT_DATA
            file_content['geolocation_data/longitude'] = DEFAULT_LON_DATA
        elif file_type == 'vl1bm':
            file_content['observation_data/M01'] = DEFAULT_FILE_DATA
            file_content['observation_data/M02'] = DEFAULT_FILE_DATA
            file_content['observation_data/M03'] = DEFAULT_FILE_DATA
            file_content['observation_data/M04'] = DEFAULT_FILE_DATA
            file_content['observation_data/M05'] = DEFAULT_FILE_DATA
            file_content['observation_data/M06'] = DEFAULT_FILE_DATA
            file_content['observation_data/M07'] = DEFAULT_FILE_DATA
            file_content['observation_data/M08'] = DEFAULT_FILE_DATA
            file_content['observation_data/M09'] = DEFAULT_FILE_DATA
            file_content['observation_data/M10'] = DEFAULT_FILE_DATA
            file_content['observation_data/M11'] = DEFAULT_FILE_DATA
            file_content['observation_data/M12'] = DEFAULT_FILE_DATA
            file_content['observation_data/M13'] = DEFAULT_FILE_DATA
            file_content['observation_data/M14'] = DEFAULT_FILE_DATA
            file_content['observation_data/M15'] = DEFAULT_FILE_DATA
            file_content['observation_data/M16'] = DEFAULT_FILE_DATA
        elif file_type == 'vl1bi':
            file_content['observation_data/I01'] = DEFAULT_FILE_DATA
            file_content['observation_data/I02'] = DEFAULT_FILE_DATA
            file_content['observation_data/I03'] = DEFAULT_FILE_DATA
            file_content['observation_data/I04'] = DEFAULT_FILE_DATA
            file_content['observation_data/I05'] = DEFAULT_FILE_DATA
        elif file_type == 'vl1bd':
            file_content['observation_data/DNB_observations'] = DEFAULT_FILE_DATA
            file_content['observation_data/DNB_observations/attr/units'] = 'Watts/cm^2/steradian'

        for k in list(file_content.keys()):
            if not k.startswith('observation_data') and not k.startswith('geolocation_data'):
                continue
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            if k[-3:] in ['M12', 'M13', 'M14', 'M15', 'M16', 'I04', 'I05']:
                file_content[k + '_brightness_temperature_lut'] = DEFAULT_FILE_DATA.ravel()
                file_content[k + '_brightness_temperature_lut/attr/units'] = 'Kelvin'
                file_content[k + '_brightness_temperature_lut/attr/valid_min'] = 0
                file_content[k + '_brightness_temperature_lut/attr/valid_max'] = 65534
                file_content[k + '_brightness_temperature_lut/attr/_FillValue'] = 65535
                file_content[k + '/attr/units'] = 'Watts/meter^2/steradian/micrometer'
            elif k[-3:] in ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08',
                            'M09', 'M10', 'M11', 'I01', 'I02', 'I03']:
                file_content[k + '/attr/radiance_units'] = 'Watts/meter^2/steradian/micrometer'
                file_content[k + '/attr/radiance_scale_factor'] = 1.1
                file_content[k + '/attr/radiance_add_offset'] = 0.1
            elif k.endswith('longitude'):
                file_content[k + '/attr/units'] = 'degrees_east'
            elif k.endswith('latitude'):
                file_content[k + '/attr/units'] = 'degrees_north'
            file_content[k + '/attr/valid_min'] = 0
            file_content[k + '/attr/valid_max'] = 65534
            file_content[k + '/attr/_FillValue'] = 65535
            file_content[k + '/attr/scale_factor'] = 1.1
            file_content[k + '/attr/add_offset'] = 0.1

        # convert to xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                if val.ndim > 1:
                    file_content[key] = DataArray(val, dims=('y', 'x'))
                else:
                    file_content[key] = DataArray(val)

        return file_content


class TestVIIRSL1BReader(unittest.TestCase):
    """Test VIIRS L1B Reader"""
    yaml_file = "viirs_l1b.yaml"

    def setUp(self):
        """Wrap NetCDF4 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_l1b import VIIRSL1BFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIIRSL1BFileHandler, '__bases__', (FakeNetCDF4FileHandler2,))
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
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_every_m_band_bt(self):
        """Test loading all M band brightness temperatures"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['M12',
                           'M13',
                           'M14',
                           'M15',
                           'M16'])
        self.assertEqual(len(datasets), 5)
        for v in datasets.values():
            self.assertEqual(v.attrs['calibration'], 'brightness_temperature')
            self.assertEqual(v.attrs['units'], 'K')

    def test_load_every_m_band_refl(self):
        """Test loading all M band reflectances"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['M01',
                           'M02',
                           'M03',
                           'M04',
                           'M05',
                           'M06',
                           'M07',
                           'M08',
                           'M09',
                           'M10',
                           'M11'])
        self.assertEqual(len(datasets), 11)
        for v in datasets.values():
            self.assertEqual(v.attrs['calibration'], 'reflectance')
            self.assertEqual(v.attrs['units'], '%')

    def test_load_every_m_band_rad(self):
        """Test loading all M bands as radiances"""
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BM_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOM_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load([DatasetID('M01', calibration='radiance'),
                           DatasetID('M02', calibration='radiance'),
                           DatasetID('M03', calibration='radiance'),
                           DatasetID('M04', calibration='radiance'),
                           DatasetID('M05', calibration='radiance'),
                           DatasetID('M06', calibration='radiance'),
                           DatasetID('M07', calibration='radiance'),
                           DatasetID('M08', calibration='radiance'),
                           DatasetID('M09', calibration='radiance'),
                           DatasetID('M10', calibration='radiance'),
                           DatasetID('M11', calibration='radiance'),
                           DatasetID('M12', calibration='radiance'),
                           DatasetID('M13', calibration='radiance'),
                           DatasetID('M14', calibration='radiance'),
                           DatasetID('M15', calibration='radiance'),
                           DatasetID('M16', calibration='radiance')])
        self.assertEqual(len(datasets), 16)
        for v in datasets.values():
            self.assertEqual(v.attrs['calibration'], 'radiance')
            self.assertEqual(v.attrs['units'], 'W m-2 um-1 sr-1')

    def test_load_dnb_radiance(self):
        """Test loading the main DNB dataset"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'VL1BD_snpp_d20161130_t012400_c20161130054822.nc',
            'VGEOD_snpp_d20161130_t012400_c20161130054822.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['DNB'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['calibration'], 'radiance')
            self.assertEqual(v.attrs['units'], 'W m-2 sr-1')


def suite():
    """The test suite for test_viirs_l1b.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSL1BReader))

    return mysuite
