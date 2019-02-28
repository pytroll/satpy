#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.acspo module.
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
        sat, inst = {
            'VIIRS_NPP': ('NPP', 'VIIRS'),
        }[filename_info['sensor_id']]

        file_content = {
            '/attr/platform': sat,
            '/attr/sensor': inst,
            '/attr/spatial_resolution': '742 m at nadir',
            '/attr/time_coverage_start': dt.strftime('%Y%m%dT%H%M%SZ'),
            '/attr/time_coverage_end': (dt + timedelta(minutes=6)).strftime('%Y%m%dT%H%M%SZ'),
        }

        file_content['lat'] = DEFAULT_LAT_DATA
        file_content['lat/attr/comment'] = 'Latitude of retrievals'
        file_content['lat/attr/long_name'] = 'latitude'
        file_content['lat/attr/standard_name'] = 'latitude'
        file_content['lat/attr/units'] = 'degrees_north'
        file_content['lat/attr/valid_min'] = -90.
        file_content['lat/attr/valid_max'] = 90.
        file_content['lat/shape'] = DEFAULT_FILE_SHAPE

        file_content['lon'] = DEFAULT_LON_DATA
        file_content['lon/attr/comment'] = 'Longitude of retrievals'
        file_content['lon/attr/long_name'] = 'longitude'
        file_content['lon/attr/standard_name'] = 'longitude'
        file_content['lon/attr/units'] = 'degrees_east'
        file_content['lon/attr/valid_min'] = -180.
        file_content['lon/attr/valid_max'] = 180.
        file_content['lon/shape'] = DEFAULT_FILE_SHAPE

        for k in ['sea_surface_temperature',
                  'satellite_zenith_angle',
                  'sea_ice_fraction',
                  'wind_speed']:
            file_content[k] = DEFAULT_FILE_DATA[None, ...]
            file_content[k + '/attr/scale_factor'] = 1.1
            file_content[k + '/attr/add_offset'] = 0.1
            file_content[k + '/attr/units'] = 'some_units'
            file_content[k + '/attr/comment'] = 'comment'
            file_content[k + '/attr/standard_name'] = 'standard_name'
            file_content[k + '/attr/long_name'] = 'long_name'
            file_content[k + '/attr/valid_min'] = 0
            file_content[k + '/attr/valid_max'] = 65534
            file_content[k + '/attr/_FillValue'] = 65534
            file_content[k + '/shape'] = (1, DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])

        file_content['l2p_flags'] = np.zeros(
            (1, DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1]),
            dtype=np.uint16)

        # convert to xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                if val.ndim > 1:
                    file_content[key] = DataArray(val, dims=tuple(x for x in 'zyx'[3-val.ndim:]))
                else:
                    file_content[key] = DataArray(val)

        return file_content


class TestACSPOReader(unittest.TestCase):
    """Test ACSPO Reader"""
    yaml_file = "acspo.yaml"

    def setUp(self):
        """Wrap NetCDF4 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.acspo import ACSPOFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(ACSPOFileHandler, '__bases__', (FakeNetCDF4FileHandler2,))
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
            '20170401174600-STAR-L2P_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40-v02.0-fv01.0.nc',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_every_dataset(self):
        """Test loading all datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            '20170401174600-STAR-L2P_GHRSST-SSTskin-VIIRS_NPP-ACSPO_V2.40-v02.0-fv01.0.nc',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['sst',
                           'satellite_zenith_angle',
                           'sea_ice_fraction',
                           'wind_speed'])
        self.assertEqual(len(datasets), 4)
        for d in datasets.values():
            self.assertTupleEqual(d.shape, DEFAULT_FILE_SHAPE)


def suite():
    """The test suite for test_acspo.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestACSPOReader))

    return mysuite
