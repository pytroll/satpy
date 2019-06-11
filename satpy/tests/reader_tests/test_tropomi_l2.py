#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.tropomi_l2 module.
"""

import os
import sys
from datetime import datetime
import numpy as np
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.readers.netcdf_utils import netCDF4

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (3246, 450)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)


class FakeNetCDF4FileHandlerTL2(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        from xarray import DataArray
        dt_s = filename_info.get('start_time', datetime(2016, 1, 1, 12, 0, 0))
        dt_e = filename_info.get('end_time', datetime(2016, 1, 1, 12, 0, 0))

        if filetype_info['file_type'] == 'tropomi_l2':
            file_content = {
                '/attr/time_coverage_start': dt_s.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                '/attr/time_coverage_end': dt_e.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                '/attr/institution': 'KNMI',
                '/attr/orbit_number': 3821,
                '/attr/sensor': 'TROPOMI',
                '/attr/platform': 'S5P',
            }

            file_content['PRODUCT/latitude'] = DEFAULT_FILE_DATA
            file_content['PRODUCT/longitude'] = DEFAULT_FILE_DATA

            if 'NO2' in filename:
                file_content['PRODUCT/nitrogen_dioxide_total_column'] = DEFAULT_FILE_DATA
            if 'SO2' in filename:
                file_content['PRODUCT/sulfurdioxide_total_vertical_column'] = DEFAULT_FILE_DATA

            for k in list(file_content.keys()):
                if not k.startswith('PRODUCT'):
                    continue
                file_content[k + '/shape'] = DEFAULT_FILE_SHAPE

            # convert to xarrays
            for key, val in file_content.items():
                if isinstance(val, np.ndarray):
                    if val.ndim > 1:
                        file_content[key] = DataArray(val, dims=('y', 'x'))
                    else:
                        file_content[key] = DataArray(val)

        else:
            print('filetype_info is: ', filetype_info)
            print('filetype is: ', filetype_info['file_type'])
            assert False

        return file_content


class TestTROPOMIL2Reader(unittest.TestCase):
    """Test TROPOMI L2 Reader"""
    yaml_file = "tropomi_l2.yaml"

    def setUp(self):
        """Wrap NetCDF4 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.tropomi_l2 import TROPOMIL2FileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(TROPOMIL2FileHandler, '__bases__', (FakeNetCDF4FileHandlerTL2,))
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
            'S5P_OFFL_L2__NO2____20180709T170334_20180709T184504_03821_01_010002_20180715T184729.nc',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_available_datasets(self, configured_datasets=None):
        """Test loading datasets."""
        from satpy.readers import load_reader
        from satpy import DatasetID
        print("test_load_available_datasets begin...")
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'S5P_OFFL_L2__NO2____20180709T170334_20180709T184504_03821_01_010002_20180715T184729.nc',
            ])
        r.create_filehandlers(loadables)

        # Determine shape of the geolocation data (lat/lon)
        lat_shape = None
        for var_name, val in self.file_content.items():
            if var_name == 'PRODUCT/latitude':
                lat_shape = self[var_name + "/shape"]
                break

        handled_variables = set()

        # update previously configured datasets
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                yield is_avail, ds_info
            var_name = ds_info.get('file_key', ds_info['name'])
            matches = self.file_type_matches(ds_info['file_type'])
            if matches and var_name in self:
                handled_variables.add(var_name)
                new_info = ds_info.copy()
                yield True, new_info
            elif is_avail is None:
                yield is_avail, ds_info

        for var_name, val in self.file_content.items():
            if isinstance(val, netCDF4.Variable):
                var_shape = self[var_name + "/shape"]
                if var_shape == lat_shape:
                    if var_name in handled_variables:
                        continue
                    handled_variables.add(var_name)
                    last_index_separator = var_name.rindex('/')
                    last_index_separator = last_index_separator + 1
                    var_name_no_path = var_name[last_index_separator:]
                    new_info = {
                        'name': var_name_no_path,
                        'file_key': var_name,
                        'coordinates': ['longitude', 'latitude'],
                        'file_type': self.filetype_info['file_type'],
                        'resolution': None,
                    }
                    datasets = r.load([DatasetID(new_info)])

        print("num datasets: ", len(datasets))
        self.assertEqual(len(datasets), 7)


def suite():
    """The test suite for test_tropomi_l2.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestTROPOMIL2Reader))

    return mysuite


if __name__ == '__main__':
    unittest.main()
