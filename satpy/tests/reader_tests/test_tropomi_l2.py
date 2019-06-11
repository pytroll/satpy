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

"""Module for testing the satpy.readers.tropomi_l2 module.
"""

import os
import sys
from datetime import datetime
import numpy as np
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
import xarray as xr

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
                '/attr/platform_shortname': 'S5P',
                '/attr/sensor': 'TROPOMI',
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
            file_content['PRODUCT/latitude'].attrs['_FillValue'] = -999.0
            file_content['PRODUCT/longitude'].attrs['_FillValue'] = -999.0
            if 'NO2' in filename:
                file_content['PRODUCT/nitrogen_dioxide_total_column'].attrs['_FillValue'] = -999.0
            if 'SO2' in filename:
                file_content['PRODUCT/sulfurdioxide_total_vertical_column'].attrs['_FillValue'] = -999.0

        else:
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

    def test_load_no2(self):
        """Load NO2 dataset"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.tropomi_l2.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'S5P_OFFL_L2__NO2____20180709T170334_20180709T184504_03821_01_010002_20180715T184729.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(['nitrogen_dioxide_total_column'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.attrs['platform_shortname'], 'S5P')
            self.assertEqual(d.attrs['sensor'], 'TROPOMI')
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_load_so2(self):
        """Load SO2 dataset"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        with mock.patch('satpy.readers.tropomi_l2.netCDF4.Variable', xr.DataArray):
            loadables = r.select_files_from_pathnames([
                'S5P_OFFL_L2__SO2____20181224T055107_20181224T073237_06198_01_010105_20181230T150634.nc',
            ])
            r.create_filehandlers(loadables)
        ds = r.load(['sulfurdioxide_total_vertical_column'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.attrs['platform_shortname'], 'S5P')
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])


def suite():
    """The test suite for test_tropomi_l2.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestTROPOMIL2Reader))

    return mysuite


if __name__ == '__main__':
    unittest.main()
