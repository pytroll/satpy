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

""" Tests for the viirs_gran Satpy reader """
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

DEFAULT_FILE_DTYPE = np.float32
DEFAULT_FILE_SHAPE = (768, 3200)
DEFAULT_FILE_DATA = \
    np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_LAT_DATA = \
    np.linspace(58, 71, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = \
    np.linspace(-93, -22, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeNetCDF4FileHandlerGRAN(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        from xarray import DataArray
        dt = filename_info.get('start_time', datetime(2019, 3, 6, 6, 48, 0))

        if filetype_info['file_type'] == 'viirs_edr_generic':
            file_content = {
                '/attr/satellite_name': 'NPP',
                '/attr/instrument_name': 'VIIRS',
                '/attr/title': 'JRR-CloudMask',
                '/attr/start_orbit_number': 38106,
                '/attr/end_orbit_number': 38106,
                '/attr/time_coverage_start': dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
                '/attr/time_coverage_end': (dt
                                            + timedelta(minutes=6)).strftime(
                                            '%Y-%m-%dT%H:%M:%SZ'),
                '/attr/resolution': '750M'
            }

            file_content['Latitude'] = DataArray(DEFAULT_LAT_DATA,
                                                 dims=('Rows', 'Columns'))
            file_content['Latitude'].attrs['units'] = 'degrees_north'
            file_content['Latitude'].attrs['_FillValue'] = -999.0
            file_content['Latitude'].attrs['valid_range'] = [-90, 90]
            file_content['Latitude/shape'] = (768, 3200)

            file_content['Longitude'] = DataArray(DEFAULT_LON_DATA,
                                                  dims=('Rows', 'Columns'))
            file_content['Longitude'].attrs['units'] = 'degrees_east'
            file_content['Longitude'].attrs['_FillValue'] = -999.0
            file_content['Longitude'].attrs['valid_range'] = [-180, 180]
            file_content['Longitude/shape'] = (768, 3200)

            file_content['CloudMask'] = DataArray(DEFAULT_FILE_DATA,
                                                  dims=('Rows', 'Columns'))
            file_content['CloudMask'].attrs['units'] = 1
            file_content['CloudMask'].attrs['_FillValue'] = -128
            file_content['CloudMask'].attrs['valid_range'] = [0, 3]
            file_content['CloudMask/shape'] = (768, 3200)

            file_content['CloudMaskBinary'] = \
                DataArray(DEFAULT_FILE_DATA,
                          dims=('Rows', 'Columns'))
            file_content['CloudMaskBinary'].attrs['units'] = 1
            file_content['CloudMaskBinary'].attrs['_FillValue'] = -128
            file_content['CloudMaskBinary'].attrs['valid_range'] = [0, 1]
            file_content['CloudMaskBinary/shape'] = (768, 3200)

            file_content['CloudMaskQualFlag'] = \
                DataArray(DEFAULT_FILE_DATA,
                          dims=('Rows', 'Columns'))
            file_content['CloudMaskQualFlag'].attrs['units'] = 1
            file_content['CloudMaskQualFlag'].attrs['_FillValue'] = -128
            file_content['CloudMaskQualFlag'].attrs['valid_range'] = [0, 6]
            file_content['CloudMaskQualFlag/shape'] = (768, 3200)

            file_content['CloudProbability'] = \
                DataArray(DEFAULT_FILE_DATA,
                          dims=('Rows', 'Columns'))
            file_content['CloudProbability'].attrs['units'] = 1
            file_content['CloudProbability'].attrs['_FillValue'] = -999.9
            file_content['CloudProbability'].attrs['valid_range'] = [0, 1]
            file_content['CloudProbability/shape'] = (768, 3200)

            file_content['Dust_Mask'] = DataArray(DEFAULT_FILE_DATA,
                                                  dims=('Rows', 'Columns'))
            file_content['Dust_Mask'].attrs['units'] = 1
            file_content['Dust_Mask'].attrs['_FillValue'] = -128
            file_content['Dust_Mask'].attrs['valid_range'] = [0, 3]
            file_content['Dust_Mask/shape'] = (768, 3200)

            file_content['Fire_Mask'] = DataArray(DEFAULT_FILE_DATA,
                                                  dims=('Rows', 'Columns'))
            file_content['Fire_Mask'].attrs['units'] = 1
            file_content['Fire_Mask'].attrs['_FillValue'] = -128
            file_content['Fire_Mask'].attrs['valid_range'] = [0, 3]
            file_content['Fire_Mask/shape'] = (768, 3200)

            file_content['Smoke_Mask'] = DataArray(DEFAULT_FILE_DATA,
                                                   dims=('Rows', 'Columns'))
            file_content['Smoke_Mask'].attrs['units'] = 1
            file_content['Smoke_Mask'].attrs['_FillValue'] = -128
            file_content['Smoke_Mask'].attrs['valid_range'] = [0, 3]
            file_content['Smoke_Mask/shape'] = (768, 3200)
        else:
            assert False

        return file_content


class TestVIIRSEDRGRANReader(unittest.TestCase):
    """ Test VIIRS EDR GRAN reader """
    yaml_file = 'viirs_edr_gran.yaml'

    def setUp(self):
        """ Wrap the netCDF4 file handler with the fake handler """
        from satpy.config import config_search_paths
        from satpy.readers.viirs_edr_gran import VIIRSEDRGRANFileHandler
        self.reader_configs = config_search_paths(
            os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(
            VIIRSEDRGRANFileHandler,
            '__bases__', (FakeNetCDF4FileHandlerGRAN,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """ Unwrap the netCDF4 file handler """
        self.p.stop()

    def test_init(self):
        """ Test basic initialization of the reader """
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'JRR-CloudMask_v2r0_npp_s201903060621591'
            '_e201903060623233_c201903060804490.nc',
            ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_example_dataset(self):
        """ Test loading a dataset """
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'JRR-CloudMask_v2r0_npp_s201903060621591'
            '_e201903060623233_c201903060804490.nc',
        ])
        r.create_filehandlers(loadables)
        example_dataset = r.load(['latitude'])
        self.assertEqual(len(example_dataset), 1)


def suite():
    """ The test suite for test_viirs_gran """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSEDRGRANReader))

    return mysuite
