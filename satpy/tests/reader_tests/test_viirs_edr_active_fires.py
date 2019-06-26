#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""VIIRS Active Fires Tests
*********************
This module implements tests for VIIRS Active Fires NetCDF and ASCII file
readers.
"""

import sys
import os
import numpy as np
import io
import dask.dataframe as dd
import pandas as pd
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.readers.file_handlers import BaseFileHandler
from satpy.tests.utils import convert_file_content_to_data_array

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


DEFAULT_FILE_SHAPE = (1, 100)

DEFAULT_LATLON_FILE_DTYPE = np.float32
DEFAULT_LATLON_FILE_DATA = np.arange(start=43, stop=45, step=0.02,
                                     dtype=DEFAULT_LATLON_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)

DEFAULT_DETECTION_FILE_DTYPE = np.ubyte
DEFAULT_DETECTION_FILE_DATA = np.arange(start=60, stop=100, step=0.4,
                                        dtype=DEFAULT_DETECTION_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)

DEFAULT_M13_FILE_DTYPE = np.float32
DEFAULT_M13_FILE_DATA = np.arange(start=300, stop=340, step=0.4,
                                  dtype=DEFAULT_M13_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)

DEFAULT_POWER_FILE_DTYPE = np.float32
DEFAULT_POWER_FILE_DATA = np.arange(start=1, stop=25, step=0.24,
                                    dtype=DEFAULT_POWER_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)


class FakeModFiresNetCDF4FileHandler(FakeNetCDF4FileHandler):
    """Swap in CDF4 file handler"""
    def get_test_content(self, filename, filename_info, filename_type):
        """Mimic reader input file content"""
        file_content = {}
        file_content['/attr/data_id'] = "AFMOD"
        file_content['satellite_name'] = "npp"
        file_content['sensor'] = 'VIIRS'

        file_content['Fire Pixels/FP_latitude'] = DEFAULT_LATLON_FILE_DATA
        file_content['Fire Pixels/FP_longitude'] = DEFAULT_LATLON_FILE_DATA
        file_content['Fire Pixels/FP_power'] = DEFAULT_POWER_FILE_DATA
        file_content['Fire Pixels/FP_T13'] = DEFAULT_M13_FILE_DATA
        file_content['Fire Pixels/FP_confidence'] = DEFAULT_DETECTION_FILE_DATA
        file_content['Fire Pixels/attr/units'] = 'none'
        file_content['Fire Pixels/shape'] = DEFAULT_FILE_SHAPE

        attrs = ('FP_latitude', 'FP_longitude',  'FP_T13', 'FP_confidence')
        convert_file_content_to_data_array(
            file_content, attrs=attrs,
            dims=('z', 'fakeDim0', 'fakeDim1'))
        return file_content


class FakeImgFiresNetCDF4FileHandler(FakeNetCDF4FileHandler):
    """Swap in CDF4 file handler"""
    def get_test_content(self, filename, filename_info, filename_type):
        """Mimic reader input file content"""
        file_content = {}
        file_content['/attr/data_id'] = "AFIMG"
        file_content['satellite_name'] = "npp"
        file_content['sensor'] = 'VIIRS'

        file_content['FP_latitude'] = DEFAULT_LATLON_FILE_DATA
        file_content['FP_longitude'] = DEFAULT_LATLON_FILE_DATA
        file_content['FP_power'] = DEFAULT_POWER_FILE_DATA
        file_content['FP_T4'] = DEFAULT_M13_FILE_DATA
        file_content['FP_confidence'] = DEFAULT_DETECTION_FILE_DATA

        attrs = ('FP_latitude', 'FP_longitude',  'FP_T13', 'FP_confidence')
        convert_file_content_to_data_array(
            file_content, attrs=attrs,
            dims=('z', 'fakeDim0', 'fakeDim1'))
        return file_content


class FakeModFiresTextFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Get fake file content from 'get_test_content'"""
        super(FakeModFiresTextFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.file_content = self.get_test_content()

        platform_key = {"NPP": "Suomi-NPP", "J01": "NOAA-20", "J02": "NOAA-21"}

        self.platform_name = platform_key.get(self.filename_info['satellite_name'].upper(), "unknown")

    def get_test_content(self):
        fake_file = io.StringIO(u'''\n\n\n\n\n\n\n\n\n\n\n\n\n\n
        24.64015007, -107.57017517,  317.38290405,   0.75,   0.75,   40,    4.28618050
        25.90660477, -100.06127167,  331.17962646,   0.75,   0.75,   81,   20.61096764''')

        return dd.from_pandas(pd.read_csv(fake_file, skiprows=15, header=None,
                                          names=["latitude", "longitude",
                                                 "T13", "Along-scan", "Along-track",
                                                 "confidence_pct",
                                                 "power"]), chunksize=1)


class FakeImgFiresTextFileHandler(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Get fake file content from 'get_test_content'"""
        super(FakeImgFiresTextFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.file_content = self.get_test_content()

    def get_test_content(self):
        fake_file = io.StringIO(u'''\n\n\n\n\n\n\n\n\n\n\n\n\n\n
        24.64015007, -107.57017517,  317.38290405,   0.75,   0.75,   40,    4.28618050
        25.90660477, -100.06127167,  331.17962646,   0.75,   0.75,   81,   20.61096764''')

        platform_key = {"NPP": "Suomi-NPP", "J01": "NOAA-20", "J02": "NOAA-21"}

        self.platform_name = platform_key.get(self.filename_info['satellite_name'].upper(), "unknown")

        return dd.from_pandas(pd.read_csv(fake_file, skiprows=15, header=None,
                                          names=["latitude", "longitude",
                                                 "T4", "Along-scan", "Along-track",
                                                 "confidence_cat",
                                                 "power"]), chunksize=1)


class TestModVIIRSActiveFiresNetCDF4(unittest.TestCase):
    """Test VIIRS Fires Reader"""
    yaml_file = 'viirs_edr_active_fires.yaml'

    def setUp(self):
        """Wrap CDF4 file handler with own fake file handler"""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_edr_active_fires import VIIRSActiveFiresFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(VIIRSActiveFiresFileHandler, '__bases__', (FakeModFiresNetCDF4FileHandler,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the CDF4 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFMOD_j02_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.nc'
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_dataset(self):
        """Test loading all datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFMOD_j02_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.nc'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['confidence_pct'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], '%')

        datasets = r.load(['T13'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')

        datasets = r.load(['power'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'MW')
            self.assertEqual(v.attrs['platform_name'], 'NOAA-21')
            self.assertEqual(v.attrs['sensor'], 'VIIRS')


class TestImgVIIRSActiveFiresNetCDF4(unittest.TestCase):
    """Test VIIRS Fires Reader"""
    yaml_file = 'viirs_edr_active_fires.yaml'

    def setUp(self):
        """Wrap CDF4 file handler with own fake file handler"""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_edr_active_fires import VIIRSActiveFiresFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(VIIRSActiveFiresFileHandler, '__bases__', (FakeImgFiresNetCDF4FileHandler,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the CDF4 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFIMG_npp_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.nc'
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_dataset(self):
        """Test loading all datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFIMG_npp_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.nc'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['confidence_cat'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], '1')
            self.assertEqual(v.attrs['flag_meanings'], ['low', 'medium', 'high'])
            self.assertEqual(v.attrs['flag_values'], [7, 8, 9])

        datasets = r.load(['T4'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')

        datasets = r.load(['power'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'MW')
            self.assertEqual(v.attrs['platform_name'], 'Suomi-NPP')
            self.assertEqual(v.attrs['sensor'], 'VIIRS')


@mock.patch('satpy.readers.viirs_edr_active_fires.dd.read_csv')
class TestModVIIRSActiveFiresText(unittest.TestCase):
    """Test VIIRS Fires Reader"""
    yaml_file = 'viirs_edr_active_fires.yaml'

    def setUp(self):
        """Wrap file handler with own fake file handler"""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_edr_active_fires import VIIRSActiveFiresTextFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(VIIRSActiveFiresTextFileHandler, '__bases__', (FakeModFiresTextFileHandler,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the text file handler"""
        self.p.stop()

    def test_init(self, mock_obj):
        """Test basic init with no extra parameters"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFEDR_j01_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.txt'
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_dataset(self, csv_mock):
        """Test loading all datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFEDR_j01_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.txt'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['confidence_pct'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], '%')

        datasets = r.load(['T13'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')

        datasets = r.load(['power'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'MW')
            self.assertEqual(v.attrs['platform_name'], 'NOAA-20')
            self.assertEqual(v.attrs['sensor'], 'VIIRS')


@mock.patch('satpy.readers.viirs_edr_active_fires.dd.read_csv')
class TestImgVIIRSActiveFiresText(unittest.TestCase):
    """Test VIIRS Fires Reader"""
    yaml_file = 'viirs_edr_active_fires.yaml'

    def setUp(self):
        """Wrap file handler with own fake file handler"""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_edr_active_fires import VIIRSActiveFiresTextFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(VIIRSActiveFiresTextFileHandler, '__bases__', (FakeImgFiresTextFileHandler,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the text file handler"""
        self.p.stop()

    def test_init(self, mock_obj):
        """Test basic init with no extra parameters"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFIMG_npp_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.txt'
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_dataset(self, mock_obj):
        """Test loading all datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'AFIMG_npp_d20180829_t2015451_e2017093_b35434_c20180829210527716708_cspp_dev.txt'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['confidence_cat'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], '1')
            self.assertEqual(v.attrs['flag_meanings'], ['low', 'medium', 'high'])
            self.assertEqual(v.attrs['flag_values'], [7, 8, 9])

        datasets = r.load(['T4'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')

        datasets = r.load(['power'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'MW')
            self.assertEqual(v.attrs['platform_name'], 'Suomi-NPP')
            self.assertEqual(v.attrs['sensor'], 'VIIRS')


def suite():
    """The test suite for testing viirs active fires
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestModVIIRSActiveFiresNetCDF4))
    mysuite.addTest(loader.loadTestsFromTestCase(TestModVIIRSActiveFiresText))
    mysuite.addTest(loader.loadTestsFromTestCase(TestImgVIIRSActiveFiresNetCDF4))
    mysuite.addTest(loader.loadTestsFromTestCase(TestImgVIIRSActiveFiresText))

    return mysuite
