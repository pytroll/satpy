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
"""Tests for the VIIRS EDR Flood reader."""
import os
import unittest
from unittest import mock
import numpy as np
from satpy.tests.reader_tests.test_hdf4_utils import FakeHDF4FileHandler


DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)


class FakeHDF4FileHandler2(FakeHDF4FileHandler):
    """Swap in HDF4 file handler."""

    def get_test_content(self, filename, filename_info, filename_type):
        """Mimic reader input file content."""
        file_content = {}
        file_content['/attr/Satellitename'] = filename_info['platform_shortname']
        file_content['/attr/SensorIdentifyCode'] = 'VIIRS'

        # only one dataset for the flood reader
        file_content['WaterDetection'] = DEFAULT_FILE_DATA
        file_content['WaterDetection/attr/_Fillvalue'] = 1
        file_content['WaterDetection/attr/scale_factor'] = 1.
        file_content['WaterDetection/attr/add_offset'] = 0.
        file_content['WaterDetection/attr/units'] = 'none'
        file_content['WaterDetection/shape'] = DEFAULT_FILE_SHAPE
        file_content['WaterDetection/attr/ProjectionMinLatitude'] = 15.
        file_content['WaterDetection/attr/ProjectionMaxLatitude'] = 68.
        file_content['WaterDetection/attr/ProjectionMinLongitude'] = -124.
        file_content['WaterDetection/attr/ProjectionMaxLongitude'] = -61.

        # convert tp xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                attrs = {}
                for a in ['_Fillvalue', 'units', 'ProjectionMinLatitude', 'ProjectionMaxLongitude',
                          'ProjectionMinLongitude', 'ProjectionMaxLatitude']:
                    if key + '/attr/' + a in file_content:
                        attrs[a] = file_content[key + '/attr/' + a]
                if val.ndim > 1:
                    file_content[key] = DataArray(val, dims=('fakeDim0', 'fakeDim1'), attrs=attrs)
                else:
                    file_content[key] = DataArray(val, attrs=attrs)

        if 'y' not in file_content['WaterDetection'].dims:
            file_content['WaterDetection'] = file_content['WaterDetection'].rename({'fakeDim0': 'x', 'fakeDim1': 'y'})
        return file_content


class TestVIIRSEDRFloodReader(unittest.TestCase):
    """Test VIIRS EDR Flood Reader."""

    yaml_file = 'viirs_edr_flood.yaml'

    def setUp(self):
        """Wrap HDF4 file handler with own fake file handler."""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_edr_flood import VIIRSEDRFlood
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(VIIRSEDRFlood, '__bases__', (FakeHDF4FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF4 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'WATER_VIIRS_Prj_SVI_npp_d20180824_t1828213_e1839433_b35361_cspp_dev_10_300_01.hdf'
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_dataset(self):
        """Test loading all datasets from a full swath file."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'WATER_VIIRS_Prj_SVI_npp_d20180824_t1828213_e1839433_b35361_cspp_dev_10_300_01.hdf'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['WaterDetection'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'none')

    def test_load_dataset_aoi(self):
        """Test loading all datasets from an area of interest file."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'WATER_VIIRS_Prj_SVI_npp_d20180824_t1828213_e1839433_b35361_cspp_dev_001_10_300_01.hdf'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['WaterDetection'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'none')
