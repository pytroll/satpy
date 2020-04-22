#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Unit tests for AMSR L2 reader"""

import os
import unittest
import numpy as np
from unittest import mock
from satpy.tests.utils import convert_file_content_to_data_array
from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler


DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 30)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        file_content = {
            '/attr/PlatformShortName': 'GCOM-W1',
            '/attr/SensorShortName': 'AMSR2',
            '/attr/StartOrbitNumber': '22210',
            '/attr/StopOrbitNumber': '22210',
        }
        k = 'Geophysical Data'
        file_content[k] = DEFAULT_FILE_DATA[:, :]
        file_content[k + '/shape'] = (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])
        file_content[k + '/attr/UNIT'] = 'K'
        file_content[k + '/attr/SCALE FACTOR'] = 1

        k = 'Latitude of Observation Point'
        file_content[k] = DEFAULT_FILE_DATA[:, :]
        file_content[k + '/shape'] = (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])
        file_content[k + '/attr/UNIT'] = 'deg'
        file_content[k + '/attr/SCALE FACTOR'] = 1
        k = 'Longitude of Observation Point'
        file_content[k] = DEFAULT_FILE_DATA[:, :]
        file_content[k + '/shape'] = (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1])
        file_content[k + '/attr/UNIT'] = 'deg'
        file_content[k + '/attr/SCALE FACTOR'] = 1

        convert_file_content_to_data_array(file_content, dims=('dim_0', 'dim_1'))
        return file_content


class TestAMSR2L2Reader(unittest.TestCase):
    """Test AMSR2 L2 Reader"""
    yaml_file = "amsr2_l2.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.amsr2_l2 import AMSR2L2FileHandler
        from satpy.readers.amsr2_l1b import AMSR2L1BFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(AMSR2L2FileHandler, '__bases__', (FakeHDF5FileHandler2,
                                                                     AMSR2L1BFileHandler))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GW1AM2_202004160129_195B_L2SNSSWLB3300300.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_basic(self):
        """Test loading of basic channels"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GW1AM2_202004160129_195B_L2SNSSWLB3300300.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        ds = r.load(['ssw'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertTupleEqual(d.shape, (DEFAULT_FILE_SHAPE[0], int(DEFAULT_FILE_SHAPE[1])))
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertTupleEqual(d.attrs['area'].lons.shape,
                                  (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1]))
            self.assertTupleEqual(d.attrs['area'].lats.shape,
                                  (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1]))
