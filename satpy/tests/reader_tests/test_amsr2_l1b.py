#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.amsr2_l1b module.
"""

import os
import sys
import numpy as np
from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler

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
        for bt_chan in [
            '(10.7GHz,H)',
            '(10.7GHz,V)',
            '(18.7GHz,H)',
            '(18.7GHz,V)',
            '(23.8GHz,H)',
            '(23.8GHz,V)',
            '(36.5GHz,H)',
            '(36.5GHz,V)',
            '(6.9GHz,H)',
            '(6.9GHz,V)',
            '(7.3GHz,H)',
            '(7.3GHz,V)',
            '(89.0GHz-A,H)',
            '(89.0GHz-A,V)',
            '(89.0GHz-B,H)',
            '(89.0GHz-B,V)',
        ]:
            k = 'Brightness Temperature {}'.format(bt_chan)
            file_content[k] = DEFAULT_FILE_DATA[:, ::2]
            file_content[k + '/shape'] = (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1] // 2)
            file_content[k + '/attr/UNIT'] = 'K'
            file_content[k + '/attr/SCALE FACTOR'] = 0.01
        for bt_chan in [
            '(89.0GHz-A,H)',
            '(89.0GHz-A,V)',
            '(89.0GHz-B,H)',
            '(89.0GHz-B,V)',
        ]:
            k = 'Brightness Temperature {}'.format(bt_chan)
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/UNIT'] = 'K'
            file_content[k + '/attr/SCALE FACTOR'] = 0.01
        for nav_chan in ['89A', '89B']:
            lon_k = 'Longitude of Observation Point for ' + nav_chan
            lat_k = 'Latitude of Observation Point for ' + nav_chan
            file_content[lon_k] = DEFAULT_LON_DATA
            file_content[lon_k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[lon_k + '/attr/SCALE FACTOR'] = 1
            file_content[lon_k + '/attr/UNIT'] = 'deg'
            file_content[lat_k] = DEFAULT_LAT_DATA
            file_content[lat_k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[lat_k + '/attr/SCALE FACTOR'] = 1
            file_content[lat_k + '/attr/UNIT'] = 'deg'

        # convert to xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                if val.ndim > 1:
                    file_content[key] = DataArray(val, dims=('y', 'x'))
                else:
                    file_content[key] = DataArray(val)

        return file_content


class TestAMSR2L1BReader(unittest.TestCase):
    """Test AMSR2 L1B Reader"""
    yaml_file = "amsr2_l1b.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.amsr2_l1b import AMSR2L1BFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(AMSR2L1BFileHandler, '__bases__', (FakeHDF5FileHandler2,))
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
            'GW1AM2_201607201808_128A_L1DLBTBR_1110110.h5',
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
            'GW1AM2_201607201808_128A_L1DLBTBR_1110110.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        ds = r.load([
            'btemp_10.7v',
            'btemp_10.7h',
            'btemp_6.9v',
            'btemp_6.9h',
            'btemp_7.3v',
            'btemp_7.3h',
            'btemp_18.7v',
            'btemp_18.7h',
            'btemp_23.8v',
            'btemp_23.8h',
            'btemp_36.5v',
            'btemp_36.5h',
        ])
        self.assertEqual(len(ds), 12)
        for d in ds.values():
            self.assertEqual(d.attrs['calibration'], 'brightness_temperature')
            self.assertTupleEqual(d.shape, (DEFAULT_FILE_SHAPE[0], int(DEFAULT_FILE_SHAPE[1] // 2)))
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertTupleEqual(d.attrs['area'].lons.shape,
                                  (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1] // 2))
            self.assertTupleEqual(d.attrs['area'].lats.shape,
                                  (DEFAULT_FILE_SHAPE[0], DEFAULT_FILE_SHAPE[1] // 2))

    def test_load_89ghz(self):
        """Test loading of 89GHz channels"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GW1AM2_201607201808_128A_L1DLBTBR_1110110.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        ds = r.load([
            'btemp_89.0av',
            'btemp_89.0ah',
            'btemp_89.0bv',
            'btemp_89.0bh',
        ])
        self.assertEqual(len(ds), 4)
        for d in ds.values():
            self.assertEqual(d.attrs['calibration'], 'brightness_temperature')
            self.assertTupleEqual(d.shape, DEFAULT_FILE_SHAPE)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertTupleEqual(d.attrs['area'].lons.shape,
                                  DEFAULT_FILE_SHAPE)
            self.assertTupleEqual(d.attrs['area'].lats.shape,
                                  DEFAULT_FILE_SHAPE)


def suite():
    """The test suite for test_amsr2_l1b.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestAMSR2L1BReader))

    return mysuite
