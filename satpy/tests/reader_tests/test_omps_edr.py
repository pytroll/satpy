#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.omps_edr module.
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
        file_content = {}
        if 'SO2NRT' in filename:
            k = 'HDFEOS/SWATHS/OMPS Column Amount SO2/Data Fields/ColumnAmountSO2_TRM'
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/ScaleFactor'] = 1.1
            file_content[k + '/attr/Offset'] = 0.1
            file_content[k + '/attr/MissingValue'] = -1
            file_content[k + '/attr/Title'] = 'Vertical Column Amount SO2 (TRM)'
            file_content[k + '/attr/Units'] = 'D.U.'
            file_content[k + '/attr/ValidRange'] = (-10, 2000)
            k = 'HDFEOS/SWATHS/OMPS Column Amount SO2/Geolocation Fields/Longitude'
            file_content[k] = DEFAULT_LON_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/ScaleFactor'] = 1.1
            file_content[k + '/attr/Offset'] = 0.1
            file_content[k + '/attr/Units'] = 'deg'
            file_content[k + '/attr/MissingValue'] = -1
            file_content[k + '/attr/Title'] = 'Geodetic Longitude'
            file_content[k + '/attr/ValidRange'] = (-180, 180)
            k = 'HDFEOS/SWATHS/OMPS Column Amount SO2/Geolocation Fields/Latitude'
            file_content[k] = DEFAULT_LAT_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/ScaleFactor'] = 1.1
            file_content[k + '/attr/Offset'] = 0.1
            file_content[k + '/attr/Units'] = 'deg'
            file_content[k + '/attr/MissingValue'] = -1
            file_content[k + '/attr/Title'] = 'Geodetic Latitude'
            file_content[k + '/attr/ValidRange'] = (-90, 90)
        else:
            for k in ['Reflectivity331', 'UVAerosolIndex']:
                k = 'SCIENCE_DATA/' + k
                file_content[k] = DEFAULT_FILE_DATA
                file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
                file_content[k + '/attr/Units'] = 'Unitless'
                if k == 'UVAerosolIndex':
                    file_content[k + '/attr/ValidRange'] = (-30, 30)
                    file_content[k + '/attr/Title'] = 'UV Aerosol Index'
                else:
                    file_content[k + '/attr/ValidRange'] = (-0.15, 1.15)
                    file_content[k + '/attr/Title'] = 'Effective Surface Reflectivity at 331 nm'
                file_content[k + '/attr/_FillValue'] = -1.
            file_content['GEOLOCATION_DATA/Longitude'] = DEFAULT_LON_DATA
            file_content['GEOLOCATION_DATA/Longitude/shape'] = DEFAULT_FILE_SHAPE
            file_content['GEOLOCATION_DATA/Longitude/attr/ValidRange'] = (-180, 180)
            file_content['GEOLOCATION_DATA/Longitude/attr/_FillValue'] = -999.
            file_content['GEOLOCATION_DATA/Longitude/attr/Title'] = 'Geodetic Longitude'
            file_content['GEOLOCATION_DATA/Longitude/attr/Units'] = 'deg'
            file_content['GEOLOCATION_DATA/Latitude'] = DEFAULT_LAT_DATA
            file_content['GEOLOCATION_DATA/Latitude/shape'] = DEFAULT_FILE_SHAPE
            file_content['GEOLOCATION_DATA/Latitude/attr/ValidRange'] = (-90, 90)
            file_content['GEOLOCATION_DATA/Latitude/attr/_FillValue'] = -999.
            file_content['GEOLOCATION_DATA/Latitude/attr/Title'] = 'Geodetic Latitude'
            file_content['GEOLOCATION_DATA/Latitude/attr/Units'] = 'deg'

        # convert to xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                if val.ndim > 1:
                    file_content[key] = DataArray(val, dims=('y', 'x'))
                else:
                    file_content[key] = DataArray(val)
        return file_content


class TestOMPSEDRReader(unittest.TestCase):
    """Test OMPS EDR Reader"""
    yaml_file = "omps_edr.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.omps_edr import EDRFileHandler, EDREOSFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(EDRFileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True
        self.p2 = mock.patch.object(EDREOSFileHandler, '__bases__', (EDRFileHandler,))
        self.fake_handler2 = self.p2.start()
        self.p2.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler"""
        self.p2.stop()
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OMPS-NPP-TC_EDR_SO2NRT-2016m0607t192031-o00001-2016m0607t192947.he5',
            'OMPS-NPP-TC_EDR_TO3-v1.0-2016m0607t192031-o00001-2016m0607t192947.h5',
        ])
        self.assertTrue(len(loadables), 2)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_basic_load_so2(self):
        """Test basic load of so2 datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OMPS-NPP-TC_EDR_SO2NRT-2016m0607t192031-o00001-2016m0607t192947.he5',
            'OMPS-NPP-TC_EDR_TO3-v1.0-2016m0607t192031-o00001-2016m0607t192947.h5',
        ])
        self.assertTrue(len(loadables), 2)
        r.create_filehandlers(loadables)
        ds = r.load(['so2_trm'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.attrs['resolution'], 50000)
            self.assertTupleEqual(d.shape, DEFAULT_FILE_SHAPE)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_basic_load_to3(self):
        """Test basic load of to3 datasets"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OMPS-NPP-TC_EDR_SO2NRT-2016m0607t192031-o00001-2016m0607t192947.he5',
            'OMPS-NPP-TC_EDR_TO3-v1.0-2016m0607t192031-o00001-2016m0607t192947.h5',
        ])
        self.assertTrue(len(loadables), 2)
        r.create_filehandlers(loadables)
        ds = r.load(['reflectivity_331', 'uvaerosol_index'])
        self.assertEqual(len(ds), 2)
        for d in ds.values():
            self.assertEqual(d.attrs['resolution'], 50000)
            self.assertTupleEqual(d.shape, DEFAULT_FILE_SHAPE)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])


def suite():
    """The test suite for test_omps_edr.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestOMPSEDRReader))

    return mysuite
