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
"""Module for testing the satpy.readers.omps_edr module."""

import os
import unittest
from unittest import mock

import numpy as np

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler
from satpy.tests.utils import convert_file_content_to_data_array

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
    """Swap-in HDF5 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        file_content = {}
        attrs = []
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
        elif 'NMSO2' in filename:
            file_content['GEOLOCATION_DATA/Longitude'] = DEFAULT_LON_DATA
            file_content['GEOLOCATION_DATA/Longitude/shape'] = DEFAULT_FILE_SHAPE
            file_content['GEOLOCATION_DATA/Longitude/attr/valid_max'] = 180
            file_content['GEOLOCATION_DATA/Longitude/attr/valid_min'] = -180
            file_content['GEOLOCATION_DATA/Longitude/attr/_FillValue'] = -1.26765e+30
            file_content['GEOLOCATION_DATA/Longitude/attr/long_name'] = 'Longitude'
            file_content['GEOLOCATION_DATA/Longitude/attr/standard_name'] = 'longitude'
            file_content['GEOLOCATION_DATA/Longitude/attr/units'] = 'degrees_east'
            file_content['GEOLOCATION_DATA/Latitude'] = DEFAULT_LAT_DATA
            file_content['GEOLOCATION_DATA/Latitude/shape'] = DEFAULT_FILE_SHAPE
            file_content['GEOLOCATION_DATA/Latitude/attr/valid_max'] = 90
            file_content['GEOLOCATION_DATA/Latitude/attr/valid_min'] = -90
            file_content['GEOLOCATION_DATA/Latitude/attr/_FillValue'] = -1.26765e+30
            file_content['GEOLOCATION_DATA/Latitude/attr/long_name'] = 'Latitude'
            file_content['GEOLOCATION_DATA/Latitude/attr/standard_name'] = 'latitude'
            file_content['GEOLOCATION_DATA/Latitude/attr/units'] = 'degress_north'

            k = 'SCIENCE_DATA/ColumnAmountSO2_TRM'
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/_FillValue'] = -1.26765e+30
            file_content[k + '/attr/long_name'] = 'Column Amount SO2 (TRM)'
            file_content[k + '/attr/units'] = 'DU'
            file_content[k + '/attr/valid_max'] = 2000
            file_content[k + '/attr/valid_min'] = -10

            k = 'SCIENCE_DATA/ColumnAmountSO2_STL'
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/_FillValue'] = -1.26765e+30
            file_content[k + '/attr/long_name'] = 'Column Amount SO2 (STL)'
            file_content[k + '/attr/units'] = 'DU'

            k = 'SCIENCE_DATA/ColumnAmountSO2_TRL'
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/_FillValue'] = -1.26765e+30
            file_content[k + '/attr/long_name'] = 'Column Amount SO2 (TRL)'
            file_content[k + '/attr/units'] = 'DU'
            file_content[k + '/attr/valid_max'] = 2000
            file_content[k + '/attr/valid_min'] = -10
            file_content[k + '/attr/DIMENSION_LIST'] = [10, 10]
            attrs = ['_FillValue', 'long_name', 'units', 'valid_max', 'valid_min', 'DIMENSION_LIST']

            k = 'SCIENCE_DATA/ColumnAmountSO2_TRU'
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/long_name'] = 'Column Amount SO2 (TRU)'
            file_content[k + '/attr/units'] = 'DU'
            file_content[k + '/attr/valid_max'] = 2000
            file_content[k + '/attr/valid_min'] = -10

            # Dataset with out unit
            k = 'SCIENCE_DATA/ColumnAmountSO2_PBL'
            file_content[k] = DEFAULT_FILE_DATA
            file_content[k + '/shape'] = DEFAULT_FILE_SHAPE
            file_content[k + '/attr/_FillValue'] = -1.26765e+30
            file_content[k + '/attr/long_name'] = 'Column Amount SO2 (PBL)'
            file_content[k + '/attr/valid_max'] = 2000
            file_content[k + '/attr/valid_min'] = -10
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

        convert_file_content_to_data_array(file_content, attrs)
        return file_content


class TestOMPSEDRReader(unittest.TestCase):
    """Test OMPS EDR Reader."""

    yaml_file = "omps_edr.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.omps_edr import EDREOSFileHandler, EDRFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(EDRFileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True
        self.p2 = mock.patch.object(EDREOSFileHandler, '__bases__', (EDRFileHandler,))
        self.fake_handler2 = self.p2.start()
        self.p2.is_local = True

    def tearDown(self):
        """Stop wrapping the NetCDF4 file handler."""
        self.p2.stop()
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OMPS-NPP-TC_EDR_SO2NRT-2016m0607t192031-o00001-2016m0607t192947.he5',
            'OMPS-NPP-TC_EDR_TO3-v1.0-2016m0607t192031-o00001-2016m0607t192947.h5',
            'OMPS-NPP_NMSO2-PCA-L2_v1.1_2018m1129t112824_o00001_2018m1129t114426.h5',
        ])
        self.assertEqual(len(loadables), 3)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_basic_load_so2(self):
        """Test basic load of so2 datasets."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OMPS-NPP-TC_EDR_SO2NRT-2016m0607t192031-o00001-2016m0607t192947.he5',
            'OMPS-NPP-TC_EDR_TO3-v1.0-2016m0607t192031-o00001-2016m0607t192947.h5',
            'OMPS-NPP_NMSO2-PCA-L2_v1.1_2018m1129t112824_o00001_2018m1129t114426.h5',
        ])
        self.assertEqual(len(loadables), 3)
        r.create_filehandlers(loadables)
        ds = r.load(['so2_trm'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.attrs['resolution'], 50000)
            self.assertTupleEqual(d.shape, DEFAULT_FILE_SHAPE)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

        ds = r.load(['tcso2_trm_sampo'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.attrs['resolution'], 50000)
            self.assertTupleEqual(d.shape, DEFAULT_FILE_SHAPE)

        ds = r.load(['tcso2_stl_sampo'])
        self.assertEqual(len(ds), 0)

        # Dataset without _FillValue
        ds = r.load(['tcso2_tru_sampo'])
        self.assertEqual(len(ds), 1)

        # Dataset without unit
        ds = r.load(['tcso2_pbl_sampo'])
        self.assertEqual(len(ds), 0)

    def test_basic_load_to3(self):
        """Test basic load of to3 datasets."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OMPS-NPP-TC_EDR_SO2NRT-2016m0607t192031-o00001-2016m0607t192947.he5',
            'OMPS-NPP-TC_EDR_TO3-v1.0-2016m0607t192031-o00001-2016m0607t192947.h5',
            'OMPS-NPP_NMSO2-PCA-L2_v1.1_2018m1129t112824_o00001_2018m1129t114426.h5',
        ])
        self.assertEqual(len(loadables), 3)
        r.create_filehandlers(loadables)
        ds = r.load(['reflectivity_331', 'uvaerosol_index'])
        self.assertEqual(len(ds), 2)
        for d in ds.values():
            self.assertEqual(d.attrs['resolution'], 50000)
            self.assertTupleEqual(d.shape, DEFAULT_FILE_SHAPE)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    @mock.patch('satpy.readers.hdf5_utils.HDF5FileHandler._get_reference')
    @mock.patch('h5py.File')
    def test_load_so2_DIMENSION_LIST(self, mock_h5py_file, mock_hdf5_utils_get_reference):
        """Test load of so2 datasets with DIMENSION_LIST."""
        from satpy.readers import load_reader
        mock_h5py_file.return_value = mock.MagicMock()
        mock_hdf5_utils_get_reference.return_value = [[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]]
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'OMPS-NPP_NMSO2-PCA-L2_v1.1_2018m1129t112824_o00001_2018m1129t114426.h5',
        ])
        r.create_filehandlers(loadables)

        ds = r.load(['tcso2_trl_sampo'])
        self.assertEqual(len(ds), 1)
