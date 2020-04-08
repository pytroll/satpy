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
"""The agri_l1 reader tests package."""

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler
import numpy as np
import dask.array as da
import xarray as xr
import os
import unittest
from unittest import mock


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def make_test_data(self, cwl, ch, prefix, dims, file_type):
        if prefix == 'CAL':
            data = xr.DataArray(
                                da.from_array((np.arange(10.) + 1.) / 10., [dims[0] * dims[1]]),
                                attrs={
                                    'Slope': 1., 'Intercept': 0.,
                                    'FillValue': -65535.0,
                                    'units': 'NUL',
                                    'center_wavelength': '{}um'.format(cwl).encode('utf-8'),
                                    'band_names': 'band{}(band number is range from 1 to 14)'
                                                  .format(ch).encode('utf-8'),
                                    'long_name': 'Calibration table of {}um Channel'.format(cwl).encode('utf-8'),
                                    'valid_range': [0, 1.5],
                                },
                                dims=('_const'))

        elif prefix == 'NOM':
            data = xr.DataArray(
                                da.from_array(np.arange(10, dtype=np.uint16).reshape((2, 5)) + 1,
                                              [dim for dim in dims]),
                                attrs={
                                    'Slope': 1., 'Intercept': 0.,
                                    'FillValue': 65535,
                                    'units': 'DN',
                                    'center_wavelength': '{}um'.format(cwl).encode('utf-8'),
                                    'band_names': 'band{}(band number is range from 1 to 14)'
                                                  .format(ch).encode('utf-8'),
                                    'long_name': 'Calibration table of {}um Channel'.format(cwl).encode('utf-8'),
                                    'valid_range': [0, 4095],
                                },
                                dims=('_RegLength', '_RegWidth'))

        elif prefix == 'COEF':
            if file_type == '500':
                data = xr.DataArray(
                                da.from_array((np.arange(2.).reshape((1, 2)) + 1.) / np.array([1E4, 1E2]), [1, 2]),
                                attrs={
                                    'Slope': 1., 'Intercept': 0.,
                                    'FillValue': 0,
                                    'units': 'NUL',
                                    'band_names': 'NUL'.format(ch).encode('utf-8'),
                                    'long_name': b'Calibration coefficient (SCALE and OFFSET)',
                                    'valid_range': [-500, 500],
                                },
                                dims=('_num_channel', '_coefs'))

            elif file_type == '1000':
                data = xr.DataArray(
                                da.from_array((np.arange(6.).reshape((3, 2)) + 1.) / np.array([1E4, 1E2]), [3, 2]),
                                attrs={
                                    'Slope': 1., 'Intercept': 0.,
                                    'FillValue': 0,
                                    'units': 'NUL',
                                    'band_names': 'NUL'.format(ch).encode('utf-8'),
                                    'long_name': b'Calibration coefficient (SCALE and OFFSET)',
                                    'valid_range': [-500, 500],
                                },
                                dims=('_num_channel', '_coefs'))

            elif file_type == '2000':
                data = xr.DataArray(
                                da.from_array((np.arange(14.).reshape((7, 2)) + 1.) / np.array([1E4, 1E2]), [7, 2]),
                                attrs={
                                    'Slope': 1., 'Intercept': 0.,
                                    'FillValue': 0,
                                    'units': 'NUL',
                                    'band_names': 'NUL'.format(ch).encode('utf-8'),
                                    'long_name': b'Calibration coefficient (SCALE and OFFSET)',
                                    'valid_range': [-500, 500],
                                },
                                dims=('_num_channel', '_coefs'))

            elif file_type == '4000':
                data = xr.DataArray(
                                da.from_array((np.arange(28.).reshape((14, 2)) + 1.)
                                              / np.array([1E4, 1E2]), [14, 2]),
                                attrs={
                                    'Slope': 1., 'Intercept': 0.,
                                    'FillValue': 0,
                                    'units': 'NUL',
                                    'band_names': 'NUL'.format(ch).encode('utf-8'),
                                    'long_name': b'Calibration coefficient (SCALE and OFFSET)',
                                    'valid_range': [-500, 500],
                                },
                                dims=('_num_channel', '_coefs'))

        return data

    def _get_500m_data(self, file_type):
        dim_0 = 2
        dim_1 = 5
        chs = [2]
        cwls = [0.65]
        data = {}
        for index, cwl in enumerate(cwls):
            data['CALChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'CAL',
                                                                           [dim_0, dim_1], file_type)
            data['NOMChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'NOM',
                                                                           [dim_0, dim_1], file_type)
            data['CALIBRATION_COEF(SCALE+OFFSET)'] = self.make_test_data(cwls[index], chs[index], 'COEF',
                                                                         [dim_0, dim_1], file_type)

        return data

    def _get_1km_data(self, file_type):
        dim_0 = 2
        dim_1 = 5
        chs = np.linspace(1, 3, 3)
        cwls = [0.47, 0.65, 0.83]
        data = {}
        for index, cwl in enumerate(cwls):
            data['CALChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'CAL',
                                                                           [dim_0, dim_1], file_type)
            data['NOMChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'NOM',
                                                                           [dim_0, dim_1], file_type)
            data['CALIBRATION_COEF(SCALE+OFFSET)'] = self.make_test_data(cwls[index], chs[index], 'COEF',
                                                                         [dim_0, dim_1], file_type)

        return data

    def _get_2km_data(self, file_type):
        dim_0 = 2
        dim_1 = 5
        chs = np.linspace(1, 7, 7)
        cwls = [0.47, 0.65, 0.83, 1.37, 1.61, 2.22, 3.72]
        data = {}
        for index, cwl in enumerate(cwls):
            data['CALChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'CAL',
                                                                           [dim_0, dim_1], file_type)
            data['NOMChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'NOM',
                                                                           [dim_0, dim_1], file_type)
            data['CALIBRATION_COEF(SCALE+OFFSET)'] = self.make_test_data(cwls[index], chs[index], 'COEF',
                                                                         [dim_0, dim_1], file_type)

        return data

    def _get_4km_data(self, file_type):
        dim_0 = 2
        dim_1 = 5
        chs = np.linspace(1, 14, 14)
        cwls = [0.47, 0.65, 0.83, 1.37, 1.61, 2.22, 3.72, 3.72, 6.25, 7.10, 8.50, 10.8, 12, 13.5]
        data = {}
        for index, cwl in enumerate(cwls):
            data['CALChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'CAL',
                                                                           [dim_0, dim_1], file_type)
            data['NOMChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'NOM',
                                                                           [dim_0, dim_1], file_type)
            data['CALIBRATION_COEF(SCALE+OFFSET)'] = self.make_test_data(cwls[index], chs[index], 'COEF',
                                                                         [dim_0, dim_1], file_type)

        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        global_attrs = {
            '/attr/NOMCenterLat': 0.0, '/attr/NOMCenterLon': 104.7, '/attr/NOMSatHeight': 3.5786E7,
            '/attr/dEA': 6378.14, '/attr/dObRecFlat': 298.257223563,
            '/attr/OBIType': 'REGC', '/attr/RegLength': 2.0, '/attr/RegWidth': 5.0,
            '/attr/Begin Line Number': 0, '/attr/End Line Number': 1,
            '/attr/Observing Beginning Date': '2019-06-03', '/attr/Observing Beginning Time': '00:30:01.807',
            '/attr/Observing Ending Date': '2019-06-03', '/attr/Observing Ending Time': '00:34:07.572',
            '/attr/Satellite Name': 'FY4A', '/attr/Sensor Identification Code': 'AGRI', '/attr/Sensor Name': 'AGRI',
        }

        data = {}
        if self.filetype_info['file_type'] == 'agri_l1_0500m':
            data = self._get_500m_data('500')
        elif self.filetype_info['file_type'] == 'agri_l1_1000m':
            data = self._get_1km_data('1000')
        elif self.filetype_info['file_type'] == 'agri_l1_2000m':
            data = self._get_2km_data('2000')
        elif self.filetype_info['file_type'] == 'agri_l1_4000m':
            data = self._get_4km_data('4000')

        test_content = {}
        test_content.update(global_attrs)
        test_content.update(data)

        return test_content


class Test_HDF_AGRI_L1_cal(unittest.TestCase):
    """Test VIRR L1B Reader."""
    yaml_file = "agri_l1.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.readers.agri_l1 import HDF_AGRI_L1
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(HDF_AGRI_L1, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_fy4a_all_resolutions(self):
        """Test loading data when all resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_0500M_V0001.HDF',
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_1000M_V0001.HDF',
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_2000M_V0001.HDF',
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_4000M_V0001.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(4, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        available_datasets = reader.available_dataset_ids

        # 500m
        band_names = ['C' + '%02d' % ch for ch in np.linspace(2, 2, 1)]
        for band_name in band_names:
            ds_id = DatasetID(name=band_name, resolution=500)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(2, len(res))

        # 1km
        band_names = ['C' + '%02d' % ch for ch in np.linspace(1, 3, 3)]
        for band_name in band_names:
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(2, len(res))

        # 2km
        band_names = ['C' + '%02d' % ch for ch in np.linspace(1, 7, 7)]
        for band_name in band_names:
            ds_id = DatasetID(name=band_name, resolution=2000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            if band_name < 'C07':
                self.assertEqual(2, len(res))
            else:
                self.assertEqual(3, len(res))

        band_names = ['C' + '%02d' % ch for ch in np.linspace(1, 14, 14)]
        res = reader.load(band_names)
        self.assertEqual(14, len(res))

        for band_name in band_names:
            self.assertEqual((2, 5), res[band_name].shape)
            if band_name < 'C07':
                self.assertEqual('reflectance', res[band_name].attrs['calibration'])
            else:
                self.assertEqual('brightness_temperature', res[band_name].attrs['calibration'])
            if band_name < 'C07':
                self.assertEqual('%', res[band_name].attrs['units'])
            else:
                self.assertEqual('K', res[band_name].attrs['units'])

    def test_fy4a_counts_calib(self):
        """Test loading data at counts calibration."""
        from satpy import DatasetID
        from satpy.readers import load_reader
        filenames = [
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_0500M_V0001.HDF',
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_1000M_V0001.HDF',
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_2000M_V0001.HDF',
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_4000M_V0001.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(4, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        ds_ids = []
        band_names = ['C' + '%02d' % ch for ch in np.linspace(1, 14, 14)]
        for band_name in band_names:
            ds_ids.append(DatasetID(name=band_name, calibration='counts'))
        res = reader.load(ds_ids)
        self.assertEqual(14, len(res))

        for band_name in band_names:
            self.assertEqual((2, 5), res[band_name].shape)
            self.assertEqual('counts', res[band_name].attrs['calibration'])
            self.assertEqual(res[band_name].dtype, np.uint16)
            self.assertEqual('1', res[band_name].attrs['units'])

    def test_fy4a_4km_resolutions(self):
        """Test loading data when only 4km resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_4000M_V0001.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        # Verify that the resolution is only 4km
        available_datasets = reader.available_dataset_ids
        band_names = ['C' + '%02d' % ch for ch in np.linspace(1, 14, 14)]

        for band_name in band_names:
            ds_id = DatasetID(name=band_name, resolution=500)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=2000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=4000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            if band_name < 'C07':
                self.assertEqual(2, len(res))
            else:
                self.assertEqual(3, len(res))

        res = reader.load(band_names)
        self.assertEqual(14, len(res))
        expected = {
                    1: np.array([[2.01, 2.02, 2.03, 2.04, 2.05], [2.06, 2.07, 2.08, 2.09, 2.1]]),
                    2: np.array([[4.03, 4.06, 4.09, 4.12, 4.15], [4.18, 4.21, 4.24, 4.27, 4.3]]),
                    3: np.array([[6.05, 6.1, 6.15, 6.2, 6.25], [6.3, 6.35, 6.4, 6.45, 6.5]]),
                    4: np.array([[8.07, 8.14, 8.21, 8.28, 8.35], [8.42, 8.49, 8.56, 8.63, 8.7]]),
                    5: np.array([[10.09, 10.18, 10.27, 10.36, 10.45], [10.54, 10.63, 10.72, 10.81, 10.9]]),
                    6: np.array([[12.11, 12.22, 12.33, 12.44, 12.55], [12.66, 12.77, 12.88, 12.99, 13.1]])
                    }
        for i in range(7, 15):
            expected[i] = np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]])

        for index, band_name in enumerate(band_names):
            self.assertEqual((2, 5), res[band_name].shape)
            if band_name < 'C07':
                self.assertEqual('reflectance', res[band_name].attrs['calibration'])
            else:
                self.assertEqual('brightness_temperature', res[band_name].attrs['calibration'])
            if band_name < 'C07':
                self.assertEqual('%', res[band_name].attrs['units'])
            else:
                self.assertEqual('K', res[band_name].attrs['units'])
            self.assertTrue(np.allclose(res[band_name].values, expected[index + 1], equal_nan=True))

    def test_fy4a_2km_resolutions(self):
        """Test loading data when only 2km resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_2000M_V0001.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        # Verify that the resolution is only 2km
        available_datasets = reader.available_dataset_ids
        band_names = ['C' + '%02d' % ch for ch in np.linspace(1, 7, 7)]

        for band_name in band_names:
            ds_id = DatasetID(name=band_name, resolution=500)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=2000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            if band_name < 'C07':
                self.assertEqual(2, len(res))
            else:
                self.assertEqual(3, len(res))
            ds_id = DatasetID(name=band_name, resolution=4000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))

        res = reader.load(band_names)
        self.assertEqual(7, len(res))
        expected = {
                    1: np.array([[2.01, 2.02, 2.03, 2.04, 2.05], [2.06, 2.07, 2.08, 2.09, 2.1]]),
                    2: np.array([[4.03, 4.06, 4.09, 4.12, 4.15], [4.18, 4.21, 4.24, 4.27, 4.3]]),
                    3: np.array([[6.05, 6.1, 6.15, 6.2, 6.25], [6.3, 6.35, 6.4, 6.45, 6.5]]),
                    4: np.array([[8.07, 8.14, 8.21, 8.28, 8.35], [8.42, 8.49, 8.56, 8.63, 8.7]]),
                    5: np.array([[10.09, 10.18, 10.27, 10.36, 10.45], [10.54, 10.63, 10.72, 10.81, 10.9]]),
                    6: np.array([[12.11, 12.22, 12.33, 12.44, 12.55], [12.66, 12.77, 12.88, 12.99, 13.1]]),
                    7: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]])
                    }

        for index, band_name in enumerate(band_names):
            self.assertEqual((2, 5), res[band_name].shape)
            if band_name < 'C07':
                self.assertEqual('reflectance', res[band_name].attrs['calibration'])
            else:
                self.assertEqual('brightness_temperature', res[band_name].attrs['calibration'])
            if band_name < 'C07':
                self.assertEqual('%', res[band_name].attrs['units'])
            else:
                self.assertEqual('K', res[band_name].attrs['units'])
            self.assertTrue(np.allclose(res[band_name].values, expected[index + 1], equal_nan=True))

    def test_fy4a_1km_resolutions(self):
        """Test loading data when only 1km resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_1000M_V0001.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        # Verify that the resolution is only 1km
        available_datasets = reader.available_dataset_ids
        band_names = ['C' + '%02d' % ch for ch in np.linspace(1, 3, 3)]

        for band_name in band_names:
            ds_id = DatasetID(name=band_name, resolution=500)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(2, len(res))
            ds_id = DatasetID(name=band_name, resolution=2000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=4000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))

        res = reader.load(band_names)
        self.assertEqual(3, len(res))
        expected = {
                    1: np.array([[2.01, 2.02, 2.03, 2.04, 2.05], [2.06, 2.07, 2.08, 2.09, 2.1]]),
                    2: np.array([[4.03, 4.06, 4.09, 4.12, 4.15], [4.18, 4.21, 4.24, 4.27, 4.3]]),
                    3: np.array([[6.05, 6.1, 6.15, 6.2, 6.25], [6.3, 6.35, 6.4, 6.45, 6.5]])
                    }

        for index, band_name in enumerate(band_names):
            self.assertEqual(1, res[band_name].attrs['sensor'].islower())
            self.assertEqual((2, 5), res[band_name].shape)
            self.assertEqual('reflectance', res[band_name].attrs['calibration'])
            self.assertEqual('%', res[band_name].attrs['units'])
            self.assertTrue(np.allclose(res[band_name].values, expected[index + 1], equal_nan=True))

    def test_fy4a_500m_resolutions(self):
        """Test loading data when only 500m resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_0500M_V0001.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        # Verify that the resolution is only 500m
        available_datasets = reader.available_dataset_ids
        band_names = ['C' + '%02d' % ch for ch in np.linspace(2, 2, 1)]

        for band_name in band_names:
            ds_id = DatasetID(name=band_name, resolution=500)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(2, len(res))
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=2000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=4000)
            res = get_key(ds_id, available_datasets, num_results=0, best=False)
            self.assertEqual(0, len(res))

        res = reader.load(band_names)
        self.assertEqual(1, len(res))
        expected = np.array([[2.01, 2.02, 2.03, 2.04, 2.05], [2.06, 2.07, 2.08, 2.09, 2.1]])

        for band_name in band_names:
            self.assertEqual((2, 5), res[band_name].shape)
            self.assertEqual('reflectance', res[band_name].attrs['calibration'])
            self.assertEqual('%', res[band_name].attrs['units'])
            self.assertTrue(np.allclose(res[band_name].values, expected, equal_nan=True))
