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
"""Tests for the 'mersi2_l1b' reader."""
import os
import unittest
from unittest import mock

import numpy as np
import dask.array as da
import xarray as xr
from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def make_test_data(self, dims):
        return xr.DataArray(da.from_array(np.ones([dim for dim in dims], dtype=np.float32) * 10, [dim for dim in dims]))

    def _get_calibration(self, num_scans, rows_per_scan):
        calibration = {
            'Calibration/VIS_Cal_Coeff':
                xr.DataArray(
                    da.ones((19, 3), chunks=1024),
                    attrs={'Slope': [1.] * 19, 'Intercept': [0.] * 19},
                    dims=('_bands', '_coeffs')),
            'Calibration/IR_Cal_Coeff':
                xr.DataArray(
                    da.ones((6, 4, num_scans), chunks=1024),
                    attrs={'Slope': [1.] * 6, 'Intercept': [0.] * 6},
                    dims=('_bands', '_coeffs', '_scans')),
        }
        return calibration

    def _get_1km_data(self, num_scans, rows_per_scan, num_cols):
        data = {
            'Data/EV_1KM_RefSB':
                xr.DataArray(
                    da.ones((15, num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 15, 'Intercept': [0.] * 15,
                        'FillValue': 65535,
                        'units': 'NO',
                        'valid_range': [0, 4095],
                        'long_name': b'1km Earth View Science Data',
                    },
                    dims=('_ref_bands', '_rows', '_cols')),
            'Data/EV_1KM_Emissive':
                xr.DataArray(
                    da.ones((4, num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 4, 'Intercept': [0.] * 4,
                        'FillValue': 65535,
                        'units': 'mW/ (m2 cm-1 sr)',
                        'valid_range': [0, 25000],
                        'long_name': b'1km Emissive Bands Earth View '
                                     b'Science Data',
                    },
                    dims=('_ir_bands', '_rows', '_cols')),
            'Data/EV_250_Aggr.1KM_RefSB':
                xr.DataArray(
                    da.ones((4, num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 4, 'Intercept': [0.] * 4,
                        'FillValue': 65535,
                        'units': 'NO',
                        'valid_range': [0, 4095],
                        'long_name': b'250m Reflective Bands Earth View '
                                     b'Science Data Aggregated to 1 km'
                    },
                    dims=('_ref250_bands', '_rows', '_cols')),
            'Data/EV_250_Aggr.1KM_Emissive':
                xr.DataArray(
                    da.ones((2, num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 2, 'Intercept': [0.] * 2,
                        'FillValue': 65535,
                        'units': 'mW/ (m2 cm-1 sr)',
                        'valid_range': [0, 4095],
                        'long_name': b'250m Emissive Bands Earth View '
                                     b'Science Data Aggregated to 1 km'
                    },
                    dims=('_ir250_bands', '_rows', '_cols')),
        }
        return data

    def _get_250m_data(self, num_scans, rows_per_scan, num_cols):
        data = {
            'Data/EV_250_RefSB_b1':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'FillValue': 65535,
                        'units': 'NO',
                        'valid_range': [0, 4095],
                    },
                    dims=('_rows', '_cols')),
            'Data/EV_250_RefSB_b2':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'FillValue': 65535,
                        'units': 'NO',
                        'valid_range': [0, 4095],
                    },
                    dims=('_rows', '_cols')),
            'Data/EV_250_RefSB_b3':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'FillValue': 65535,
                        'units': 'NO',
                        'valid_range': [0, 4095],
                    },
                    dims=('_rows', '_cols')),
            'Data/EV_250_RefSB_b4':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'FillValue': 65535,
                        'units': 'NO',
                        'valid_range': [0, 4095],
                    },
                    dims=('_rows', '_cols')),
            'Data/EV_250_Emissive_b24':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'FillValue': 65535,
                        'units': 'mW/ (m2 cm-1 sr)',
                        'valid_range': [0, 4095],
                    },
                    dims=('_rows', '_cols')),
            'Data/EV_250_Emissive_b25':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024,
                            dtype=np.uint16),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'FillValue': 65535,
                        'units': 'mW/ (m2 cm-1 sr)',
                        'valid_range': [0, 4095],
                    },
                    dims=('_rows', '_cols')),
        }
        return data

    def _get_geo_data(self, num_scans, rows_per_scan, num_cols, prefix='Geolocation/'):
        geo = {
            prefix + 'Longitude':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'units': 'degree',
                        'valid_range': [-90, 90],
                    },
                    dims=('_rows', '_cols')),
            prefix + 'Latitude':
                xr.DataArray(
                    da.ones((num_scans * rows_per_scan, num_cols), chunks=1024),
                    attrs={
                        'Slope': [1.] * 1, 'Intercept': [0.] * 1,
                        'units': 'degree',
                        'valid_range': [-180, 180],
                    },
                    dims=('_rows', '_cols')),
        }
        return geo

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        rows_per_scan = self.filetype_info.get('rows_per_scan', 10)
        num_scans = 2
        num_cols = 2048
        global_attrs = {
            '/attr/Observing Beginning Date': '2019-01-01',
            '/attr/Observing Ending Date': '2019-01-01',
            '/attr/Observing Beginning Time': '18:27:39.720',
            '/attr/Observing Ending Time': '18:38:36.728',
            '/attr/Satellite Name': 'FY-3D',
            '/attr/Sensor Identification Code': 'MERSI',
        }

        data = {}
        if self.filetype_info['file_type'] == 'mersi2_l1b_1000':
            data = self._get_1km_data(num_scans, rows_per_scan, num_cols)
            global_attrs['/attr/TBB_Trans_Coefficient_A'] = [1.0] * 6
            global_attrs['/attr/TBB_Trans_Coefficient_B'] = [0.0] * 6
        elif self.filetype_info['file_type'] == 'mersi2_l1b_250':
            data = self._get_250m_data(num_scans, rows_per_scan, num_cols * 2)
            global_attrs['/attr/TBB_Trans_Coefficient_A'] = [0.0] * 6
            global_attrs['/attr/TBB_Trans_Coefficient_B'] = [0.0] * 6
        elif self.filetype_info['file_type'] == 'mersi2_l1b_1000_geo':
            data = self._get_geo_data(num_scans, rows_per_scan, num_cols)
        elif self.filetype_info['file_type'] == 'mersi2_l1b_250_geo':
            data = self._get_geo_data(num_scans, rows_per_scan, num_cols * 2,
                                      prefix='')

        test_content = {}
        test_content.update(global_attrs)
        test_content.update(data)
        test_content.update(self._get_calibration(num_scans, rows_per_scan))
        return test_content


class TestMERSI2L1BReader(unittest.TestCase):
    """Test MERSI2 L1B Reader."""
    yaml_file = "mersi2_l1b.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.readers.mersi2_l1b import MERSI2L1B
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(MERSI2L1B, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_fy3d_all_resolutions(self):
        """Test loading data when all resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'tf2019071182739.FY3D-X_MERSI_0250M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_1000M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEO1K_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEOQK_L1B.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(4, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ('1', '2', '3', '4', '24', '25'):
            if band_name in ('24', '25'):
                # don't know how to get radiance for IR bands
                num_results = 2
            else:
                num_results = 3
            ds_id = DatasetID(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            self.assertEqual(num_results, len(res))
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            self.assertEqual(num_results, len(res))

        res = reader.load(['1', '2', '3', '4', '5', '20', '24', '25'])
        self.assertEqual(8, len(res))
        self.assertEqual((2 * 40, 2048 * 2), res['1'].shape)
        self.assertEqual('reflectance', res['1'].attrs['calibration'])
        self.assertEqual('%', res['1'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['2'].shape)
        self.assertEqual('reflectance', res['2'].attrs['calibration'])
        self.assertEqual('%', res['2'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['3'].shape)
        self.assertEqual('reflectance', res['3'].attrs['calibration'])
        self.assertEqual('%', res['3'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['4'].shape)
        self.assertEqual('reflectance', res['4'].attrs['calibration'])
        self.assertEqual('%', res['4'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['5'].shape)
        self.assertEqual('reflectance', res['5'].attrs['calibration'])
        self.assertEqual('%', res['5'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['20'].shape)
        self.assertEqual('brightness_temperature', res['20'].attrs['calibration'])
        self.assertEqual('K', res['20'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['24'].shape)
        self.assertEqual('brightness_temperature', res['24'].attrs['calibration'])
        self.assertEqual('K', res['24'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['25'].shape)
        self.assertEqual('brightness_temperature', res['25'].attrs['calibration'])
        self.assertEqual('K', res['25'].attrs['units'])

    def test_fy3d_counts_calib(self):
        """Test loading data at counts calibration."""
        from satpy import DatasetID
        from satpy.readers import load_reader
        filenames = [
            'tf2019071182739.FY3D-X_MERSI_0250M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_1000M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEO1K_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEOQK_L1B.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(4, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        ds_ids = []
        for band_name in ['1', '2', '3', '4', '5', '20', '24', '25']:
            ds_ids.append(DatasetID(name=band_name, calibration='counts'))
        res = reader.load(ds_ids)
        self.assertEqual(8, len(res))
        self.assertEqual((2 * 40, 2048 * 2), res['1'].shape)
        self.assertEqual('counts', res['1'].attrs['calibration'])
        self.assertEqual(res['1'].dtype, np.uint16)
        self.assertEqual('1', res['1'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['2'].shape)
        self.assertEqual('counts', res['2'].attrs['calibration'])
        self.assertEqual(res['2'].dtype, np.uint16)
        self.assertEqual('1', res['2'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['3'].shape)
        self.assertEqual('counts', res['3'].attrs['calibration'])
        self.assertEqual(res['3'].dtype, np.uint16)
        self.assertEqual('1', res['3'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['4'].shape)
        self.assertEqual('counts', res['4'].attrs['calibration'])
        self.assertEqual(res['4'].dtype, np.uint16)
        self.assertEqual('1', res['4'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['5'].shape)
        self.assertEqual('counts', res['5'].attrs['calibration'])
        self.assertEqual(res['5'].dtype, np.uint16)
        self.assertEqual('1', res['5'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['20'].shape)
        self.assertEqual('counts', res['20'].attrs['calibration'])
        self.assertEqual(res['20'].dtype, np.uint16)
        self.assertEqual('1', res['20'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['24'].shape)
        self.assertEqual('counts', res['24'].attrs['calibration'])
        self.assertEqual(res['24'].dtype, np.uint16)
        self.assertEqual('1', res['24'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['25'].shape)
        self.assertEqual('counts', res['25'].attrs['calibration'])
        self.assertEqual(res['25'].dtype, np.uint16)
        self.assertEqual('1', res['25'].attrs['units'])

    def test_fy3d_rad_calib(self):
        """Test loading data at radiance calibration."""
        from satpy import DatasetID
        from satpy.readers import load_reader
        filenames = [
            'tf2019071182739.FY3D-X_MERSI_0250M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_1000M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEO1K_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEOQK_L1B.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(4, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        ds_ids = []
        for band_name in ['1', '2', '3', '4', '5']:
            ds_ids.append(DatasetID(name=band_name, calibration='radiance'))
        res = reader.load(ds_ids)
        self.assertEqual(5, len(res))
        self.assertEqual((2 * 40, 2048 * 2), res['1'].shape)
        self.assertEqual('radiance', res['1'].attrs['calibration'])
        self.assertEqual('mW/ (m2 cm-1 sr)', res['1'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['2'].shape)
        self.assertEqual('radiance', res['2'].attrs['calibration'])
        self.assertEqual('mW/ (m2 cm-1 sr)', res['2'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['3'].shape)
        self.assertEqual('radiance', res['3'].attrs['calibration'])
        self.assertEqual('mW/ (m2 cm-1 sr)', res['3'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['4'].shape)
        self.assertEqual('radiance', res['4'].attrs['calibration'])
        self.assertEqual('mW/ (m2 cm-1 sr)', res['4'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['5'].shape)
        self.assertEqual('radiance', res['5'].attrs['calibration'])
        self.assertEqual('mW/ (m2 cm-1 sr)', res['5'].attrs['units'])

    def test_fy3d_1km_resolutions(self):
        """Test loading data when only 1km resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'tf2019071182739.FY3D-X_MERSI_1000M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEO1K_L1B.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(4, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ('1', '2', '3', '4', '24', '25'):
            if band_name in ('24', '25'):
                # don't know how to get radiance for IR bands
                num_results = 2
            else:
                num_results = 3
            ds_id = DatasetID(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            self.assertEqual(0, len(res))
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            self.assertEqual(num_results, len(res))

        res = reader.load(['1', '2', '3', '4', '5', '20', '24', '25'])
        self.assertEqual(8, len(res))
        self.assertEqual((2 * 10, 2048), res['1'].shape)
        self.assertEqual('reflectance', res['1'].attrs['calibration'])
        self.assertEqual('%', res['1'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['2'].shape)
        self.assertEqual('reflectance', res['2'].attrs['calibration'])
        self.assertEqual('%', res['2'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['3'].shape)
        self.assertEqual('reflectance', res['3'].attrs['calibration'])
        self.assertEqual('%', res['3'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['4'].shape)
        self.assertEqual('reflectance', res['4'].attrs['calibration'])
        self.assertEqual('%', res['4'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['5'].shape)
        self.assertEqual('reflectance', res['5'].attrs['calibration'])
        self.assertEqual('%', res['5'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['20'].shape)
        self.assertEqual('brightness_temperature', res['20'].attrs['calibration'])
        self.assertEqual('K', res['20'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['24'].shape)
        self.assertEqual('brightness_temperature', res['24'].attrs['calibration'])
        self.assertEqual('K', res['24'].attrs['units'])
        self.assertEqual((2 * 10, 2048), res['25'].shape)
        self.assertEqual('brightness_temperature', res['25'].attrs['calibration'])
        self.assertEqual('K', res['25'].attrs['units'])

    def test_fy3d_250_resolutions(self):
        """Test loading data when only 250m resolutions are available."""
        from satpy import DatasetID
        from satpy.readers import load_reader, get_key
        filenames = [
            'tf2019071182739.FY3D-X_MERSI_0250M_L1B.HDF',
            'tf2019071182739.FY3D-X_MERSI_GEOQK_L1B.HDF',
        ]
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(4, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        # Verify that we have multiple resolutions for:
        #     - Bands 1-4 (visible)
        #     - Bands 24-25 (IR)
        available_datasets = reader.available_dataset_ids
        for band_name in ('1', '2', '3', '4', '24', '25'):
            if band_name in ('24', '25'):
                # don't know how to get radiance for IR bands
                num_results = 2
            else:
                num_results = 3
            ds_id = DatasetID(name=band_name, resolution=250)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            self.assertEqual(num_results, len(res))
            ds_id = DatasetID(name=band_name, resolution=1000)
            res = get_key(ds_id, available_datasets,
                          num_results=num_results, best=False)
            self.assertEqual(0, len(res))

        res = reader.load(['1', '2', '3', '4', '5', '20', '24', '25'])
        self.assertEqual(6, len(res))
        self.assertRaises(KeyError, res.__getitem__, '5')
        self.assertRaises(KeyError, res.__getitem__, '20')
        self.assertEqual((2 * 40, 2048 * 2), res['1'].shape)
        self.assertEqual('reflectance', res['1'].attrs['calibration'])
        self.assertEqual('%', res['1'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['2'].shape)
        self.assertEqual('reflectance', res['2'].attrs['calibration'])
        self.assertEqual('%', res['2'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['3'].shape)
        self.assertEqual('reflectance', res['3'].attrs['calibration'])
        self.assertEqual('%', res['3'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['4'].shape)
        self.assertEqual('reflectance', res['4'].attrs['calibration'])
        self.assertEqual('%', res['4'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['24'].shape)
        self.assertEqual('brightness_temperature', res['24'].attrs['calibration'])
        self.assertEqual('K', res['24'].attrs['units'])
        self.assertEqual((2 * 40, 2048 * 2), res['25'].shape)
        self.assertEqual('brightness_temperature', res['25'].attrs['calibration'])
        self.assertEqual('K', res['25'].attrs['units'])
