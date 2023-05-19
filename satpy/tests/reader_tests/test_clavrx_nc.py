#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Module for testing the satpy.readers.clavrx module."""

import os
import unittest
from unittest import mock

import numpy as np
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy.readers import load_reader
from satpy.tests.reader_tests.test_netCDF_utils import FakeNetCDF4FileHandler

ABI_FILE = 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s20231021601173.level2.nc'
DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FLAGS = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                               dtype=np.byte).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
ABI_FILE = 'clavrx_OR_ABI-L1b-RadC-M6C01_G16_s20231021601173.level2.nc'
FILL_VALUE = -32768


def fake_dataset():
    """Mimic reader input file content."""
    attrs = {
        'platform': 'G16',
        'sensor': 'ABI',
        # this is a Level 2 file that came from a L1B file
        'L1B': '"clavrx_OR_ABI-L1b-RadC-M6C01_G16_s20231021601173',
    }

    longitude = xr.DataArray(DEFAULT_LON_DATA,
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'_FillValue': -999.,
                                    'SCALED': 0,
                                    'scale_factor': 1.,
                                    'add_offset': 0.,
                                    'standard_name': 'longitude',
                                    'units': 'degrees_east'
                                    })

    latitude = xr.DataArray(DEFAULT_LAT_DATA,
                            dims=('scan_lines_along_track_direction',
                                  'pixel_elements_along_scan_direction'),
                            attrs={'_FillValue': -999.,
                                   'SCALED': 0,
                                   'scale_factor': 1.,
                                   'add_offset': 0.,
                                   'standard_name': 'latitude',
                                   'units': 'degrees_south'
                                   })

    variable1 = xr.DataArray(DEFAULT_FILE_DATA.astype(np.int8),
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'_FillValue': -127,
                                    'SCALED': 0,
                                    'units': '1',
                                    })

    # data with fill values and a file_type alias
    variable2 = xr.DataArray(DEFAULT_FILE_DATA.astype(np.int16),
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'_FillValue': FILL_VALUE,
                                    'SCALED': 1,
                                    'scale_factor': 0.001861629,
                                    'add_offset': 59.,
                                    'units': '%',
                                    'valid_range': [-32767, 32767],
                                    })
    variable2 = variable2.where(variable2 % 2 != 0, FILL_VALUE)

    # category
    variable3 = xr.DataArray(DEFAULT_FILE_FLAGS.astype(np.int8),
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'SCALED': 0,
                                    '_FillValue': -127,
                                    'units': '1',
                                    'flag_values': [0, 1, 2, 3]})

    ds_vars = {
        'longitude': longitude,
        'latitude': latitude,
        'variable1': variable1,
        'refl_0_65um_nom': variable2,
        'variable3': variable3
    }

    ds = xr.Dataset(ds_vars, attrs=attrs)
    ds = ds.assign_coords({"latitude": latitude, "longitude": longitude})

    return ds


class FakeNetCDF4FileHandlerCLAVRx(FakeNetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler."""

    def get_test_content(self, filename, filename_info, filetype_info):
        """Get a fake dataset."""
        return fake_dataset()


class TestCLAVRXReaderNetCDF(unittest.TestCase):
    """Test CLAVR-X Reader with NetCDF files."""

    yaml_file = "clavrx.yaml"
    filename = ABI_FILE
    loadable_ids = list(fake_dataset().keys())

    def setUp(self):
        """Wrap NetCDF file handler with a fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.clavrx import CLAVRXNetCDFFileHandler

        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(CLAVRXNetCDFFileHandler, '__bases__',
                                   (FakeNetCDF4FileHandlerCLAVRx,), spec=True)
        self.fake_open_dataset = mock.patch('satpy.readers.clavrx.xr.open_dataset',
                                            return_value=fake_dataset()).start()
        self.fake_handler = self.p.start()
        self.p.is_local = True

        self.addCleanup(mock.patch.stopall)

    def test_init(self):
        """Test basic init with no extra parameters."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([ABI_FILE])
        self.assertEqual(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_available_datasets(self):
        """Test that variables are dynamically discovered."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([ABI_FILE])
        r.create_filehandlers(loadables)
        avails = list(r.available_dataset_names)
        expected_datasets = self.loadable_ids + ["latitude", "longitude"]
        self.assertEqual(avails.sort(), expected_datasets.sort())

    def test_load_all_new_donor(self):
        """Test loading all test datasets with new donor."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([ABI_FILE])
        r.create_filehandlers(loadables)
        with mock.patch('satpy.readers.clavrx.glob') as g, \
                mock.patch('satpy.readers.clavrx.netCDF4.Dataset') as d:
            g.return_value = ['fake_donor.nc']
            x = np.linspace(-0.1518, 0.1518, 300)
            y = np.linspace(0.1518, -0.1518, 10)
            proj = mock.Mock(
                semi_major_axis=6378137,
                semi_minor_axis=6356752.3142,
                perspective_point_height=35791000,
                longitude_of_projection_origin=-137.2,
                sweep_angle_axis='x',
            )
            d.return_value = fake_donor = mock.MagicMock(
                variables={'goes_imager_projection': proj, 'x': x, 'y': y},
            )
            fake_donor.__getitem__.side_effect = lambda key: fake_donor.variables[key]

            datasets = r.load(self.loadable_ids + ["C02"])
            self.assertEqual(len(datasets), len(self.loadable_ids)+1)

            # should have file variable and one alias for reflectance
            self.assertNotIn("valid_range", datasets["variable1"].attrs)
            self.assertNotIn("_FillValue", datasets["variable1"].attrs)
            self.assertEqual(np.float64, datasets["variable1"].dtype)

            assert np.issubdtype(datasets["variable3"].dtype, np.integer)
            self.assertIsNotNone(datasets['variable3'].attrs.get('flag_meanings'))
            self.assertEqual('<flag_meanings_unknown>',
                             datasets['variable3'].attrs.get('flag_meanings'),
                             )

            self.assertIsInstance(datasets["refl_0_65um_nom"].valid_range, list)
            self.assertEqual(np.float64, datasets["refl_0_65um_nom"].dtype)
            self.assertNotIn("_FillValue", datasets["refl_0_65um_nom"].attrs)

            self.assertEqual("refl_0_65um_nom", datasets["C02"].file_key)
            self.assertNotIn("_FillValue", datasets["C02"].attrs)

            for v in datasets.values():
                self.assertIsInstance(v.area, AreaDefinition)
                self.assertEqual(v.platform_name, 'GOES-16')
                self.assertEqual(v.sensor, 'abi')

                self.assertNotIn('calibration', v.attrs)
                self.assertIn("units", v.attrs)
                self.assertNotIn('rows_per_scan', v.coords.get('longitude').attrs)

    def test_yaml_datasets(self):
        """Test available_datasets with fake variables from YAML."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([ABI_FILE])
        r.create_filehandlers(loadables)
        # mimic the YAML file being configured for more datasets
        fake_dataset_info = [
            (None, {'name': 'yaml1', 'resolution': None, 'file_type': ['clavrx_nc']}),
            (True, {'name': 'yaml2', 'resolution': 0.5, 'file_type': ['clavrx_nc']}),
        ]
        new_ds_infos = list(r.file_handlers['clavrx_nc'][0].available_datasets(
            fake_dataset_info))
        self.assertEqual(len(new_ds_infos), 9)

        # we have this and can provide the resolution
        self.assertTrue(new_ds_infos[0][0])
        self.assertEqual(new_ds_infos[0][1]['resolution'], 2004)  # hardcoded

        # we have this, but previous file handler said it knew about it
        # and it is producing the same resolution as what we have
        self.assertTrue(new_ds_infos[1][0])
        self.assertEqual(new_ds_infos[1][1]['resolution'], 0.5)

        # we have this, but don't want to change the resolution
        # because a previous handler said it has it
        self.assertTrue(new_ds_infos[2][0])
        self.assertEqual(new_ds_infos[2][1]['resolution'], 2004)
