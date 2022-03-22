#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
AHI_FILE = 'clavrx_H08_20210603_1500_B01_FLDK_R.level2.nc'


def fake_test_content(filename, **kwargs):
    """Mimic reader input file content."""
    attrs = {
        'platform': 'HIM8',
        'sensor': 'AHI',
        # this is a Level 2 file that came from a L1B file
        'L1B': 'clavrx_H08_20210603_1500_B01_FLDK_R',
    }

    longitude = xr.DataArray(DEFAULT_LON_DATA,
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'_FillValue': np.nan,
                                    'scale_factor': 1.,
                                    'add_offset': 0.,
                                    'standard_name': 'longitude',
                                    'units': 'degrees_east'
                                    })

    latitude = xr.DataArray(DEFAULT_LAT_DATA,
                            dims=('scan_lines_along_track_direction',
                                  'pixel_elements_along_scan_direction'),
                            attrs={'_FillValue': np.nan,
                                   'scale_factor': 1.,
                                   'add_offset': 0.,
                                   'standard_name': 'latitude',
                                   'units': 'degrees_south'
                                   })

    variable1 = xr.DataArray(DEFAULT_FILE_DATA.astype(np.float32),
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'_FillValue': np.nan,
                                    'scale_factor': 1.,
                                    'add_offset': 0.,
                                    'units': '1',
                                    'valid_range': [-32767, 32767],
                                    })

    # data with fill values
    variable2 = xr.DataArray(DEFAULT_FILE_DATA.astype(np.float32),
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'_FillValue': np.nan,
                                    'scale_factor': 1.,
                                    'add_offset': 0.,
                                    'units': '1',
                                    'valid_range': [-32767, 32767],
                                    })
    variable2 = variable2.where(variable2 % 2 != 0)

    # category
    variable3 = xr.DataArray(DEFAULT_FILE_DATA.astype(np.byte),
                             dims=('scan_lines_along_track_direction',
                                   'pixel_elements_along_scan_direction'),
                             attrs={'SCALED': 0,
                                    '_FillValue': -128,
                                    'flag_meanings': 'clear water supercooled mixed ice unknown',
                                    'flag_values': [0, 1, 2, 3, 4, 5],
                                    'units': '1',
                                    })

    ds_vars = {
        'longitude': longitude,
        'latitude': latitude,
        'variable1': variable1,
        'variable2': variable2,
        'variable3': variable3
    }

    ds = xr.Dataset(ds_vars, attrs=attrs)
    ds = ds.assign_coords({"latitude": latitude, "longitude": longitude})

    return ds


class TestCLAVRXReaderGeo:
    """Test CLAVR-X Reader with Geo files."""

    yaml_file = "clavrx.yaml"

    def setup_method(self):
        """Read fake data."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

    @pytest.mark.parametrize(
        ("filenames", "expected_loadables"),
        [([AHI_FILE], 1)]
    )
    def test_reader_creation(self, filenames, expected_loadables):
        """Test basic initialization."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.clavrx.xr.open_dataset') as od:
            od.side_effect = fake_test_content
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
            assert len(loadables) == expected_loadables
            r.create_filehandlers(loadables)
            # make sure we have some files
            assert r.file_handlers

    @pytest.mark.parametrize(
        ("filenames", "expected_datasets"),
        [([AHI_FILE], ['variable1', 'variable2', 'variable3']), ]
    )
    def test_available_datasets(self, filenames, expected_datasets):
        """Test that variables are dynamically discovered."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.clavrx.xr.open_dataset') as od:
            od.side_effect = fake_test_content
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
            r.create_filehandlers(loadables)
            avails = list(r.available_dataset_names)
            for var_name in expected_datasets:
                assert var_name in avails

    @pytest.mark.parametrize(
        ("filenames", "loadable_ids"),
        [([AHI_FILE], ['variable1', 'variable2', 'variable3']), ]
    )
    def test_load_all_new_donor(self, filenames, loadable_ids):
        """Test loading all test datasets with new donor."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.clavrx.xr.open_dataset') as od:
            od.side_effect = fake_test_content
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
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
                    longitude_of_projection_origin=140.7,
                    sweep_angle_axis='y',
                )
                d.return_value = fake_donor = mock.MagicMock(
                    variables={'goes_imager_projection': proj, 'x': x, 'y': y},
                )
                fake_donor.__getitem__.side_effect = lambda key: fake_donor.variables[key]
                datasets = r.load(loadable_ids)
                assert len(datasets) == 3
                for v in datasets.values():
                    assert 'calibration' not in v.attrs
                    assert v.attrs['units'] == '1'
                    assert isinstance(v.attrs['area'], AreaDefinition)
                    assert v.attrs['platform_name'] == 'himawari8'
                    assert v.attrs['sensor'] == 'AHI'
                    assert 'rows_per_scan' not in v.coords.get('longitude').attrs
                    if v.attrs["name"] in ["variable1", "variable2"]:
                        assert isinstance(v.attrs["valid_range"], list)
                        assert v.dtype == np.float32
                    else:
                        assert (datasets['variable3'].attrs.get('flag_meanings')) is not None
                        assert "_FillValue" in v.attrs.keys()
                        assert np.issubdtype(v.dtype, np.integer)
