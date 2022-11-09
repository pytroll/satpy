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
"""Tests for the 'amsr2_l2_gaasp' reader."""

import os
from datetime import datetime
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

MBT_FILENAME = "AMSR2-MBT_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
PRECIP_FILENAME = "AMSR2-PRECIP_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
OCEAN_FILENAME = "AMSR2-OCEAN_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SEAICE_NH_FILENAME = "AMSR2-SEAICE-NH_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SEAICE_SH_FILENAME = "AMSR2-SEAICE-SH_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SNOW_FILENAME = "AMSR2-SNOW_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SOIL_FILENAME = "AMSR2-SOIL_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"

EXAMPLE_FILENAMES = [
    MBT_FILENAME,
    PRECIP_FILENAME,
    OCEAN_FILENAME,
    SEAICE_NH_FILENAME,
    SEAICE_SH_FILENAME,
    SNOW_FILENAME,
    SOIL_FILENAME,
]


def _get_shared_global_attrs(filename):
    attrs = {
        'time_coverage_start': '2020-08-12T05:58:31.0Z',
        'time_coverage_end': '2020-08-12T06:07:01.0Z',
        'platform_name': 'GCOM-W1',
        'instrument_name': 'AMSR2',
    }
    return attrs


def _create_two_res_gaasp_dataset(filename):
    """Represent files with two resolution of variables in them (ex. OCEAN)."""
    lon_var_hi = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_hi_rez_FOVs'),
                              attrs={'standard_name': 'longitude'})
    lat_var_hi = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_hi_rez_FOVs'),
                              attrs={'standard_name': 'latitude'})
    lon_var_lo = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                              attrs={'standard_name': 'longitude'})
    lat_var_lo = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                              attrs={'standard_name': 'latitude'})
    swath_var1 = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_hi_rez_FOVs'),
                              coords={'some_longitude_hi': lon_var_hi, 'some_latitude_hi': lat_var_hi},
                              attrs={'_FillValue': -9999.,
                                     'scale_factor': 0.5, 'add_offset': 2.0})
    swath_var2 = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                              coords={'some_longitude_lo': lon_var_lo, 'some_latitude_lo': lat_var_lo},
                              attrs={'_FillValue': -9999.})
    swath_int_var = xr.DataArray(da.zeros((10, 10), dtype=np.uint16),
                                 dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                                 attrs={'_FillValue': 100, 'comment': 'Some comment'})
    not_xy_dim_var = xr.DataArray(da.zeros((10, 5), dtype=np.float32),
                                  dims=('Number_of_Scans', 'Time_Dimension'))
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32),
                            dims=('Time_Dimension',))
    ds_vars = {
        'swath_var_hi': swath_var1,
        'swath_var_low': swath_var2,
        'swath_var_low_int': swath_int_var,
        'some_longitude_hi': lon_var_hi,
        'some_latitude_hi': lat_var_hi,
        'some_longitude_lo': lon_var_lo,
        'some_latitude_lo': lat_var_lo,
        'not_xy_dim_var': not_xy_dim_var,
        'time_var': time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    ds = xr.Dataset(ds_vars, attrs=attrs)
    return ds


def _create_gridded_gaasp_dataset(filename):
    """Represent files with gridded products."""
    grid_var = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                            dims=('Number_of_Y_Dimension', 'Number_of_X_Dimension'),
                            attrs={
                                '_FillValue': -9999.,
                                'scale_factor': 0.5, 'add_offset': 2.0
                            })
    latency_var = xr.DataArray(da.zeros((10, 10), dtype=np.timedelta64),
                               dims=('Number_of_Y_Dimension', 'Number_of_X_Dimension'),
                               attrs={
                                   '_FillValue': -9999,
                               })
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32),
                            dims=('Time_Dimension',))
    ds_vars = {
        'grid_var': grid_var,
        'latency_var': latency_var,
        'time_var': time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    return xr.Dataset(ds_vars, attrs=attrs)


def _create_one_res_gaasp_dataset(filename):
    """Represent files with one resolution of variables in them (ex. SOIL)."""
    lon_var_lo = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                              attrs={'standard_name': 'longitude'})
    lat_var_lo = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                              attrs={'standard_name': 'latitude'})
    swath_var2 = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                              coords={'some_longitude_lo': lon_var_lo, 'some_latitude_lo': lat_var_lo},
                              attrs={
                                  '_FillValue': -9999.,
                                  'scale_factor': 0.5, 'add_offset': 2.0
                              })
    swath_int_var = xr.DataArray(da.zeros((10, 10), dtype=np.uint16),
                                 dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'),
                                 attrs={'_FillValue': 100, 'comment': 'Some comment'})
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32),
                            dims=('Time_Dimension',))
    ds_vars = {
        'swath_var': swath_var2,
        'swath_var_int': swath_int_var,
        'some_longitude_lo': lon_var_lo,
        'some_latitude_lo': lat_var_lo,
        'time_var': time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    return xr.Dataset(ds_vars, attrs=attrs)


def fake_open_dataset(filename, **kwargs):
    """Create a Dataset similar to reading an actual file with xarray.open_dataset."""
    if filename in [MBT_FILENAME, PRECIP_FILENAME, OCEAN_FILENAME]:
        return _create_two_res_gaasp_dataset(filename)
    if filename in [SEAICE_NH_FILENAME, SEAICE_SH_FILENAME]:
        return _create_gridded_gaasp_dataset(filename)
    return _create_one_res_gaasp_dataset(filename)


class TestGAASPReader:
    """Tests for the GAASP reader."""

    yaml_file = 'amsr2_l2_gaasp.yaml'

    def setup_method(self):
        """Wrap pygrib to read fake data."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

    @pytest.mark.parametrize(
        ("filenames", "expected_loadables"),
        [
            (EXAMPLE_FILENAMES, 7),
            ([MBT_FILENAME], 1),
            ([PRECIP_FILENAME], 1),
            ([OCEAN_FILENAME], 1),
            ([SEAICE_NH_FILENAME], 1),
            ([SEAICE_SH_FILENAME], 1),
            ([SNOW_FILENAME], 1),
            ([SOIL_FILENAME], 1),
        ]
    )
    def test_reader_creation(self, filenames, expected_loadables):
        """Test basic initialization."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.amsr2_l2_gaasp.xr.open_dataset') as od:
            od.side_effect = fake_open_dataset
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
            assert len(loadables) == expected_loadables
            r.create_filehandlers(loadables)
            # make sure we have some files
            assert r.file_handlers

    @pytest.mark.parametrize(
        ("filenames", "expected_datasets"),
        [
            (EXAMPLE_FILENAMES, ['swath_var_hi', 'swath_var_low',
                                 'swath_var_low_int', 'swath_var',
                                 'swath_var_int',
                                 'grid_var_NH', 'grid_var_SH',
                                 'latency_var_NH', 'latency_var_SH']),
            ([MBT_FILENAME], ['swath_var_hi', 'swath_var_low',
                              'swath_var_low_int']),
            ([PRECIP_FILENAME], ['swath_var_hi', 'swath_var_low',
                                 'swath_var_low_int']),
            ([OCEAN_FILENAME], ['swath_var_hi', 'swath_var_low',
                                'swath_var_low_int']),
            ([SEAICE_NH_FILENAME], ['grid_var_NH', 'latency_var_NH']),
            ([SEAICE_SH_FILENAME], ['grid_var_SH', 'latency_var_SH']),
            ([SNOW_FILENAME], ['swath_var', 'swath_var_int']),
            ([SOIL_FILENAME], ['swath_var', 'swath_var_int']),
        ])
    def test_available_datasets(self, filenames, expected_datasets):
        """Test that variables are dynamically discovered."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.amsr2_l2_gaasp.xr.open_dataset') as od:
            od.side_effect = fake_open_dataset
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
            r.create_filehandlers(loadables)
            avails = list(r.available_dataset_names)
            for var_name in expected_datasets:
                assert var_name in avails
            assert 'not_xy_dim_var' not in expected_datasets

    @staticmethod
    def _check_area(data_id, data_arr):
        from pyresample.geometry import AreaDefinition, SwathDefinition
        area = data_arr.attrs['area']
        if 'grid_var' in data_id['name'] or 'latency_var' in data_id['name']:
            assert isinstance(area, AreaDefinition)
        else:
            assert isinstance(area, SwathDefinition)

    @staticmethod
    def _check_fill(data_id, data_arr):
        if 'int' in data_id['name']:
            assert data_arr.attrs['_FillValue'] == 100
            assert np.issubdtype(data_arr.dtype, np.integer)
        else:
            assert '_FillValue' not in data_arr.attrs
            if np.issubdtype(data_arr.dtype, np.floating):
                # we started with float32, it should stay that way
                assert data_arr.dtype.type == np.float32

    @staticmethod
    def _check_attrs(data_arr):
        attrs = data_arr.attrs
        assert 'scale_factor' not in attrs
        assert 'add_offset' not in attrs
        assert attrs['platform_name'] == 'GCOM-W1'
        assert attrs['sensor'] == 'amsr2'
        assert attrs['start_time'] == datetime(2020, 8, 12, 5, 58, 31)
        assert attrs['end_time'] == datetime(2020, 8, 12, 6, 7, 1)

    @pytest.mark.parametrize(
        ("filenames", "loadable_ids"),
        [
            (EXAMPLE_FILENAMES, ['swath_var_hi', 'swath_var_low',
                                 'swath_var_low_int', 'swath_var',
                                 'swath_var_int',
                                 'grid_var_NH', 'grid_var_SH',
                                 'latency_var_NH', 'latency_var_SH']),
            ([MBT_FILENAME], ['swath_var_hi', 'swath_var_low', 'swath_var_low_int']),
            ([PRECIP_FILENAME], ['swath_var_hi', 'swath_var_low', 'swath_var_low_int']),
            ([OCEAN_FILENAME], ['swath_var_hi', 'swath_var_low', 'swath_var_low_int']),
            ([SEAICE_NH_FILENAME], ['grid_var_NH', 'latency_var_NH']),
            ([SEAICE_SH_FILENAME], ['grid_var_SH', 'latency_var_SH']),
            ([SNOW_FILENAME], ['swath_var', 'swath_var_int']),
            ([SOIL_FILENAME], ['swath_var', 'swath_var_int']),
        ])
    def test_basic_load(self, filenames, loadable_ids):
        """Test that variables are loaded properly."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.amsr2_l2_gaasp.xr.open_dataset') as od:
            od.side_effect = fake_open_dataset
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
            r.create_filehandlers(loadables)
            loaded_data_arrs = r.load(loadable_ids)
            assert loaded_data_arrs
            for data_id, data_arr in loaded_data_arrs.items():
                self._check_area(data_id, data_arr)
                self._check_fill(data_id, data_arr)
                self._check_attrs(data_arr)
