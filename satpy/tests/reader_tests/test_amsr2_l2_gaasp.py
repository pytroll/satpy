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
from unittest import mock
import pytest
import xarray as xr
import dask.array as da
import numpy as np

MBT_FILENAME = "AMSR2-MBT_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
OCEAN_FILENAME = "AMSR2-OCEAN_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SEAICE_NH_FILENAME = "AMSR2-SEAICE-NH_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SEAICE_SH_FILENAME = "AMSR2-SEAICE-SH_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SNOW_FILENAME = "AMSR2-SNOW_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"
SOIL_FILENAME = "AMSR2-SOIL_v2r2_GW1_s202008120558310_e202008120607010_c202008120637340.nc"

EXAMPLE_FILENAMES = [
    MBT_FILENAME,
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
    }
    return attrs


def _create_two_rez_gaasp_dataset(filename):
    swath_var1 = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_hi_rez_FOVs'))
    swath_var2 = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'))
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32),
                            dims=('Time_Dimension',))
    vars = {
        'swath_var_hi': swath_var1,
        'swath_var_low': swath_var2,
        'time_var': time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    ds = xr.Dataset(vars, attrs=attrs)
    return ds


def _create_gridded_gaasp_dataset(filename):
    grid_var = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                            dims=('Number_of_Y_Dimension', 'Number_of_X_Dimension'))
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32),
                            dims=('Time_Dimension',))
    vars = {
        'grid_var': grid_var,
        'time_var': time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    return xr.Dataset(vars, attrs=attrs)


def _create_one_rez_gaasp_dataset(filename):
    swath_var2 = xr.DataArray(da.zeros((10, 10), dtype=np.float32),
                              dims=('Number_of_Scans', 'Number_of_low_rez_FOVs'))
    time_var = xr.DataArray(da.zeros((5,), dtype=np.float32),
                            dims=('Time_Dimension',))
    vars = {
        'swath_var': swath_var2,
        'time_var': time_var,
    }
    attrs = _get_shared_global_attrs(filename)
    return xr.Dataset(vars, attrs=attrs)


def fake_open_dataset(filename, **kwargs):
    """Create a Dataset similar to reading an actual file with xarray.open_dataset."""
    if filename in [MBT_FILENAME, OCEAN_FILENAME]:
        return _create_two_rez_gaasp_dataset(filename)
    if filename in [SEAICE_NH_FILENAME, SEAICE_SH_FILENAME]:
        return _create_gridded_gaasp_dataset(filename)
    return _create_one_rez_gaasp_dataset(filename)


class TestGAASPReader:
    """Tests for the GAASP reader."""

    yaml_file = 'amsr2_l2_gaasp.yaml'

    def setup_method(self):
        """Wrap pygrib to read fake data."""
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

    @pytest.mark.parametrize(
        ("filenames", "expected_loadables"),
        [
            (EXAMPLE_FILENAMES, 6),
            ([MBT_FILENAME], 1),
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
            (EXAMPLE_FILENAMES, ['swath_var_hi', 'swath_var_low', 'swath_var', 'grid_var_NH', 'grid_var_SH']),
            ([MBT_FILENAME], ['swath_var_hi', 'swath_var_low']),
            ([OCEAN_FILENAME], ['swath_var_hi', 'swath_var_low']),
            ([SEAICE_NH_FILENAME], ['grid_var_NH']),
            ([SEAICE_SH_FILENAME], ['grid_var_SH']),
            ([SNOW_FILENAME], ['swath_var']),
            ([SOIL_FILENAME], ['swath_var']),
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
