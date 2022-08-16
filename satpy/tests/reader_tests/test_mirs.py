#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# This file is part of Satpy.
#
# Satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Module for testing the satpy.readers.tropomi_l2 module."""

import os
from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import xarray as xr

METOP_FILE = "IMG_SX.M2.D17037.S1601.E1607.B0000001.WE.HR.ORB.nc"
NPP_MIRS_L2_SWATH = "NPR-MIRS-IMG_v11r6_npp_s201702061601000_e201702061607000_c202012201658410.nc"
N20_MIRS_L2_SWATH = "NPR-MIRS-IMG_v11r4_n20_s201702061601000_e201702061607000_c202012201658410.nc"
OTHER_MIRS_L2_SWATH = "NPR-MIRS-IMG_v11r4_gpm_s201702061601000_e201702061607000_c202010080001310.nc"

EXAMPLE_FILES = [METOP_FILE, NPP_MIRS_L2_SWATH, OTHER_MIRS_L2_SWATH]

N_CHANNEL = 22
N_FOV = 96
N_SCANLINE = 100
DEFAULT_FILE_DTYPE = np.float64
DEFAULT_2D_SHAPE = (N_SCANLINE, N_FOV)
DEFAULT_DATE = datetime(2019, 6, 19, 13, 0)
DEFAULT_LAT = np.linspace(23.09356, 36.42844, N_SCANLINE * N_FOV,
                          dtype=DEFAULT_FILE_DTYPE)
DEFAULT_LON = np.linspace(127.6879, 144.5284, N_SCANLINE * N_FOV,
                          dtype=DEFAULT_FILE_DTYPE)
FREQ = xr.DataArray([23.8, 31.4, 50.3, 51.76, 52.8, 53.596, 54.4, 54.94, 55.5,
                     57.29, 57.29, 57.29, 57.29, 57.29, 57.29, 88.2, 165.5,
                     183.31, 183.31, 183.31, 183.31, 183.31][:N_CHANNEL],
                    dims='Channel',
                    attrs={'description': "Central Frequencies (GHz)"})
POLO = xr.DataArray([2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3,
                     3, 3, 3][:N_CHANNEL], dims='Channel',
                    attrs={'description': "Polarizations"})

DS_IDS = ['RR', 'longitude', 'latitude']
TEST_VARS = ['btemp_88v', 'btemp_165h',
             'btemp_23v', 'RR', 'Sfc_type']
DEFAULT_UNITS = {'btemp_88v': 'K', 'btemp_165h': 'K',
                 'btemp_23v': 'K', 'RR': 'mm/hr', 'Sfc_type': "1"}
PLATFORM = {"M2": "metop-a", "NPP": "npp", "GPM": "gpm"}
SENSOR = {"m2": "amsu-mhs", "npp": "atms", "gpm": "GPI"}

START_TIME = datetime(2017, 2, 6, 16, 1, 0)
END_TIME = datetime(2017, 2, 6, 16, 7, 0)


def fake_coeff_from_fn(fn):
    """Create Fake Coefficients."""
    ameans = np.random.uniform(261, 267, N_CHANNEL)
    locations = [
        [1, 2],
        [1, 2],
        [3, 4, 5],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8],
        [9, 10, 11],
        [10, 11],
        [10, 11, 12],
        [11, 12, 13],
        [12, 13],
        [12, 13, 14],
        [14, 15],
        [1, 16],
        [17, 18],
        [18, 19],
        [18, 19, 20],
        [19, 20, 21],
        [20, 21, 22],
        [21, 22],
    ]
    all_nchx = [len(loc) for loc in locations]

    coeff_str = []
    for idx in range(1, N_CHANNEL + 1):
        nx = idx - 1
        coeff_str.append('\n')
        next_line = '   {}  {} {}\n'.format(idx, all_nchx[nx], ameans[nx])
        coeff_str.append(next_line)
        next_line = '   {}\n'.format("   ".join([str(x) for x in locations[idx - 1]]))
        coeff_str.append(next_line)
        for fov in range(1, N_FOV+1):
            random_coeff = np.random.rand(all_nchx[nx])
            random_coeff = np.ones(all_nchx[nx])
            str_coeff = '  '.join([str(x) for x in random_coeff])
            random_means = np.random.uniform(261, 267, all_nchx[nx])
            random_means = np.zeros(all_nchx[nx])
            str_means = ' '.join([str(x) for x in random_means])
            error_val = np.random.uniform(0, 4)
            coeffs_line = ' {:>2} {:>2}  {} {}   {}\n'.format(idx, fov,
                                                              str_coeff,
                                                              str_means,
                                                              error_val)
            coeff_str.append(coeffs_line)

    return coeff_str


def _get_datasets_with_attributes(**kwargs):
    """Represent files with two resolution of variables in them (ex. OCEAN)."""
    bt = xr.DataArray(np.linspace(1830, 3930, N_SCANLINE * N_FOV * N_CHANNEL).
                      reshape(N_SCANLINE, N_FOV, N_CHANNEL),
                      attrs={'long_name': "Channel Temperature (K)",
                             'units': "Kelvin",
                             'coordinates': "Longitude Latitude Freq",
                             'scale_factor': 0.01,
                             '_FillValue': -999,
                             'valid_range': [0, 50000]},
                      dims=('Scanline', 'Field_of_view', 'Channel'))
    rr = xr.DataArray(np.random.randint(100, 500, size=(N_SCANLINE, N_FOV)),
                      attrs={'long_name': "Rain Rate (mm/hr)",
                             'units': "mm/hr",
                             'coordinates': "Longitude Latitude",
                             'scale_factor': 0.1,
                             '_FillValue': -999,
                             'valid_range': [0, 1000]},
                      dims=('Scanline', 'Field_of_view'))
    sfc_type = xr.DataArray(np.random.randint(0, 4, size=(N_SCANLINE, N_FOV)),
                            attrs={'description': "type of surface:0-ocean," +
                                                  "1-sea ice,2-land,3-snow",
                                   'units': "1",
                                   'coordinates': "Longitude Latitude",
                                   '_FillValue': -999,
                                   'valid_range': [0, 3]
                                   },
                            dims=('Scanline', 'Field_of_view'))
    latitude = xr.DataArray(DEFAULT_LAT.reshape(DEFAULT_2D_SHAPE),
                            attrs={'long_name':
                                   "Latitude of the view (-90,90)"},
                            dims=('Scanline', 'Field_of_view'))
    longitude = xr.DataArray(DEFAULT_LON.reshape(DEFAULT_2D_SHAPE),
                             attrs={'long_name':
                                    "Longitude of the view (-180,180)"},
                             dims=('Scanline', 'Field_of_view'))

    ds_vars = {
        'Freq': FREQ,
        'Polo': POLO,
        'BT': bt,
        'RR': rr,
        'Sfc_type': sfc_type,
        'Latitude': latitude,
        'Longitude': longitude
    }

    attrs = {'missing_value': -999.}
    ds = xr.Dataset(ds_vars, attrs=attrs)
    ds = ds.assign_coords({"Freq": FREQ, "Latitude": latitude, "Longitude": longitude})
    return ds


def _get_datasets_with_less_attributes():
    """Represent files with two resolution of variables in them (ex. OCEAN)."""
    bt = xr.DataArray(np.linspace(1830, 3930, N_SCANLINE * N_FOV * N_CHANNEL).
                      reshape(N_SCANLINE, N_FOV, N_CHANNEL),
                      attrs={'long_name': "Channel Temperature (K)",
                             'scale_factor': 0.01},
                      dims=('Scanline', 'Field_of_view', 'Channel'))
    rr = xr.DataArray(np.random.randint(100, 500, size=(N_SCANLINE, N_FOV)),
                      attrs={'long_name': "Rain Rate (mm/hr)",
                             'scale_factor': 0.1},
                      dims=('Scanline', 'Field_of_view'))

    sfc_type = xr.DataArray(np.random.randint(0, 4, size=(N_SCANLINE, N_FOV)),
                            attrs={'description': "type of surface:0-ocean," +
                                                  "1-sea ice,2-land,3-snow"},
                            dims=('Scanline', 'Field_of_view'))
    latitude = xr.DataArray(DEFAULT_LAT.reshape(DEFAULT_2D_SHAPE),
                            attrs={'long_name':
                                   "Latitude of the view (-90,90)"},
                            dims=('Scanline', 'Field_of_view'))
    longitude = xr.DataArray(DEFAULT_LON.reshape(DEFAULT_2D_SHAPE),
                             attrs={"long_name":
                                    "Longitude of the view (-180,180)"},
                             dims=('Scanline', 'Field_of_view'))

    ds_vars = {
        'Freq': FREQ,
        'Polo': POLO,
        'BT': bt,
        'RR': rr,
        'Sfc_type': sfc_type,
        'Longitude': longitude,
        'Latitude': latitude
    }

    attrs = {'missing_value': -999.}
    ds = xr.Dataset(ds_vars, attrs=attrs)
    ds = ds.assign_coords({"Freq": FREQ, "Latitude": latitude, "Longitude": longitude})
    return ds


def fake_open_dataset(filename, **kwargs):
    """Create a Dataset similar to reading an actual file with xarray.open_dataset."""
    if filename == METOP_FILE:
        return _get_datasets_with_less_attributes()
    return _get_datasets_with_attributes()


class TestMirsL2_NcReader:
    """Test mirs Reader."""

    yaml_file = "mirs.yaml"

    def setup_method(self):
        """Read fake data."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

    @pytest.mark.parametrize(
        ("filenames", "expected_loadables"),
        [
            ([METOP_FILE], 1),
            ([NPP_MIRS_L2_SWATH], 1),
            ([OTHER_MIRS_L2_SWATH], 1),
        ]
    )
    def test_reader_creation(self, filenames, expected_loadables):
        """Test basic initialization."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.mirs.xr.open_dataset') as od:
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
            ([METOP_FILE], DS_IDS),
            ([NPP_MIRS_L2_SWATH], DS_IDS),
            ([OTHER_MIRS_L2_SWATH], DS_IDS),
        ]
    )
    def test_available_datasets(self, filenames, expected_datasets):
        """Test that variables are dynamically discovered."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.mirs.xr.open_dataset') as od:
            od.side_effect = fake_open_dataset
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
            r.create_filehandlers(loadables)
            avails = list(r.available_dataset_names)
            for var_name in expected_datasets:
                assert var_name in avails

    @staticmethod
    def _check_area(data_arr):
        from pyresample.geometry import SwathDefinition
        area = data_arr.attrs['area']
        assert isinstance(area, SwathDefinition)

    @staticmethod
    def _check_fill(data_arr):
        assert '_FillValue' not in data_arr.attrs
        if np.issubdtype(data_arr.dtype, np.floating):
            # we started with float32, it should stay that way
            assert data_arr.dtype.type == np.float64

    @staticmethod
    def _check_valid_range(data_arr, test_valid_range):
        # valid_range is popped out of data_arr.attrs when it is applied
        assert 'valid_range' not in data_arr.attrs
        assert data_arr.data.min() >= test_valid_range[0]
        assert data_arr.data.max() <= test_valid_range[1]

    @staticmethod
    def _check_fill_value(data_arr, test_fill_value):
        assert '_FillValue' not in data_arr.attrs
        assert not (data_arr.data == test_fill_value).any()

    @staticmethod
    def _check_attrs(data_arr, platform_name):
        attrs = data_arr.attrs
        assert 'scale_factor' not in attrs
        assert 'platform_name' in attrs
        assert attrs['platform_name'] == platform_name
        assert attrs['start_time'] == START_TIME
        assert attrs['end_time'] == END_TIME

    @pytest.mark.parametrize(
        ("filenames", "loadable_ids", "platform_name"),
        [
            ([METOP_FILE], TEST_VARS, "metop-a"),
            ([NPP_MIRS_L2_SWATH], TEST_VARS, "npp"),
            ([N20_MIRS_L2_SWATH], TEST_VARS, "noaa-20"),
            ([OTHER_MIRS_L2_SWATH], TEST_VARS, "gpm"),
        ]
    )
    @pytest.mark.parametrize('reader_kw', [{}, {'limb_correction': False}])
    def test_basic_load(self, filenames, loadable_ids,
                        platform_name, reader_kw):
        """Test that variables are loaded properly."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.mirs.xr.open_dataset') as od:
            od.side_effect = fake_open_dataset
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames(filenames)
            r.create_filehandlers(loadables,  fh_kwargs=reader_kw)
            with mock.patch('satpy.readers.mirs.read_atms_coeff_to_string') as \
                    fd, mock.patch('satpy.readers.mirs.retrieve'):
                fd.side_effect = fake_coeff_from_fn
                loaded_data_arrs = r.load(loadable_ids)
            assert len(loaded_data_arrs) == len(loadable_ids)

            test_data = fake_open_dataset(filenames[0])
            for _data_id, data_arr in loaded_data_arrs.items():
                data_arr = data_arr.compute()
                var_name = data_arr.attrs["name"]
                if var_name not in ['latitude', 'longitude']:
                    self._check_area(data_arr)
                self._check_fill(data_arr)
                self._check_attrs(data_arr, platform_name)

                input_fake_data = test_data['BT'] if "btemp" in var_name \
                    else test_data[var_name]
                if "valid_range" in input_fake_data.attrs:
                    valid_range = input_fake_data.attrs["valid_range"]
                    self._check_valid_range(data_arr, valid_range)
                if "_FillValue" in input_fake_data.attrs:
                    fill_value = input_fake_data.attrs["_FillValue"]
                    self._check_fill_value(data_arr, fill_value)

                sensor = data_arr.attrs['sensor']
                if reader_kw.get('limb_correction', True) and sensor == 'atms':
                    fd.assert_called()
                else:
                    fd.assert_not_called()
                assert data_arr.attrs['units'] == DEFAULT_UNITS[var_name]
