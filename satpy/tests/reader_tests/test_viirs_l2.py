#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Tests for the VIIRS CSPP L2 readers."""

import os
import shutil
import tempfile

import numpy as np
import xarray as xr

from satpy import Scene

CLOUD_MASK_FILE = "JRR-CloudMask_v3r0_npp_s202212070905565_e202212070907207_c202212071932513.nc"
NUM_COLUMNS = 3200
NUM_ROWS = 768
DATASETS = ['Latitude', 'Longitude', 'CloudMask', 'CloudMaskBinary']


class JRRCloudMaskFile:
    """Write a test JRR CloudMask file."""

    def __init__(self):
        """Initialize the class."""
        self._dir = tempfile.mkdtemp()
        self.name = os.path.join(self._dir, CLOUD_MASK_FILE)

    def __enter__(self):
        """Create the file when entering the context manager."""
        self._write_file()
        return self.name

    def _write_file(self):
        dset = xr.Dataset()
        dset.attrs = _get_global_attrs()
        dset['Latitude'] = _get_lat_arr()
        dset['Longitude'] = _get_lon_arr()
        dset['CloudMask'] = _get_cloud_mask_arr()
        dset['CloudMaskBinary'] = _get_cloud_mask_binary_arr()
        dset.to_netcdf(self.name, 'w')

    def __exit__(self, type, value, traceback):
        """Delete the file when exiting the context manager."""
        shutil.rmtree(self._dir, ignore_errors=True)


def _get_global_attrs():
    return {
        'time_coverage_start': '2022-12-07T09:05:56Z',
        'time_coverage_end': '2022-12-07T09:07:20Z',
        'start_orbit_number': np.array(57573),
        'end_orbit_number': np.array(57573),
        'instrument_name': 'VIIRS',
    }


def _get_lat_arr():
    arr = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.float32)
    attrs = {
        'long_name': 'Latitude',
        'units': 'degrees_north',
        'valid_range': np.array([-90, 90], dtype=np.float32),
        '_FillValue': -999.
    }
    return xr.DataArray(arr, attrs=attrs, dims=('Rows', 'Columns'))


def _get_lon_arr():
    arr = np.zeros((NUM_ROWS, NUM_COLUMNS), dtype=np.float32)
    attrs = {
        'long_name': 'Longitude',
        'units': 'degrees_east',
        'valid_range': np.array([-180, 180], dtype=np.float32),
        '_FillValue': -999.
    }
    return xr.DataArray(arr, attrs=attrs, dims=('Rows', 'Columns'))


def _get_cloud_mask_arr():
    arr = np.random.randint(0, 4, (NUM_ROWS, NUM_COLUMNS), dtype=np.byte)
    attrs = {
        'long_name': 'Cloud Mask',
        '_FillValue': np.byte(-128),
        'valid_range': np.array([0, 3], dtype=np.byte),
        'units': '1',
        'flag_values': np.array([0, 1, 2, 3], dtype=np.byte),
        'flag_meanings': 'clear probably_clear probably_cloudy cloudy',
    }
    return xr.DataArray(arr, attrs=attrs, dims=('Rows', 'Columns'))


def _get_cloud_mask_binary_arr():
    arr = np.random.randint(0, 2, (NUM_ROWS, NUM_COLUMNS), dtype=np.byte)
    attrs = {
        'long_name': 'Cloud Mask Binary',
        '_FillValue': np.byte(-128),
        'valid_range': np.array([0, 1], dtype=np.byte),
        'units': '1',
    }
    return xr.DataArray(arr, attrs=attrs, dims=('Rows', 'Columns'))


def test_cloud_mask_read_latitude():
    """Test reading latitude dataset."""
    with JRRCloudMaskFile() as fname:
        data = _read_viirs_l2_cloud_mask_nc_data(fname, 'latitude')
        _assert_common(data)


def test_cloud_mask_read_longitude():
    """Test reading longitude dataset."""
    with JRRCloudMaskFile() as fname:
        data = _read_viirs_l2_cloud_mask_nc_data(fname, 'longitude')
        _assert_common(data)


def test_cloud_mask_read_cloud_mask():
    """Test reading cloud mask dataset."""
    with JRRCloudMaskFile() as fname:
        data = _read_viirs_l2_cloud_mask_nc_data(fname, 'cloud_mask')
        _assert_common(data)
        np.testing.assert_equal(data.attrs['flag_values'], [0, 1, 2, 3])
        assert data.attrs['flag_meanings'] == ['clear', 'probably_clear', 'probably_cloudy', 'cloudy']


def test_cloud_mas_read_binary_cloud_mask():
    """Test reading binary cloud mask dataset."""
    with JRRCloudMaskFile() as fname:
        data = _read_viirs_l2_cloud_mask_nc_data(fname, 'cloud_mask_binary')
        _assert_common(data)


def _read_viirs_l2_cloud_mask_nc_data(fname, dset_name):
    scn = Scene(reader="viirs_l2_cloud_mask_nc", filenames=[fname])
    scn.load([dset_name])
    return scn[dset_name]


def _assert_common(data):
    assert data.dims == ('y', 'x')
