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
"""Module for testing the satpy.readers.viirs_l2_jrr module.

Note: This is adapted from the test_slstr_l2.py code.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample import SwathDefinition

from satpy.readers.viirs_edr import VIIRSJRRFileHandler

I_COLS = 64  # real-world 6400
I_ROWS = 32  # one scan
M_COLS = 32  # real-world 3200
M_ROWS = 16  # one scan
START_TIME = datetime(2023, 5, 30, 17, 55, 41, 0)
END_TIME = datetime(2023, 5, 30, 17, 57, 5, 0)


@pytest.fixture(scope="module")
def surface_reflectance_file(tmp_path_factory) -> Path:
    """Generate fake surface reflectance EDR file."""
    tmp_path = tmp_path_factory.mktemp("viirs_edr_tmp")
    fn = f"SurfRefl_v1r2_npp_s{START_TIME:%Y%m%d%H%M%S}0_e{END_TIME:%Y%m%d%H%M%S}0_c202305302025590.nc"
    file_path = tmp_path / fn
    sr_vars = _create_surf_refl_variables()
    ds = _create_fake_dataset(sr_vars)
    ds.to_netcdf(file_path)
    return file_path


def _create_fake_dataset(vars_dict: dict[str, xr.DataArray]) -> xr.Dataset:
    ds = xr.Dataset(
        vars_dict,
        attrs={}
    )
    return ds


def _create_surf_refl_variables() -> dict[str, xr.DataArray]:
    dim_y_750 = "Along_Track_750m"
    dim_x_750 = "Along_Scan_750m"
    m_dims = (dim_y_750, dim_x_750)
    dim_y_375 = "Along_Track_375m"
    dim_x_375 = "Along_Scan_375m"
    i_dims = (dim_y_375, dim_x_375)

    lon_attrs = {"standard_name": "longitude", "units": "degrees_east", "_FillValue": -999.9}
    lat_attrs = {"standard_name": "latitude", "units": "degrees_north", "_FillValue": -999.9}
    sr_attrs = {"units": "unitless", "_FillValue": -9999, "scale_factor": 0.0001, "add_offset": 0.0}

    i_data = np.zeros((I_ROWS, I_COLS), dtype=np.float32)
    m_data = np.zeros((M_ROWS, M_COLS), dtype=np.float32)
    data_arrs = {
        "Longitude_at_375m_resolution": xr.DataArray(i_data, dims=i_dims, attrs=lon_attrs),
        "Latitude_at_375m_resolution": xr.DataArray(i_data, dims=i_dims, attrs=lat_attrs),
        "Longitude_at_750m_resolution": xr.DataArray(i_data, dims=i_dims, attrs=lon_attrs),
        "Latitude_at_750m_resolution": xr.DataArray(i_data, dims=i_dims, attrs=lat_attrs),
        "375m Surface Reflectance Band I1": xr.DataArray(i_data, dims=i_dims, attrs=sr_attrs),
        "750m Surface Reflectance Band M1": xr.DataArray(m_data, dims=m_dims, attrs=sr_attrs),
    }
    for data_arr in data_arrs.values():
        if "scale_factor" not in data_arr.attrs:
            continue
        data_arr.encoding["dtype"] = np.int16
    return data_arrs


class TestVIIRSJRRReader:
    """Test the VIIRS JRR L2 reader."""

    def test_get_dataset_surf_refl(self, surface_reflectance_file):
        """Test retrieval of datasets."""
        from satpy import Scene
        bytes_in_m_row = 4 * 3200
        with dask.config.set({"array.chunk-size": f"{bytes_in_m_row * 4}B"}):
            scn = Scene(reader="viirs_edr", filenames=[surface_reflectance_file])
            scn.load(["surf_refl_I01", "surf_refl_M01"])
        assert scn.start_time == START_TIME
        assert scn.end_time == END_TIME
        _check_surf_refl_data_arr(scn["surf_refl_I01"])
        _check_surf_refl_data_arr(scn["surf_refl_M01"])

    @pytest.mark.parametrize(
        ("filename_platform", "exp_shortname"),
        [
            ("npp", "Suomi-NPP"),
            ("JPSS-1", "NOAA-20"),
            ("J01", "NOAA-20")
        ])
    def test_get_platformname(self, surface_reflectance_file, filename_platform, exp_shortname):
        """Test finding start and end times of granules."""
        from satpy import Scene
        new_name = str(surface_reflectance_file).replace("npp", filename_platform)
        if new_name != str(surface_reflectance_file):
            shutil.copy(surface_reflectance_file, new_name)
        scn = Scene(reader="viirs_edr", filenames=[new_name])
        scn.load(["surf_refl_I01"])
        assert scn["surf_refl_I01"].attrs["platform_name"] == exp_shortname

    @mock.patch('xarray.open_dataset')
    def test_get_dataset(self, mocked_dataset):
        """Test retrieval of datasets."""
        filename_info = {'platform_shortname': 'npp'}
        tmp = MagicMock(start_time='20191120T125002Z', stop_time='20191120T125002Z')
        xr.open_dataset.return_value = tmp
        test = VIIRSJRRFileHandler('somedir/somefile.nc', filename_info, None)
        test.nc = {'Longitude': xr.Dataset(),
                   'Latitude': xr.Dataset(),
                   'smoke_concentration': xr.Dataset(),
                   'fire_mask': xr.Dataset(),
                   }
        test.get_dataset('longitude', {'file_key': 'Longitude'})
        test.get_dataset('latitude', {'file_key': 'Latitude'})
        test.get_dataset('smoke_concentration', {'file_key': 'smoke_concentration'})
        test.get_dataset('fire_mask', {'file_key': 'fire_mask'})
        with pytest.raises(KeyError):
            test.get_dataset('erroneous dataset', {'file_key': 'erroneous dataset'})
        mocked_dataset.assert_called()


def _check_surf_refl_data_arr(data_arr: xr.DataArray) -> None:
    assert data_arr.dims == ("y", "x")
    assert isinstance(data_arr.attrs["area"], SwathDefinition)
    assert isinstance(data_arr.data, da.Array)
    assert np.issubdtype(data_arr.data.dtype, np.float32)
    is_m_band = "I" not in data_arr.attrs["name"]
    exp_shape = (M_ROWS, M_COLS) if is_m_band else (I_ROWS, I_COLS)
    assert data_arr.shape == exp_shape
    exp_row_chunks = 4 if is_m_band else 8
    assert all(c == exp_row_chunks for c in data_arr.chunks[0])
    assert data_arr.chunks[1] == (exp_shape[1],)
    assert data_arr.attrs["units"] == "1"
