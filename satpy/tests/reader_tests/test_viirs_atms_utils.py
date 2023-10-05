#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Satpy Developers


# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Test common VIIRS/ATMS SDR reader functions."""

import logging

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.readers.viirs_atms_sdr_base import _get_file_units, _get_scale_factors_for_units
from satpy.tests.utils import make_dataid

DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)


def test_get_file_units(caplog):
    """Test get the file-units from the dataset info."""
    did = make_dataid(name='some_variable', modifiers=())
    ds_info = {'file_units': None}
    with caplog.at_level(logging.DEBUG):
        file_units = _get_file_units(did, ds_info)

    assert file_units is None
    log_output = "Unknown units for file key 'DataID(name='some_variable', modifiers=())'"
    assert log_output in caplog.text


def test_get_scale_factors_for_units_unsupported_units():
    """Test get scale factors for units, when units are not supported."""
    factors = xr.DataArray(da.from_array(DEFAULT_FILE_FACTORS, chunks=1))
    file_units = 'unknown unit'
    output_units = '%'
    with pytest.raises(ValueError) as exec_info:
        _ = _get_scale_factors_for_units(factors, file_units, output_units)

    expected = "Don't know how to convert 'unknown unit' to '%'"
    assert str(exec_info.value) == expected


def test_get_scale_factors_for_units_reflectances(caplog):
    """Test get scale factors for units, when variable is supposed to be a reflectance."""
    factors = xr.DataArray(da.from_array(DEFAULT_FILE_FACTORS, chunks=1))
    file_units = '1'
    output_units = '%'
    with caplog.at_level(logging.DEBUG):
        retv = _get_scale_factors_for_units(factors, file_units, output_units)

    log_output = "Adjusting scaling factors to convert '1' to '%'"
    assert log_output in caplog.text
    np.testing.assert_allclose(retv, np.array([200., 100.]))


def test_get_scale_factors_for_units_tbs(caplog):
    """Test get scale factors for units, when variable is supposed to be a brightness temperature."""
    factors = xr.DataArray(da.from_array(DEFAULT_FILE_FACTORS, chunks=1))
    file_units = 'W cm-2 sr-1'
    output_units = 'W m-2 sr-1'
    with caplog.at_level(logging.DEBUG):
        retv = _get_scale_factors_for_units(factors, file_units, output_units)

    log_output = "Adjusting scaling factors to convert 'W cm-2 sr-1' to 'W m-2 sr-1'"
    assert log_output in caplog.text
    np.testing.assert_allclose(retv, np.array([20000., 10000.]))
