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
"""Tests for the 'fci_l1c_nc' reader."""
import contextlib
import logging
import os
from typing import Dict, List, Union
from unittest import mock

import dask.array as da
import numpy as np
import numpy.testing
import pytest
import xarray as xr
from netCDF4 import default_fillvals
from pytest_lazyfixture import lazy_fixture

from satpy.readers.fci_l1c_nc import FCIL1cNCFileHandler
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - caplog

GRID_TYPE_INFO_FOR_TEST_CONTENT = {
    '500m': {
        'nrows': 400,
        'ncols': 22272,
        'scale_factor': 1.39717881644274e-05,
        'add_offset': 1.55596818893146e-01,
    },
    '1km': {
        'nrows': 200,
        'ncols': 11136,
        'scale_factor': 2.79435763233999e-05,
        'add_offset': 1.55603804756852e-01,
    },
    '2km': {
        'nrows': 100,
        'ncols': 5568,
        'scale_factor': 5.58871526031607e-05,
        'add_offset': 1.55617776423501e-01,
    },
}


# ----------------------------------------------------
# Filehandlers preparation ---------------------------
# ----------------------------------------------------

class FakeH5Variable:
    """Class for faking h5netcdf.Variable class."""

    def __init__(self, data, dims=(), attrs=None):
        """Initialize the class."""
        self.dimensions = dims
        self.name = "name"
        self.attrs = attrs if attrs else {}
        self.dtype = None
        self._data = data
        self._set_meta()

    def _set_meta(self):
        if hasattr(self._data, "dtype"):
            self.dtype = self._data.dtype

    def __array__(self):
        """Get the array data.."""
        return self._data.__array__()

    def __getitem__(self, key):
        """Get item for the key."""
        return self._data[key]

    @property
    def shape(self):
        """Get the shape."""
        return self._data.shape

    @property
    def ndim(self):
        """Get the number of dimensions."""
        return self._data.ndim


def _get_test_calib_for_channel_ir(data, meas_path):
    from pyspectral.blackbody import C_SPEED as c
    from pyspectral.blackbody import H_PLANCK as h
    from pyspectral.blackbody import K_BOLTZMANN as k
    data[meas_path + "/radiance_to_bt_conversion_coefficient_wavenumber"] = FakeH5Variable(da.array(955))
    data[meas_path + "/radiance_to_bt_conversion_coefficient_a"] = FakeH5Variable(da.array(1))
    data[meas_path + "/radiance_to_bt_conversion_coefficient_b"] = FakeH5Variable(da.array(0.4))
    data[meas_path + "/radiance_to_bt_conversion_constant_c1"] = FakeH5Variable(da.array(1e11 * 2 * h * c ** 2))
    data[meas_path + "/radiance_to_bt_conversion_constant_c2"] = FakeH5Variable(da.array(1e2 * h * c / k))
    return data


def _get_test_calib_for_channel_vis(data, meas):
    data["state/celestial/earth_sun_distance"] = FakeH5Variable(
        da.repeat(da.array([149597870.7]), 6000), dims=("x"))
    data[meas + "/channel_effective_solar_irradiance"] = FakeH5Variable(da.array(50))
    return data


def _get_test_calib_data_for_channel(data, ch_str):
    meas_path = "data/{:s}/measured".format(ch_str)
    if ch_str.startswith("ir") or ch_str.startswith("wv"):
        _get_test_calib_for_channel_ir(data, meas_path)
    elif ch_str.startswith("vis") or ch_str.startswith("nir"):
        _get_test_calib_for_channel_vis(data, meas_path)
    data[meas_path + "/radiance_unit_conversion_coefficient"] = xr.DataArray(da.array(1234.56))


def _get_test_image_data_for_channel(data, ch_str, n_rows_cols):
    ch_path = "data/{:s}/measured/effective_radiance".format(ch_str)

    common_attrs = {
        "scale_factor": 5,
        "add_offset": 10,
        "long_name": "Effective Radiance",
        "units": "mW.m-2.sr-1.(cm-1)-1",
        "ancillary_variables": "pixel_quality"
    }
    if "38" in ch_path:
        fire_line = da.ones((1, n_rows_cols[1]), dtype="uint16", chunks=1024) * 5000
        data_without_fires = da.ones((n_rows_cols[0] - 1, n_rows_cols[1]), dtype="uint16", chunks=1024)
        d = FakeH5Variable(
            da.concatenate([fire_line, data_without_fires], axis=0),
            dims=('y', 'x'),
            attrs={
                "valid_range": [0, 8191],
                "warm_scale_factor": 2,
                "warm_add_offset": -300,
                **common_attrs
            }
        )
    else:
        d = FakeH5Variable(
            da.ones(n_rows_cols, dtype="uint16", chunks=1024),
            dims=("y", "x"),
            attrs={
                "valid_range": [0, 4095],
                "warm_scale_factor": 1,
                "warm_add_offset": 0,
                **common_attrs
            }
        )

    data[ch_path] = d
    data[ch_path + '/shape'] = n_rows_cols


def _get_test_segment_position_for_channel(data, ch_str, n_rows_cols):
    pos = "data/{:s}/measured/{:s}_position_{:s}"
    data[pos.format(ch_str, "start", "row")] = FakeH5Variable(da.array(1))
    data[pos.format(ch_str, "start", "column")] = FakeH5Variable(da.array(1))
    data[pos.format(ch_str, "end", "row")] = FakeH5Variable(da.array(n_rows_cols[0]))
    data[pos.format(ch_str, "end", "column")] = FakeH5Variable(da.array(n_rows_cols[1]))


def _get_test_index_map_for_channel(data, ch_str, n_rows_cols):
    index_map_path = "data/{:s}/measured/index_map".format(ch_str)
    data[index_map_path] = xr.DataArray(
        (da.ones(n_rows_cols)) * 110, dims=("y", "x"))


def _get_test_pixel_quality_for_channel(data, ch_str, n_rows_cols):
    qual_path = "data/{:s}/measured/pixel_quality".format(ch_str)
    data[qual_path] = xr.DataArray((da.ones(n_rows_cols)) * 3, dims=("y", "x"))


def _get_test_geolocation_for_channel(data, ch_str, grid_type, n_rows_cols):
    x_path = "data/{:s}/measured/x".format(ch_str)
    data[x_path] = xr.DataArray(
        da.arange(1, n_rows_cols[1] + 1, dtype=np.dtype("uint16")),
        dims=("x",),
        attrs={
            "scale_factor": -GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['scale_factor'],
            "add_offset": GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['add_offset'],
        }
    )

    y_path = "data/{:s}/measured/y".format(ch_str)
    data[y_path] = xr.DataArray(
        da.arange(1, n_rows_cols[0] + 1, dtype=np.dtype("uint16")),
        dims=("y",),
        attrs={
            "scale_factor": GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['scale_factor'],
            "add_offset": -GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['add_offset'],
        }
    )


def _get_test_content_areadef():
    data = {}

    proj = "data/mtg_geos_projection"

    attrs = {
        "sweep_angle_axis": "y",
        "perspective_point_height": "35786400.0",
        "semi_major_axis": "6378137.0",
        "longitude_of_projection_origin": "0.0",
        "inverse_flattening": "298.257223563",
        "units": "m"}
    data[proj] = xr.DataArray(
        0,
        dims=(),
        attrs=attrs)

    # also set attributes cached, as this may be how they are accessed with
    # the NetCDF4FileHandler
    for (k, v) in attrs.items():
        data[proj + "/attr/" + k] = v

    return data


def _get_test_content_aux_data():
    from satpy.readers.fci_l1c_nc import AUX_DATA
    data = {}
    indices_dim = 6000
    for key, value in AUX_DATA.items():
        # skip population of earth_sun_distance as this is already defined for reflectance calculation
        if key == 'earth_sun_distance':
            continue
        data[value] = xr.DataArray(da.arange(indices_dim, dtype="float32"), dims=("index"))

    # compute the last data entry to simulate the FCI caching
    # data[list(AUX_DATA.values())[-1]] = data[list(AUX_DATA.values())[-1]].compute()

    data['index'] = xr.DataArray(
        da.ones(indices_dim, dtype="uint16") * 100, dims=("index"))
    return data


def _get_global_attributes():
    data = {}
    attrs = {"platform": "MTI1"}
    for (k, v) in attrs.items():
        data["attr/" + k] = v
    return data


def _get_test_content_for_channel(ch_str, grid_type):
    nrows = GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows']
    ncols = GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols']
    n_rows_cols = (nrows, ncols)

    data = {}

    _get_test_image_data_for_channel(data, ch_str, n_rows_cols)
    _get_test_calib_data_for_channel(data, ch_str)
    _get_test_geolocation_for_channel(data, ch_str, grid_type, n_rows_cols)
    _get_test_pixel_quality_for_channel(data, ch_str, n_rows_cols)
    _get_test_index_map_for_channel(data, ch_str, n_rows_cols)
    _get_test_segment_position_for_channel(data, ch_str, n_rows_cols)

    return data


class FakeFCIFileHandlerBase(FakeNetCDF4FileHandler):
    """Class for faking the NetCDF4 Filehandler."""

    cached_file_content: Dict[str, xr.DataArray] = {}
    # overwritten by FDHSI and HRFI FIle Handlers
    chan_patterns: Dict[str, Dict[str, Union[List[int], str]]] = {}

    def _get_test_content_all_channels(self):
        data = {}
        for pat in self.chan_patterns:
            for ch in self.chan_patterns[pat]['channels']:
                data.update(_get_test_content_for_channel(pat.format(ch), self.chan_patterns[pat]['grid_type']))
        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        """Get the content of the test data."""
        D = {}
        D.update(self._get_test_content_all_channels())
        D.update(_get_test_content_areadef())
        D.update(_get_test_content_aux_data())
        D.update(_get_global_attributes())
        return D


class FakeFCIFileHandlerFDHSI(FakeFCIFileHandlerBase):
    """Mock FDHSI data."""

    chan_patterns = {
        "vis_{:>02d}": {'channels': [4, 5, 6, 8, 9],
                        'grid_type': '1km'},
        "nir_{:>02d}": {'channels': [13, 16, 22],
                        'grid_type': '1km'},
        "ir_{:>02d}": {'channels': [38, 87, 97, 105, 123, 133],
                       'grid_type': '2km'},
        "wv_{:>02d}": {'channels': [63, 73],
                       'grid_type': '2km'},
    }


class FakeFCIFileHandlerWithBadData(FakeFCIFileHandlerFDHSI):
    """Mock bad data."""

    def _get_test_content_all_channels(self):
        data = super()._get_test_content_all_channels()
        v = xr.DataArray(default_fillvals["f4"])

        data.update({"data/ir_105/measured/radiance_to_bt_conversion_coefficient_wavenumber": v,
                     "data/ir_105/measured/radiance_to_bt_conversion_coefficient_a": v,
                     "data/ir_105/measured/radiance_to_bt_conversion_coefficient_b": v,
                     "data/ir_105/measured/radiance_to_bt_conversion_constant_c1": v,
                     "data/ir_105/measured/radiance_to_bt_conversion_constant_c2": v,
                     "data/vis_06/measured/channel_effective_solar_irradiance": v})

        return data


class FakeFCIFileHandlerWithBadIDPFData(FakeFCIFileHandlerFDHSI):
    """Mock bad data for IDPF TO-DO's."""

    def _get_test_content_all_channels(self):
        data = super()._get_test_content_all_channels()
        data['data/vis_06/measured/x'].attrs['scale_factor'] *= -1
        data['data/vis_06/measured/x'].attrs['scale_factor'] = \
            np.float32(data['data/vis_06/measured/x'].attrs['scale_factor'])
        data['data/vis_06/measured/x'].attrs['add_offset'] = \
            np.float32(data['data/vis_06/measured/x'].attrs['add_offset'])
        data['data/vis_06/measured/y'].attrs['scale_factor'] = \
            np.float32(data['data/vis_06/measured/y'].attrs['scale_factor'])
        data['data/vis_06/measured/y'].attrs['add_offset'] = \
            np.float32(data['data/vis_06/measured/y'].attrs['add_offset'])

        data["state/celestial/earth_sun_distance"] = xr.DataArray(da.repeat(da.array([30000000]), 6000))
        return data


class FakeFCIFileHandlerHRFI(FakeFCIFileHandlerBase):
    """Mock HRFI data."""

    chan_patterns = {
        "vis_{:>02d}_hr": {'channels': [6],
                           'grid_type': '500m'},
        "nir_{:>02d}_hr": {'channels': [22],
                           'grid_type': '500m'},
        "ir_{:>02d}_hr": {'channels': [38, 105],
                          'grid_type': '1km'},
    }


# ----------------------------------------------------
# Fixtures preparation -------------------------------
# ----------------------------------------------------

@pytest.fixture
def reader_configs():
    """Return reader configs for FCI."""
    from satpy._config import config_search_paths
    return config_search_paths(
        os.path.join("readers", "fci_l1c_nc.yaml"))


def _get_reader_with_filehandlers(filenames, reader_configs):
    from satpy.readers import load_reader
    reader = load_reader(reader_configs)
    loadables = reader.select_files_from_pathnames(filenames)
    reader.create_filehandlers(loadables)
    clear_cache(reader)
    return reader


def clear_cache(reader):
    """Clear the cache for file handlres in reader."""
    for key in reader.file_handlers:
        fhs = reader.file_handlers[key]
        for fh in fhs:
            fh.cached_file_content = {}


_chans_fdhsi = {"solar": ["vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
                          "nir_13", "nir_16", "nir_22"],
                "solar_grid_type": ["1km"] * 8,
                "terran": ["ir_38", "wv_63", "wv_73", "ir_87", "ir_97", "ir_105",
                           "ir_123", "ir_133"],
                "terran_grid_type": ["2km"] * 8}

_chans_hrfi = {"solar": ["vis_06", "nir_22"],
               "solar_grid_type": ["500m"] * 2,
               "terran": ["ir_38", "ir_105"],
               "terran_grid_type": ["1km"] * 2}

_test_filenames = {'fdhsi': [
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
    "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
    "20170410113925_20170410113934_N__C_0070_0067.nc"
],
    'hrfi': [
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--"
        "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
        "20170410113925_20170410113934_N__C_0070_0067.nc"
    ]
}


@contextlib.contextmanager
def mocked_basefilehandler(filehandler):
    """Mock patch the base class of the FCIL1cNCFileHandler with the content of our fake files (filehandler)."""
    p = mock.patch.object(FCIL1cNCFileHandler, "__bases__", (filehandler,))
    with p:
        p.is_local = True
        yield


@pytest.fixture
def FakeFCIFileHandlerFDHSI_fixture():
    """Get a fixture for the fake FDHSI filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerFDHSI):
        param_dict = {
            'filetype': 'fci_l1c_fdhsi',
            'channels': _chans_fdhsi,
            'filenames': _test_filenames['fdhsi']
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerHRFI_fixture():
    """Get a fixture for the fake HRFI filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerHRFI):
        param_dict = {
            'filetype': 'fci_l1c_hrfi',
            'channels': _chans_hrfi,
            'filenames': _test_filenames['hrfi']
        }
        yield param_dict


# ----------------------------------------------------
# Tests ----------------------------------------------
# ----------------------------------------------------


class TestFCIL1cNCReader:
    """Test FCI L1c NetCDF reader with nominal data."""

    fh_param_for_filetype = {'hrfi': {'channels': _chans_hrfi,
                                      'filenames': _test_filenames['hrfi']},
                             'fdhsi': {'channels': _chans_fdhsi,
                                       'filenames': _test_filenames['fdhsi']}}

    @pytest.mark.parametrize('filenames', [_test_filenames['fdhsi'], _test_filenames['hrfi']])
    def test_file_pattern(self, reader_configs, filenames):
        """Test file pattern matching."""
        from satpy.readers import load_reader

        reader = load_reader(reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 1

    @pytest.mark.parametrize('filenames', [_test_filenames['fdhsi'][0].replace('BODY', 'TRAIL'),
                                           _test_filenames['hrfi'][0].replace('BODY', 'TRAIL')])
    def test_file_pattern_for_TRAIL_file(self, reader_configs, filenames):
        """Test file pattern matching for TRAIL files, which should not be picked up."""
        from satpy.readers import load_reader

        reader = load_reader(reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 0

    @pytest.mark.parametrize('fh_param,expected_res_n', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), 16),
                                                         (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), 4)])
    def test_load_counts(self, reader_configs, fh_param,
                         expected_res_n):
        """Test loading with counts."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(
            [make_dataid(name=name, calibration="counts") for name in
             fh_param['channels']["solar"] + fh_param['channels']["terran"]], pad_data=False)
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param['channels']["solar"] + fh_param['channels']["terran"],
                                 fh_param['channels']["solar_grid_type"] +
                                 fh_param['channels']["terran_grid_type"]):
            assert res[ch].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows'],
                                     GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols'])
            assert res[ch].dtype == np.uint16
            assert res[ch].attrs["calibration"] == "counts"
            assert res[ch].attrs["units"] == "count"
            if ch == 'ir_38':
                numpy.testing.assert_array_equal(res[ch][-1], 1)
                numpy.testing.assert_array_equal(res[ch][0], 5000)
            else:
                numpy.testing.assert_array_equal(res[ch], 1)

    @pytest.mark.parametrize('fh_param,expected_res_n', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), 16),
                                                         (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), 4)])
    def test_load_radiance(self, reader_configs, fh_param,
                           expected_res_n):
        """Test loading with radiance."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(
            [make_dataid(name=name, calibration="radiance") for name in
             fh_param['channels']["solar"] + fh_param['channels']["terran"]], pad_data=False)
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param['channels']["solar"] + fh_param['channels']["terran"],
                                 fh_param['channels']["solar_grid_type"] +
                                 fh_param['channels']["terran_grid_type"]):
            assert res[ch].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows'],
                                     GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols'])
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "radiance"
            assert res[ch].attrs["units"] == 'mW m-2 sr-1 (cm-1)-1'
            assert res[ch].attrs["radiance_unit_conversion_coefficient"] == 1234.56
            if ch == 'ir_38':
                numpy.testing.assert_array_equal(res[ch][-1], 15)
                numpy.testing.assert_array_equal(res[ch][0], 9700)
            else:
                numpy.testing.assert_array_equal(res[ch], 15)

    @pytest.mark.parametrize('fh_param,expected_res_n', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), 8),
                                                         (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), 2)])
    def test_load_reflectance(self, reader_configs, fh_param,
                              expected_res_n):
        """Test loading with reflectance."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(
            [make_dataid(name=name, calibration="reflectance") for name in
             fh_param['channels']["solar"]], pad_data=False)
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param['channels']["solar"], fh_param['channels']["solar_grid_type"]):
            assert res[ch].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows'],
                                     GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols'])
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "reflectance"
            assert res[ch].attrs["units"] == "%"
            numpy.testing.assert_array_almost_equal(res[ch], 100 * 15 * 1 * np.pi / 50)

    @pytest.mark.parametrize('fh_param,expected_res_n', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), 8),
                                                         (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), 2)])
    def test_load_bt(self, reader_configs, caplog, fh_param,
                     expected_res_n):
        """Test loading with bt."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        with caplog.at_level(logging.WARNING):
            res = reader.load(
                [make_dataid(name=name, calibration="brightness_temperature") for
                 name in fh_param['channels']["terran"]], pad_data=False)
            assert caplog.text == ""
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param['channels']["terran"], fh_param['channels']["terran_grid_type"]):
            assert res[ch].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows'],
                                     GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols'])
            assert res[ch].dtype == np.float64
            assert res[ch].attrs["calibration"] == "brightness_temperature"
            assert res[ch].attrs["units"] == "K"

            if ch == 'ir_38':
                numpy.testing.assert_array_almost_equal(res[ch][-1], 209.68274099)
                numpy.testing.assert_array_almost_equal(res[ch][0], 1888.851296)
            else:
                numpy.testing.assert_array_almost_equal(res[ch], 209.68274099)

    @pytest.mark.parametrize('fh_param', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture')),
                                          (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'))])
    def test_orbital_parameters_attr(self, reader_configs, fh_param):
        """Test the orbital parameter attribute."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(
            [make_dataid(name=name) for name in
             fh_param['channels']["solar"] + fh_param['channels']["terran"]], pad_data=False)

        for ch in fh_param['channels']["solar"] + fh_param['channels']["terran"]:
            assert res[ch].attrs["orbital_parameters"] == {
                'satellite_actual_longitude': np.mean(np.arange(6000)),
                'satellite_actual_latitude': np.mean(np.arange(6000)),
                'satellite_actual_altitude': np.mean(np.arange(6000)),
                'satellite_nominal_longitude': 0.0,
                'satellite_nominal_latitude': 0,
                'satellite_nominal_altitude': 35786400.0,
                'projection_longitude': 0.0,
                'projection_latitude': 0,
                'projection_altitude': 35786400.0,
            }

    expected_pos_info_for_filetype = {
        'fdhsi': {'1km': {'start_position_row': 1,
                          'end_position_row': 200,
                          'segment_height': 200,
                          'grid_width': 11136},
                  '2km': {'start_position_row': 1,
                          'end_position_row': 100,
                          'segment_height': 100,
                          'grid_width': 5568}},
        'hrfi': {'500m': {'start_position_row': 1,
                          'end_position_row': 400,
                          'segment_height': 400,
                          'grid_width': 22272},
                 '1km': {'start_position_row': 1,
                         'end_position_row': 200,
                         'grid_width': 11136,
                         'segment_height': 200}}
    }

    @pytest.mark.parametrize('fh_param, expected_pos_info', [
        (lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), expected_pos_info_for_filetype['fdhsi']),
        (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), expected_pos_info_for_filetype['hrfi'])
    ])
    def test_get_segment_position_info(self, reader_configs, fh_param, expected_pos_info):
        """Test the segment position info method."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        for filetype_handler in list(reader.file_handlers.values())[0]:
            segpos_info = filetype_handler.get_segment_position_info()
            assert segpos_info == expected_pos_info

    @pytest.mark.parametrize('fh_param,expected_res_n', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), 16),
                                                         (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), 4)])
    def test_load_index_map(self, reader_configs, fh_param, expected_res_n):
        """Test loading of index_map."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(
            [name + '_index_map' for name in
             fh_param['channels']["solar"] + fh_param['channels']["terran"]], pad_data=False)
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param['channels']["solar"] + fh_param['channels']["terran"],
                                 fh_param['channels']["solar_grid_type"] +
                                 fh_param['channels']["terran_grid_type"]):
            assert res[ch + '_index_map'].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows'],
                                                    GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols'])
            numpy.testing.assert_array_equal(res[ch + '_index_map'][1, 1], 110)

    @pytest.mark.parametrize('fh_param', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture')),
                                          (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'))])
    def test_load_aux_data(self, reader_configs, fh_param):
        """Test loading of auxiliary data."""
        from satpy.readers.fci_l1c_nc import AUX_DATA
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load([fh_param['channels']['solar'][0] + '_' + key for key in AUX_DATA.keys()],
                          pad_data=False)
        grid_type = fh_param['channels']['solar_grid_type'][0]
        for aux in [fh_param['channels']['solar'][0] + '_' + key for key in AUX_DATA.keys()]:
            assert res[aux].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows'],
                                      GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols'])
            if aux == fh_param['channels']['solar'][0] + '_earth_sun_distance':
                numpy.testing.assert_array_equal(res[aux][1, 1], 149597870.7)
            else:
                numpy.testing.assert_array_equal(res[aux][1, 1], 10)

    @pytest.mark.parametrize('fh_param,expected_res_n', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), 16),
                                                         (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), 4)])
    def test_load_quality_only(self, reader_configs, fh_param, expected_res_n):
        """Test that loading quality only works."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(
            [name + '_pixel_quality' for name in
             fh_param['channels']["solar"] + fh_param['channels']["terran"]], pad_data=False)
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param['channels']["solar"] + fh_param['channels']["terran"],
                                 fh_param['channels']["solar_grid_type"] +
                                 fh_param['channels']["terran_grid_type"]):
            assert res[ch + '_pixel_quality'].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['nrows'],
                                                        GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]['ncols'])
            numpy.testing.assert_array_equal(res[ch + '_pixel_quality'][1, 1], 3)
            assert res[ch + '_pixel_quality'].attrs["name"] == ch + '_pixel_quality'

    @pytest.mark.parametrize('fh_param', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture')),
                                          (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'))])
    def test_platform_name(self, reader_configs, fh_param):
        """Test that platform name is exposed.

        Test that the FCI reader exposes the platform name.  Corresponds
        to GH issue 1014.
        """
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(["vis_06"], pad_data=False)
        assert res["vis_06"].attrs["platform_name"] == "MTG-I1"

    @pytest.mark.parametrize('fh_param, expected_area', [
        (lazy_fixture('FakeFCIFileHandlerFDHSI_fixture'), ['mtg_fci_fdss_1km', 'mtg_fci_fdss_2km']),
        (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'), ['mtg_fci_fdss_500m', 'mtg_fci_fdss_1km']),
    ])
    def test_area_definition_computation(self, reader_configs, fh_param, expected_area):
        """Test that the geolocation computation is correct."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)
        res = reader.load(['ir_105', 'vis_06'], pad_data=False)

        # test that area_ids are harmonisation-conform <platform>_<instrument>_<service>_<resolution>
        assert res['vis_06'].attrs['area'].area_id == expected_area[0]
        assert res['ir_105'].attrs['area'].area_id == expected_area[1]

        area_def = res['ir_105'].attrs['area']
        # test area extents computation
        np.testing.assert_array_almost_equal(np.array(area_def.area_extent),
                                             np.array([-5567999.994203, -5367999.994411,
                                                       5567999.994203, -5567999.994203]),
                                             decimal=2)

        # check that the projection is read in properly
        assert area_def.crs.coordinate_operation.method_name == 'Geostationary Satellite (Sweep Y)'
        assert area_def.crs.coordinate_operation.params[0].value == 0.0  # projection origin longitude
        assert area_def.crs.coordinate_operation.params[1].value == 35786400.0  # projection height
        assert area_def.crs.ellipsoid.semi_major_metre == 6378137.0
        assert area_def.crs.ellipsoid.inverse_flattening == 298.257223563
        assert area_def.crs.ellipsoid.is_semi_minor_computed

    @pytest.mark.parametrize('fh_param', [(lazy_fixture('FakeFCIFileHandlerFDHSI_fixture')),
                                          (lazy_fixture('FakeFCIFileHandlerHRFI_fixture'))])
    def test_excs(self, reader_configs, fh_param):
        """Test that exceptions are raised where expected."""
        reader = _get_reader_with_filehandlers(fh_param['filenames'], reader_configs)

        with pytest.raises(ValueError):
            reader.file_handlers[fh_param['filetype']][0].get_dataset(make_dataid(name="invalid"), {})
        with pytest.raises(ValueError):
            reader.file_handlers[fh_param['filetype']][0].get_dataset(
                make_dataid(name="ir_123", calibration="unknown"),
                {"units": "unknown"})

    def test_load_composite(self):
        """Test that composites are loadable."""
        # when dedicated composites for FCI are implemented in satpy,
        # this method should probably move to a dedicated class and module
        # in the tests.compositor_tests package

        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        comps, mods = load_compositor_configs_for_sensors(['fci'])
        assert len(comps["fci"]) > 0
        assert len(mods["fci"]) > 0


class TestFCIL1cNCReaderBadData:
    """Test the FCI L1c NetCDF Reader for bad data input."""

    def test_handling_bad_data_ir(self, reader_configs, caplog):
        """Test handling of bad IR data."""
        with mocked_basefilehandler(FakeFCIFileHandlerWithBadData):
            reader = _get_reader_with_filehandlers(_test_filenames['fdhsi'], reader_configs)
            with caplog.at_level(logging.ERROR):
                reader.load([make_dataid(
                    name="ir_105",
                    calibration="brightness_temperature")], pad_data=False)
                assert "cannot produce brightness temperature" in caplog.text

    def test_handling_bad_data_vis(self, reader_configs, caplog):
        """Test handling of bad VIS data."""
        with mocked_basefilehandler(FakeFCIFileHandlerWithBadData):
            reader = _get_reader_with_filehandlers(_test_filenames['fdhsi'], reader_configs)
            with caplog.at_level(logging.ERROR):
                reader.load([make_dataid(
                    name="vis_06",
                    calibration="reflectance")], pad_data=False)
                assert "cannot produce reflectance" in caplog.text


class TestFCIL1cNCReaderBadDataFromIDPF:
    """Test the FCI L1c NetCDF Reader for bad data input, specifically the IDPF issues."""

    def test_handling_bad_earthsun_distance(self, reader_configs):
        """Test handling of bad earth-sun distance data."""
        with mocked_basefilehandler(FakeFCIFileHandlerWithBadIDPFData):
            reader = _get_reader_with_filehandlers(_test_filenames['fdhsi'], reader_configs)
            res = reader.load([make_dataid(name=["vis_06"], calibration="reflectance")], pad_data=False)

            numpy.testing.assert_array_almost_equal(res["vis_06"], 100 * 15 * 1 * np.pi / 50)

    def test_bad_xy_coords(self, reader_configs):
        """Test that the geolocation computation is correct."""
        with mocked_basefilehandler(FakeFCIFileHandlerWithBadIDPFData):
            reader = _get_reader_with_filehandlers(_test_filenames['fdhsi'], reader_configs)
            res = reader.load(['vis_06'], pad_data=False)

            area_def = res['vis_06'].attrs['area']
            # test area extents computation
            np.testing.assert_array_almost_equal(np.array(area_def.area_extent),
                                                 np.array([-5568000.227139, -5368000.221262,
                                                           5568000.100073, -5568000.227139]),
                                                 decimal=2)
