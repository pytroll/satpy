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
import datetime
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
from pytest_lazy_fixtures import lf as lazy_fixture

from satpy.readers.fci_l1c_nc import FCIL1cNCFileHandler
from satpy.tests.reader_tests.test_netcdf_utils import FakeNetCDF4FileHandler
from satpy.tests.utils import make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - caplog

GRID_TYPE_INFO_FOR_TEST_CONTENT = {
    "500m": {
        "nrows": 400,
        "ncols": 22272,
        "scale_factor": 1.39717881644274e-05,
        "add_offset": 1.55596818893146e-01,
    },
    "1km": {
        "nrows": 200,
        "ncols": 11136,
        "scale_factor": 2.79435763233999e-05,
        "add_offset": 1.55603804756852e-01,
    },
    "2km": {
        "nrows": 100,
        "ncols": 5568,
        "scale_factor": 5.58871526031607e-05,
        "add_offset": 1.55617776423501e-01,
    },
    "3km": {
        "nrows": 67,
        "ncols": 3712,
        "scale_factor": 8.38307287956433e-05,
        "add_offset": 0.155631748009112,
    },
}

LIST_CHANNEL_SOLAR = ["vis_04", "vis_05", "vis_06", "vis_08", "vis_09",
                      "nir_13", "nir_16", "nir_22"]
LIST_CHANNEL_TERRAN = ["ir_38", "wv_63", "wv_73", "ir_87", "ir_97", "ir_105",
                       "ir_123", "ir_133"]
LIST_TOTAL_CHANNEL = LIST_CHANNEL_SOLAR + LIST_CHANNEL_TERRAN
LIST_RESOLUTION_VIS06_AF = ["1km", "3km"]
LIST_RESOLUTION_AF = ["3km"]
EXPECTED_POS_INFO_FOR_FILETYPE = {
    "fdhsi": {"1km": {"start_position_row": 1,
                      "end_position_row": 200,
                      "segment_height": 200,
                      "grid_width": 11136},
              "2km": {"start_position_row": 1,
                      "end_position_row": 100,
                      "segment_height": 100,
                      "grid_width": 5568}},
    "hrfi": {"500m": {"start_position_row": 1,
                      "end_position_row": 400,
                      "segment_height": 400,
                      "grid_width": 22272},
             "1km": {"start_position_row": 1,
                     "end_position_row": 200,
                     "grid_width": 11136,
                     "segment_height": 200}},
    "fci_af": {"3km": {"start_position_row": 1,
                       "end_position_row": 67,
                       "segment_height": 67,
                       "grid_width": 3712
                       },
               },
    "fci_af_vis_06": {"3km": {"start_position_row": 1,
                              "end_position_row": 67,
                              "segment_height": 67,
                              "grid_width": 3712
                              },
                      "1km": {"start_position_row": 1,
                              "end_position_row": 200,
                              "grid_width": 11136,
                              "segment_height": 200}
                      }
}

CHANS_FDHSI = {"solar": LIST_CHANNEL_SOLAR,
               "solar_grid_type": ["1km"] * 8,
               "terran": LIST_CHANNEL_TERRAN,
               "terran_grid_type": ["2km"] * 8}

CHANS_HRFI = {"solar": ["vis_06", "nir_22"],
              "solar_grid_type": ["500m"] * 2,
              "terran": ["ir_38", "ir_105"],
              "terran_grid_type": ["1km"] * 2}

DICT_CALIBRATION = {"radiance": {"dtype": np.float32,
                                 "value_1": 15,
                                 "value_0": 9700,
                                 "attrs_dict": {"calibration": "radiance",
                                                "units": "mW m-2 sr-1 (cm-1)-1",
                                                "radiance_unit_conversion_coefficient": np.float32(1234.56)
                                                },
                                 },

                    "reflectance": {"dtype": np.float32,
                                    "attrs_dict": {"calibration": "reflectance",
                                                   "units": "%"
                                                   },
                                    },

                    "counts": {"dtype": np.uint16,
                               "value_1": 1,
                               "value_0": 5000,
                               "attrs_dict": {"calibration": "counts",
                                              "units": "count",
                                              },
                               },

                    "brightness_temperature": {"dtype": np.float32,
                                               "value_1": np.float32(209.68275),
                                               "value_0": np.float32(1888.8513),
                                               "attrs_dict": {"calibration": "brightness_temperature",
                                                              "units": "K",
                                                              },
                                               },
                    }
TEST_FILENAMES = {"fdhsi": [
    "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
    "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
    "20170410113925_20170410113934_N__C_0070_0067.nc"
],
    "fdhsi_error": [
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FDD--"
        "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
        "20170410113925_20170410113934_N__C_0070_0067.nc"
    ],
    "fdhsi_iqti": [
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--"
        "CHK-BODY--MON-NC4_C_EUMT_20240307233956_IQTI_DEV_"
        "20231016125007_20231016125017_N__C_0078_0001.nc"
    ],
    "hrfi": [
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--"
        "CHK-BODY--L2P-NC4E_C_EUMT_20170410114434_GTT_DEV_"
        "20170410113925_20170410113934_N__C_0070_0067.nc"
    ],
    "hrfi_iqti": [
        "W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-FD--"
        "CHK-BODY--MON-NC4_C_EUMT_20240307233956_IQTI_DEV_"
        "20231016125007_20231016125017_N__C_0078_0001.nc"
    ],
    "fdhsi_q4": ["W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-Q4--"
                 "CHK-BODY--DIS-NC4E_C_EUMT_20230723025408_IDPFI_DEV_"
                 "20230722120000_20230722120027_N_JLS_C_0289_0001.nc"
                 ],
    "hrfi_q4": ["W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-Q4--"
                "CHK-BODY--DIS-NC4E_C_EUMT_20230723025408_IDPFI_DEV"
                "_20230722120000_20230722120027_N_JLS_C_0289_0001.nc"]
}


def resolutions_AF_products(channel):
    """Get the resolutions of the African products."""
    if channel == "vis_06":
        return LIST_RESOLUTION_VIS06_AF
    else:
        return LIST_RESOLUTION_AF


def fill_chans_af():
    """Fill the dict CHANS_AF and the list TEST_FILENAMES with the right channel and resolution."""
    chans_af = {}
    for channel in LIST_TOTAL_CHANNEL:
        list_resol = resolutions_AF_products(channel)
        if channel in ["wv_63", "wv_73"]:
            ch_name_for_file = channel.replace("wv", "ir").replace("_", "").upper()
        else:
            ch_name_for_file = channel.replace("_", "").upper()
        for resol in list_resol:
            TEST_FILENAMES[f"af_{channel}_{resol}"] = [f"W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1-FCI-1C-RRAD"
                                                       f"-{resol.upper()}-AF-{ch_name_for_file}-x-x---NC4E_C_EUMT_"
                                                       f"20240125144655_DT_OPE"
                                                       f"_20240109080007_20240109080924_N_JLS_T_0049_0000.nc"]
            if channel.split("_")[0] in ["vis", "nir"]:
                chans_af[f"{channel}_{resol}"] = {"solar": [channel],
                                                  "solar_grid_type": [resol]}
            elif channel.split("_")[0] in ["ir", "wv"]:
                chans_af[f"{channel}_{resol}"] = {"terran": [channel],
                                                  "terran_grid_type": [resol]}
    return chans_af


CHANS_AF = fill_chans_af()


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
        """Get the array data."""
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
    data[meas_path + "/radiance_to_bt_conversion_coefficient_wavenumber"] = FakeH5Variable(
        da.array(955.0, dtype=np.float32))
    data[meas_path + "/radiance_to_bt_conversion_coefficient_a"] = FakeH5Variable(da.array(1.0, dtype=np.float32))
    data[meas_path + "/radiance_to_bt_conversion_coefficient_b"] = FakeH5Variable(da.array(0.4, dtype=np.float32))
    data[meas_path + "/radiance_to_bt_conversion_constant_c1"] = FakeH5Variable(
        da.array(1e11 * 2 * h * c ** 2, dtype=np.float32))
    data[meas_path + "/radiance_to_bt_conversion_constant_c2"] = FakeH5Variable(
        da.array(1e2 * h * c / k, dtype=np.float32))
    return data


def _get_test_calib_for_channel_vis(data, meas):
    data["state/celestial/earth_sun_distance"] = FakeH5Variable(
        da.repeat(da.array([149597870.7]), 6000), dims="index")
    data[meas + "/channel_effective_solar_irradiance"] = FakeH5Variable(da.array(50.0, dtype=np.float32))
    return data


def _get_test_calib_data_for_channel(data, ch_str):
    meas_path = "data/{:s}/measured".format(ch_str)
    if ch_str.startswith("ir") or ch_str.startswith("wv"):
        _get_test_calib_for_channel_ir(data, meas_path)
    elif ch_str.startswith("vis") or ch_str.startswith("nir"):
        _get_test_calib_for_channel_vis(data, meas_path)
    data[meas_path + "/radiance_unit_conversion_coefficient"] = xr.DataArray(da.array(1234.56, dtype=np.float32))


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
            dims=("y", "x"),
            attrs={
                "valid_range": [0, 8191],
                "warm_scale_factor": np.float32(2.0),
                "warm_add_offset": np.float32(-300.0),
                **common_attrs
            }
        )
    else:
        d = FakeH5Variable(
            da.ones(n_rows_cols, dtype="uint16", chunks=1024),
            dims=("y", "x"),
            attrs={
                "valid_range": [0, 4095],
                "warm_scale_factor": np.float32(1.0),
                "warm_add_offset": np.float32(0.0),
                **common_attrs
            }
        )

    data[ch_path] = d
    data[ch_path + "/shape"] = n_rows_cols


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
            "scale_factor": -GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["scale_factor"],
            "add_offset": GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["add_offset"],
        }
    )

    y_path = "data/{:s}/measured/y".format(ch_str)
    data[y_path] = xr.DataArray(
        da.arange(1, n_rows_cols[0] + 1, dtype=np.dtype("uint16")),
        dims=("y",),
        attrs={
            "scale_factor": GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["scale_factor"],
            "add_offset": -GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["add_offset"],
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
        if key == "earth_sun_distance":
            continue
        data[value] = xr.DataArray(da.arange(indices_dim, dtype=np.dtype("float32")), dims="index")

    # compute the last data entry to simulate the FCI caching
    # data[list(AUX_DATA.values())[-1]] = data[list(AUX_DATA.values())[-1]].compute()

    data["index"] = xr.DataArray(
        da.ones(indices_dim, dtype="uint16") * 100, dims="index")
    return data


def _get_global_attributes():
    data = {}
    attrs = {"platform": "MTI1"}
    for (k, v) in attrs.items():
        data["attr/" + k] = v
    return data


def _get_test_content_for_channel(ch_str, grid_type):
    nrows = GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["nrows"]
    ncols = GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["ncols"]
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
            for ch in self.chan_patterns[pat]["channels"]:
                data.update(_get_test_content_for_channel(pat.format(ch), self.chan_patterns[pat]["grid_type"]))
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
        "vis_{:>02d}": {"channels": [4, 5, 6, 8, 9],
                        "grid_type": "1km"},
        "nir_{:>02d}": {"channels": [13, 16, 22],
                        "grid_type": "1km"},
        "ir_{:>02d}": {"channels": [38, 87, 97, 105, 123, 133],
                       "grid_type": "2km"},
        "wv_{:>02d}": {"channels": [63, 73],
                       "grid_type": "2km"},
    }


class FakeFCIFileHandlerFDHSIIQTI(FakeFCIFileHandlerFDHSI):
    """Mock IQTI for FHDSI data."""

    def _get_test_content_all_channels(self):
        data = super()._get_test_content_all_channels()
        data.update({"state/celestial/earth_sun_distance": FakeH5Variable(
            da.repeat(da.array([np.nan]), 6000), dims="index")})
        return data


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


class FakeFCIFileHandlerHRFI(FakeFCIFileHandlerBase):
    """Mock HRFI data."""

    chan_patterns = {
        "vis_{:>02d}_hr": {"channels": [6],
                           "grid_type": "500m"},
        "nir_{:>02d}_hr": {"channels": [22],
                           "grid_type": "500m"},
        "ir_{:>02d}_hr": {"channels": [38, 105],
                          "grid_type": "1km"},
    }


class FakeFCIFileHandlerHRFIIQTI(FakeFCIFileHandlerHRFI):
    """Mock IQTI for HRFI data."""

    def _get_test_content_all_channels(self):
        data = super()._get_test_content_all_channels()
        data.update({"state/celestial/earth_sun_distance": FakeH5Variable(
            da.repeat(da.array([np.nan]), 6000), dims="x")})
        return data


class FakeFCIFileHandlerAF(FakeFCIFileHandlerBase):
    """Mock AF data."""
    chan_patterns = {}


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


def get_list_channel_calibration(calibration):
    """Get the channel's list according the calibration."""
    if calibration == "reflectance":
        return LIST_CHANNEL_SOLAR
    elif calibration == "brightness_temperature":
        return LIST_CHANNEL_TERRAN
    else:
        return LIST_TOTAL_CHANNEL


def generate_parameters(calibration):
    """Generate dynamically the parameters."""
    for channel in get_list_channel_calibration(calibration):
        for resolution in resolutions_AF_products(channel):
            yield channel, resolution


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
            "filetype": "fci_l1c_fdhsi",
            "channels": CHANS_FDHSI,
            "filenames": TEST_FILENAMES["fdhsi"]
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerFDHSIError_fixture():
    """Get a fixture for the fake FDHSI filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerFDHSI):
        param_dict = {
            "filetype": "fci_l1c_fdhsi",
            "channels": CHANS_FDHSI,
            "filenames": TEST_FILENAMES["fdhsi_error"]
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerFDHSIIQTI_fixture():
    """Get a fixture for the fake FDHSI IQTI filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerFDHSIIQTI):
        param_dict = {
            "filetype": "fci_l1c_fdhsi",
            "channels": CHANS_FDHSI,
            "filenames": TEST_FILENAMES["fdhsi_iqti"]
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerFDHSIQ4_fixture():
    """Get a fixture for the fake FDHSI Q4 filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerFDHSI):
        param_dict = {
            "filetype": "fci_l1c_fdhsi",
            "channels": CHANS_FDHSI,
            "filenames": TEST_FILENAMES["fdhsi_q4"]
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerHRFI_fixture():
    """Get a fixture for the fake HRFI filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerHRFI):
        param_dict = {
            "filetype": "fci_l1c_hrfi",
            "channels": CHANS_HRFI,
            "filenames": TEST_FILENAMES["hrfi"]
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerHRFIIQTI_fixture():
    """Get a fixture for the fake HRFI IQTI filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerHRFIIQTI):
        param_dict = {
            "filetype": "fci_l1c_hrfi",
            "channels": CHANS_HRFI,
            "filenames": TEST_FILENAMES["hrfi_iqti"]
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerHRFIQ4_fixture():
    """Get a fixture for the fake HRFI Q4 filehandler, including channel and file names."""
    with mocked_basefilehandler(FakeFCIFileHandlerHRFI):
        param_dict = {
            "filetype": "fci_l1c_hrfi",
            "channels": CHANS_HRFI,
            "filenames": TEST_FILENAMES["hrfi_q4"]
        }
        yield param_dict


@pytest.fixture
def FakeFCIFileHandlerAF_fixture(channel, resolution):
    """Get a fixture for the fake AF filehandler, it contains only one channel and one resolution."""
    chan_patterns = {channel.split("_")[0] + "_{:>02d}": {"channels": [int(channel.split("_")[1])],
                                                          "grid_type": f"{resolution}"}, }
    FakeFCIFileHandlerAF.chan_patterns = chan_patterns
    with mocked_basefilehandler(FakeFCIFileHandlerAF):
        param_dict = {
            "filetype": "fci_l1c_af",
            "channels": CHANS_AF[f"{channel}_{resolution}"],
            "filenames": TEST_FILENAMES[f"af_{channel}_{resolution}"],
        }
        yield param_dict


# ----------------------------------------------------
# Tests ----------------------------------------------
# ----------------------------------------------------
class ModuleTestFCIL1cNcReader:
    """Class containing parameters and modules useful for the test related to L1c reader."""
    fh_param_for_filetype = {"hrfi": {"channels": CHANS_HRFI,
                                      "filenames": TEST_FILENAMES["hrfi"]},
                             "fdhsi": {"channels": CHANS_FDHSI,
                                       "filenames": TEST_FILENAMES["fdhsi"]},
                             "fdhsi_iqti": {"channels": CHANS_FDHSI,
                                            "filenames": TEST_FILENAMES["fdhsi_iqti"]},
                             "hrfi_q4": {"channels": CHANS_HRFI,
                                         "filenames": TEST_FILENAMES["hrfi_q4"]},
                             "hrfi_iqti": {"channels": CHANS_HRFI,
                                           "filenames": TEST_FILENAMES["hrfi_iqti"]},
                             "fdhsi_q4": {"channels": CHANS_FDHSI,
                                          "filenames": TEST_FILENAMES["fdhsi_q4"]}}

    @staticmethod
    def _get_type_ter_AF(channel):
        """Get the type_ter."""
        if channel.split("_")[0] in ["vis", "nir"]:
            return "solar"
        elif channel.split("_")[0] in ["wv", "ir"]:
            return "terran"

    @staticmethod
    def _get_assert_attrs(res, ch, attrs_dict):
        """Test the different attributes values."""
        for key, item in attrs_dict.items():
            assert res[ch].attrs[key] == item

    @staticmethod
    def _get_assert_erased_attrs(res, ch):
        """Test that the attributes listed have been erased."""
        LIST_ATTRIBUTES = ["add_offset", "warm_add_offset", "scale_factor",
                           "warm_scale_factor", "valid_range"]
        for atr in LIST_ATTRIBUTES:
            assert atr not in res[ch].attrs

    @staticmethod
    def _reflectance_test(tab, filenames):
        """Test of with the reflectance test."""
        if "IQTI" in filenames:
            numpy.testing.assert_array_almost_equal(tab,
                                                    93.6462, decimal=4)
        else:
            numpy.testing.assert_array_almost_equal(tab,
                                                    100 * 15 * 1 * np.pi / 50)

    @staticmethod
    def _other_calibration_test(res, ch, dict_arg):
        """Test of other calibration test."""
        if ch == "ir_38":
            numpy.testing.assert_array_equal(res[ch][-1], dict_arg["value_1"])
            numpy.testing.assert_array_equal(res[ch][0], dict_arg["value_0"])
        else:
            numpy.testing.assert_array_equal(res[ch], dict_arg["value_1"])

    @staticmethod
    def _shape_test(res, ch, grid_type, dict_arg):
        """Test the shape."""
        assert res[ch].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["nrows"],
                                 GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["ncols"])
        assert res[ch].dtype == dict_arg["dtype"]

    def _get_assert_load(self, res, ch, dict_arg, filenames):
        """Test the value for different channels."""
        self._get_assert_attrs(res, ch, dict_arg["attrs_dict"])
        if dict_arg["attrs_dict"]["calibration"] in ["radiance", "brightness_temperature", "reflectance"]:
            self._get_assert_erased_attrs(res, ch)
        if dict_arg["attrs_dict"]["calibration"] == "reflectance":
            self._reflectance_test(res[ch], filenames)
        else:
            self._other_calibration_test(res, ch, dict_arg)

    def _get_res_AF(self, channel, fh_param, calibration, reader_configs):
        """Load the reader for AF data."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        type_ter = self._get_type_ter_AF(channel)
        res = reader.load([make_dataid(name=name, calibration=calibration)
                           for name in fh_param["channels"][type_ter]], pad_data=False)
        return res

    @staticmethod
    def _compare_sun_earth_distance(filetype, fh_param, reader_configs):
        """Test the sun earth distance calculation."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        if "IQTI" in fh_param["filenames"][0]:
            np.testing.assert_almost_equal(
                reader.file_handlers[filetype][0]._compute_sun_earth_distance,
                0.996803423, decimal=7)
        else:
            np.testing.assert_almost_equal(
                reader.file_handlers[filetype][0]._compute_sun_earth_distance,
                1.0, decimal=7)

    @staticmethod
    def _compare_rc_period_min_count_in_repeat_cycle(filetype, fh_param,
                                                     reader_configs, compare_parameters_tuple):
        """Test the count_in_repeat_cycle, rc_period_min."""
        count_in_repeat_cycle_imp, rc_period_min_imp, start_nominal_time, end_nominal_time = compare_parameters_tuple
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        assert count_in_repeat_cycle_imp == \
               reader.file_handlers[filetype][0].filename_info["count_in_repeat_cycle"]
        assert rc_period_min_imp == \
               reader.file_handlers[filetype][0].rc_period_min
        assert start_nominal_time == reader.file_handlers[filetype][0].nominal_start_time
        assert end_nominal_time == reader.file_handlers[filetype][0].nominal_end_time


class TestFCIL1cNCReader(ModuleTestFCIL1cNcReader):
    """Test FCI L1c NetCDF reader with nominal data."""

    @pytest.mark.parametrize("filenames", [TEST_FILENAMES[filename] for filename in TEST_FILENAMES.keys()])
    def test_file_pattern(self, reader_configs, filenames):
        """Test file pattern matching."""
        from satpy.readers import load_reader

        reader = load_reader(reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 1

    @pytest.mark.parametrize("filenames", [TEST_FILENAMES["fdhsi"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["hrfi"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["hrfi_q4"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["fdhsi_q4"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["fdhsi_iqti"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["hrfi_iqti"][0].replace("BODY", "TRAIL")])
    def test_file_pattern_for_TRAIL_file(self, reader_configs, filenames):
        """Test file pattern matching for TRAIL files, which should not be picked up."""
        from satpy.readers import load_reader

        reader = load_reader(reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 0

    @pytest.mark.parametrize("calibration", ["counts", "radiance", "brightness_temperature", "reflectance"])
    @pytest.mark.parametrize(("fh_param", "res_type"), [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), "hdfi"),
                                                        (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), "hrfi"),
                                                        (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), "hrfi"),
                                                        (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), "hdfi"),
                                                        (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), "hrfi"),
                                                        (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), "hdfi")])
    def test_load_calibration(self, reader_configs, fh_param,
                              caplog, calibration, res_type):
        """Test loading with counts,radiance,reflectance and bt."""
        expected_res_n = {}
        if calibration == "reflectance":
            list_chan = fh_param["channels"]["solar"]
            list_grid = fh_param["channels"]["solar_grid_type"]
            expected_res_n["hdfi"] = 8
            expected_res_n["hrfi"] = 2
        elif calibration == "brightness_temperature":
            list_chan = fh_param["channels"]["terran"]
            list_grid = fh_param["channels"]["terran_grid_type"]
            expected_res_n["hdfi"] = 8
            expected_res_n["hrfi"] = 2
        else:
            list_chan = fh_param["channels"]["solar"] + fh_param["channels"]["terran"]
            list_grid = fh_param["channels"]["solar_grid_type"] + fh_param["channels"]["terran_grid_type"]
            expected_res_n["hdfi"] = 16
            expected_res_n["hrfi"] = 4
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        with caplog.at_level(logging.WARNING):
            res = reader.load(
                [make_dataid(name=name, calibration=calibration) for name in
                 list_chan], pad_data=False)
            assert caplog.text == ""
        assert expected_res_n[res_type] == len(res)
        for ch, grid_type in zip(list_chan,
                                 list_grid):
            self._shape_test(res, ch, grid_type, DICT_CALIBRATION[calibration])
            self._get_assert_load(res, ch, DICT_CALIBRATION[calibration],
                                  fh_param["filenames"][0])

    @pytest.mark.parametrize(("calibration", "channel", "resolution"), [
        (calibration, channel, resolution)
        for calibration in ["counts", "radiance", "brightness_temperature", "reflectance"]
        for channel, resolution in generate_parameters(calibration)
    ])
    def test_load_calibration_af(self, FakeFCIFileHandlerAF_fixture, reader_configs, channel, calibration, caplog):
        """Test loading with counts,radiance,reflectance and bt for AF files."""
        expected_res_n = 1
        fh_param = FakeFCIFileHandlerAF_fixture
        type_ter = self._get_type_ter_AF(channel)
        with caplog.at_level(logging.WARNING):
            res = self._get_res_AF(channel, fh_param, calibration, reader_configs)
            assert caplog.text == ""
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param["channels"][type_ter],
                                 fh_param["channels"][f"{type_ter}_grid_type"]):
            self._shape_test(res, ch, grid_type, DICT_CALIBRATION[calibration])
            self._get_assert_load(res, ch, DICT_CALIBRATION[calibration],
                                  fh_param["filenames"][0])

    @pytest.mark.parametrize("fh_param", [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"))])
    def test_orbital_parameters_attr(self, reader_configs, fh_param):
        """Test the orbital parameter attribute."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        res = reader.load(
            [make_dataid(name=name) for name in
             fh_param["channels"]["solar"] + fh_param["channels"]["terran"]], pad_data=False)

        for ch in fh_param["channels"]["solar"] + fh_param["channels"]["terran"]:
            assert res[ch].attrs["orbital_parameters"] == {
                "satellite_actual_longitude": np.mean(np.arange(6000)) if "IQTI" not in
                                                                          fh_param["filenames"][0] else 0.0,
                "satellite_actual_latitude": np.mean(np.arange(6000)) if "IQTI" not in
                                                                         fh_param["filenames"][0] else 0.0,
                "satellite_actual_altitude": np.mean(np.arange(6000)) if "IQTI" not in
                                                                         fh_param["filenames"][0] else 35786400.0,
                "satellite_nominal_longitude": 0.0,
                "satellite_nominal_latitude": 0,
                "satellite_nominal_altitude": 35786400.0,
                "projection_longitude": 0.0,
                "projection_latitude": 0,
                "projection_altitude": 35786400.0,
            }

    @pytest.mark.parametrize(("fh_param", "expected_pos_info"), [
        (lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["fdhsi"]),
        (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["hrfi"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["hrfi"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["fdhsi"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["hrfi"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["fdhsi"])
    ])
    def test_get_segment_position_info(self, reader_configs, fh_param, expected_pos_info):
        """Test the segment position info method."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        for filetype_handler in list(reader.file_handlers.values())[0]:
            segpos_info = filetype_handler.get_segment_position_info()
            assert segpos_info == expected_pos_info

    @pytest.mark.parametrize(("channel", "resolution"), generate_parameters("radiance"))
    def test_not_get_segment_info_called_af(self, FakeFCIFileHandlerAF_fixture, reader_configs, channel, resolution):
        """Test that checks that the get_segment_position_info has not been called for AF data."""
        with mock.patch("satpy.readers.fci_l1c_nc.FCIL1cNCFileHandler.get_segment_position_info") as gspi:
            fh_param = FakeFCIFileHandlerAF_fixture
            reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
            with contextlib.suppress(KeyError):
                # attempt to load the channel
                # If get_segment_position_info is called, the code will fail with a KeyError because of the mocking.
                # However, the point of the test is to check if the function has been called, not if the function
                # would work with this case, so the expected KeyError is suppressed, and we assert_not_called below.
                reader.load([channel])
            gspi.assert_not_called()

    @pytest.mark.parametrize("calibration", ["index_map", "pixel_quality"])
    @pytest.mark.parametrize(("fh_param", "expected_res_n"), [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), 16),
                                                              (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), 4),
                                                              (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), 4),
                                                              (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), 16),
                                                              (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), 4),
                                                              (
                                                              lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), 16)])
    def test_load_map_and_pixel(self, reader_configs, fh_param, expected_res_n, calibration):
        """Test loading of index_map and pixel_quality."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        res = reader.load(
            [f"{name}_{calibration}" for name in
             fh_param["channels"]["solar"] + fh_param["channels"]["terran"]], pad_data=False)
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param["channels"]["solar"] + fh_param["channels"]["terran"],
                                 fh_param["channels"]["solar_grid_type"] +
                                 fh_param["channels"]["terran_grid_type"]):
            assert res[f"{ch}_{calibration}"].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["nrows"],
                                                        GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["ncols"])
            if calibration == "index_map":
                numpy.testing.assert_array_equal(res[f"{ch}_{calibration}"][1, 1], 110)
            elif calibration == "pixel_quality":
                numpy.testing.assert_array_equal(res[f"{ch}_{calibration}"][1, 1], 3)
                assert res[f"{ch}_{calibration}"].attrs["name"] == ch + "_pixel_quality"

    @pytest.mark.parametrize(("calibration", "channel", "resolution"), [
        (calibration, channel, resolution)
        for calibration in ["index_map", "pixel_quality"]
        for channel, resolution in generate_parameters(calibration)
    ])
    def test_load_map_and_pixel_af(self, FakeFCIFileHandlerAF_fixture, reader_configs, channel, calibration):
        """Test loading with of index_map and pixel_quality for AF files."""
        expected_res_n = 1
        fh_param = FakeFCIFileHandlerAF_fixture
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        type_ter = self._get_type_ter_AF(channel)
        res = reader.load([f"{name}_{calibration}"
                           for name in fh_param["channels"][type_ter]], pad_data=False)
        assert expected_res_n == len(res)
        for ch, grid_type in zip(fh_param["channels"][type_ter],
                                 fh_param["channels"][f"{type_ter}_grid_type"]):
            assert res[f"{ch}_{calibration}"].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["nrows"],
                                                        GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["ncols"])
            if calibration == "index_map":
                numpy.testing.assert_array_equal(res[f"{ch}_{calibration}"][1, 1], 110)
            elif calibration == "pixel_quality":
                numpy.testing.assert_array_equal(res[f"{ch}_{calibration}"][1, 1], 3)
                assert res[f"{ch}_{calibration}"].attrs["name"] == ch + "_pixel_quality"

    @pytest.mark.parametrize("fh_param", [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"))])
    def test_load_aux_data(self, reader_configs, fh_param):
        """Test loading of auxiliary data."""
        from satpy.readers.fci_l1c_nc import AUX_DATA
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        res = reader.load([fh_param["channels"]["solar"][0] + "_" + key for key in AUX_DATA.keys()],
                          pad_data=False)
        grid_type = fh_param["channels"]["solar_grid_type"][0]
        for aux in [fh_param["channels"]["solar"][0] + "_" + key for key in AUX_DATA.keys()]:
            assert res[aux].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["nrows"],
                                      GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["ncols"])
            if (aux == fh_param["channels"]["solar"][0] + "_earth_sun_distance") and ("IQTI" not in
                                                                                      fh_param["filenames"][0]):
                numpy.testing.assert_array_equal(res[aux][1, 1], 149597870.7)
            elif aux == fh_param["channels"]["solar"][0] + "_earth_sun_distance":
                numpy.testing.assert_array_equal(res[aux][1, 1], np.nan)
            else:
                numpy.testing.assert_array_equal(res[aux][1, 1], 10)

    @pytest.mark.parametrize("fh_param", [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture")),
                                          ])
    def test_platform_name(self, reader_configs, fh_param):
        """Test that platform name is exposed.

        Test that the FCI reader exposes the platform name.  Corresponds
        to GH issue 1014.
        """
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        res = reader.load(["vis_06"], pad_data=False)
        assert res["vis_06"].attrs["platform_name"] == "MTG-I1"

    @pytest.mark.parametrize(("fh_param", "compare_tuples"),
                             [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), (67, 10,
                                                                                 datetime.datetime.strptime(
                                                                                     "2017-04-10 11:30:00",
                                                                                     "%Y-%m-%d %H:%M:%S"),
                                                                                 datetime.datetime.strptime(
                                                                                     "2017-04-10 11:40:00",
                                                                                     "%Y-%m-%d %H:%M:%S"))),
                              (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), (67, 10,
                                                                                datetime.datetime.strptime(
                                                                                    "2017-04-10 11:30:00",
                                                                                    "%Y-%m-%d %H:%M:%S"),
                                                                                datetime.datetime.strptime(
                                                                                    "2017-04-10 11:40:00",
                                                                                    "%Y-%m-%d %H:%M:%S"))),
                              (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), (29, 2.5,
                                                                                  datetime.datetime.strptime(
                                                                                      "2023-07-22 12:00:00",
                                                                                      "%Y-%m-%d %H:%M:%S"),
                                                                                  datetime.datetime.strptime(
                                                                                      "2023-07-22 12:02:30",
                                                                                      "%Y-%m-%d %H:%M:%S"))),
                              (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), (29, 2.5,
                                                                                   datetime.datetime.strptime(
                                                                                       "2023-07-22 12:00:00",
                                                                                       "%Y-%m-%d %H:%M:%S"),
                                                                                   datetime.datetime.strptime(
                                                                                       "2023-07-22 12:02:30",
                                                                                       "%Y-%m-%d %H:%M:%S"))),
                              (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), (1, 10,
                                                                                    datetime.datetime.strptime(
                                                                                        "2023-10-16 12:50:00",
                                                                                        "%Y-%m-%d %H:%M:%S"),
                                                                                    datetime.datetime.strptime(
                                                                                        "2023-10-16 13:00:00",
                                                                                        "%Y-%m-%d %H:%M:%S"))),
                              (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), (1, 10,
                                                                                     datetime.datetime.strptime(
                                                                                         "2023-10-16 12:50:00",
                                                                                         "%Y-%m-%d %H:%M:%S"),
                                                                                     datetime.datetime.strptime(
                                                                                         "2023-10-16 13:00:00",
                                                                                         "%Y-%m-%d %H:%M:%S"))),
                              ])
    def test_count_in_repeat_cycle_rc_period_min(self, reader_configs, fh_param, compare_tuples):
        """Test the rc_period_min value for each configuration."""
        self._compare_rc_period_min_count_in_repeat_cycle(fh_param["filetype"], fh_param,
                                                          reader_configs, compare_tuples)

    @pytest.mark.parametrize(("channel", "resolution", "compare_tuples"),
                             [("vis_06", "3km", (1, 10,
                                                 datetime.datetime.strptime("2024-01-09 08:00:00", "%Y-%m-%d %H:%M:%S"),
                                                 datetime.datetime.strptime("2024-01-09 08:10:00",
                                                                            "%Y-%m-%d %H:%M:%S"))),
                              ("vis_06", "1km", (1, 10,
                                                 datetime.datetime.strptime("2024-01-09 08:00:00", "%Y-%m-%d %H:%M:%S"),
                                                 datetime.datetime.strptime("2024-01-09 08:10:00",
                                                                            "%Y-%m-%d %H:%M:%S")))
                              ])
    def test_count_in_repeat_cycle_rc_period_min_AF(self, FakeFCIFileHandlerAF_fixture, reader_configs,
                                                    channel, resolution, compare_tuples):
        """Test the rc_period_min value for each configuration."""
        fh_param = FakeFCIFileHandlerAF_fixture
        self._compare_rc_period_min_count_in_repeat_cycle(f"{fh_param['filetype']}_{channel}_{resolution}", fh_param,
                                                          reader_configs, compare_tuples)

    @pytest.mark.parametrize(("fh_param"),
                             [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")),
                              (lazy_fixture("FakeFCIFileHandlerHRFI_fixture")),
                              (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture")),
                              (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture")),
                              (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture")),
                              (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture")),
                              ])
    def test_compute_earth_sun_parameter(self, reader_configs, fh_param):
        """Test the computation of the sun_earth_parameter."""
        self._compare_sun_earth_distance(fh_param["filetype"], fh_param, reader_configs)

    @pytest.mark.parametrize(("channel", "resolution"), [("vis_06", "3km"),
                                                         ("vis_06", "1km")])
    def test_compute_earth_sun_parameter_AF(self, FakeFCIFileHandlerAF_fixture, reader_configs,
                                            channel, resolution):
        """Test the rc_period_min value for each configuration."""
        fh_param = FakeFCIFileHandlerAF_fixture
        self._compare_sun_earth_distance(f"{fh_param['filetype']}_{channel}_{resolution}", fh_param, reader_configs)

    @pytest.mark.parametrize(("fh_param"), [(lazy_fixture("FakeFCIFileHandlerFDHSIError_fixture"))])
    def test_rc_period_min_error(self, reader_configs, fh_param):
        """Test the rc_period_min error."""
        with pytest.raises(NotImplementedError):
            _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)

    @pytest.mark.parametrize(("fh_param", "expected_area"), [
        (lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), ["mtg_fci_fdss_1km", "mtg_fci_fdss_2km"]),
        (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), ["mtg_fci_fdss_500m", "mtg_fci_fdss_1km"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), ["mtg_fci_fdss_500m", "mtg_fci_fdss_1km"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), ["mtg_fci_fdss_1km", "mtg_fci_fdss_2km"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), ["mtg_fci_fdss_500m", "mtg_fci_fdss_1km"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), ["mtg_fci_fdss_1km", "mtg_fci_fdss_2km"])
    ])
    def test_area_definition_computation(self, reader_configs, fh_param, expected_area):
        """Test that the geolocation computation is correct."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        res = reader.load(["ir_105", "vis_06"], pad_data=False)

        # test that area_ids are harmonisation-conform <platform>_<instrument>_<service>_<resolution>
        assert res["vis_06"].attrs["area"].area_id == expected_area[0]
        assert res["ir_105"].attrs["area"].area_id == expected_area[1]

        area_def = res["ir_105"].attrs["area"]
        # test area extents computation
        np.testing.assert_array_almost_equal(np.array(area_def.area_extent),
                                             np.array([-5567999.994203, -5367999.994411,
                                                       5567999.994203, -5567999.994203]),
                                             decimal=2)

        # check that the projection is read in properly
        assert area_def.crs.coordinate_operation.method_name == "Geostationary Satellite (Sweep Y)"
        assert area_def.crs.coordinate_operation.params[0].value == 0.0  # projection origin longitude
        assert area_def.crs.coordinate_operation.params[1].value == 35786400.0  # projection height
        assert area_def.crs.ellipsoid.semi_major_metre == 6378137.0
        assert area_def.crs.ellipsoid.inverse_flattening == 298.257223563
        assert area_def.crs.ellipsoid.is_semi_minor_computed

    @pytest.mark.parametrize("fh_param", [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture")),
                                          ])
    def test_excs(self, reader_configs, fh_param):
        """Test that exceptions are raised where expected."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)

        with pytest.raises(ValueError, match="Unknown dataset key, not a channel, quality or auxiliary data: invalid"):
            reader.file_handlers[fh_param["filetype"]][0].get_dataset(make_dataid(name="invalid"), {})
        with pytest.raises(ValueError, match="unknown invalid value for <enum 'calibration'>"):
            reader.file_handlers[fh_param["filetype"]][0].get_dataset(
                make_dataid(name="ir_123", calibration="unknown"),
                {"units": "unknown"})

    def test_load_composite(self):
        """Test that composites are loadable."""
        # when dedicated composites for FCI are implemented in satpy,
        # this method should probably move to a dedicated class and module
        # in the tests.compositor_tests package

        from satpy.composites.config_loader import load_compositor_configs_for_sensors
        comps, mods = load_compositor_configs_for_sensors(["fci"])
        assert len(comps["fci"]) > 0
        assert len(mods["fci"]) > 0


class TestFCIL1cNCReaderBadData:
    """Test the FCI L1c NetCDF Reader for bad data input."""

    def test_handling_bad_data_ir(self, reader_configs, caplog):
        """Test handling of bad IR data."""
        with mocked_basefilehandler(FakeFCIFileHandlerWithBadData):
            reader = _get_reader_with_filehandlers(TEST_FILENAMES["fdhsi"], reader_configs)
            with caplog.at_level(logging.ERROR):
                reader.load([make_dataid(
                    name="ir_105",
                    calibration="brightness_temperature")], pad_data=False)
                assert "cannot produce brightness temperature" in caplog.text

    def test_handling_bad_data_vis(self, reader_configs, caplog):
        """Test handling of bad VIS data."""
        with mocked_basefilehandler(FakeFCIFileHandlerWithBadData):
            reader = _get_reader_with_filehandlers(TEST_FILENAMES["fdhsi"], reader_configs)
            with caplog.at_level(logging.ERROR):
                reader.load([make_dataid(
                    name="vis_06",
                    calibration="reflectance")], pad_data=False)
                assert "cannot produce reflectance" in caplog.text
