"""Tests for the 'fci_l1c_nc' reader."""
import contextlib
import datetime
import functools
import logging
import os
import shutil
from unittest import mock

import netCDF4
import numpy as np
import numpy.testing
import pytest
import xarray as xr
from netCDF4 import default_fillvals
from pytest_lazy_fixtures import lf as lazy_fixture

from satpy.tests.utils import make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - caplog
# - tmp_path_factory

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
                                 "value_2": -5,
                                 "attrs_dict": {"calibration": "radiance",
                                                "units": "mW m-2 sr-1 (cm-1)-1",
                                                "radiance_unit_conversion_coefficient": np.float32(1234.56)
                                                },
                                 },

                    # 8< v1.0
                    "reflectance": {"dtype": np.float32,
                                    "attrs_dict": {"calibration": "reflectance",
                                                   "units": "%"
                                                   },
                                    },
                    # >8 v1.0
                    "unnormalized_reflectance": {"dtype": np.float32,
                                        "attrs_dict": {"calibration": "unnormalized_reflectance",
                                                       "units": "%"
                                                       },
                                        },

                    "counts": {"dtype": np.uint16,
                               "value_1": 5,
                               "value_0": 5000,
                               "value_2": 1,
                               "attrs_dict": {"calibration": "counts",
                                              "units": "count",
                                              },
                               },

                    "brightness_temperature": {"dtype": np.float32,
                                               "value_1": np.float32(209.68275),
                                               "value_0": np.float32(1888.8513),
                                               "value_2": np.float32("nan"),
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
                "_20230722120000_20230722120027_N_JLS_C_0289_0001.nc"],
    "hrfi_other_coverage": ["W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-HRFI-xx--"
                "CHK-BODY--DIS-NC4E_C_EUMT_20230723025408_IDPFI_DEV"
                "_20230722120000_20230722120027_N_JLS_C_0289_0001.nc"],
    "fdhsi_other_coverage": ["W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-xx--"
                 "CHK-BODY--DIS-NC4E_C_EUMT_20230723025408_IDPFI_DEV_"
                 "20230722120000_20230722120027_N_JLS_C_0289_0001.nc"
                 ],
}

REQUIRED_ATTR_KEYS = {
    "time_parameters",
    "orbital_parameters",
    "file_type",
    "name",
    "sensor",
    "resolution",
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

# Engines to read the test files with explicitly. Tests without an engine use the reader's default list of engines.
ENGINES = ["netcdf4", "h5netcdf"]

# Size of the "index" dimension of the auxiliary data vectors
AUX_INDEX_SIZE = 6000
# dtypes of the auxiliary data vectors in real files, which are float32 otherwise
AUX_DATA_DTYPES = {"time": np.float64, "swath_number": np.uint16, "swath_direction": np.uint16}
EARTH_SUN_DISTANCE_KM = 149597870.7
FILL_VALUE_F4 = np.float32(default_fillvals["f4"])


# ----------------------------------------------------
# Test file content ----------------------------------
# ----------------------------------------------------

# The content of a test file is a dict of netCDF paths to DataArrays (variables, with their attributes and
# dimension names) and of "attr/<name>" keys to global attributes. The dtypes and attributes follow real files.

def _get_coefficient(value):
    """Get a scalar calibration coefficient."""
    return xr.DataArray(np.array(value, dtype=np.float32), attrs={"_FillValue": FILL_VALUE_F4})


def _get_aux_vector(values, dtype=np.float32):
    """Get an auxiliary data vector along the "index" dimension, with a fill value if it is a float."""
    values = np.asarray(values, dtype=dtype)
    attrs = {}
    if values.dtype.kind == "f":
        attrs["_FillValue"] = values.dtype.type(default_fillvals[values.dtype.str[1:]])
    return xr.DataArray(values, dims="index", attrs=attrs)


def _get_earth_sun_distance(value):
    return _get_aux_vector(np.full(AUX_INDEX_SIZE, value))


def _get_test_calib_for_channel_ir(data, meas_path):
    from pyspectral.blackbody import C_SPEED as c
    from pyspectral.blackbody import H_PLANCK as h
    from pyspectral.blackbody import K_BOLTZMANN as k
    data[meas_path + "/radiance_to_bt_conversion_coefficient_wavenumber"] = _get_coefficient(955.0)
    data[meas_path + "/radiance_to_bt_conversion_coefficient_a"] = _get_coefficient(1.0)
    data[meas_path + "/radiance_to_bt_conversion_coefficient_b"] = _get_coefficient(0.4)
    data[meas_path + "/radiance_to_bt_conversion_constant_c1"] = _get_coefficient(1e11 * 2 * h * c ** 2)
    data[meas_path + "/radiance_to_bt_conversion_constant_c2"] = _get_coefficient(1e2 * h * c / k)
    return data


def _get_test_calib_for_channel_vis(data, meas):
    data["state/celestial/earth_sun_distance"] = _get_earth_sun_distance(EARTH_SUN_DISTANCE_KM)
    data[meas + "/channel_effective_solar_irradiance"] = _get_coefficient(50.0)
    return data


def _get_test_calib_data_for_channel(data, ch_str):
    meas_path = "data/{:s}/measured".format(ch_str)
    if ch_str.startswith("ir") or ch_str.startswith("wv"):
        _get_test_calib_for_channel_ir(data, meas_path)
    elif ch_str.startswith("vis") or ch_str.startswith("nir"):
        _get_test_calib_for_channel_vis(data, meas_path)
    data[meas_path + "/radiance_unit_conversion_coefficient"] = _get_coefficient(1234.56)


def _get_test_image_data_for_channel(data, ch_str, n_rows_cols):
    ch_path = "data/{:s}/measured/effective_radiance".format(ch_str)

    common_attrs = {
        "scale_factor": np.float32(5),
        "add_offset": np.float32(-10),
        "long_name": "Effective Radiance",
        "units": "mW.m-2.sr-1.(cm-1)-1",
        "ancillary_variables": "pixel_quality",
        "_FillValue": np.uint16(65535)
    }
    counts = np.full(n_rows_cols, 5, dtype=np.uint16)
    if "38" in ch_path:
        counts[0] = 5000  # fire line
        counts[-1] = 1  # negative radiances
        d = xr.DataArray(
            counts,
            dims=("y", "x"),
            attrs={
                "valid_range": np.array([0, 8191], dtype=np.uint16),
                "warm_scale_factor": np.float32(2.0),
                "warm_add_offset": np.float32(-300.0),
                **common_attrs
            }
        )
    else:
        d = xr.DataArray(
            counts,
            dims=("y", "x"),
            attrs={
                "valid_range": np.array([0, 4095], dtype=np.uint16),
                "warm_scale_factor": np.float32(1.0),
                "warm_add_offset": np.float32(0.0),
                **common_attrs
            }
        )

    data[ch_path] = d


def _get_segment_position(value):
    return xr.DataArray(np.array(value, dtype=np.uint16), attrs={"_FillValue": np.uint16(65535)})


def _get_test_segment_position_for_channel(data, ch_str, n_rows_cols):
    pos = "data/{:s}/measured/{:s}_position_{:s}"
    data[pos.format(ch_str, "start", "row")] = _get_segment_position(1)
    data[pos.format(ch_str, "start", "column")] = _get_segment_position(1)
    data[pos.format(ch_str, "end", "row")] = _get_segment_position(n_rows_cols[0])
    data[pos.format(ch_str, "end", "column")] = _get_segment_position(n_rows_cols[1])


def _get_test_index_map_for_channel(data, ch_str, n_rows_cols):
    index_map_path = "data/{:s}/measured/index_map".format(ch_str)
    data[index_map_path] = xr.DataArray(
        np.full(n_rows_cols, 110, dtype=np.uint16), dims=("y", "x"), attrs={"_FillValue": np.uint16(65535)})


def _get_test_pixel_quality_for_channel(data, ch_str, n_rows_cols):
    qual_path = "data/{:s}/measured/pixel_quality".format(ch_str)
    data[qual_path] = xr.DataArray(
        np.full(n_rows_cols, 3, dtype=np.uint8), dims=("y", "x"),
        attrs={"_FillValue": np.uint8(0), "valid_range": np.array([0, 255], dtype=np.uint8)})


def _get_coordinate(dim, size, scale_factor, add_offset):
    return xr.DataArray(
        np.arange(1, size + 1, dtype=np.int16),
        dims=(dim,),
        attrs={
            "valid_range": np.array([1, size], dtype=np.int16),
            "scale_factor": np.float64(scale_factor),
            "add_offset": np.float64(add_offset),
        }
    )


def _get_test_geolocation_for_channel(data, ch_str, grid_type, n_rows_cols):
    grid_info = GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]
    data["data/{:s}/measured/x".format(ch_str)] = _get_coordinate(
        "x", n_rows_cols[1], -grid_info["scale_factor"], grid_info["add_offset"])
    data["data/{:s}/measured/y".format(ch_str)] = _get_coordinate(
        "y", n_rows_cols[0], grid_info["scale_factor"], -grid_info["add_offset"])


def _get_test_content_areadef():
    attrs = {
        "sweep_angle_axis": "y",
        "perspective_point_height": np.float64(35786400.0),
        "semi_major_axis": np.float64(6378137.0),
        "longitude_of_projection_origin": np.float64(0.0),
        "inverse_flattening": np.float64(298.257223563),
        "units": "m"}
    return {"data/mtg_geos_projection": xr.DataArray(np.array(0, dtype=np.int32), attrs=attrs)}


def _get_test_content_aux_data():
    from satpy.readers.fci_l1c_nc import AUX_DATA
    data = {}
    for key, value in AUX_DATA.items():
        # skip population of earth_sun_distance as this is already defined for reflectance calculation
        if key == "earth_sun_distance":
            continue
        data[value] = _get_aux_vector(np.arange(AUX_INDEX_SIZE), AUX_DATA_DTYPES.get(key, np.float32))

    data["index"] = xr.DataArray(np.full(AUX_INDEX_SIZE, 100, dtype=np.uint16), dims="index")
    return data


def _get_global_attributes():
    return {"attr/platform": "MTI1"}


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


def _get_test_content(chans, channel_suffix=""):
    """Get the content of a test file with the channels and grid types of ``chans`` (like ``CHANS_FDHSI``).

    The names of the channel groups in the file are the channel names followed by ``channel_suffix``.
    """
    data = {}
    for kind in ("solar", "terran"):
        for ch_str, grid_type in zip(chans.get(kind, []), chans.get(f"{kind}_grid_type", [])):
            data.update(_get_test_content_for_channel(ch_str + channel_suffix, grid_type))
    data.update(_get_test_content_areadef())
    data.update(_get_test_content_aux_data())
    data.update(_get_global_attributes())
    return data


def _get_test_content_iqti(chans, channel_suffix=""):
    """Get the content of an IQTI test file, which has no valid earth-sun distance."""
    data = _get_test_content(chans, channel_suffix)
    data["state/celestial/earth_sun_distance"] = _get_earth_sun_distance(np.nan)
    return data


def _get_test_content_with_bad_data():
    """Get the content of an FDHSI test file with calibration coefficients set to their fill value."""
    data = _get_test_content(CHANS_FDHSI)
    for path in ["data/ir_105/measured/radiance_to_bt_conversion_coefficient_wavenumber",
                 "data/ir_105/measured/radiance_to_bt_conversion_coefficient_a",
                 "data/ir_105/measured/radiance_to_bt_conversion_coefficient_b",
                 "data/ir_105/measured/radiance_to_bt_conversion_constant_c1",
                 "data/ir_105/measured/radiance_to_bt_conversion_constant_c2",
                 "data/vis_06/measured/channel_effective_solar_irradiance"]:
        data[path] = _get_coefficient(FILL_VALUE_F4)
    return data


# ----------------------------------------------------
# Test files -----------------------------------------
# ----------------------------------------------------

def _write_test_file(content, path):
    """Write the test file ``content`` (see above) to a netCDF4 file at ``path`` and return ``path``.

    The values are written as they are, without masking or packing them. The dimensions are created
    where real files have them: ``x`` and ``y`` in the channel group (the parent of the ``measured``
    group of the variables using them), ``index`` in the root group and any other in the group of
    the variable.
    """
    with netCDF4.Dataset(path, "w") as nc:
        for key, val in content.items():
            if key.startswith("attr/"):
                _set_nc_attr(nc, key.removeprefix("attr/"), val)
            else:
                _write_variable(nc, key, val)
    return path


def _write_variable(nc, key, val):
    group_path, _, name = key.rpartition("/")
    group = nc
    for group_name in group_path.split("/") if group_path else []:
        if group_name not in group.groups:
            group.createGroup(group_name)
        group = group.groups[group_name]
    for dim_name, size in zip(val.dims, val.shape):
        dim_group = _get_dimension_group(nc, group, dim_name)
        if dim_name not in dim_group.dimensions:
            dim_group.createDimension(dim_name, size)
    attrs = dict(val.attrs)
    var = group.createVariable(name, val.dtype, val.dims, fill_value=attrs.pop("_FillValue", None),
                               zlib=val.ndim == 2, complevel=1)
    var.set_auto_maskandscale(False)
    for attr_name, attr_val in attrs.items():
        _set_nc_attr(var, attr_name, attr_val)
    var[...] = val.values


def _get_dimension_group(nc, group, dim_name):
    if dim_name == "index":
        return nc
    if dim_name in ("x", "y") and group.name == "measured":
        return group.parent
    return group


def _set_nc_attr(obj, name, value):
    if isinstance(value, str):
        # real files have string attributes, not character arrays
        obj.setncattr_string(name, value)
    else:
        obj.setncattr(name, value)


def _copy_test_file(path, filename):
    """Copy the test file at ``path`` to ``filename`` in the same directory and return the new path."""
    return shutil.copyfile(path, path.parent / filename)


@pytest.fixture(scope="module")
def fci_l1c_dir(tmp_path_factory):
    """Get the directory of the test files."""
    return tmp_path_factory.mktemp("fci_l1c")


@pytest.fixture(scope="module")
def fdhsi_file(fci_l1c_dir):
    """Write the FDHSI test file."""
    return _write_test_file(_get_test_content(CHANS_FDHSI), fci_l1c_dir / TEST_FILENAMES["fdhsi"][0])


@pytest.fixture(scope="module")
def fdhsi_iqti_file(fci_l1c_dir):
    """Write the FDHSI IQTI test file."""
    return _write_test_file(_get_test_content_iqti(CHANS_FDHSI), fci_l1c_dir / TEST_FILENAMES["fdhsi_iqti"][0])


@pytest.fixture(scope="module")
def fdhsi_bad_data_file(tmp_path_factory):
    """Write the FDHSI test file with bad calibration coefficients.

    It has the file name of the FDHSI test file, so it is in a directory of its own.
    """
    return _write_test_file(_get_test_content_with_bad_data(),
                            tmp_path_factory.mktemp("fci_l1c_bad_data") / TEST_FILENAMES["fdhsi"][0])


@pytest.fixture(scope="module")
def hrfi_file(fci_l1c_dir):
    """Write the HRFI test file."""
    return _write_test_file(_get_test_content(CHANS_HRFI, "_hr"), fci_l1c_dir / TEST_FILENAMES["hrfi"][0])


@pytest.fixture(scope="module")
def hrfi_iqti_file(fci_l1c_dir):
    """Write the HRFI IQTI test file."""
    return _write_test_file(_get_test_content_iqti(CHANS_HRFI, "_hr"), fci_l1c_dir / TEST_FILENAMES["hrfi_iqti"][0])


@pytest.fixture(scope="module")
def af_file_writer(fci_l1c_dir):
    """Get a function returning the AF test file of a channel and resolution, which it writes the first time."""
    @functools.cache
    def get_af_file(channel, resolution):
        return _write_test_file(_get_test_content(CHANS_AF[f"{channel}_{resolution}"]),
                                fci_l1c_dir / TEST_FILENAMES[f"af_{channel}_{resolution}"][0])
    return get_af_file


# ----------------------------------------------------
# Fixtures preparation -------------------------------
# ----------------------------------------------------

@pytest.fixture
def reader_configs():
    """Return reader configs for FCI."""
    from satpy._config import config_search_paths
    return config_search_paths(
        os.path.join("readers", "fci_l1c_nc.yaml"))


def _get_reader_with_filehandlers(filenames, reader_configs, engine=None, **reader_kwargs):
    from satpy.readers.core.loading import load_reader
    if engine is not None:
        reader_kwargs["engine"] = engine
    reader = load_reader(reader_configs)
    loadables = reader.select_files_from_pathnames(filenames)
    reader.create_filehandlers(loadables, fh_kwargs=reader_kwargs)
    return reader


def get_list_channel_calibration(calibration):
    """Get the channel's list according the calibration."""
    if calibration == "unnormalized_reflectance":
        return LIST_CHANNEL_SOLAR
    # 8< v1.0
    elif calibration == "reflectance":
        return LIST_CHANNEL_SOLAR
    # >8 v1.0
    elif calibration == "brightness_temperature":
        return LIST_CHANNEL_TERRAN
    else:
        return LIST_TOTAL_CHANNEL


def generate_parameters(calibration):
    """Generate dynamically the parameters."""
    for channel in get_list_channel_calibration(calibration):
        for resolution in resolutions_AF_products(channel):
            yield channel, resolution


def _get_fh_param(filetype, channels, path):
    return {"filetype": filetype, "channels": channels, "filenames": [str(path)]}


@pytest.fixture(scope="module")
def FakeFCIFileHandlerFDHSI_fixture(fdhsi_file):
    """Get the FDHSI test file, including file type and channels."""
    return _get_fh_param("fci_l1c_fdhsi", CHANS_FDHSI, fdhsi_file)


@pytest.fixture(scope="module")
def FakeFCIFileHandlerFDHSIIQTI_fixture(fdhsi_iqti_file):
    """Get the FDHSI IQTI test file, including file type and channels."""
    return _get_fh_param("fci_l1c_fdhsi", CHANS_FDHSI, fdhsi_iqti_file)


@pytest.fixture(scope="module")
def FakeFCIFileHandlerFDHSIOtherCoverage_fixture(fdhsi_file):
    """Get the FDHSI test file for coverages not defined, including file type and channels."""
    return _get_fh_param("fci_l1c_fdhsi", CHANS_FDHSI,
                         _copy_test_file(fdhsi_file, TEST_FILENAMES["fdhsi_other_coverage"][0]))


@pytest.fixture(scope="module")
def FakeFCIFileHandlerFDHSIQ4_fixture(fdhsi_file):
    """Get the FDHSI Q4 test file, including file type and channels."""
    return _get_fh_param("fci_l1c_fdhsi", CHANS_FDHSI, _copy_test_file(fdhsi_file, TEST_FILENAMES["fdhsi_q4"][0]))


@pytest.fixture(scope="module")
def FakeFCIFileHandlerHRFI_fixture(hrfi_file):
    """Get the HRFI test file, including file type and channels."""
    return _get_fh_param("fci_l1c_hrfi", CHANS_HRFI, hrfi_file)


@pytest.fixture(scope="module")
def FakeFCIFileHandlerHRFIOtherCoverage_fixture(hrfi_file):
    """Get the HRFI test file for coverages not defined, including file type and channels."""
    return _get_fh_param("fci_l1c_hrfi", CHANS_HRFI,
                         _copy_test_file(hrfi_file, TEST_FILENAMES["hrfi_other_coverage"][0]))


@pytest.fixture(scope="module")
def FakeFCIFileHandlerHRFIIQTI_fixture(hrfi_iqti_file):
    """Get the HRFI IQTI test file, including file type and channels."""
    return _get_fh_param("fci_l1c_hrfi", CHANS_HRFI, hrfi_iqti_file)


@pytest.fixture(scope="module")
def FakeFCIFileHandlerHRFIQ4_fixture(hrfi_file):
    """Get the HRFI Q4 test file, including file type and channels."""
    return _get_fh_param("fci_l1c_hrfi", CHANS_HRFI, _copy_test_file(hrfi_file, TEST_FILENAMES["hrfi_q4"][0]))


@pytest.fixture
def FakeFCIFileHandlerAF_fixture(af_file_writer, channel, resolution):
    """Get the AF test file of a channel and resolution, including file type and channels."""
    return _get_fh_param("fci_l1c_af", CHANS_AF[f"{channel}_{resolution}"], af_file_writer(channel, resolution))


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
                                          "filenames": TEST_FILENAMES["fdhsi_q4"]},
                             "fdhsi_other_coverage": {"channels": CHANS_FDHSI,
                                          "filenames": TEST_FILENAMES["fdhsi_other_coverage"]},
                             "hrfi_other_coverage": {"channels": CHANS_HRFI,
                                         "filenames": TEST_FILENAMES["hrfi_other_coverage"]}}

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
    def _unnormalized_reflectance_test(tab, filenames):
        """Test of with the unnormalized_reflectance test."""
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
            numpy.testing.assert_array_equal(res[ch][-1], dict_arg["value_2"])
            numpy.testing.assert_array_equal(res[ch][-2], dict_arg["value_1"])
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
        if dict_arg["attrs_dict"]["calibration"] in ["radiance", "brightness_temperature",
                                                     # 8< v1.0
                                                     "reflectance",
                                                     # >8 v1.0
                                                     "unnormalized_reflectance"]:
            self._get_assert_erased_attrs(res, ch)
        if dict_arg["attrs_dict"]["calibration"] in [
                # 8< v1.0
                "reflectance",
                # >8 v1.0
                "unnormalized_reflectance"]:
            self._unnormalized_reflectance_test(res[ch], filenames)
        else:
            self._other_calibration_test(res, ch, dict_arg)

    def _get_res_AF(self, channel, fh_param, calibration, reader_configs):
        """Load the reader for AF data.

        The reader is returned too, as the loaded data can't be computed once it is deleted.
        """
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        type_ter = self._get_type_ter_AF(channel)
        res = reader.load([make_dataid(name=name, calibration=calibration)
                           for name in fh_param["channels"][type_ter]], pad_data=False)
        return reader, res

    @staticmethod
    def _compare_sun_earth_distance(filetype, fh_param, reader_configs):
        """Test the sun earth distance calculation."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        if "IQTI" in fh_param["filenames"][0]:
            np.testing.assert_almost_equal(
                reader.file_handlers[filetype][0]._compute_sun_earth_distance,
                0.996803423, decimal=7)
        else:
            # the mean of the float32 distances in the file, without float32 rounding errors
            np.testing.assert_allclose(
                reader.file_handlers[filetype][0]._compute_sun_earth_distance,
                float(np.float32(EARTH_SUN_DISTANCE_KM)) / EARTH_SUN_DISTANCE_KM, rtol=1e-12)

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
        from satpy.readers.core.loading import load_reader

        reader = load_reader(reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 1

    @pytest.mark.parametrize("filenames", [TEST_FILENAMES["fdhsi"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["hrfi"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["hrfi_q4"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["fdhsi_q4"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["fdhsi_iqti"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["hrfi_iqti"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["hrfi_other_coverage"][0].replace("BODY", "TRAIL"),
                                           TEST_FILENAMES["fdhsi_other_coverage"][0].replace("BODY", "TRAIL")])
    def test_file_pattern_for_TRAIL_file(self, reader_configs, filenames):
        """Test file pattern matching for TRAIL files, which should not be picked up."""
        from satpy.readers.core.loading import load_reader

        reader = load_reader(reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 0

    @pytest.mark.parametrize("calibration", ["counts", "radiance", "brightness_temperature",
                                             # 8< v1.0
                                             "reflectance",
                                             # >8 v1.0
                                             "unnormalized_reflectance"])
    @pytest.mark.parametrize(("fh_param", "res_type", "engine"), [
            *[(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), "hdfi", engine) for engine in ENGINES],
            *[(lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), "hrfi", engine) for engine in ENGINES],
            (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), "hrfi", None),
            (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), "hdfi", None),
            (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), "hrfi", None),
            (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), "hdfi", None),
            (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture"), "hdfi", None),
            (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"), "hrfi", None),
                                                        ])
    def test_load_calibration(self, reader_configs, fh_param,
                              caplog, calibration, res_type, engine):
        """Test loading with counts, radiance, unnormalized_reflectance and bt."""
        expected_res_n = {}
        if calibration == "unnormalized_reflectance":
            list_chan = fh_param["channels"]["solar"]
            list_grid = fh_param["channels"]["solar_grid_type"]
            expected_res_n["hdfi"] = 8
            expected_res_n["hrfi"] = 2
        # 8< v1.0
        elif calibration == "reflectance":
            list_chan = fh_param["channels"]["solar"]
            list_grid = fh_param["channels"]["solar_grid_type"]
            expected_res_n["hdfi"] = 8
            expected_res_n["hrfi"] = 2
        # >8 v1.0
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
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs, engine=engine)
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

    @pytest.mark.parametrize("fh_param", [lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")])
    def test_load_calibration_negative_rad(self, reader_configs, fh_param):
        """Test calibrating negative radiances.

        See https://github.com/pytroll/satpy/issues/3009.
        """
        import satpy
        reader = _get_reader_with_filehandlers(fh_param["filenames"],
                                               reader_configs,
                                               clip_negative_radiances=True)
        did = make_dataid(name="ir_38", calibration="radiance")
        res = reader.load([did], pad_data=False)
        with satpy.config.set({"readers.clip_negative_radiances": True}):
            reader2 = _get_reader_with_filehandlers(fh_param["filenames"],
                                                    reader_configs)
            res2 = reader2.load([did], pad_data=False)
        numpy.testing.assert_array_equal(res["ir_38"][-1, :], 5)  # smallest positive radiance
        numpy.testing.assert_array_equal(res2["ir_38"][-1, :], 5)  # smallest positive radiance
        assert res["ir_38"].dtype == res2["ir_38"].dtype == np.dtype("float32")

    # Satpy's chunks are not aligned with the chunks of the file, as with real files
    @pytest.mark.filterwarnings("ignore:The specified chunks separate the stored chunks:UserWarning")
    @pytest.mark.parametrize("fh_param", [lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")])
    def test_load_with_shared_store(self, reader_configs, fh_param):
        """Test that the "shared_store" open strategy gives the same data as the reader's default one."""
        dataids = [make_dataid(name="ir_38", calibration="radiance"),
                   make_dataid(name="ir_105", calibration="brightness_temperature"),
                   make_dataid(name="vis_06", calibration="unnormalized_reflectance"),
                   "vis_06_pixel_quality", "vis_06_index_map", "vis_06_subsatellite_latitude"]
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs, open_strategy="shared_store")
        res = reader.load(dataids, pad_data=False)
        default_reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        expected = default_reader.load(dataids, pad_data=False)

        assert len(res) == len(expected) == len(dataids)
        for key, expected_data in expected.items():
            assert res[key].dtype == expected_data.dtype
            numpy.testing.assert_array_equal(res[key].values, expected_data.values)

    @pytest.mark.parametrize("fh_param", [lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")])
    def test_drop_fill_value(self, reader_configs, fh_param):
        """Test that _FillValue attribute is dropped for floats.

        See https://github.com/pytroll/satpy/issues/3058.
        """
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        did = make_dataid(name="ir_105", calibration="brightness_temperature")
        res = reader.load([did], pad_data=False)
        assert "_FillValue" not in res["ir_105"].attrs

        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        did = make_dataid(name="vis_06", calibration="reflectance")
        res = reader.load([did], pad_data=False)
        assert "_FillValue" not in res["vis_06"].attrs

        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        did = make_dataid(name="vis_06", calibration="radiance")
        res = reader.load([did], pad_data=False)
        assert "_FillValue" not in res["vis_06"].attrs

        # but for counts it should still be there
        did = make_dataid(name="ir_105", calibration="counts")
        res = reader.load([did], pad_data=False)
        assert "_FillValue" in res["ir_105"].attrs

    @pytest.mark.parametrize(("calibration", "channel", "resolution"), [
        (calibration, channel, resolution)
        for calibration in ["counts",
                            "radiance",
                            "brightness_temperature",
                            # 8< v1.0
                            "reflectance",
                            # >8 v1.0
                            "unnormalized_reflectance"]
        for channel, resolution in generate_parameters(calibration)
    ])
    def test_load_calibration_af(self, FakeFCIFileHandlerAF_fixture, reader_configs, channel, calibration, caplog):
        """Test loading with counts, radiance, unnormalized_reflectance and bt for AF files."""
        expected_res_n = 1
        fh_param = FakeFCIFileHandlerAF_fixture
        type_ter = self._get_type_ter_AF(channel)
        with caplog.at_level(logging.WARNING):
            _reader, res = self._get_res_AF(channel, fh_param, calibration, reader_configs)
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
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"))
                                          ])
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

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize(("fh_param", "expected_pos_info"), [
        (lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["fdhsi"]),
        (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["hrfi"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["hrfi"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["fdhsi"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["hrfi"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["fdhsi"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["fdhsi"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"), EXPECTED_POS_INFO_FOR_FILETYPE["hrfi"])
    ])
    def test_get_segment_position_info(self, reader_configs, fh_param, expected_pos_info, engine):
        """Test the segment position info method."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs, engine=engine)
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
    @pytest.mark.parametrize(("fh_param", "expected_res_n"), [
        (lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), 16),
        (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), 4),
        (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), 4),
        (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), 16),
        (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), 4),
        (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), 16),
        (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture"), 16),
        (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"), 4)])
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
            # Verify that attributes are propagated/inserted
            for attr_key in REQUIRED_ATTR_KEYS:
                assert attr_key in res[f"{ch}_{calibration}"].attrs.keys(), f"Missing attribute: {attr_key}"
            assert res[f"{ch}_{calibration}"].attrs["platform_name"] == "Meteosat-12"

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
            # Verify that attributes are propagated/inserted
            for attr_key in REQUIRED_ATTR_KEYS:
                assert attr_key in res[f"{ch}_{calibration}"].attrs.keys(), f"Missing attribute: {attr_key}"
            assert res[f"{ch}_{calibration}"].attrs["platform_name"] == "Meteosat-12"

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("fh_param", [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"))])
    def test_load_aux_data(self, reader_configs, fh_param, engine):
        """Test loading of auxiliary data."""
        from satpy.readers.fci_l1c_nc import AUX_DATA
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs, engine=engine)
        res = reader.load([fh_param["channels"]["solar"][0] + "_" + key for key in AUX_DATA.keys()],
                          pad_data=False)
        grid_type = fh_param["channels"]["solar_grid_type"][0]
        for aux in [fh_param["channels"]["solar"][0] + "_" + key for key in AUX_DATA.keys()]:
            assert res[aux].shape == (GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["nrows"],
                                      GRID_TYPE_INFO_FOR_TEST_CONTENT[grid_type]["ncols"])
            if (aux == fh_param["channels"]["solar"][0] + "_earth_sun_distance") and ("IQTI" not in
                                                                                      fh_param["filenames"][0]):
                # the earth-sun distance is a float32 in the file
                numpy.testing.assert_array_equal(res[aux][1, 1], np.float32(EARTH_SUN_DISTANCE_KM))
            elif aux == fh_param["channels"]["solar"][0] + "_earth_sun_distance":
                numpy.testing.assert_array_equal(res[aux][1, 1], np.nan)
            else:
                numpy.testing.assert_array_equal(res[aux][1, 1], 10)
            # Verify that attributes are propagated/inserted
            assert res[aux].attrs["platform_name"] == "Meteosat-12"
            assert "time_parameters" in res[aux].attrs.keys()
            assert "orbital_parameters" in res[aux].attrs.keys()
            assert "file_type" in res[aux].attrs.keys()
            assert "name" in res[aux].attrs.keys()
            assert "sensor" in res[aux].attrs.keys()
            assert "resolution" in res[aux].attrs.keys()

    @pytest.mark.parametrize("fh_param", [(lazy_fixture("FakeFCIFileHandlerFDHSI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"))
                                          ])
    def test_platform_name(self, reader_configs, fh_param):
        """Test that platform name is exposed.

        Test that the FCI reader exposes the platform name.  Corresponds
        to GH issue 1014.
        """
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        res = reader.load(["vis_06"], pad_data=False)
        assert res["vis_06"].attrs["platform_name"] == "Meteosat-12"

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
                              (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture"), (2, None,
                                                                                    datetime.datetime.strptime(
                                                                                         "2023-07-22 12:00:00",
                                                                                         "%Y-%m-%d %H:%M:%S"),
                                                                                    datetime.datetime.strptime(
                                                                                         "2023-07-22 12:00:27",
                                                                                         "%Y-%m-%d %H:%M:%S"))),
                              (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"), (2, None,
                                                                                    datetime.datetime.strptime(
                                                                                         "2023-07-22 12:00:00",
                                                                                         "%Y-%m-%d %H:%M:%S"),
                                                                                    datetime.datetime.strptime(
                                                                                         "2023-07-22 12:00:27",
                                                                                         "%Y-%m-%d %H:%M:%S")))
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
                              (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture")),
                              (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"))
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

    @pytest.mark.parametrize(("fh_param"), [ (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture")),
                                             (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"))])
    def test_rc_period_min_log_message_other_coverage(self, reader_configs, fh_param, caplog):
        """Test the rc_period_min error."""
        with caplog.at_level(logging.DEBUG):
            _get_reader_with_filehandlers(fh_param["filenames"], reader_configs)
        assert any('Coverage "xx" not recognised. Using observation times for nominal times.'
                   in message for message in caplog.messages)

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize(("fh_param", "expected_area"), [
        (lazy_fixture("FakeFCIFileHandlerFDHSI_fixture"), ["mtg_fci_fdss_1km", "mtg_fci_fdss_2km"]),
        (lazy_fixture("FakeFCIFileHandlerHRFI_fixture"), ["mtg_fci_fdss_500m", "mtg_fci_fdss_1km"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIQ4_fixture"), ["mtg_fci_fdss_500m", "mtg_fci_fdss_1km"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIQ4_fixture"), ["mtg_fci_fdss_1km", "mtg_fci_fdss_2km"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIIQTI_fixture"), ["mtg_fci_fdss_500m", "mtg_fci_fdss_1km"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIIQTI_fixture"), ["mtg_fci_fdss_1km", "mtg_fci_fdss_2km"]),
        (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture"), ["mtg_fci_fdss_1km", "mtg_fci_fdss_2km"]),
        (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"), ["mtg_fci_fdss_500m", "mtg_fci_fdss_1km"])
    ])
    def test_area_definition_computation(self, reader_configs, fh_param, expected_area, engine):
        """Test that the geolocation computation is correct."""
        reader = _get_reader_with_filehandlers(fh_param["filenames"], reader_configs, engine=engine)
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
                                          (lazy_fixture("FakeFCIFileHandlerFDHSIOtherCoverage_fixture")),
                                          (lazy_fixture("FakeFCIFileHandlerHRFIOtherCoverage_fixture"))
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

    @pytest.mark.parametrize("engine", ENGINES)
    def test_handling_bad_data_ir(self, reader_configs, caplog, fdhsi_bad_data_file, engine):
        """Test handling of bad IR data."""
        reader = _get_reader_with_filehandlers([str(fdhsi_bad_data_file)], reader_configs, engine=engine)
        with caplog.at_level(logging.ERROR):
            reader.load([make_dataid(
                name="ir_105",
                calibration="brightness_temperature")], pad_data=False)
            assert "cannot produce brightness temperature" in caplog.text

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("calibration", [
        # 8< v1.0
        "reflectance",
        # >8 v1.0
        "unnormalized_reflectance"])
    def test_handling_bad_data_vis(self, reader_configs, caplog, calibration, fdhsi_bad_data_file, engine):
        """Test handling of bad VIS data."""
        reader = _get_reader_with_filehandlers([str(fdhsi_bad_data_file)], reader_configs, engine=engine)
        with caplog.at_level(logging.ERROR):
            reader.load([make_dataid(
                name="vis_06",
                calibration=calibration)], pad_data=False)
            assert f"cannot produce {calibration}" in caplog.text
