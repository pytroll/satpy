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

import datetime as dt
import shutil
from pathlib import Path
from typing import Iterable

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from pyresample import SwathDefinition
from pytest import TempPathFactory  # noqa: PT013
from pytest_lazy_fixtures import lf as lazy_fixture

from satpy.tests.utils import RANDOM_GEN

I_COLS = 6400
I_ROWS = 32  # one scan
M_COLS = 3200
M_ROWS = 16  # one scan
START_TIME = dt.datetime(2023, 5, 30, 17, 55, 41, 0)
END_TIME = dt.datetime(2023, 5, 30, 17, 57, 5, 0)
QF1_FLAG_MEANINGS = """
\tBits are listed from the MSB (bit 7) to the LSB (bit 0):
\tBit    Description
\t6-7    SUN GLINT;
\t       00 -- none
\t       01 -- geometry based
\t       10 -- wind speed based
\t       11 -- geometry & wind speed based
\t5      low sun mask;
\t       0 -- high
\t       1 -- low
\t4      day/night;
\t       0 -- day
\t       1 -- night
\t2-3    cloud detection & confidence;
\t       00 -- confident clear
\t       01 -- probably clear
\t       10 -- probably cloudy
\t       11 -- confident cloudy
\t0-1    cloud mask quality;
\t       00 -- poor
\t       01 -- low
\t       10 -- medium
\t       11 -- high
"""


@pytest.fixture(scope="module")
def surface_reflectance_file(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake surface reflectance EDR file."""
    return _create_surface_reflectance_file(tmp_path_factory, START_TIME, include_veg_indices=False)


@pytest.fixture(scope="module")
def surface_reflectance_file2(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake surface reflectance EDR file."""
    return _create_surface_reflectance_file(tmp_path_factory, START_TIME + dt.timedelta(minutes=5),
                                            include_veg_indices=False)


@pytest.fixture(scope="module")
def multiple_surface_reflectance_files(surface_reflectance_file, surface_reflectance_file2) -> list[Path]:
    """Get two multiple surface reflectance files."""
    return [surface_reflectance_file, surface_reflectance_file2]


@pytest.fixture(scope="module")
def surface_reflectance_with_veg_indices_file(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake surface reflectance EDR file with vegetation indexes included."""
    return _create_surface_reflectance_file(tmp_path_factory, START_TIME, include_veg_indices=True)


@pytest.fixture(scope="module")
def surface_reflectance_with_veg_indices_file2(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake surface reflectance EDR file with vegetation indexes included."""
    return _create_surface_reflectance_file(tmp_path_factory, START_TIME + dt.timedelta(minutes=5),
                                            include_veg_indices=True)


@pytest.fixture(scope="module")
def multiple_surface_reflectance_files_with_veg_indices(surface_reflectance_with_veg_indices_file,
                                                        surface_reflectance_with_veg_indices_file2) -> list[Path]:
    """Get two multiple surface reflectance files with vegetation indexes included."""
    return [surface_reflectance_with_veg_indices_file, surface_reflectance_with_veg_indices_file2]


def _create_surface_reflectance_file(
        tmp_path_factory: TempPathFactory,
        start_time: dt.datetime,
        include_veg_indices: bool = False,
) -> Path:
    fn = f"SurfRefl_v1r2_npp_s{start_time:%Y%m%d%H%M%S}0_e{END_TIME:%Y%m%d%H%M%S}0_c202305302025590.nc"
    sr_vars = _create_surf_refl_variables()
    if include_veg_indices:
        sr_vars.update(_create_veg_index_variables())
    return _create_fake_file(tmp_path_factory, fn, sr_vars)


def _create_surf_refl_variables() -> dict[str, xr.DataArray]:
    dim_y_750 = "Along_Track_750m"
    dim_x_750 = "Along_Scan_750m"
    m_dims = (dim_y_750, dim_x_750)
    dim_y_375 = "Along_Track_375m"
    dim_x_375 = "Along_Scan_375m"
    i_dims = (dim_y_375, dim_x_375)

    lon_attrs = {"standard_name": "longitude", "units": "degrees_east", "_FillValue": -999.9,
                 "valid_min": -180.0, "valid_max": 180.0}
    lat_attrs = {"standard_name": "latitude", "units": "degrees_north", "_FillValue": -999.9,
                 "valid_min": -90.0, "valid_max": 90.0}
    sr_attrs = {"units": "unitless", "_FillValue": -9999,
                "scale_factor": np.float32(0.0001), "add_offset": np.float32(0.0)}

    i_data = RANDOM_GEN.random((I_ROWS, I_COLS)).astype(np.float32)
    m_data = RANDOM_GEN.random((M_ROWS, M_COLS)).astype(np.float32)
    lon_i_data = (i_data * 360) - 180.0
    lon_m_data = (m_data * 360) - 180.0
    lat_i_data = (i_data * 180) - 90.0
    lat_m_data = (m_data * 180) - 90.0
    for geo_var in (lon_i_data, lon_m_data, lat_i_data, lat_m_data):
        geo_var[0, 0] = -999.9
        geo_var[0, 1] = -999.3
    data_arrs = {
        "Longitude_at_375m_resolution": xr.DataArray(lon_i_data, dims=i_dims, attrs=lon_attrs),
        "Latitude_at_375m_resolution": xr.DataArray(lat_i_data, dims=i_dims, attrs=lat_attrs),
        "Longitude_at_750m_resolution": xr.DataArray(lon_m_data, dims=m_dims, attrs=lon_attrs),
        "Latitude_at_750m_resolution": xr.DataArray(lat_m_data, dims=m_dims, attrs=lat_attrs),
        "375m Surface Reflectance Band I1": xr.DataArray(i_data, dims=i_dims, attrs=sr_attrs),
        "750m Surface Reflectance Band M1": xr.DataArray(m_data, dims=m_dims, attrs=sr_attrs),
    }
    for data_arr in data_arrs.values():
        data_arr.encoding["chunksizes"] = data_arr.shape
        if "scale_factor" not in data_arr.attrs:
            continue
        data_arr.encoding["dtype"] = np.int16
        data_arr.encoding["scale_factor"] = data_arr.attrs.pop("scale_factor")
        data_arr.encoding["add_offset"] = data_arr.attrs.pop("add_offset")
    return data_arrs


def _create_veg_index_variables() -> dict[str, xr.DataArray]:
    dim_y_750 = "Along_Track_750m"
    dim_x_750 = "Along_Scan_750m"
    m_dims = (dim_y_750, dim_x_750)
    dim_y_375 = "Along_Track_375m"
    dim_x_375 = "Along_Scan_375m"
    i_dims = (dim_y_375, dim_x_375)

    vi_data = np.zeros((I_ROWS, I_COLS), dtype=np.float32)
    vi_data[0, :7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    data_arrs = {
        "NDVI": xr.DataArray(vi_data, dims=i_dims, attrs={"units": "unitless"}),
        "EVI": xr.DataArray(vi_data, dims=i_dims, attrs={"units": "unitless"}),
    }
    data_arrs["NDVI"].encoding["dtype"] = np.float32
    data_arrs["EVI"].encoding["dtype"] = np.float32

    # Quality Flags are from the Surface Reflectance data, but only used for VI products in the reader
    for qf_num in range(1, 8):
        qf_name = f"QF{qf_num} Surface Reflectance"
        qf_data = np.zeros((M_ROWS, M_COLS), dtype=np.uint8)
        bad_qf_start = 4  # 0.5x the last test pixel set in "vi_data" above (I-band versus M-band index)
        if qf_num == 1:
            qf_data[:, :] |= 0b00000010  # medium cloud mask quality everywhere
            qf_data[0, bad_qf_start] |= 0b11000000  # sun glint
            qf_data[0, bad_qf_start + 1] |= 0b00001100  # cloudy
            qf_data[0, bad_qf_start + 2] = 0b00000001  # low cloud mask quality
        elif qf_num == 2:
            qf_data[:, :] |= 0b00000011  # desert everywhere
            qf_data[0, bad_qf_start + 3] |= 0b00100000  # snow or ice
            qf_data[0, bad_qf_start + 4] |= 0b00001000  # cloud shadow
            qf_data[0, bad_qf_start + 5] = 0b00000001  # deep ocean
        elif qf_num == 7:
            qf_data[0, bad_qf_start + 6] |= 0b00001100  # high aerosol
            qf_data[0, bad_qf_start + 7] |= 0b00000010  # adjacent to cloud

        data_arr = xr.DataArray(qf_data, dims=m_dims, attrs={"flag_meanings": QF1_FLAG_MEANINGS})
        data_arr.encoding["dtype"] = np.uint8
        data_arrs[qf_name] = data_arr
    return data_arrs


@pytest.fixture(scope="module")
def cloud_height_file(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake CloudHeight VIIRS EDR file."""
    fn = f"JRR-CloudHeight_v3r2_npp_s{START_TIME:%Y%m%d%H%M%S}0_e{END_TIME:%Y%m%d%H%M%S}0_c202307231023395.nc"
    data_vars = _create_continuous_variables(
        ("CldTopTemp", "CldTopHght", "CldTopPres")
    )
    lon_pc = data_vars["Longitude"].copy(deep=True)
    lat_pc = data_vars["Latitude"].copy(deep=True)
    lon_pc.attrs["long_name"] = "BAD"
    lat_pc.attrs["long_name"] = "BAD"
    del lon_pc.encoding["_FillValue"]
    del lat_pc.encoding["_FillValue"]
    data_vars["Longitude_Pc"] = lon_pc
    data_vars["Latitude_Pc"] = lat_pc
    return _create_fake_file(tmp_path_factory, fn, data_vars)


@pytest.fixture(scope="module")
def aod_file(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake AOD VIIRs EDR file."""
    fn = f"JRR-AOD_v3r2_npp_s{START_TIME:%Y%m%d%H%M%S}0_e{END_TIME:%Y%m%d%H%M%S}0_c202307231023395.nc"
    data_vars = _create_continuous_variables(
        ("AOD550",),
        data_attrs={
            "valid_range": [-0.5, 0.5],
            "units": "1",
            "_FillValue": -999.999,
        }
    )
    qc_data = np.zeros(data_vars["AOD550"].shape, dtype=np.int8)
    qc_data[-1, -1] = 2
    data_vars["QCAll"] = xr.DataArray(
        qc_data,
        dims=data_vars["AOD550"].dims,
        attrs={"valid_range": [0, 3]},
    )
    data_vars["QCAll"].encoding["_FillValue"] = -128
    return _create_fake_file(tmp_path_factory, fn, data_vars)


@pytest.fixture(scope="module")
def lst_file(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake VLST EDR file."""
    fn = f"LST_v2r0_npp_s{START_TIME:%Y%m%d%H%M%S}0_e{END_TIME:%Y%m%d%H%M%S}0_c202307241854058.nc"
    data_vars = _create_lst_variables()
    return _create_fake_file(tmp_path_factory, fn, data_vars)


def _create_lst_variables() -> dict[str, xr.DataArray]:
    data_vars = _create_continuous_variables(("VLST",))

    # VLST scale factors
    data_vars["VLST"].data = (data_vars["VLST"].data / 0.0001).astype(np.int16)
    data_vars["VLST"].encoding.pop("scale_factor")
    data_vars["VLST"].encoding.pop("add_offset")
    data_vars["LST_ScaleFact"] = xr.DataArray(np.float32(0.0001))
    data_vars["LST_Offset"] = xr.DataArray(np.float32(0.0))

    return data_vars


@pytest.fixture(scope="module")
def volcanic_ash_file(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake *partial* Volcanic Ash VIIRs EDR file."""
    fn = f"JRR-VolcanicAsh_v3r0_j01_s{START_TIME:%Y%m%d%H%M%S}0_e{END_TIME:%Y%m%d%H%M%S}0_c202307231023395.nc"
    data_vars = _create_continuous_variables(
        ("AshBeta",),
        data_attrs={
            "units": "1",
            "_FillValue": -999.,
        }
    )
    # The 'Det_QF_Size' variable is actually a scalar, but the there's no way to check it is dropped other than
    # making it 2D
    data_vars["Det_QF_Size"] = xr.DataArray(np.array([[1, 2]], dtype=np.int32),
                                            attrs={"_FillValue": -999, "units": "1"})
    return _create_fake_file(tmp_path_factory, fn, data_vars)


@pytest.fixture(scope="module")
def cloud_phase_file(tmp_path_factory: TempPathFactory) -> Path:
    """Generate fake AOD VIIRs EDR file."""
    fn = f"JRR-CloudPhase_v3r2_npp_s{START_TIME:%Y%m%d%H%M%S}0_e{END_TIME:%Y%m%d%H%M%S}0_c202307231023395.nc"

    # get lon/lat variables
    data_vars = _create_continuous_variables([])
    phase_data = (RANDOM_GEN.random((M_ROWS, M_COLS)) * 6).astype(np.int8)
    cloud_phase = xr.DataArray(
        phase_data,
        dims=("Rows", "Columns"),
        attrs={
            "valid_range": [np.int8(0), np.int8(5)],
            "units": "1",
            "coordinates": "Longitude Latitude",
        },
    )
    cloud_phase.encoding["_FillValue"] = np.int8(-128)
    cloud_phase.encoding["dtype"] = np.int8
    data_vars["CloudPhase"] = cloud_phase
    return _create_fake_file(tmp_path_factory, fn, data_vars)


def _create_continuous_variables(
        var_names: Iterable[str],
        data_attrs: None | dict = None
) -> dict[str, xr.DataArray]:
    dims = ("Rows", "Columns")

    lon_attrs = {"standard_name": "longitude", "units": "degrees_east", "_FillValue": -999.9}
    lat_attrs = {"standard_name": "latitude", "units": "degrees_north", "_FillValue": -999.9}
    cont_attrs = data_attrs
    if cont_attrs is None:
        cont_attrs = {"units": "Kelvin", "_FillValue": -9999,
                      "scale_factor": np.float32(0.0001), "add_offset": np.float32(0.0)}

    m_data = RANDOM_GEN.random((M_ROWS, M_COLS)).astype(np.float32)
    data_arrs = {
        "Longitude": xr.DataArray(m_data, dims=dims, attrs=lon_attrs),
        "Latitude": xr.DataArray(m_data, dims=dims, attrs=lat_attrs),
    }
    cont_data = m_data
    if "valid_range" in cont_attrs:
        valid_range = cont_attrs["valid_range"]
        # scale 0-1 random data to fit in valid_range
        cont_data = cont_data * (valid_range[1] - valid_range[0]) + valid_range[0]
    for var_name in var_names:
        data_arrs[var_name] = xr.DataArray(cont_data, dims=dims, attrs=cont_attrs)
    for data_arr in data_arrs.values():
        if "_FillValue" in data_arr.attrs:
            data_arr.encoding["_FillValue"] = data_arr.attrs.pop("_FillValue")
        data_arr.encoding["coordinates"] = "Longitude Latitude"
        if "scale_factor" not in data_arr.attrs:
            continue
        data_arr.encoding["dtype"] = np.int16
        data_arr.encoding["scale_factor"] = data_arr.attrs.pop("scale_factor")
        data_arr.encoding["add_offset"] = data_arr.attrs.pop("add_offset")
    return data_arrs


def _create_fake_file(tmp_path_factory: TempPathFactory, filename: str, data_arrs: dict[str, xr.DataArray]) -> Path:
    tmp_path = tmp_path_factory.mktemp("viirs_edr_tmp")
    file_path = tmp_path / filename
    ds = _create_fake_dataset(data_arrs)
    ds.to_netcdf(file_path)
    return file_path


def _create_fake_dataset(vars_dict: dict[str, xr.DataArray]) -> xr.Dataset:
    ds = xr.Dataset(
        vars_dict,
        attrs={}
    )
    return ds


def test_available_datasets(aod_file):
    """Test that available datasets doesn't claim non-filetype datasets.

    For example, if a YAML-configured dataset's file type is not loaded
    then the available status is `None` and should remain `None`. This
    means no file type knows what to do with this dataset. If it is
    `False` then that means that a file type knows of the dataset, but
    that the variable is not available in the file. In the below test
    this isn't the case so the YAML-configured dataset should be
    provided once and have a `None` availability.

    """
    from satpy.readers.viirs_edr import VIIRSJRRFileHandler
    file_handler = VIIRSJRRFileHandler(
        aod_file,
        {"platform_shortname": "npp"},
        {"file_type": "jrr_aod"},
    )
    fake_yaml_datasets = [
        (None, {"file_key": "fake", "file_type": "fake_file", "name": "fake"}),
    ]
    available_datasets = list(file_handler.available_datasets(configured_datasets=fake_yaml_datasets))
    fake_availables = [avail_tuple for avail_tuple in available_datasets if avail_tuple[1]["name"] == "fake"]
    assert len(fake_availables) == 1
    assert fake_availables[0][0] is None


class TestVIIRSJRRReader:
    """Test the VIIRS JRR L2 reader."""

    @pytest.mark.parametrize(
        "data_files",
        [
            lazy_fixture("surface_reflectance_file"),
            lazy_fixture("multiple_surface_reflectance_files"),
        ],
    )
    def test_get_dataset_surf_refl(self, data_files):
        """Test retrieval of datasets."""
        from satpy import Scene

        if not isinstance(data_files, list):
            data_files = [data_files]
        is_multiple = len(data_files) > 1
        bytes_in_m_row = 4 * 3200
        with dask.config.set({"array.chunk-size": f"{bytes_in_m_row * 4}B"}):
            scn = Scene(reader="viirs_edr", filenames=data_files)
            scn.load(["surf_refl_I01", "surf_refl_M01"])
        assert scn.start_time == START_TIME
        assert scn.end_time == END_TIME
        _check_surf_refl_data_arr(scn["surf_refl_I01"], multiple_files=is_multiple)
        _check_surf_refl_data_arr(scn["surf_refl_M01"], multiple_files=is_multiple)

    @pytest.mark.parametrize("filter_veg", [False, True])
    @pytest.mark.parametrize(
        "data_files",
        [
            lazy_fixture("surface_reflectance_with_veg_indices_file2"),
            lazy_fixture("multiple_surface_reflectance_files_with_veg_indices"),
        ],
    )
    def test_get_dataset_surf_refl_with_veg_idx(
            self,
            data_files,
            filter_veg,
    ):
        """Test retrieval of vegetation indices from surface reflectance files."""
        from satpy import Scene

        if not isinstance(data_files, list):
            data_files = [data_files]
        is_multiple = len(data_files) > 1
        bytes_in_m_row = 4 * 3200
        with dask.config.set({"array.chunk-size": f"{bytes_in_m_row * 4}B"}):
            scn = Scene(reader="viirs_edr", filenames=data_files,
                        reader_kwargs={"filter_veg": filter_veg})
            scn.load(["NDVI", "EVI", "surf_refl_qf1"])
        _check_vi_data_arr(scn["NDVI"], filter_veg, is_multiple)
        _check_vi_data_arr(scn["EVI"], filter_veg, is_multiple)
        _check_surf_refl_qf_data_arr(scn["surf_refl_qf1"], is_multiple)

    @pytest.mark.parametrize(
        ("var_names", "data_file"),
        [
            (("CldTopTemp", "CldTopHght", "CldTopPres"), lazy_fixture("cloud_height_file")),
            (("VLST",), lazy_fixture("lst_file")),
        ]
    )
    def test_get_dataset_generic(self, var_names, data_file):
        """Test datasets from cloud height files."""
        from satpy import Scene
        bytes_in_m_row = 4 * 3200
        with dask.config.set({"array.chunk-size": f"{bytes_in_m_row * 4}B"}):
            scn = Scene(reader="viirs_edr", filenames=[data_file])
            scn.load(var_names)
        for var_name in var_names:
            _check_continuous_data_arr(scn[var_name])

    def test_get_dataset_category(self, cloud_phase_file):
        """Test loading category (integer) data products."""
        from satpy import Scene
        bytes_in_m_row = 4 * 3200
        with dask.config.set({"array.chunk-size": f"{bytes_in_m_row * 4}B"}):
            scn = Scene(reader="viirs_edr", filenames=[cloud_phase_file])
            scn.load(["CloudPhase"])
        data_arr = scn["CloudPhase"]
        _array_checks(data_arr, dtype=np.int8)
        _shared_metadata_checks(data_arr)

    @pytest.mark.parametrize(
        ("aod_qc_filter", "exp_masked_pixel"),
        [
            (None, False),
            (0, True),
            (2, False)
        ],
    )
    def test_get_aod_filtered(self, aod_file, aod_qc_filter, exp_masked_pixel):
        """Test that the AOD product can be loaded and filtered."""
        from satpy import Scene
        bytes_in_m_row = 4 * 3200
        with dask.config.set({"array.chunk-size": f"{bytes_in_m_row * 4}B"}):
            scn = Scene(reader="viirs_edr", filenames=[aod_file], reader_kwargs={"aod_qc_filter": aod_qc_filter})
            scn.load(["AOD550"])
        _check_continuous_data_arr(scn["AOD550"])
        data_np = scn["AOD550"].data.compute()
        pixel_is_nan = np.isnan(data_np[-1, -1])
        assert pixel_is_nan if exp_masked_pixel else not pixel_is_nan

        # filtering should never affect geolocation
        lons, lats = scn["AOD550"].attrs["area"].get_lonlats()
        assert not np.isnan(lons[-1, -1].compute())
        assert not np.isnan(lats[-1, -1].compute())

    @pytest.mark.parametrize(
        ("data_file", "exp_available"),
        [
            (lazy_fixture("surface_reflectance_file"), False),
            (lazy_fixture("surface_reflectance_with_veg_indices_file"), True),
        ]
    )
    def test_availability_veg_idx(self, data_file, exp_available):
        """Test that vegetation indexes aren't available when they aren't present."""
        from satpy import Scene
        scn = Scene(reader="viirs_edr", filenames=[data_file])
        avail = scn.available_dataset_names()
        if exp_available:
            assert "NDVI" in avail
            assert "EVI" in avail
        else:
            assert "NDVI" not in avail
            assert "EVI" not in avail

    @pytest.mark.parametrize(
        ("filename_platform", "exp_shortname"),
        [
            ("npp", "Suomi-NPP"),
            ("JPSS-1", "NOAA-20"),
            ("J01", "NOAA-20"),
            ("n21", "NOAA-21")
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

    def test_volcanic_ash_drop_variables(self, volcanic_ash_file):
        """Test that Det_QF_Size variable is dropped when reading VolcanicAsh products.

        The said variable is also used as a dimension in v3r0 files, so the reading fails
        if it is not dropped.
        """
        from satpy import Scene
        scn = Scene(reader="viirs_edr", filenames=[volcanic_ash_file])
        available = scn.available_dataset_names()
        assert "Det_QF_Size" not in available


def _check_surf_refl_qf_data_arr(data_arr: xr.DataArray, multiple_files: bool) -> None:
    _array_checks(data_arr, dtype=np.uint8, multiple_files=multiple_files)
    _shared_metadata_checks(data_arr)
    assert data_arr.attrs["units"] == "1"
    assert data_arr.attrs["standard_name"] == "quality_flag"


def _check_vi_data_arr(data_arr: xr.DataArray, is_filtered: bool, multiple_files: bool) -> None:
    _array_checks(data_arr, multiple_files=multiple_files)
    _shared_metadata_checks(data_arr)
    assert data_arr.attrs["units"] == "1"
    assert data_arr.attrs["standard_name"] == "normalized_difference_vegetation_index"

    data = data_arr.data.compute()
    if is_filtered:
        np.testing.assert_allclose(data[0, :7], [np.nan, -1.0, -0.5, 0.0, 0.5, 1.0, np.nan])
        np.testing.assert_allclose(data[0, 8:8 + 16], np.nan)
        np.testing.assert_allclose(data[0, 8 + 16:], 0.0)
    else:
        np.testing.assert_allclose(data[0, :7], [np.nan, -1.0, -0.5, 0.0, 0.5, 1.0, np.nan])
        np.testing.assert_allclose(data[0, 8:], 0.0)


def _check_surf_refl_data_arr(
        data_arr: xr.DataArray,
        dtype: npt.DType = np.float32,
        multiple_files: bool = False
) -> None:
    _array_checks(data_arr, dtype, multiple_files=multiple_files)
    data = data_arr.data.compute()
    assert data.max() > 1.0  # random 0-1 test data multiplied by 100

    _shared_metadata_checks(data_arr)
    assert data_arr.attrs["units"] == "%"
    assert data_arr.attrs["standard_name"] == "surface_bidirectional_reflectance"


def _check_continuous_data_arr(data_arr: xr.DataArray) -> None:
    _array_checks(data_arr)

    if "valid_range" not in data_arr.attrs and "valid_min" not in data_arr.attrs:
        # random sample should be between 0 and 1 only if factor/offset applied
        exp_range = (0, 1)
    else:
        # if there is a valid range then we shouldn't be outside it
        exp_range = data_arr.attrs.get("valid_range",
                                       (data_arr.attrs.get("valid_min"), data_arr.attrs.get("valid_max")))
    data = data_arr.data.compute()
    assert not (data < exp_range[0]).any()
    assert not (data > exp_range[1]).any()

    _shared_metadata_checks(data_arr)


def _array_checks(data_arr: xr.DataArray, dtype: npt.Dtype = np.float32, multiple_files: bool = False) -> None:
    assert data_arr.dims == ("y", "x")
    assert isinstance(data_arr.attrs["area"], SwathDefinition)
    assert data_arr.attrs["area"].shape == data_arr.shape
    assert isinstance(data_arr.data, da.Array)
    assert np.issubdtype(data_arr.data.dtype, dtype)
    data_np = data_arr.data.compute()
    assert data_np.dtype == data_arr.dtype
    is_mband_res = _is_mband_res(data_arr)
    shape_multiplier = 1 + int(multiple_files)
    exp_shape = (M_ROWS * shape_multiplier, M_COLS) if is_mband_res else (I_ROWS * shape_multiplier, I_COLS)
    assert data_arr.shape == exp_shape
    exp_row_chunks = 4 if is_mband_res else 8
    assert all(c == exp_row_chunks for c in data_arr.chunks[0])
    assert data_arr.chunks[1] == (exp_shape[1],)


def _shared_metadata_checks(data_arr: xr.DataArray) -> None:
    is_mband_res = _is_mband_res(data_arr)
    exp_rps = 16 if is_mband_res else 32
    assert data_arr.attrs["sensor"] == "viirs"
    assert data_arr.attrs["rows_per_scan"] == exp_rps

    lons = data_arr.attrs["area"].lons
    lats = data_arr.attrs["area"].lats
    assert lons.attrs["rows_per_scan"] == exp_rps
    assert lats.attrs["rows_per_scan"] == exp_rps
    assert lons.min() >= -180.0
    assert lons.max() <= 180.0
    assert lats.min() >= -90.0
    assert lats.max() <= 90.0
    # Some files (ex. CloudHeight) have other lon/lats that shouldn't be used
    assert lons.attrs.get("long_name") != "BAD"
    assert lats.attrs.get("long_name") != "BAD"

    if "valid_range" in data_arr.attrs:
        valid_range = data_arr.attrs["valid_range"]
        assert isinstance(valid_range, tuple)
        assert len(valid_range) == 2

    if np.issubdtype(data_arr.dtype, np.floating):
        # floating point arrays always use NaN as fill and should not specify it in attrs
        assert "_FillValue" not in data_arr.attrs
    if "_FillValue" in data_arr.attrs:
        # make sure fill vlaue is scalar
        assert not hasattr(data_arr.attrs["_FillValue"], "shape")


def _is_mband_res(data_arr: xr.DataArray) -> bool:
    return "I" not in data_arr.attrs["name"]  # includes NDVI and EVI
