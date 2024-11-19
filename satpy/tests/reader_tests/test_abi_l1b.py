#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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

"""The abi_l1b reader tests package."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest
import xarray as xr
from pytest_lazy_fixtures import lf as lazy_fixture

from satpy import DataQuery
from satpy.readers.abi_l1b import NC_ABI_L1B
from satpy.readers.yaml_reader import FileYAMLReader
from satpy.utils import ignore_pyproj_proj_warnings

RAD_SHAPE = {
    500: (3000, 5000),  # conus - 500m
}
RAD_SHAPE[1000] = (RAD_SHAPE[500][0] // 2, RAD_SHAPE[500][1] // 2)
RAD_SHAPE[2000] = (RAD_SHAPE[500][0] // 4, RAD_SHAPE[500][1] // 4)


def _create_fake_rad_dataarray(
    rad: xr.DataArray | None = None,
    resolution: int = 2000,
) -> xr.DataArray:
    x_image = xr.DataArray(0.0)
    y_image = xr.DataArray(0.0)
    time = xr.DataArray(0.0)
    shape = RAD_SHAPE[resolution]
    if rad is None:
        rad_data = (np.arange(shape[0] * shape[1]).reshape(shape) + 1.0) * 50.0
        rad_data = (rad_data + 1.0) / 0.5
        rad_data = rad_data.astype(np.int16)
        rad = xr.DataArray(
            da.from_array(rad_data, chunks=226),
            dims=("y", "x"),
            attrs={
                "scale_factor": 0.5,
                "add_offset": -1.0,
                "_FillValue": 1002,
                "units": "W m-2 um-1 sr-1",
                "valid_range": (0, 4095),
            },
        )
    rad.coords["t"] = time
    rad.coords["x_image"] = x_image
    rad.coords["y_image"] = y_image
    return rad


def _create_fake_rad_dataset(rad: xr.DataArray, resolution: int) -> xr.Dataset:
    rad = _create_fake_rad_dataarray(rad=rad, resolution=resolution)

    x__ = xr.DataArray(
        range(rad.shape[1]),
        attrs={"scale_factor": 2.0, "add_offset": -1.0},
        dims=("x",),
    )
    y__ = xr.DataArray(
        range(rad.shape[0]),
        attrs={"scale_factor": -2.0, "add_offset": 1.0},
        dims=("y",),
    )
    proj = xr.DataArray(
        np.int64(0),
        attrs={
            "semi_major_axis": 1.0,
            "semi_minor_axis": 1.0,
            "perspective_point_height": 1.0,
            "longitude_of_projection_origin": -90.0,
            "latitude_of_projection_origin": 0.0,
            "sweep_angle_axis": "x",
        },
    )

    fake_dataset = xr.Dataset(
        data_vars={
            "Rad": rad,
            "band_id": np.array(8),
            # 'x': x__,
            # 'y': y__,
            "x_image": xr.DataArray(0.0),
            "y_image": xr.DataArray(0.0),
            "goes_imager_projection": proj,
            "yaw_flip_flag": np.array([1]),
            "planck_fk1": np.array(13432.1),
            "planck_fk2": np.array(1497.61),
            "planck_bc1": np.array(0.09102),
            "planck_bc2": np.array(0.99971),
            "esun": np.array(2017),
            "nominal_satellite_subpoint_lat": np.array(0.0),
            "nominal_satellite_subpoint_lon": np.array(-89.5),
            "nominal_satellite_height": np.array(35786.02),
            "earth_sun_distance_anomaly_in_AU": np.array(0.99),
        },
        coords={
            "t": rad.coords["t"],
            "x": x__,
            "y": y__,
        },
        attrs={
            "time_coverage_start": "2017-09-20T17:30:40.8Z",
            "time_coverage_end": "2017-09-20T17:41:17.5Z",
        },
    )
    return fake_dataset


def generate_l1b_filename(chan_name: str) -> str:
    """Generate a l1b filename."""
    return f"OR_ABI-L1b-RadC-M4{chan_name}_G16_s20161811540362_e20161811545170_c20161811545230_suffix.nc"


@pytest.fixture
def c01_refl(tmp_path) -> xr.DataArray:
    """Load c01 reflectances."""
    with _apply_dask_chunk_size():
        reader = _create_reader_for_data(tmp_path, "C01", None, 1000)
        return reader.load(["C01"])["C01"]


@pytest.fixture
def c01_rad(tmp_path) -> xr.DataArray:
    """Load c01 radiances."""
    with _apply_dask_chunk_size():
        reader = _create_reader_for_data(tmp_path, "C01", None, 1000)
        return reader.load([DataQuery(name="C01", calibration="radiance")])["C01"]


@pytest.fixture
def c01_rad_h5netcdf(tmp_path) -> xr.DataArray:
    """Load c01 radiances through h5netcdf."""
    shape = RAD_SHAPE[1000]
    rad_data = (np.arange(shape[0] * shape[1]).reshape(shape) + 1.0) * 50.0
    rad_data = (rad_data + 1.0) / 0.5
    rad_data = rad_data.astype(np.int16)
    rad = xr.DataArray(
        da.from_array(rad_data, chunks=226),
        dims=("y", "x"),
        attrs={
            "scale_factor": 0.5,
            "add_offset": -1.0,
            "_FillValue": np.array([1002]),
            "units": "W m-2 um-1 sr-1",
            "valid_range": (0, 4095),
        },
    )
    with _apply_dask_chunk_size():
        reader = _create_reader_for_data(tmp_path, "C01", rad, 1000)
        return reader.load([DataQuery(name="C01", calibration="radiance")])["C01"]


@pytest.fixture
def c01_counts(tmp_path) -> xr.DataArray:
    """Load c01 counts."""
    with _apply_dask_chunk_size():
        reader = _create_reader_for_data(tmp_path, "C01", None, 1000)
        return reader.load([DataQuery(name="C01", calibration="counts")])["C01"]


@pytest.fixture
def c07_bt_creator(tmp_path) -> Callable:
    """Create a loader for c07 brightness temperatures."""
    def _load_data_array(
        clip_negative_radiances: bool = False,
    ):
        rad = _fake_c07_data()
        with _apply_dask_chunk_size():
            reader = _create_reader_for_data(
                tmp_path,
                "C07",
                rad,
                2000,
                {"clip_negative_radiances": clip_negative_radiances},
            )
            return reader.load(["C07"])["C07"]

    return _load_data_array


def _fake_c07_data() -> xr.DataArray:
    shape = RAD_SHAPE[2000]
    values = np.arange(shape[0] * shape[1])
    rad_data = (values.reshape(shape) + 1.0) * 50.0
    rad_data[0, 0] = -0.0001  # introduce below minimum expected radiance
    rad_data = (rad_data + 1.3) / 0.5
    data = rad_data.astype(np.int16)
    rad = xr.DataArray(
        da.from_array(data, chunks=226),
        dims=("y", "x"),
        attrs={
            "scale_factor": 0.5,
            "add_offset": -1.3,
            "_FillValue": np.int16(
                np.floor(((9 + 1) * 50.0 + 1.3) / 0.5)
            ),  # last rad_data value
        },
    )
    return rad


def _create_reader_for_data(
        tmp_path: Path,
        channel_name: str,
        rad: xr.DataArray | None,
        resolution: int,
        reader_kwargs: dict[str, Any] | None = None,
) -> FileYAMLReader:
    filename = generate_l1b_filename(channel_name)
    data_path = tmp_path / filename
    dataset = _create_fake_rad_dataset(rad=rad, resolution=resolution)
    dataset.to_netcdf(
        data_path,
        encoding={
            "Rad": {"chunksizes": [226, 226]},
        },
    )
    from satpy.readers import load_readers
    return load_readers([str(data_path)], "abi_l1b", reader_kwargs=reader_kwargs)["abi_l1b"]


def _apply_dask_chunk_size():
    # 226 on-disk chunk size
    # 8 on-disk chunks for 500 meter data
    # Square (**2) for 2D size
    # 4 bytes for 32-bit floats
    return dask.config.set({"array.chunk-size": ((226 * 8) ** 2) * 4})


def _get_and_check_array(data_arr: xr.DataArray, exp_dtype: npt.DTypeLike) -> npt.NDArray:
    data_np = data_arr.data.compute()
    assert isinstance(data_arr, xr.DataArray)
    assert isinstance(data_arr.data, da.Array)
    assert isinstance(data_np, np.ndarray)
    res = 1000 if RAD_SHAPE[1000][0] == data_np.shape[0] else 2000
    assert data_arr.chunks[0][0] == 226 * (8 / (res / 500))
    assert data_arr.chunks[1][0] == 226 * (8 / (res / 500))

    assert data_np.dtype == data_arr.dtype
    assert data_np.dtype == exp_dtype
    return data_np


def _check_area(data_arr: xr.DataArray) -> None:
    from pyresample.geometry import AreaDefinition

    area_def = data_arr.attrs["area"]
    assert isinstance(area_def, AreaDefinition)

    with ignore_pyproj_proj_warnings():
        proj_dict = area_def.crs.to_dict()
        exp_dict = {
            "h": 1.0,
            "lon_0": -90.0,
            "proj": "geos",
            "sweep": "x",
            "units": "m",
        }
        if "R" in proj_dict:
            assert proj_dict["R"] == 1
        else:
            assert proj_dict["a"] == 1
            assert proj_dict["b"] == 1
        for proj_key, proj_val in exp_dict.items():
            assert proj_dict[proj_key] == proj_val

    assert area_def.shape == data_arr.shape
    if area_def.shape[0] == RAD_SHAPE[1000][0]:
        exp_extent = (-2.0, -2998.0, 4998.0, 2.0)
    else:
        exp_extent = (-2.0, -1498.0, 2498.0, 2.0)
    assert area_def.area_extent == exp_extent


def _check_dims_and_coords(data_arr: xr.DataArray) -> None:
    assert "y" in data_arr.dims
    assert "x" in data_arr.dims

    # we remove any time dimension information
    assert "t" not in data_arr.coords
    assert "t" not in data_arr.dims
    assert "time" not in data_arr.coords
    assert "time" not in data_arr.dims


@pytest.mark.parametrize(
    ("channel", "suffix"),
    [
        ("C{:02d}".format(num), suffix)
        for num in range(1, 17)
        for suffix in ("", "_test_suffix")
    ],
)
def test_file_patterns_match(channel, suffix):
    """Test that the configured file patterns work."""
    from satpy.readers import configs_for_reader, load_reader

    reader_configs = list(configs_for_reader("abi_l1b"))[0]
    reader = load_reader(reader_configs)
    fn1 = (
        "OR_ABI-L1b-RadM1-M3{}_G16_s20182541300210_e20182541300267"
        "_c20182541300308{}.nc"
    ).format(channel, suffix)
    loadables = reader.select_files_from_pathnames([fn1])
    assert len(loadables) == 1
    if not suffix and channel in ["C01", "C02", "C03", "C05"]:
        fn2 = (
            "OR_ABI-L1b-RadM1-M3{}_G16_s20182541300210_e20182541300267"
            "_c20182541300308-000000_0.nc"
        ).format(channel)
        loadables = reader.select_files_from_pathnames([fn2])
        assert len(loadables) == 1


@pytest.mark.parametrize(
    "c01_data_arr", [lazy_fixture("c01_rad"), lazy_fixture("c01_rad_h5netcdf")]
)
class Test_NC_ABI_L1B:
    """Test the NC_ABI_L1B reader."""

    def test_get_dataset(self, c01_data_arr):
        """Test the get_dataset method."""
        exp = {
            "calibration": "radiance",
            "instrument_ID": None,
            "modifiers": (),
            "name": "C01",
            "observation_type": "Rad",
            "orbital_parameters": {
                "projection_altitude": 1.0,
                "projection_latitude": 0.0,
                "projection_longitude": -90.0,
                "satellite_nominal_altitude": 35786020.0,
                "satellite_nominal_latitude": 0.0,
                "satellite_nominal_longitude": -89.5,
                "yaw_flip": True,
            },
            "orbital_slot": None,
            "platform_name": "GOES-16",
            "platform_shortname": "G16",
            "production_site": None,
            "reader": "abi_l1b",
            "resolution": 1000,
            "scan_mode": "M4",
            "scene_abbr": "C",
            "scene_id": None,
            "sensor": "abi",
            "timeline_ID": None,
            "suffix": "suffix",
            "units": "W m-2 um-1 sr-1",
            "start_time": dt.datetime(2017, 9, 20, 17, 30, 40, 800000),
            "end_time": dt.datetime(2017, 9, 20, 17, 41, 17, 500000),
        }

        res = c01_data_arr
        _get_and_check_array(res, np.float32)
        _check_area(res)
        _check_dims_and_coords(res)
        for exp_key, exp_val in exp.items():
            assert res.attrs[exp_key] == exp_val


@pytest.mark.parametrize("clip_negative_radiances", [False, True])
def test_ir_calibrate(c07_bt_creator, clip_negative_radiances):
    """Test IR calibration."""
    res = c07_bt_creator(clip_negative_radiances=clip_negative_radiances)
    clipped_ir = 134.68753 if clip_negative_radiances else np.nan
    expected = np.array(
        [
            clipped_ir,
            304.97037,
            332.22778,
            354.6147,
            374.08688,
            391.58655,
            407.64786,
            422.60635,
            436.68802,
            np.nan,
        ]
    )
    data_np = _get_and_check_array(res, np.float32)
    _check_area(res)
    _check_dims_and_coords(res)
    np.testing.assert_allclose(
        data_np[0, :10], expected, equal_nan=True, atol=1e-04
    )

    # make sure the attributes from the file are in the data array
    assert "scale_factor" not in res.attrs
    assert "_FillValue" not in res.attrs
    assert res.attrs["standard_name"] == "toa_brightness_temperature"
    assert res.attrs["long_name"] == "Brightness Temperature"


def test_vis_calibrate(c01_refl):
    """Test VIS calibration."""
    res = c01_refl
    expected = np.array(
        [
            7.632808,
            15.265616,
            22.898426,
            30.531233,
            38.164043,
            45.796852,
            53.429657,
            61.062466,
            68.695274,
            np.nan,
        ]
    )
    data_np = _get_and_check_array(res, np.float32)
    _check_area(res)
    _check_dims_and_coords(res)
    np.testing.assert_allclose(data_np[0, :10], expected, equal_nan=True)
    assert "scale_factor" not in res.attrs
    assert "_FillValue" not in res.attrs
    assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"
    assert res.attrs["long_name"] == "Bidirectional Reflectance"


def test_raw_calibrate(c01_counts):
    """Test RAW calibration."""
    res = c01_counts

    # We expect the raw data to be unchanged
    _get_and_check_array(res, np.int16)
    _check_area(res)
    _check_dims_and_coords(res)

    # check for the presence of typical attributes
    assert "scale_factor" in res.attrs
    assert "add_offset" in res.attrs
    assert "_FillValue" in res.attrs
    assert "orbital_parameters" in res.attrs
    assert "platform_shortname" in res.attrs
    assert "scene_id" in res.attrs

    # determine if things match their expected values/types.
    assert res.attrs["standard_name"] == "counts"
    assert res.attrs["long_name"] == "Raw Counts"


@mock.patch("satpy.readers.abi_base.xr")
def test_open_dataset(_):  # noqa: PT019
    """Test opening a dataset."""
    openable_thing = mock.MagicMock()

    NC_ABI_L1B(openable_thing, {"platform_shortname": "g16"}, {})
    openable_thing.open.assert_called()
