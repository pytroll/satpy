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

import contextlib
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.readers.abi_l1b import NC_ABI_L1B
from satpy.tests.utils import make_dataid

RAD_SHAPE = {
    500: (3000, 5000),  # conus - 500m
    1000: (1500, 2500),  # conus - 1km
    2000: (750, 1250),  # conus - 2km
}


def _create_fake_rad_dataarray(
    rad: xr.DataArray | None = None,
    # resolution: int = 2000,
) -> xr.DataArray:
    x_image = xr.DataArray(0.0)
    y_image = xr.DataArray(0.0)
    time = xr.DataArray(0.0)
    shape = (2, 5)  # RAD_SHAPE[resolution]
    if rad is None:
        rad_data = (np.arange(shape[0] * shape[1]).reshape(shape) + 1.0) * 50.0
        rad_data = (rad_data + 1.0) / 0.5
        rad_data = rad_data.astype(np.int16)
        rad = xr.DataArray(
            da.from_array(rad_data),
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


def _create_fake_rad_dataset(rad=None):
    rad = _create_fake_rad_dataarray(rad=rad)

    x__ = xr.DataArray(
        range(5), attrs={"scale_factor": 2.0, "add_offset": -1.0}, dims=("x",)
    )
    y__ = xr.DataArray(
        range(2), attrs={"scale_factor": -2.0, "add_offset": 1.0}, dims=("y",)
    )
    proj = xr.DataArray(
        [],
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
    return f"OR_ABI-L1b-RadC-M4{chan_name}_G16_s20161811540362_e20161811545170_c20161811545230.nc"


@pytest.fixture(scope="module")
def l1b_c01_file(tmp_path_factory) -> list[Path]:
    filename = generate_l1b_filename("C01")
    data_path = tmp_path_factory.mktemp("abi_l1b").join(filename)
    dataset = _create_fake_rad_dataset()
    dataset.to_netcdf(data_path)
    return [data_path]


@pytest.fixture(scope="module")
def l1b_all_files(
    l1b_c01_file,
) -> list[Path]:
    return l1b_c01_file


@contextlib.contextmanager
def create_file_handler(
    rad: xr.DataArray | None = None,
    clip_negative_radiances: bool = False,
    filetype_resolution: int = 0,
) -> Iterator[NC_ABI_L1B]:
    """Create a fake dataset using the given radiance data."""

    ft_info: dict[str, Any] = {"filetype": "info"}
    if filetype_resolution:
        ft_info["resolution"] = filetype_resolution

    with mock.patch("satpy.readers.abi_base.xr") as xr_:
        xr_.open_dataset.return_value = _create_fake_rad_dataset(rad=rad)
        file_handler = NC_ABI_L1B(
            "filename",
            {
                "platform_shortname": "G16",
                "observation_type": "Rad",
                "suffix": "custom",
                "scene_abbr": "C",
                "scan_mode": "M3",
            },
            ft_info,
            clip_negative_radiances=clip_negative_radiances,
        )
        yield file_handler


class TestABIYAML:
    """Tests for the ABI L1b reader's YAML configuration."""

    @pytest.mark.parametrize(
        ("channel", "suffix"),
        [
            ("C{:02d}".format(num), suffix)
            for num in range(1, 17)
            for suffix in ("", "_test_suffix")
        ],
    )
    def test_file_patterns_match(self, channel, suffix):
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


class Test_NC_ABI_L1B:
    """Test the NC_ABI_L1B reader."""

    @property
    def fake_rad(self):
        """Create fake data for these tests.

        Needs to be an instance method so the subclass can override it.

        """
        return None  # use default from file handler creator

    def test_basic_attributes(self):
        """Test getting basic file attributes."""
        from datetime import datetime

        with create_file_handler(rad=self.fake_rad) as file_handler:
            assert file_handler.start_time == datetime(2017, 9, 20, 17, 30, 40, 800000)
            assert file_handler.end_time == datetime(2017, 9, 20, 17, 41, 17, 500000)

    def test_get_dataset(self):
        """Test the get_dataset method."""
        key = make_dataid(name="Rad", calibration="radiance")
        with create_file_handler(rad=self.fake_rad) as file_handler:
            res = file_handler.get_dataset(key, {"info": "info"})
        exp = {
            "calibration": "radiance",
            "instrument_ID": None,
            "modifiers": (),
            "name": "Rad",
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
            "scan_mode": "M3",
            "scene_abbr": "C",
            "scene_id": None,
            "sensor": "abi",
            "timeline_ID": None,
            "suffix": "custom",
            "units": "W m-2 um-1 sr-1",
        }

        assert res.attrs == exp
        # we remove any time dimension information
        assert "t" not in res.coords
        assert "t" not in res.dims
        assert "time" not in res.coords
        assert "time" not in res.dims

    @mock.patch("satpy.readers.abi_base.geometry.AreaDefinition")
    def test_get_area_def(self, adef):
        """Test the area generation."""
        with create_file_handler(rad=self.fake_rad) as file_handler:
            file_handler.get_area_def(None)

            assert adef.call_count == 1
            call_args = tuple(adef.call_args)[0]
            assert call_args[3] == {
                "a": 1.0,
                "b": 1.0,
                "h": 1.0,
                "lon_0": -90.0,
                "proj": "geos",
                "sweep": "x",
                "units": "m",
            }
            assert call_args[4] == file_handler.ncols
            assert call_args[5] == file_handler.nlines
            np.testing.assert_allclose(call_args[6], (-2, -2, 8, 2))


class Test_NC_ABI_L1B_ir_cal:
    """Test the NC_ABI_L1B reader's default IR calibration."""

    @pytest.mark.parametrize("clip_negative_radiances", [False, True])
    def test_ir_calibrate(self, clip_negative_radiances):
        """Test IR calibration."""
        with _ir_file_handler(
            clip_negative_radiances=clip_negative_radiances
        ) as file_handler:
            res = file_handler.get_dataset(
                make_dataid(name="C07", calibration="brightness_temperature"), {}
            )
            assert file_handler.clip_negative_radiances == clip_negative_radiances

        clipped_ir = 134.68753 if clip_negative_radiances else np.nan
        expected = np.array(
            [
                [clipped_ir, 304.97037, 332.22778, 354.6147, 374.08688],
                [391.58655, 407.64786, 422.60635, 436.68802, np.nan],
            ]
        )
        np.testing.assert_allclose(res.data, expected, equal_nan=True, atol=1e-04)

        # make sure the attributes from the file are in the data array
        assert "scale_factor" not in res.attrs
        assert "_FillValue" not in res.attrs
        assert res.attrs["standard_name"] == "toa_brightness_temperature"
        assert res.attrs["long_name"] == "Brightness Temperature"


@contextlib.contextmanager
def _ir_file_handler(clip_negative_radiances: bool = False):
    """Create fake data for the tests."""
    values = np.arange(10.0)
    rad_data = (values.reshape((2, 5)) + 1.0) * 50.0
    rad_data[0, 0] = -0.0001  # introduce below minimum expected radiance
    rad_data = (rad_data + 1.3) / 0.5
    data = rad_data.astype(np.int16)

    rad = xr.DataArray(
        da.from_array(data),
        dims=("y", "x"),
        attrs={
            "scale_factor": 0.5,
            "add_offset": -1.3,
            "_FillValue": np.int16(
                np.floor(((9 + 1) * 50.0 + 1.3) / 0.5)
            ),  # last rad_data value
        },
    )
    with create_file_handler(
        rad=rad,
        clip_negative_radiances=clip_negative_radiances,
        filetype_resolution=2000,
    ) as file_handler:
        yield file_handler


class Test_NC_ABI_L1B_vis_cal:
    """Test the NC_ABI_L1B reader."""

    def test_vis_calibrate(self):
        """Test VIS calibration."""
        rad_data = np.arange(10.0).reshape((2, 5)) + 1.0
        rad_data = (rad_data + 1.0) / 0.5
        rad_data = rad_data.astype(np.int16)
        rad = xr.DataArray(
            da.from_array(rad_data),
            dims=("y", "x"),
            attrs={
                "scale_factor": 0.5,
                "add_offset": -1.0,
                "_FillValue": 20,
            },
        )
        with create_file_handler(rad=rad, filetype_resolution=1000) as file_handler:
            res = file_handler.get_dataset(
                make_dataid(name="C05", calibration="reflectance"), {}
            )

        expected = np.array(
            [
                [0.15265617, 0.30531234, 0.45796851, 0.61062468, 0.76328085],
                [0.91593702, 1.06859319, 1.22124936, np.nan, 1.52656171],
            ]
        )
        assert np.allclose(res.data, expected, equal_nan=True)
        assert "scale_factor" not in res.attrs
        assert "_FillValue" not in res.attrs
        assert res.attrs["standard_name"] == "toa_bidirectional_reflectance"
        assert res.attrs["long_name"] == "Bidirectional Reflectance"


class Test_NC_ABI_L1B_raw_cal:
    """Test the NC_ABI_L1B reader raw calibration."""

    def test_raw_calibrate(self):
        """Test RAW calibration."""
        rad_data = np.arange(10.0).reshape((2, 5)) + 1.0
        rad_data = (rad_data + 1.0) / 0.5
        rad_data = rad_data.astype(np.int16)
        rad = xr.DataArray(
            da.from_array(rad_data),
            dims=("y", "x"),
            attrs={
                "scale_factor": 0.5,
                "add_offset": -1.0,
                "_FillValue": 20,
            },
        )
        with create_file_handler(rad=rad) as file_handler:
            res = file_handler.get_dataset(
                make_dataid(name="C05", calibration="counts"), {}
            )

        # We expect the raw data to be unchanged
        expected = res.data
        assert np.allclose(res.data, expected, equal_nan=True)

        # check for the presence of typical attributes
        assert "scale_factor" in res.attrs
        assert "add_offset" in res.attrs
        assert "_FillValue" in res.attrs
        assert "orbital_parameters" in res.attrs
        assert "platform_shortname" in res.attrs
        assert "scene_id" in res.attrs

        # determine if things match their expected values/types.
        assert res.data.dtype == np.int16
        assert res.attrs["standard_name"] == "counts"
        assert res.attrs["long_name"] == "Raw Counts"


class Test_NC_ABI_File:
    """Test file opening."""

    @mock.patch("satpy.readers.abi_base.xr")
    def test_open_dataset(self, _):  # noqa: PT019
        """Test openning a dataset."""
        from satpy.readers.abi_l1b import NC_ABI_L1B

        openable_thing = mock.MagicMock()

        NC_ABI_L1B(openable_thing, {"platform_shortname": "g16"}, {})
        openable_thing.open.assert_called()


class Test_NC_ABI_L1B_H5netcdf(Test_NC_ABI_L1B):
    """Allow h5netcdf peculiarities."""

    @property
    def fake_rad(self):
        """Create fake data for the tests."""
        rad_data = np.int16(50)
        rad = xr.DataArray(
            da.from_array(rad_data),
            attrs={
                "scale_factor": 0.5,
                "add_offset": -1.0,
                "_FillValue": np.array([1002]),
                "units": "W m-2 um-1 sr-1",
                "valid_range": (0, 4095),
            },
        )
        return rad
