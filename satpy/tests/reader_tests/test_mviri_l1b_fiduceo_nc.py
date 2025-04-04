#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Unit tests for the FIDUCEO MVIRI FCDR Reader."""

from __future__ import annotations

import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS
from pyresample.geometry import AreaDefinition

from satpy.readers.mviri_l1b_fiduceo_nc import (
    ALTITUDE,
    EQUATOR_RADIUS,
    POLE_RADIUS,
    FiduceoMviriEasyFcdrFileHandler,
    FiduceoMviriFullFcdrFileHandler,
    Interpolator,
    preprocess_dataset,
)
from satpy.tests.utils import make_dataid

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - request

fill_val = np.uint32(429496729) # FillValue lower than in dataset to be windows-compatible

attrs_exp: dict = {
    "platform": "MET7",
    "raw_metadata": {"foo": "bar"},
    "sensor": "MVIRI",
    "orbital_parameters": {
        "projection_longitude": 57.0,
        "projection_latitude": 0.0,
        "projection_altitude": 35785860.0,
        "satellite_actual_longitude": 57.1,
        "satellite_actual_latitude": 0.1,
    }

}
attrs_refl_exp = attrs_exp.copy()
attrs_refl_exp.update(
    {"sun_earth_distance_correction_applied": True,
     "sun_earth_distance_correction_factor": 1.}
)
acq_time_vis_exp = [np.datetime64("NaT").astype("datetime64[ns]"),
                    np.datetime64("NaT").astype("datetime64[ns]"),
                    np.datetime64("1970-01-01 02:30").astype("datetime64[ns]"),
                    np.datetime64("1970-01-01 02:30").astype("datetime64[ns]")]
vis_counts_exp = xr.DataArray(
    np.array(
        [[0., 17., 34., 51.],
         [68., 85., 102., 119.],
         [136., 153., np.nan, 187.],
         [204., 221., 238., 255]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_vis_exp),
    },
    attrs=attrs_exp
)

vis_rad_exp = xr.DataArray(
    np.array(
        [[np.nan, 18.56, 38.28, 58.],
         [77.72, 97.44, 117.16, 136.88],
         [156.6, 176.32, np.nan, 215.76],
         [235.48, 255.2, 274.92, 294.64]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_vis_exp),
    },
    attrs=attrs_exp
)
vis_refl_exp = xr.DataArray(
    np.array(
        [[np.nan, 23.440929, np.nan, np.nan],
         [40.658744, 66.602233, 147.970867, np.nan],
         [75.688217, 92.240733, np.nan, np.nan],
         [np.nan, np.nan, np.nan, np.nan]],
        dtype=np.float32
    ),
    # (0, 0) and (2, 2) are NaN because radiance is NaN
    # (0, 2) is NaN because SZA >= 90 degrees
    # Last row/col is NaN due to SZA interpolation
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_vis_exp),
    },
    attrs=attrs_refl_exp
)
u_vis_refl_exp = xr.DataArray(
    np.array(
        [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8],
         [0.9, 1.0, 1.1, 1.2],
         [1.3, 1.4, 1.5, 1.6]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_vis_exp),
    },
    attrs=attrs_exp
)

u_struct_refl_exp = u_vis_refl_exp.copy()

acq_time_ir_wv_exp = [np.datetime64("NaT"),
                      np.datetime64("1970-01-01 02:30").astype("datetime64[ns]")]
wv_counts_exp = xr.DataArray(
    np.array(
        [[0, 85],
         [170, 255]],
        dtype=np.uint8
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
wv_rad_exp = xr.DataArray(
    np.array(
        [[np.nan, 3.75],
         [8, 12.25]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
wv_bt_exp = xr.DataArray(
    np.array(
        [[np.nan, 230.461366],
         [252.507448, 266.863289]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
ir_counts_exp = xr.DataArray(
    np.array(
        [[0, 85],
         [170, 255]],
        dtype=np.uint8
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
ir_rad_exp = xr.DataArray(
    np.array(
        [[np.nan, 80],
         [165, 250]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
ir_bt_exp = xr.DataArray(
    np.array(
        [[np.nan, 178.00013189],
         [204.32955838, 223.28709913]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_ir_wv_exp),
    },
    attrs=attrs_exp
)
quality_pixel_bitmask_exp = xr.DataArray(
    np.array(
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0]],
        dtype=np.uint8
    ),
    dims=("y", "x"),
    coords={
        "acq_time": ("y", acq_time_vis_exp),
    },
    attrs=attrs_exp
)
sza_vis_exp = xr.DataArray(
    np.array(
        [[45., 67.5, 90., np.nan],
         [22.5, 45., 67.5, np.nan],
         [0., 22.5, 45., np.nan],
         [np.nan, np.nan, np.nan, np.nan]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    attrs=attrs_exp
)
sza_ir_wv_exp = xr.DataArray(
    np.array(
        [[45, 90],
         [0, 45]],
        dtype=np.float32
    ),
    dims=("y", "x"),
    attrs=attrs_exp
)
projection = CRS(f"+proj=geos +lon_0=57.0 +h={ALTITUDE} +a={EQUATOR_RADIUS} +b={POLE_RADIUS}")
area_vis_exp = AreaDefinition(
    area_id="geos_mviri_4x4",
    proj_id="geos_mviri_4x4",
    description="MVIRI Geostationary Projection",
    projection=projection,
    width=4,
    height=4,
    area_extent=[5621229.74392, 5621229.74392, -5621229.74392, -5621229.74392]
)
area_ir_wv_exp = area_vis_exp.copy(
    area_id="geos_mviri_2x2",
    proj_id="geos_mviri_2x2",
    width=2,
    height=2
)


@pytest.fixture(name="time_fake_dataset")
def fixture_time_fake_dataset():
    """Create time for fake dataset."""
    time = np.arange(4) * 60 * 60
    time[0] = fill_val
    time[1] = fill_val
    time = time.reshape(2, 2)

    return time


@pytest.fixture(name="fake_dataset")
def fixture_fake_dataset(time_fake_dataset):
    """Create fake dataset."""
    count_ir = da.linspace(0, 255, 4, dtype=np.uint8).reshape(2, 2)
    count_wv = da.linspace(0, 255, 4, dtype=np.uint8).reshape(2, 2)
    count_vis = da.linspace(0, 255, 16, dtype=np.uint8).reshape(4, 4)
    sza = da.from_array(
        np.array(
            [[45, 90], [0, 45]],
            dtype=np.float32
        )
    )
    mask = da.from_array(
        np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], # 1 = "invalid"
            dtype=np.uint8
        )
    )

    cov = da.from_array([[1, 2], [3, 4]])

    ds = xr.Dataset(
        data_vars={
            "count_vis": (("y", "x"), count_vis),
            "count_wv": (("y_ir_wv", "x_ir_wv"), count_wv),
            "count_ir": (("y_ir_wv", "x_ir_wv"), count_ir),
            "toa_bidirectional_reflectance_vis": vis_refl_exp / 100,
            "u_independent_toa_bidirectional_reflectance": u_vis_refl_exp / 100,
            "u_structured_toa_bidirectional_reflectance": u_vis_refl_exp / 100,
            "quality_pixel_bitmask": (("y", "x"), mask),
            "solar_zenith_angle": (("y_tie", "x_tie"), sza),
            "time_ir_wv": (("y_ir_wv", "x_ir_wv"), time_fake_dataset),
            "a_ir": -5.0,
            "b_ir": 1.0,
            "bt_a_ir": 10.0,
            "bt_b_ir": -1000.0,
            "a_wv": -0.5,
            "b_wv": 0.05,
            "bt_a_wv": 10.0,
            "bt_b_wv": -2000.0,
            "years_since_launch": 20.0,
            "a0_vis": 1.0,
            "a1_vis": 0.01,
            "a2_vis": -0.0001,
            "mean_count_space_vis": 1.0,
            "distance_sun_earth": 1.0,
            "solar_irradiance_vis": 650.0,
            "sub_satellite_longitude_start": 57.1,
            "sub_satellite_longitude_end": np.nan,
            "sub_satellite_latitude_start": np.nan,
            "sub_satellite_latitude_end": 0.1,
            "covariance_spectral_response_function_vis": (("srf_size", "srf_size"), cov),
            "channel_correlation_matrix_independent": (("channel", "channel"), cov),
            "channel_correlation_matrix_structured": (("channel", "channel"), cov)
        },
        coords={
            "y": [1, 2, 3, 4],
            "x": [1, 2, 3, 4],
            "y_ir_wv": [1, 2],
            "x_ir_wv": [1, 2],
            "y_tie": [1, 2],
            "x_tie": [1, 2],
        },
        attrs={"foo": "bar"}
    )
    ds["count_ir"].attrs["ancillary_variables"] = "a_ir b_ir"
    ds["count_wv"].attrs["ancillary_variables"] = "a_wv b_wv"
    ds["quality_pixel_bitmask"].encoding["chunksizes"] = (2, 2)
    ds["time_ir_wv"].attrs["_FillValue"] = fill_val
    ds["time_ir_wv"].attrs["add_offset"] = 0

    return ds


@pytest.fixture(name="projection_longitude", params=["57.0"])
def fixture_projection_longitude(request):
    """Get projection longitude as string."""
    return request.param


@pytest.fixture(name="fake_file")
def fixture_fake_file(fake_dataset, tmp_path):
    """Create test file."""
    filename = tmp_path / "test_mviri_fiduceo.nc"
    fake_dataset.to_netcdf(filename)
    return filename


@pytest.fixture(
    name="file_handler",
    params=[FiduceoMviriEasyFcdrFileHandler,
            FiduceoMviriFullFcdrFileHandler]
)
def fixture_file_handler(fake_file, request, projection_longitude):
    """Create mocked file handler."""
    marker = request.node.get_closest_marker("file_handler_data")
    mask_bad_quality = True
    if marker:
        mask_bad_quality = marker.kwargs["mask_bad_quality"]
    fh_class = request.param
    return fh_class(
        filename=fake_file,
        filename_info={"platform": "MET7",
                       "sensor": "MVIRI",
                       "projection_longitude": projection_longitude},
        filetype_info={"foo": "bar"},
        mask_bad_quality=mask_bad_quality
    )


@pytest.fixture(name="reader")
def fixture_reader():
    """Return MVIRI FIDUCEO FCDR reader."""
    from satpy._config import config_search_paths
    from satpy.readers import load_reader

    reader_configs = config_search_paths(
        os.path.join("readers", "mviri_l1b_fiduceo_nc.yaml"))
    reader = load_reader(reader_configs)
    return reader


class TestFiduceoMviriFileHandlers:
    """Unit tests for FIDUCEO MVIRI file handlers."""

    @pytest.mark.parametrize("projection_longitude", ["57.0", "5700"], indirect=True)
    def test_init(self, file_handler, projection_longitude):
        """Test file handler initialization."""
        assert file_handler.projection_longitude == 57.0
        assert file_handler.mask_bad_quality is True

    @pytest.mark.parametrize(
        ("name", "calibration", "resolution", "expected"),
        [
            ("VIS", "counts", 2250, vis_counts_exp),
            ("VIS", "radiance", 2250, vis_rad_exp),
            ("VIS", "reflectance", 2250, vis_refl_exp),
            ("WV", "counts", 4500, wv_counts_exp),
            ("WV", "radiance", 4500, wv_rad_exp),
            ("WV", "brightness_temperature", 4500, wv_bt_exp),
            ("IR", "counts", 4500, ir_counts_exp),
            ("IR", "radiance", 4500, ir_rad_exp),
            ("IR", "brightness_temperature", 4500, ir_bt_exp),
            ("quality_pixel_bitmask", None, 2250, quality_pixel_bitmask_exp),
            ("solar_zenith_angle", None, 2250, sza_vis_exp),
            ("solar_zenith_angle", None, 4500, sza_ir_wv_exp),
            ("u_independent_toa_bidirectional_reflectance", None, 4500, u_vis_refl_exp),
            ("u_structured_toa_bidirectional_reflectance", None, 4500, u_struct_refl_exp)
        ]
    )
    def test_get_dataset(self, file_handler, name, calibration, resolution,
                         expected):
        """Test getting datasets."""
        id_keys = {"name": name, "resolution": resolution}
        if calibration:
            id_keys["calibration"] = calibration
        dataset_id = make_dataid(**id_keys)
        dataset_info = {"platform": "MET7"}

        is_easy = isinstance(file_handler, FiduceoMviriEasyFcdrFileHandler)
        is_vis = name == "VIS"
        is_refl = calibration == "reflectance"
        if is_easy and is_vis and not is_refl:
            # VIS counts/radiance not available in easy FCDR
            with pytest.raises(ValueError, match="Cannot calibrate to .*. Easy FCDR provides reflectance only."):
                file_handler.get_dataset(dataset_id, dataset_info)
        else:
            ds = file_handler.get_dataset(dataset_id, dataset_info)
            xr.testing.assert_allclose(ds, expected)
            assert ds.dtype == expected.dtype
            assert ds.attrs == expected.attrs

    def test_get_dataset_corrupt(self, file_handler):
        """Test getting datasets with known corruptions."""
        # Satellite position might be missing
        file_handler.nc.ds = file_handler.nc.ds.drop_vars(
            ["sub_satellite_longitude_start"]
        )

        dataset_id = make_dataid(
            name="VIS",
            calibration="reflectance",
            resolution=2250
        )
        ds = file_handler.get_dataset(dataset_id, {"platform": "MET7"})
        assert "actual_satellite_longitude" not in ds.attrs["orbital_parameters"]
        assert "actual_satellite_latitude" not in ds.attrs["orbital_parameters"]
        xr.testing.assert_allclose(ds, vis_refl_exp)

    @mock.patch(
        "satpy.readers.mviri_l1b_fiduceo_nc.Interpolator.interp_acq_time"
    )
    def test_time_cache(self, interp_acq_time, file_handler):
        """Test caching of acquisition times."""
        dataset_id = make_dataid(
            name="VIS",
            resolution=2250,
            calibration="reflectance"
        )
        info = {}
        interp_acq_time.return_value = xr.DataArray([1, 2, 3, 4], dims="y")

        # Cache init
        file_handler.get_dataset(dataset_id, info)
        interp_acq_time.assert_called()

        # Cache hit
        interp_acq_time.reset_mock()
        file_handler.get_dataset(dataset_id, info)
        interp_acq_time.assert_not_called()

        # Cache miss
        interp_acq_time.return_value = xr.DataArray([1, 2], dims="y")
        another_id = make_dataid(
            name="IR",
            resolution=4500,
            calibration="brightness_temperature"
        )
        interp_acq_time.reset_mock()
        file_handler.get_dataset(another_id, info)
        interp_acq_time.assert_called()

    @mock.patch(
        "satpy.readers.mviri_l1b_fiduceo_nc.Interpolator.interp_tiepoints"
    )
    def test_angle_cache(self, interp_tiepoints, file_handler):
        """Test caching of angle datasets."""
        dataset_id = make_dataid(name="solar_zenith_angle",
                                 resolution=2250)
        info = {}

        # Cache init
        file_handler.get_dataset(dataset_id, info)
        interp_tiepoints.assert_called()

        # Cache hit
        interp_tiepoints.reset_mock()
        file_handler.get_dataset(dataset_id, info)
        interp_tiepoints.assert_not_called()

        # Cache miss
        another_id = make_dataid(name="solar_zenith_angle",
                                 resolution=4500)
        interp_tiepoints.reset_mock()
        file_handler.get_dataset(another_id, info)
        interp_tiepoints.assert_called()

    @pytest.mark.parametrize(
        ("name", "resolution", "area_exp"),
        [
            ("VIS", 2250, area_vis_exp),
            ("WV", 4500, area_ir_wv_exp),
            ("IR", 4500, area_ir_wv_exp),
            ("quality_pixel_bitmask", 2250, area_vis_exp),
            ("solar_zenith_angle", 2250, area_vis_exp),
            ("solar_zenith_angle", 4500, area_ir_wv_exp)
        ]
    )
    def test_get_area_definition(self, file_handler, name, resolution,
                                 area_exp):
        """Test getting area definitions."""
        dataset_id = make_dataid(name=name, resolution=resolution)
        area = file_handler.get_area_def(dataset_id)

        assert area.crs == area_exp.crs
        np.testing.assert_allclose(area.area_extent, area_exp.area_extent)

    def test_calib_exceptions(self, file_handler):
        """Test calibration exceptions."""
        with pytest.raises(KeyError):
            file_handler.get_dataset(
                make_dataid(name="solar_zenith_angle", calibration="counts"),
                {}
            )
        with pytest.raises(KeyError):
            file_handler.get_dataset(
                make_dataid(
                    name="VIS",
                    resolution=2250,
                    calibration="brightness_temperature"),
                {}
            )
        with pytest.raises(KeyError):
            file_handler.get_dataset(
                make_dataid(
                    name="IR",
                    resolution=4500,
                    calibration="reflectance"),
                {}
            )
        if isinstance(file_handler, FiduceoMviriEasyFcdrFileHandler):
            with pytest.raises(KeyError):
                file_handler.get_dataset(
                    {"name": "VIS", "calibration": "counts"},
                    {}
                )  # not available in easy FCDR

    @pytest.mark.file_handler_data(mask_bad_quality=False)
    def test_bad_quality_warning(self, file_handler):
        """Test warning about bad VIS quality."""
        file_handler.nc.ds["quality_pixel_bitmask"] = 2
        vis = make_dataid(name="VIS", resolution=2250,
                          calibration="reflectance")
        with pytest.warns(UserWarning):
            file_handler.get_dataset(vis, {})

    def test_file_pattern(self, reader):
        """Test file pattern matching."""
        filenames = [
            "FIDUCEO_FCDR_L15_MVIRI_MET7-57.0_201701201000_201701201030_FULL_v2.6_fv3.1.nc",
            "FIDUCEO_FCDR_L15_MVIRI_MET7-57.0_201701201000_201701201030_EASY_v2.6_fv3.1.nc",
            "FIDUCEO_FCDR_L15_MVIRI_MET7-00.0_201701201000_201701201030_EASY_v2.6_fv3.1.nc",
            "MVIRI_FCDR-EASY_L15_MET7-E0000_200607060600_200607060630_0200.nc",
            "MVIRI_FCDR-EASY_L15_MET7-E5700_200607060600_200607060630_0200.nc",
            "MVIRI_FCDR-FULL_L15_MET7-E0000_200607060600_200607060630_0200.nc",
            "abcde",
        ]

        files = reader.select_files_from_pathnames(filenames)
        assert len(files) == 6


class TestDatasetPreprocessor:
    """Test dataset preprocessing."""

    @pytest.fixture(name="dataset")
    def fixture_dataset(self):
        """Get dataset before preprocessing.

        - Encoded timestamps including fill values
        - Duplicate dimension names
        - x/y coordinates not assigned
        """
        time = 60*60
        return xr.Dataset(
            data_vars={
                "covariance_spectral_response_function_vis": (("srf_size", "srf_size"), [[1, 2], [3, 4]]),
                "channel_correlation_matrix_independent": (("channel", "channel"), [[1, 2], [3, 4]]),
                "channel_correlation_matrix_structured": (("channel", "channel"), [[1, 2], [3, 4]]),
                "time_ir_wv": (("y", "x"), [[time, fill_val], [time, time]],
                               {"_FillValue": fill_val, "add_offset": 0})
            }
        )

    @pytest.fixture(name="dataset_exp")
    def fixture_dataset_exp(self):
        """Get expected dataset after preprocessing.

        - Timestamps should have been converted to datetime64
        - Time dimension should have been renamed
        - Duplicate dimensions should have been removed
        - x/y coordinates should have been assigned
        """
        time_exp = np.datetime64("1970-01-01 01:00").astype("datetime64[ns]")
        return xr.Dataset(
            data_vars={
                "covariance_spectral_response_function_vis": (("srf_size_1", "srf_size_2"), [[1, 2], [3, 4]]),
                "channel_correlation_matrix_independent": (("channel_1", "channel_2"), [[1, 2], [3, 4]]),
                "channel_correlation_matrix_structured": (("channel_1", "channel_2"), [[1, 2], [3, 4]]),
                "time": (("y", "x"), [[time_exp, np.datetime64("NaT")], [time_exp, time_exp]])
            },
            coords={
                "y": [0, 1],
                "x": [0, 1]
            }
        )

    def test_preprocess(self, dataset, dataset_exp):
        """Test dataset preprocessing."""
        preprocessed = preprocess_dataset(dataset)
        xr.testing.assert_allclose(preprocessed, dataset_exp)


class TestInterpolator:
    """Unit tests for Interpolator class."""
    @pytest.fixture(name="time_ir_wv")
    def fixture_time_ir_wv(self):
        """Returns time_ir_wv."""
        time_ir_wv = xr.DataArray(
            [
              [np.datetime64("1970-01-01 01:00"), np.datetime64("1970-01-01 02:00")],
              [np.datetime64("1970-01-01 03:00"), np.datetime64("1970-01-01 04:00")],
              [np.datetime64("NaT"), np.datetime64("1970-01-01 06:00")],
              [np.datetime64("NaT"), np.datetime64("NaT")],
            ],
            dims=("y", "x"),
            coords={"y": [1, 3, 5, 7]}
        )
        return time_ir_wv.astype("datetime64[ns]")

    @pytest.fixture(name="acq_time_exp")
    def fixture_acq_time_exp(self):
        """Returns acq_time_vis_exp."""
        vis = xr.DataArray(
            [
                np.datetime64("1970-01-01 01:30"),
                np.datetime64("1970-01-01 01:30"),
                np.datetime64("1970-01-01 03:30"),
                np.datetime64("1970-01-01 03:30"),
                np.datetime64("1970-01-01 06:00"),
                np.datetime64("1970-01-01 06:00"),
                np.datetime64("NaT"),
                np.datetime64("NaT")
            ],
            dims="y",
            coords={"y": [1, 2, 3, 4, 5, 6, 7, 8]}
        )

        ir = xr.DataArray(
            [
                np.datetime64("1970-01-01 01:30"),
                np.datetime64("1970-01-01 03:30"),
                np.datetime64("1970-01-01 06:00"),
                np.datetime64("NaT"),
            ],
            dims="y",
            coords={"y": [1, 3, 5, 7]}
        )

        return vis, ir

    def test_interp_acq_time(self, time_ir_wv, acq_time_exp):
        """Tests time interpolation."""
        res_vis = Interpolator.interp_acq_time(time_ir_wv, target_y=acq_time_exp[0].coords["y"])
        res_ir = Interpolator.interp_acq_time(time_ir_wv, target_y=acq_time_exp[1].coords["y"])

        xr.testing.assert_allclose(res_vis, acq_time_exp[0])
        xr.testing.assert_allclose(res_ir, acq_time_exp[1])
