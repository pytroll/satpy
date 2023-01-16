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

"""Tests for the angles in modifiers."""
import contextlib
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from glob import glob
from typing import Optional, Union
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition, StackedAreaDefinition

import satpy
from satpy.utils import PerformanceWarning

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


def _angle_cache_area_def():
    area = AreaDefinition(
        "test", "", "",
        {"proj": "merc"},
        5, 5,
        (-2500, -2500, 2500, 2500),
    )
    return area


def _angle_cache_stacked_area_def():
    area1 = AreaDefinition(
        "test", "", "",
        {"proj": "merc"},
        5, 2,
        (2500, 500, 7500, 2500),
    )
    area2 = AreaDefinition(
        "test", "", "",
        {"proj": "merc"},
        5, 3,
        (2500, -2500, 7500, 500),
    )
    return StackedAreaDefinition(area1, area2)


def _get_angle_test_data(area_def: Optional[Union[AreaDefinition, StackedAreaDefinition]] = None,
                         chunks: Optional[Union[int, tuple]] = 2,
                         shape: tuple = (5, 5),
                         dims: Optional[tuple] = None,
                         ) -> xr.DataArray:
    if area_def is None:
        area_def = _angle_cache_area_def()
    orb_params = {
        "satellite_nominal_altitude": 12345678,
        "satellite_nominal_longitude": 10.0,
        "satellite_nominal_latitude": 0.0,
    }
    stime = datetime(2020, 1, 1, 12, 0, 0)
    data = da.zeros(shape, chunks=chunks)
    vis = xr.DataArray(data,
                       dims=dims,
                       attrs={
                           'area': area_def,
                           'start_time': stime,
                           'orbital_parameters': orb_params,
                       })
    return vis


def _get_stacked_angle_test_data():
    return _get_angle_test_data(area_def=_angle_cache_stacked_area_def(),
                                chunks=(5, (2, 2, 1)))


def _get_angle_test_data_rgb():
    return _get_angle_test_data(shape=(5, 5, 3), chunks=((2, 2, 1), (2, 2, 1), (1, 1, 1)),
                                dims=("y", "x", "bands"))


def _get_angle_test_data_rgb_nodims():
    return _get_angle_test_data(shape=(3, 5, 5), chunks=((1, 1, 1), (2, 2, 1), (2, 2, 1)))


def _get_angle_test_data_odd_chunks():
    return _get_angle_test_data(chunks=((2, 1, 2), (1, 1, 2, 1)))


def _similar_sat_pos_datetime(orig_data, lon_offset=0.04):
    # change data slightly
    new_data = orig_data.copy()
    old_lon = new_data.attrs["orbital_parameters"]["satellite_nominal_longitude"]
    new_data.attrs["orbital_parameters"]["satellite_nominal_longitude"] = old_lon + lon_offset
    new_data.attrs["start_time"] = new_data.attrs["start_time"] + timedelta(hours=36)
    return new_data


def _diff_sat_pos_datetime(orig_data):
    return _similar_sat_pos_datetime(orig_data, lon_offset=0.05)


def _glob_reversed(pat):
    """Behave like glob but force results to be in the wrong order."""
    return sorted(glob(pat), reverse=True)


@contextlib.contextmanager
def _mock_glob_if(mock_glob):
    if mock_glob:
        with mock.patch("satpy.modifiers.angles.glob", _glob_reversed):
            yield
    else:
        yield


def _assert_allclose_if(expect_equal, arr1, arr2):
    if not expect_equal:
        pytest.raises(AssertionError, np.testing.assert_allclose, arr1, arr2)
    else:
        np.testing.assert_allclose(arr1, arr2)


class TestAngleGeneration:
    """Test the angle generation utility functions."""

    @pytest.mark.parametrize(
        ("input_func", "exp_calls"),
        [
            (_get_angle_test_data, 9),
            (_get_stacked_angle_test_data, 3),
            (_get_angle_test_data_rgb, 9),
            (_get_angle_test_data_rgb_nodims, 9),
        ],
    )
    def test_get_angles(self, input_func, exp_calls):
        """Test sun and satellite angle calculation."""
        from satpy.modifiers.angles import get_angles
        data = input_func()

        from pyorbital.orbital import get_observer_look
        with mock.patch("satpy.modifiers.angles.get_observer_look", wraps=get_observer_look) as gol:
            angles = get_angles(data)
            assert all(isinstance(x, xr.DataArray) for x in angles)
            da.compute(angles)

        # get_observer_look should have been called once per array chunk
        assert gol.call_count == exp_calls
        # Check arguments of get_orbserver_look() call, especially the altitude
        # unit conversion from meters to kilometers
        args = gol.call_args[0]
        assert args[:4] == (10.0, 0.0, 12345.678, data.attrs["start_time"])

    @pytest.mark.parametrize("forced_preference", ["actual", "nadir"])
    def test_get_angles_satpos_preference(self, forced_preference):
        """Test that 'actual' satellite position is used for generating sensor angles."""
        from satpy.modifiers.angles import get_angles

        input_data1 = _get_angle_test_data()
        # add additional satellite position metadata
        input_data1.attrs["orbital_parameters"]["nadir_longitude"] = 9.0
        input_data1.attrs["orbital_parameters"]["nadir_latitude"] = 0.01
        input_data1.attrs["orbital_parameters"]["satellite_actual_longitude"] = 9.5
        input_data1.attrs["orbital_parameters"]["satellite_actual_latitude"] = 0.005
        input_data1.attrs["orbital_parameters"]["satellite_actual_altitude"] = 12345679
        input_data2 = input_data1.copy(deep=True)
        input_data2.attrs = deepcopy(input_data1.attrs)
        input_data2.attrs["orbital_parameters"]["nadir_longitude"] = 9.1
        input_data2.attrs["orbital_parameters"]["nadir_latitude"] = 0.02
        input_data2.attrs["orbital_parameters"]["satellite_actual_longitude"] = 9.5
        input_data2.attrs["orbital_parameters"]["satellite_actual_latitude"] = 0.005
        input_data2.attrs["orbital_parameters"]["satellite_actual_altitude"] = 12345679

        from pyorbital.orbital import get_observer_look
        with mock.patch("satpy.modifiers.angles.get_observer_look", wraps=get_observer_look) as gol, \
                satpy.config.set(sensor_angles_position_preference=forced_preference):
            angles1 = get_angles(input_data1)
            da.compute(angles1)
            angles2 = get_angles(input_data2)
            da.compute(angles2)

        # get_observer_look should have been called once per array chunk
        assert gol.call_count == input_data1.data.blocks.size * 2
        if forced_preference == "actual":
            exp_call = mock.call(9.5, 0.005, 12345.679, input_data1.attrs["start_time"], mock.ANY, mock.ANY, 0)
            all_same_calls = [exp_call] * gol.call_count
            gol.assert_has_calls(all_same_calls)
            # the dask arrays should have the same name to prove they are the same computation
            for angle_arr1, angle_arr2 in zip(angles1, angles2):
                assert angle_arr1.data.name == angle_arr2.data.name
        else:
            # nadir 1
            gol.assert_any_call(9.0, 0.01, 12345.679, input_data1.attrs["start_time"], mock.ANY, mock.ANY, 0)
            # nadir 2
            gol.assert_any_call(9.1, 0.02, 12345.679, input_data1.attrs["start_time"], mock.ANY, mock.ANY, 0)

    @pytest.mark.parametrize("force_bad_glob", [False, True])
    @pytest.mark.parametrize(
        ("input2_func", "exp_equal_sun", "exp_num_zarr"),
        [
            (lambda x: x, True, 4),
            (_similar_sat_pos_datetime, False, 4),
            (_diff_sat_pos_datetime, False, 6),
        ]
    )
    @pytest.mark.parametrize(
        ("input_func", "num_normalized_chunks", "exp_zarr_chunks"),
        [
            (_get_angle_test_data, 9, ((2, 2, 1), (2, 2, 1))),
            (_get_stacked_angle_test_data, 3, ((5,), (2, 2, 1))),
            (_get_angle_test_data_odd_chunks, 9, ((2, 1, 2), (1, 1, 2, 1))),
            (_get_angle_test_data_rgb, 9, ((2, 2, 1), (2, 2, 1))),
            (_get_angle_test_data_rgb_nodims, 9, ((2, 2, 1), (2, 2, 1))),
        ])
    def test_cache_get_angles(
            self,
            input_func, num_normalized_chunks, exp_zarr_chunks,
            input2_func, exp_equal_sun, exp_num_zarr,
            force_bad_glob, tmp_path):
        """Test get_angles when caching is enabled."""
        from satpy.modifiers.angles import STATIC_EARTH_INERTIAL_DATETIME, get_angles

        # Patch methods
        data = input_func()
        additional_cache = exp_num_zarr > 4

        # Compute angles
        from pyorbital.orbital import get_observer_look
        with mock.patch("satpy.modifiers.angles.get_observer_look", wraps=get_observer_look) as gol, \
                satpy.config.set(cache_lonlats=True, cache_sensor_angles=True, cache_dir=str(tmp_path)), \
                warnings.catch_warnings(record=True) as caught_warnings:
            res = get_angles(data)
            self._check_cached_result(res, exp_zarr_chunks)

            # call again, should be cached
            new_data = input2_func(data)
            with _mock_glob_if(force_bad_glob):
                res2 = get_angles(new_data)
            self._check_cached_result(res2, exp_zarr_chunks)

            res_numpy, res2_numpy = da.compute(res, res2)
            for r1, r2 in zip(res_numpy[:2], res2_numpy[:2]):
                _assert_allclose_if(not additional_cache, r1, r2)
            for r1, r2 in zip(res_numpy[2:], res2_numpy[2:]):
                _assert_allclose_if(exp_equal_sun, r1, r2)

            self._check_cache_and_clear(tmp_path, exp_num_zarr)

        if "odd_chunks" in input_func.__name__:
            assert any(w.category is PerformanceWarning for w in caught_warnings)
        else:
            assert not any(w.category is PerformanceWarning for w in caught_warnings)
        assert gol.call_count == num_normalized_chunks * (int(additional_cache) + 1)
        args = gol.call_args_list[0][0]
        assert args[:4] == (10.0, 0.0, 12345.678, STATIC_EARTH_INERTIAL_DATETIME)
        exp_sat_lon = 10.1 if additional_cache else 10.0
        args = gol.call_args_list[-1][0]
        assert args[:4] == (exp_sat_lon, 0.0, 12345.678, STATIC_EARTH_INERTIAL_DATETIME)

    @staticmethod
    def _check_cached_result(results, exp_zarr_chunks):
        assert all(isinstance(x, xr.DataArray) for x in results)
        # output chunks should be consistent
        for angle_data_arr in results:
            assert angle_data_arr.chunks == exp_zarr_chunks

    @staticmethod
    def _check_cache_and_clear(tmp_path, exp_num_zarr):
        from satpy.modifiers.angles import _get_sensor_angles_from_sat_pos, _get_valid_lonlats
        zarr_dirs = glob(str(tmp_path / "*.zarr"))
        assert len(zarr_dirs) == exp_num_zarr  # two for lon/lat, one for sata, one for satz

        _get_valid_lonlats.cache_clear()
        _get_sensor_angles_from_sat_pos.cache_clear()
        zarr_dirs = glob(str(tmp_path / "*.zarr"))
        assert len(zarr_dirs) == 0

    def test_cached_no_chunks_fails(self, tmp_path):
        """Test that trying to pass non-dask arrays and no chunks fails."""
        from satpy.modifiers.angles import _sanitize_args_with_chunks, cache_to_zarr_if

        @cache_to_zarr_if("cache_lonlats", sanitize_args_func=_sanitize_args_with_chunks)
        def _fake_func(data, tuple_arg, chunks):
            return da.from_array(data)

        data = list(range(5))
        with pytest.raises(RuntimeError), \
                satpy.config.set(cache_lonlats=True, cache_dir=str(tmp_path)):
            _fake_func(data, (1, 2, 3), 5)

    def test_cached_result_numpy_fails(self, tmp_path):
        """Test that trying to cache with non-dask arrays fails."""
        from satpy.modifiers.angles import _sanitize_args_with_chunks, cache_to_zarr_if

        @cache_to_zarr_if("cache_lonlats", sanitize_args_func=_sanitize_args_with_chunks)
        def _fake_func(shape, chunks):
            return np.zeros(shape)

        with pytest.raises(ValueError), \
                satpy.config.set(cache_lonlats=True, cache_dir=str(tmp_path)):
            _fake_func((5, 5), ((5,), (5,)))

    def test_no_cache_dir_fails(self, tmp_path):
        """Test that 'cache_dir' not being set fails."""
        from satpy.modifiers.angles import _get_sensor_angles_from_sat_pos, get_angles
        data = _get_angle_test_data()
        with pytest.raises(RuntimeError), \
                satpy.config.set(cache_lonlats=True, cache_sensor_angles=True, cache_dir=None):
            get_angles(data)
        with pytest.raises(RuntimeError), \
                satpy.config.set(cache_lonlats=True, cache_sensor_angles=True, cache_dir=None):
            _get_sensor_angles_from_sat_pos.cache_clear()

    def test_relative_azimuth_calculation(self):
        """Test relative azimuth calculation."""
        from satpy.modifiers.angles import compute_relative_azimuth

        saa = xr.DataArray(np.array([-120, 40., 0.04, 179.4, 94.2, 12.1]))
        vaa = xr.DataArray(np.array([60., 57.7, 175.1, 234.18, 355.4, 12.1]))

        expected_raa = xr.DataArray(np.array([180., 17.7, 175.06, 54.78, 98.8, 0.]))

        raa = compute_relative_azimuth(vaa, saa)
        assert isinstance(raa, xr.DataArray)
        np.testing.assert_allclose(expected_raa, raa)

    def test_solazi_correction(self):
        """Test that solar azimuth angles are corrected into the right range."""
        from datetime import datetime

        from satpy.modifiers.angles import _get_sun_azimuth_ndarray

        lats = np.array([-80, 40, 0, 40, 80])
        lons = np.array([-80, 40, 0, 40, 80])

        dt = datetime(2022, 1, 5, 12, 50, 0)

        azi = _get_sun_azimuth_ndarray(lats, lons, dt)

        assert np.all(azi > 0)
