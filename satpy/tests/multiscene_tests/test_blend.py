#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018-2023 Satpy developers
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

"""Unit tests for blending datasets with the Multiscene object."""

from datetime import datetime

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy import DataQuery, Scene
from satpy.multiscene import stack, timeseries
from satpy.tests.multiscene_tests.test_utils import (
    DEFAULT_SHAPE,
    _create_test_area,
    _create_test_dataset,
    _create_test_int8_dataset,
)
from satpy.tests.utils import make_dataid

NUM_TEST_ROWS = 2
NUM_TEST_COLS = 3


def _get_expected_stack_select(scene1: Scene, scene2: Scene) -> xr.DataArray:
    expected = scene2['polar-ct']
    expected[..., NUM_TEST_ROWS, :] = scene1['geo-ct'][..., NUM_TEST_ROWS, :]
    expected[..., :, NUM_TEST_COLS] = scene1['geo-ct'][..., :, NUM_TEST_COLS]
    expected[..., -1, :] = scene1['geo-ct'][..., -1, :]
    return expected.compute()


def _get_expected_stack_blend(scene1: Scene, scene2: Scene) -> xr.DataArray:
    expected = scene2['polar-ct'].copy().compute().astype(np.float64)
    expected[..., NUM_TEST_ROWS, :] = 5 / 3  # (1*2 + 3*1) / (2 + 1)
    expected[..., :, NUM_TEST_COLS] = 5 / 3
    expected[..., -1, :] = np.nan  # (1*0 + 0*1) / (0 + 1)
    # weight of 1 is masked to 0 because invalid overlay value:
    expected[..., -1, NUM_TEST_COLS] = 2 / 2  # (1*2 + 0*1) / (2 + 0)
    return expected


@pytest.fixture
def test_area():
    """Get area definition used by test DataArrays."""
    return _create_test_area()


@pytest.fixture(params=[np.int8, np.float32])
def data_type(request):
    """Get array data type of the DataArray being tested."""
    return request.param


@pytest.fixture(params=["", "L", "RGB", "RGBA"])
def image_mode(request):
    """Get image mode of the main DataArray being tested."""
    return request.param


@pytest.fixture
def cloud_type_data_array1(test_area, data_type, image_mode):
    """Get DataArray for cloud type in the first test Scene."""
    dsid1 = make_dataid(
        name="geo-ct",
        resolution=3000,
        modifiers=()
    )
    shape = DEFAULT_SHAPE if len(image_mode) == 0 else (len(image_mode),) + DEFAULT_SHAPE
    dims = ("y", "x") if len(image_mode) == 0 else ("bands", "y", "x")
    if data_type is np.int8:
        data_arr = _create_test_int8_dataset(name='geo-ct', shape=shape, area=test_area, values=1, dims=dims)
    else:
        data_arr = _create_test_dataset(name='geo-ct', shape=shape, area=test_area, values=1.0, dims=dims)

    data_arr.attrs['platform_name'] = 'Meteosat-11'
    data_arr.attrs['sensor'] = {'seviri'}
    data_arr.attrs['units'] = '1'
    data_arr.attrs['long_name'] = 'NWC GEO CT Cloud Type'
    data_arr.attrs['orbital_parameters'] = {
        'satellite_nominal_altitude': 35785863.0,
        'satellite_nominal_longitude': 0.0,
        'satellite_nominal_latitude': 0,
    }
    data_arr.attrs['start_time'] = datetime(2023, 1, 16, 11, 9, 17)
    data_arr.attrs['end_time'] = datetime(2023, 1, 16, 11, 12, 22)
    data_arr.attrs["_satpy_id"] = dsid1
    return data_arr


@pytest.fixture
def cloud_type_data_array2(test_area, data_type, image_mode):
    """Get DataArray for cloud type in the second test Scene."""
    dsid1 = make_dataid(
        name="polar-ct",
        resolution=1000,
        modifiers=()
    )
    shape = DEFAULT_SHAPE if len(image_mode) == 0 else (len(image_mode),) + DEFAULT_SHAPE
    dims = ("y", "x") if len(image_mode) == 0 else ("bands", "y", "x")
    if data_type is np.int8:
        data_arr = _create_test_int8_dataset(name='polar-ct', shape=shape, area=test_area, values=3, dims=dims)
        data_arr[..., -1, :] = data_arr.attrs['_FillValue']
    else:
        data_arr = _create_test_dataset(name='polar-ct', shape=shape, area=test_area, values=3.0, dims=dims)
        data_arr[..., -1, :] = np.nan
    data_arr.attrs['platform_name'] = 'NOAA-18'
    data_arr.attrs['sensor'] = {'avhrr-3'}
    data_arr.attrs['units'] = '1'
    data_arr.attrs['long_name'] = 'SAFNWC PPS CT Cloud Type'
    data_arr.attrs['start_time'] = datetime(2023, 1, 16, 11, 12, 57, 500000)
    data_arr.attrs['end_time'] = datetime(2023, 1, 16, 11, 28, 1, 900000)
    data_arr.attrs["_satpy_id"] = dsid1
    return data_arr


@pytest.fixture
def scene1_with_weights(cloud_type_data_array1, test_area):
    """Create first test scene with a dataset of weights."""
    from satpy import Scene

    scene = Scene()
    scene[cloud_type_data_array1.attrs["_satpy_id"]] = cloud_type_data_array1

    wgt1 = _create_test_dataset(name='geo-ct-wgt', area=test_area, values=0)

    wgt1[NUM_TEST_ROWS, :] = 2
    wgt1[:, NUM_TEST_COLS] = 2

    dsid2 = make_dataid(
        name="geo-cma",
        resolution=3000,
        modifiers=()
    )
    scene[dsid2] = _create_test_int8_dataset(name='geo-cma', area=test_area, values=2)
    scene[dsid2].attrs['start_time'] = datetime(2023, 1, 16, 11, 9, 17)
    scene[dsid2].attrs['end_time'] = datetime(2023, 1, 16, 11, 12, 22)

    wgt2 = _create_test_dataset(name='geo-cma-wgt', area=test_area, values=0)

    return scene, [wgt1, wgt2]


@pytest.fixture
def scene2_with_weights(cloud_type_data_array2, test_area):
    """Create second test scene."""
    from satpy import Scene

    scene = Scene()
    scene[cloud_type_data_array2.attrs["_satpy_id"]] = cloud_type_data_array2

    wgt1 = _create_test_dataset(name='polar-ct-wgt', area=test_area, values=1)

    dsid2 = make_dataid(
        name="polar-cma",
        resolution=1000,
        modifiers=()
    )
    scene[dsid2] = _create_test_int8_dataset(name='polar-cma', area=test_area, values=4)
    scene[dsid2].attrs['start_time'] = datetime(2023, 1, 16, 11, 12, 57, 500000)
    scene[dsid2].attrs['end_time'] = datetime(2023, 1, 16, 11, 28, 1, 900000)

    wgt2 = _create_test_dataset(name='polar-cma-wgt', area=test_area, values=1)
    return scene, [wgt1, wgt2]


@pytest.fixture
def multi_scene_and_weights(scene1_with_weights, scene2_with_weights):
    """Create small multi-scene for testing."""
    from satpy import MultiScene
    scene1, weights1 = scene1_with_weights
    scene2, weights2 = scene2_with_weights

    return MultiScene([scene1, scene2]), [weights1, weights2]


@pytest.fixture
def groups():
    """Get group definitions for the MultiScene."""
    return {
        DataQuery(name='CloudType'): ['geo-ct', 'polar-ct'],
        DataQuery(name='CloudMask'): ['geo-cma', 'polar-cma']
    }


class TestBlendFuncs:
    """Test individual functions used for blending."""

    def test_blend_two_scenes_using_stack(self, multi_scene_and_weights, groups,
                                          scene1_with_weights, scene2_with_weights):
        """Test blending two scenes by stacking them on top of each other using function 'stack'."""
        multi_scene, weights = multi_scene_and_weights
        scene1, weights1 = scene1_with_weights
        scene2, weights2 = scene2_with_weights

        multi_scene.group(groups)

        resampled = multi_scene
        stacked = resampled.blend(blend_function=stack)
        result = stacked['CloudType'].compute()

        expected = scene2['polar-ct'].copy()
        expected[..., -1, :] = scene1['geo-ct'][..., -1, :]

        xr.testing.assert_equal(result, expected.compute())
        _check_stacked_metadata(result, "CloudType")
        assert result.attrs['start_time'] == datetime(2023, 1, 16, 11, 9, 17)
        assert result.attrs['end_time'] == datetime(2023, 1, 16, 11, 28, 1, 900000)

    def test_blend_two_scenes_bad_blend_type(self, multi_scene_and_weights, groups):
        """Test exception is raised when bad 'blend_type' is used."""
        from functools import partial

        multi_scene, weights = multi_scene_and_weights

        simple_groups = {DataQuery(name='CloudType'): groups[DataQuery(name='CloudType')]}
        multi_scene.group(simple_groups)
        weights = [weights[0][0], weights[1][0]]
        stack_func = partial(stack, weights=weights, blend_type="i_dont_exist")
        with pytest.raises(ValueError):
            multi_scene.blend(blend_function=stack_func)

    @pytest.mark.parametrize(
        ("blend_func", "exp_result_func"),
        [
            ("select_with_weights", _get_expected_stack_select),
            ("blend_with_weights", _get_expected_stack_blend),
        ])
    @pytest.mark.parametrize("combine_times", [False, True])
    def test_blend_two_scenes_using_stack_weighted(self, multi_scene_and_weights, groups,
                                                   scene1_with_weights, scene2_with_weights,
                                                   combine_times, blend_func, exp_result_func):
        """Test stacking two scenes using weights.

        Here we test that the start and end times can be combined so that they
        describe the start and times of the entire data series. We also test
        the various types of weighted stacking functions (ex. select, blend).

        """
        from functools import partial

        multi_scene, weights = multi_scene_and_weights
        scene1, weights1 = scene1_with_weights
        scene2, weights2 = scene2_with_weights

        simple_groups = {DataQuery(name='CloudType'): groups[DataQuery(name='CloudType')]}
        multi_scene.group(simple_groups)

        weights = [weights[0][0], weights[1][0]]
        stack_func = partial(stack, weights=weights, blend_type=blend_func, combine_times=combine_times)
        weighted_blend = multi_scene.blend(blend_function=stack_func)

        expected = exp_result_func(scene1, scene2)
        result = weighted_blend['CloudType'].compute()
        # result has NaNs and xarray's xr.testing.assert_equal doesn't support NaN comparison
        np.testing.assert_allclose(result.data, expected.data)

        _check_stacked_metadata(result, "CloudType")
        if combine_times:
            assert result.attrs['start_time'] == datetime(2023, 1, 16, 11, 9, 17)
            assert result.attrs['end_time'] == datetime(2023, 1, 16, 11, 28, 1, 900000)
        else:
            assert result.attrs['start_time'] == datetime(2023, 1, 16, 11, 11, 7, 250000)
            assert result.attrs['end_time'] == datetime(2023, 1, 16, 11, 20, 11, 950000)

    @pytest.fixture
    def datasets_and_weights(self):
        """X-Array datasets with area definition plus weights for input to tests."""
        shape = (8, 12)
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                              shape[1], shape[0], [-200, -200, 200, 200])

        ds1 = xr.DataArray(da.ones(shape, chunks=-1), dims=('y', 'x'),
                           attrs={'start_time': datetime(2018, 1, 1, 0, 0, 0), 'area': area})
        ds2 = xr.DataArray(da.ones(shape, chunks=-1) * 2, dims=('y', 'x'),
                           attrs={'start_time': datetime(2018, 1, 1, 1, 0, 0), 'area': area})
        ds3 = xr.DataArray(da.ones(shape, chunks=-1) * 3, dims=('y', 'x'),
                           attrs={'start_time': datetime(2018, 1, 1, 1, 0, 0), 'area': area})

        ds4 = xr.DataArray(da.zeros(shape, chunks=-1), dims=('y', 'time'),
                           attrs={'start_time': datetime(2018, 1, 1, 0, 0, 0), 'area': area})
        ds5 = xr.DataArray(da.zeros(shape, chunks=-1), dims=('y', 'time'),
                           attrs={'start_time': datetime(2018, 1, 1, 1, 0, 0), 'area': area})

        wgt1 = xr.DataArray(da.ones(shape, chunks=-1), dims=('y', 'x'),
                            attrs={'start_time': datetime(2018, 1, 1, 0, 0, 0), 'area': area})
        wgt2 = xr.DataArray(da.zeros(shape, chunks=-1), dims=('y', 'x'),
                            attrs={'start_time': datetime(2018, 1, 1, 0, 0, 0), 'area': area})
        wgt3 = xr.DataArray(da.zeros(shape, chunks=-1), dims=('y', 'x'),
                            attrs={'start_time': datetime(2018, 1, 1, 0, 0, 0), 'area': area})

        datastruct = {'shape': shape,
                      'area': area,
                      'datasets': [ds1, ds2, ds3, ds4, ds5],
                      'weights': [wgt1, wgt2, wgt3]}
        return datastruct

    @pytest.mark.parametrize(('line', 'column',),
                             [(2, 3), (4, 5)]
                             )
    def test_blend_function_stack_weighted(self, datasets_and_weights, line, column):
        """Test the 'stack_weighted' function."""
        from functools import partial

        from satpy.dataset import combine_metadata

        input_data = datasets_and_weights

        input_data['weights'][1][line, :] = 2
        input_data['weights'][2][:, column] = 2

        stack_with_weights = partial(stack, weights=input_data['weights'], combine_times=False)
        blend_result = stack_with_weights(input_data['datasets'][0:3])

        ds1 = input_data['datasets'][0]
        ds2 = input_data['datasets'][1]
        ds3 = input_data['datasets'][2]
        expected = ds1.copy()
        expected[:, column] = ds3[:, column]
        expected[line, :] = ds2[line, :]
        expected.attrs = combine_metadata(*[x.attrs for x in input_data['datasets'][0:3]])

        xr.testing.assert_equal(blend_result.compute(), expected.compute())
        assert expected.attrs == blend_result.attrs

    def test_blend_function_stack(self, datasets_and_weights):
        """Test the 'stack' function."""
        input_data = datasets_and_weights

        ds1 = input_data['datasets'][0]
        ds2 = input_data['datasets'][1]

        res = stack([ds1, ds2])
        expected = ds2.copy()
        expected.attrs["start_time"] = ds1.attrs["start_time"]

        xr.testing.assert_equal(res.compute(), expected.compute())
        assert expected.attrs == res.attrs

    def test_timeseries(self, datasets_and_weights):
        """Test the 'timeseries' function."""
        input_data = datasets_and_weights

        ds1 = input_data['datasets'][0]
        ds2 = input_data['datasets'][1]
        ds4 = input_data['datasets'][2]
        ds4 = input_data['datasets'][3]
        ds5 = input_data['datasets'][4]

        res = timeseries([ds1, ds2])
        res2 = timeseries([ds4, ds5])
        assert isinstance(res, xr.DataArray)
        assert isinstance(res2, xr.DataArray)
        assert (2, ds1.shape[0], ds1.shape[1]) == res.shape
        assert (ds4.shape[0], ds4.shape[1]+ds5.shape[1]) == res2.shape


def _check_stacked_metadata(data_arr: xr.DataArray, exp_name: str) -> None:
    assert data_arr.attrs['units'] == '1'
    assert data_arr.attrs['name'] == exp_name
    if "_FillValue" in data_arr.attrs:
        assert data_arr.attrs['_FillValue'] == 255
        assert data_arr.attrs['valid_range'] == [1, 15]

    expected_area = _create_test_area()
    assert data_arr.attrs['area'] == expected_area

    # these metadata items don't match between all inputs
    assert 'sensor' not in data_arr.attrs
    assert 'platform_name' not in data_arr.attrs
    assert 'long_name' not in data_arr.attrs


class TestTemporalRGB:
    """Test the temporal RGB blending method."""

    @pytest.fixture
    def nominal_data(self):
        """Return the input arrays for the nominal use case."""
        da1 = xr.DataArray([1, 0, 0], attrs={'start_time': datetime(2023, 5, 22, 9, 0, 0)})
        da2 = xr.DataArray([0, 1, 0], attrs={'start_time': datetime(2023, 5, 22, 10, 0, 0)})
        da3 = xr.DataArray([0, 0, 1], attrs={'start_time': datetime(2023, 5, 22, 11, 0, 0)})

        return [da1, da2, da3]

    @pytest.fixture
    def expected_result(self):
        """Return the expected result arrays."""
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    @staticmethod
    def _assert_results(res, expected_start_time, expected_result):
        assert res.attrs['start_time'] == expected_start_time
        np.testing.assert_equal(res.data[0, :], expected_result[0])
        np.testing.assert_equal(res.data[1, :], expected_result[1])
        np.testing.assert_equal(res.data[2, :], expected_result[2])

    def test_nominal(self, nominal_data, expected_result):
        """Test that nominal usage with 3 datasets works."""
        from satpy.multiscene import temporal_rgb

        res = temporal_rgb(nominal_data)

        self._assert_results(res, nominal_data[-1].attrs['start_time'], expected_result)

    def test_extra_datasets(self, nominal_data, expected_result):
        """Test that only the first three arrays affect the usage."""
        from satpy.multiscene import temporal_rgb

        da4 = xr.DataArray([0, 0, 1], attrs={'start_time': datetime(2023, 5, 22, 12, 0, 0)})

        res = temporal_rgb(nominal_data + [da4,])

        self._assert_results(res, nominal_data[-1].attrs['start_time'], expected_result)
