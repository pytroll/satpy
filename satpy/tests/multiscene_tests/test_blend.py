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
import pytest
import xarray as xr
from pyresample.geometry import AreaDefinition

from satpy import DataQuery
from satpy.multiscene import stack
from satpy.tests.multiscene_tests.test_utils import _create_test_area, _create_test_dataset, _create_test_int8_dataset
from satpy.tests.utils import make_dataid


class TestBlendFuncs:
    """Test individual functions used for blending."""

    def setup_method(self):
        """Set up test functions."""
        self._line = 2
        self._column = 3

    @pytest.fixture
    def scene1_with_weights(self):
        """Create first test scene with a dataset of weights."""
        from satpy import Scene

        area = _create_test_area()
        scene = Scene()
        dsid1 = make_dataid(
            name="geo-ct",
            resolution=3000,
            modifiers=()
        )
        scene[dsid1] = _create_test_int8_dataset(name='geo-ct', area=area, values=1)
        scene[dsid1].attrs['platform_name'] = 'Meteosat-11'
        scene[dsid1].attrs['sensor'] = set({'seviri'})
        scene[dsid1].attrs['units'] = '1'
        scene[dsid1].attrs['long_name'] = 'NWC GEO CT Cloud Type'
        scene[dsid1].attrs['orbital_parameters'] = {'satellite_nominal_altitude': 35785863.0,
                                                    'satellite_nominal_longitude': 0.0,
                                                    'satellite_nominal_latitude': 0}
        scene[dsid1].attrs['start_time'] = datetime(2023, 1, 16, 11, 9, 17)
        scene[dsid1].attrs['end_time'] = datetime(2023, 1, 16, 11, 12, 22)

        wgt1 = _create_test_dataset(name='geo-ct-wgt', area=area, values=0)

        wgt1[self._line, :] = 2
        wgt1[:, self._column] = 2

        dsid2 = make_dataid(
            name="geo-cma",
            resolution=3000,
            modifiers=()
        )
        scene[dsid2] = _create_test_int8_dataset(name='geo-cma', area=area, values=2)
        scene[dsid2].attrs['start_time'] = datetime(2023, 1, 16, 11, 9, 17)
        scene[dsid2].attrs['end_time'] = datetime(2023, 1, 16, 11, 12, 22)

        wgt2 = _create_test_dataset(name='geo-cma-wgt', area=area, values=0)

        return scene, [wgt1, wgt2]

    @pytest.fixture
    def scene2_with_weights(self):
        """Create second test scene."""
        from satpy import Scene

        area = _create_test_area()
        scene = Scene()
        dsid1 = make_dataid(
            name="polar-ct",
            resolution=1000,
            modifiers=()
        )
        scene[dsid1] = _create_test_int8_dataset(name='polar-ct', area=area, values=3)
        scene[dsid1].attrs['platform_name'] = 'NOAA-18'
        scene[dsid1].attrs['sensor'] = set({'avhrr-3'})
        scene[dsid1].attrs['units'] = '1'
        scene[dsid1].attrs['long_name'] = 'SAFNWC PPS CT Cloud Type'
        scene[dsid1][-1, :] = scene[dsid1].attrs['_FillValue']
        scene[dsid1].attrs['start_time'] = datetime(2023, 1, 16, 11, 12, 57, 500000)
        scene[dsid1].attrs['end_time'] = datetime(2023, 1, 16, 11, 28, 1, 900000)

        wgt1 = _create_test_dataset(name='polar-ct-wgt', area=area, values=1)

        dsid2 = make_dataid(
            name="polar-cma",
            resolution=1000,
            modifiers=()
        )
        scene[dsid2] = _create_test_int8_dataset(name='polar-cma', area=area, values=4)
        scene[dsid2].attrs['start_time'] = datetime(2023, 1, 16, 11, 12, 57, 500000)
        scene[dsid2].attrs['end_time'] = datetime(2023, 1, 16, 11, 28, 1, 900000)

        wgt2 = _create_test_dataset(name='polar-cma-wgt', area=area, values=1)
        return scene, [wgt1, wgt2]

    @pytest.fixture
    def multi_scene_and_weights(self, scene1_with_weights, scene2_with_weights):
        """Create small multi-scene for testing."""
        from satpy import MultiScene
        scene1, weights1 = scene1_with_weights
        scene2, weights2 = scene2_with_weights

        return MultiScene([scene1, scene2]), [weights1, weights2]

    @pytest.fixture
    def groups(self):
        """Get group definitions for the MultiScene."""
        return {
            DataQuery(name='CloudType'): ['geo-ct', 'polar-ct'],
            DataQuery(name='CloudMask'): ['geo-cma', 'polar-cma']
        }

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
        expected[-1, :] = scene1['geo-ct'][-1, :]

        xr.testing.assert_equal(result, expected.compute())
        assert result.attrs['platform_name'] == 'Meteosat-11'
        assert result.attrs['sensor'] == set({'seviri'})
        assert result.attrs['long_name'] == 'NWC GEO CT Cloud Type'
        assert result.attrs['units'] == '1'
        assert result.attrs['name'] == 'CloudType'
        assert result.attrs['_FillValue'] == 255
        assert result.attrs['valid_range'] == [1, 15]

        assert result.attrs['start_time'] == datetime(2023, 1, 16, 11, 9, 17)
        assert result.attrs['end_time'] == datetime(2023, 1, 16, 11, 12, 22)

    def test_blend_two_scenes_using_stack_weighted(self, multi_scene_and_weights, groups,
                                                   scene1_with_weights, scene2_with_weights):
        """Test stacking two scenes using weights - testing that metadata are combined correctly.

        Here we test that the start and end times can be combined so that they
        describe the start and times of the entire data series.

        """
        from functools import partial

        multi_scene, weights = multi_scene_and_weights
        scene1, weights1 = scene1_with_weights
        scene2, weights2 = scene2_with_weights

        simple_groups = {DataQuery(name='CloudType'): groups[DataQuery(name='CloudType')]}
        multi_scene.group(simple_groups)

        weights = [weights[0][0], weights[1][0]]
        stack_with_weights = partial(stack, weights=weights)
        weighted_blend = multi_scene.blend(blend_function=stack_with_weights)

        expected = scene2['polar-ct']
        expected[self._line, :] = scene1['geo-ct'][self._line, :]
        expected[:, self._column] = scene1['geo-ct'][:, self._column]
        expected[-1, :] = scene1['geo-ct'][-1, :]

        result = weighted_blend['CloudType'].compute()
        xr.testing.assert_equal(result, expected.compute())

        expected_area = _create_test_area()
        assert result.attrs['area'] == expected_area
        assert 'sensor' not in result.attrs
        assert 'platform_name' not in result.attrs
        assert 'long_name' not in result.attrs
        assert result.attrs['units'] == '1'
        assert result.attrs['name'] == 'CloudType'
        assert result.attrs['_FillValue'] == 255
        assert result.attrs['valid_range'] == [1, 15]

        assert result.attrs['start_time'] == datetime(2023, 1, 16, 11, 9, 17)
        assert result.attrs['end_time'] == datetime(2023, 1, 16, 11, 28, 1, 900000)

    def test_blend_two_scenes_using_stack_weighted_no_time_combination(self, multi_scene_and_weights, groups,
                                                                       scene1_with_weights, scene2_with_weights):
        """Test stacking two scenes using weights - test that the start and end times are averaged and not combined."""
        from functools import partial

        multi_scene, weights = multi_scene_and_weights
        scene1, weights1 = scene1_with_weights
        scene2, weights2 = scene2_with_weights

        simple_groups = {DataQuery(name='CloudType'): groups[DataQuery(name='CloudType')]}
        multi_scene.group(simple_groups)

        weights = [weights[0][0], weights[1][0]]
        stack_with_weights = partial(stack, weights=weights, combine_times=False)
        weighted_blend = multi_scene.blend(blend_function=stack_with_weights)

        result = weighted_blend['CloudType'].compute()

        expected_area = _create_test_area()
        assert result.attrs['area'] == expected_area
        assert 'sensor' not in result.attrs
        assert 'platform_name' not in result.attrs
        assert 'long_name' not in result.attrs
        assert result.attrs['units'] == '1'
        assert result.attrs['name'] == 'CloudType'
        assert result.attrs['_FillValue'] == 255
        assert result.attrs['valid_range'] == [1, 15]

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
        from satpy.multiscene import stack

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
        from satpy.multiscene import stack

        input_data = datasets_and_weights

        ds1 = input_data['datasets'][0]
        ds2 = input_data['datasets'][1]

        res = stack([ds1, ds2])
        expected = ds2.copy()

        xr.testing.assert_equal(res.compute(), expected.compute())

    def test_timeseries(self, datasets_and_weights):
        """Test the 'timeseries' function."""
        from satpy.multiscene import timeseries

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
