#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""Tests CF-compliant DataArray creation."""

import datetime
from collections import OrderedDict

import numpy as np
import xarray as xr

from satpy.tests.utils import make_dsq


def test_preprocess_dataarray_name():
    """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix include orig name."""
    from satpy import Scene
    from satpy.writers.cf.dataarray import _preprocess_dataarray_name

    scn = Scene()
    scn['1'] = xr.DataArray([1, 2, 3])
    dataarray = scn['1']
    # If numeric_name_prefix is a string, test add the original_name attributes
    out_da = _preprocess_dataarray_name(dataarray, numeric_name_prefix="TEST", include_orig_name=True)
    assert out_da.attrs['original_name'] == '1'

    # If numeric_name_prefix is empty string, False or None, test do not add original_name attributes
    out_da = _preprocess_dataarray_name(dataarray, numeric_name_prefix="", include_orig_name=True)
    assert "original_name" not in out_da.attrs

    out_da = _preprocess_dataarray_name(dataarray, numeric_name_prefix=False, include_orig_name=True)
    assert "original_name" not in out_da.attrs

    out_da = _preprocess_dataarray_name(dataarray, numeric_name_prefix=None, include_orig_name=True)
    assert "original_name" not in out_da.attrs


def test_make_cf_dataarray_lonlat():
    """Test correct CF encoding for area with lon/lat units."""
    from pyresample import create_area_def

    from satpy.resample import add_crs_xy_coords
    from satpy.writers.cf.dataarray import make_cf_dataarray

    area = create_area_def("mavas", 4326, shape=(5, 5),
                           center=(0, 0), resolution=(1, 1))
    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"),
        attrs={"area": area})
    da = add_crs_xy_coords(da, area)
    new_da = make_cf_dataarray(da)
    assert new_da["x"].attrs["units"] == "degrees_east"
    assert new_da["y"].attrs["units"] == "degrees_north"


class TestCFWriter:
    """Test creation of CF DataArray."""

    def get_test_attrs(self):
        """Create some dataset attributes for testing purpose.

        Returns:
            Attributes, encoded attributes, encoded and flattened attributes

        """
        # TODO: also used by cf/test_attrs.py
        attrs = {'name': 'IR_108',
                 'start_time': datetime.datetime(2018, 1, 1, 0),
                 'end_time': datetime.datetime(2018, 1, 1, 0, 15),
                 'int': 1,
                 'float': 1.0,
                 'none': None,  # should be dropped
                 'numpy_int': np.uint8(1),
                 'numpy_float': np.float32(1),
                 'numpy_bool': True,
                 'numpy_void': np.void(0),
                 'numpy_bytes': np.bytes_('test'),
                 'numpy_string': np.string_('test'),
                 'list': [1, 2, np.float64(3)],
                 'nested_list': ["1", ["2", [3]]],
                 'bool': True,
                 'array': np.array([1, 2, 3], dtype='uint8'),
                 'array_bool': np.array([True, False, True]),
                 'array_2d': np.array([[1, 2], [3, 4]]),
                 'array_3d': np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
                 'dict': {'a': 1, 'b': 2},
                 'nested_dict': {'l1': {'l2': {'l3': np.array([1, 2, 3], dtype='uint8')}}},
                 'raw_metadata': OrderedDict([
                     ('recarray', np.zeros(3, dtype=[('x', 'i4'), ('y', 'u1')])),
                     ('flag', np.bool_(True)),
                     ('dict', OrderedDict([('a', 1), ('b', np.array([1, 2, 3], dtype='uint8'))]))
                 ])}
        encoded = {'name': 'IR_108',
                   'start_time': '2018-01-01 00:00:00',
                   'end_time': '2018-01-01 00:15:00',
                   'int': 1,
                   'float': 1.0,
                   'numpy_int': np.uint8(1),
                   'numpy_float': np.float32(1),
                   'numpy_bool': 'true',
                   'numpy_void': '[]',
                   'numpy_bytes': 'test',
                   'numpy_string': 'test',
                   'list': [1, 2, np.float64(3)],
                   'nested_list': '["1", ["2", [3]]]',
                   'bool': 'true',
                   'array': np.array([1, 2, 3], dtype='uint8'),
                   'array_bool': ['true', 'false', 'true'],
                   'array_2d': '[[1, 2], [3, 4]]',
                   'array_3d': '[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]',
                   'dict': '{"a": 1, "b": 2}',
                   'nested_dict': '{"l1": {"l2": {"l3": [1, 2, 3]}}}',
                   'raw_metadata': '{"recarray": [[0, 0], [0, 0], [0, 0]], '
                                   '"flag": "true", "dict": {"a": 1, "b": [1, 2, 3]}}'}
        encoded_flat = {'name': 'IR_108',
                        'start_time': '2018-01-01 00:00:00',
                        'end_time': '2018-01-01 00:15:00',
                        'int': 1,
                        'float': 1.0,
                        'numpy_int': np.uint8(1),
                        'numpy_float': np.float32(1),
                        'numpy_bool': 'true',
                        'numpy_void': '[]',
                        'numpy_bytes': 'test',
                        'numpy_string': 'test',
                        'list': [1, 2, np.float64(3)],
                        'nested_list': '["1", ["2", [3]]]',
                        'bool': 'true',
                        'array': np.array([1, 2, 3], dtype='uint8'),
                        'array_bool': ['true', 'false', 'true'],
                        'array_2d': '[[1, 2], [3, 4]]',
                        'array_3d': '[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]',
                        'dict_a': 1,
                        'dict_b': 2,
                        'nested_dict_l1_l2_l3': np.array([1, 2, 3], dtype='uint8'),
                        'raw_metadata_recarray': '[[0, 0], [0, 0], [0, 0]]',
                        'raw_metadata_flag': 'true',
                        'raw_metadata_dict_a': 1,
                        'raw_metadata_dict_b': np.array([1, 2, 3], dtype='uint8')}
        return attrs, encoded, encoded_flat

    def assertDictWithArraysEqual(self, d1, d2):
        """Check that dicts containing arrays are equal."""
        # TODO: also used by cf/test_attrs.py
        assert set(d1.keys()) == set(d2.keys())
        for key, val1 in d1.items():
            val2 = d2[key]
            if isinstance(val1, np.ndarray):
                np.testing.assert_array_equal(val1, val2)
                assert val1.dtype == val2.dtype
            else:
                assert val1 == val2
                if isinstance(val1, (np.floating, np.integer, np.bool_)):
                    assert isinstance(val2, np.generic)
                    assert val1.dtype == val2.dtype

    def test_make_cf_dataarray(self):
        """Test the conversion of a DataArray to a CF-compatible DataArray."""
        from satpy.writers.cf.dataarray import make_cf_dataarray

        # Create set of test attributes
        attrs, attrs_expected, attrs_expected_flat = self.get_test_attrs()
        attrs['area'] = 'some_area'
        attrs['prerequisites'] = [make_dsq(name='hej')]
        attrs['_satpy_id_name'] = 'myname'

        # Adjust expected attributes
        expected_prereq = ("DataQuery(name='hej')")
        update = {'prerequisites': [expected_prereq], 'long_name': attrs['name']}

        attrs_expected.update(update)
        attrs_expected_flat.update(update)

        attrs_expected.pop('name')
        attrs_expected_flat.pop('name')

        # Create test data array
        arr = xr.DataArray(np.array([[1, 2], [3, 4]]), attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [1, 2], 'acq_time': ('y', [3, 4])})

        # Test conversion to something cf-compliant
        res = make_cf_dataarray(arr)
        np.testing.assert_array_equal(res['x'], arr['x'])
        np.testing.assert_array_equal(res['y'], arr['y'])
        np.testing.assert_array_equal(res['acq_time'], arr['acq_time'])
        assert res['x'].attrs == {'units': 'm', 'standard_name': 'projection_x_coordinate'}
        assert res['y'].attrs == {'units': 'm', 'standard_name': 'projection_y_coordinate'}
        self.assertDictWithArraysEqual(res.attrs, attrs_expected)

        # Test attribute kwargs
        res_flat = make_cf_dataarray(arr, flatten_attrs=True, exclude_attrs=['int'])
        attrs_expected_flat.pop('int')
        self.assertDictWithArraysEqual(res_flat.attrs, attrs_expected_flat)

    def test_make_cf_dataarray_one_dimensional_array(self):
        """Test the conversion of an 1d DataArray to a CF-compatible DataArray."""
        from satpy.writers.cf.dataarray import make_cf_dataarray

        arr = xr.DataArray(np.array([1, 2, 3, 4]), attrs={}, dims=('y',),
                           coords={'y': [0, 1, 2, 3], 'acq_time': ('y', [0, 1, 2, 3])})
        _ = make_cf_dataarray(arr)
