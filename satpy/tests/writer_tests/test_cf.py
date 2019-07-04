#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Tests for the CF writer."""

from collections import OrderedDict
import os
import sys
from datetime import datetime
import tempfile
from satpy import DatasetID

import numpy as np

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

try:
    from pyproj import CRS
except ImportError:
    CRS = None


class TempFile(object):
    """A temporary filename class."""

    def __init__(self):
        """Initialize."""
        self.filename = None

    def __enter__(self):
        """Enter."""
        self.handle, self.filename = tempfile.mkstemp()
        os.close(self.handle)
        return self.filename

    def __exit__(self, *args):
        """Exit."""
        os.remove(self.filename)


class TestCFWriter(unittest.TestCase):
    """Test case for CF writer."""

    def test_init(self):
        """Test initializing the CFWriter class."""
        from satpy.writers.cf_writer import CFWriter
        import satpy.config
        CFWriter(config_files=[os.path.join(satpy.config.CONFIG_PATH,
                                            'writers', 'cf.yaml')])

    def test_save_array(self):
        """Test saving an array to netcdf/cf."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[DatasetID('hej')]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                self.assertTrue(np.all(f['test-array'][:] == [1, 2, 3]))
                expected_prereq = ("DatasetID(name='hej', wavelength=None, "
                                   "resolution=None, polarization=None, "
                                   "calibration=None, level=None, modifiers=())")
                self.assertEqual(f['test-array'].attrs['prerequisites'],
                                 expected_prereq)

    def test_save_with_compression(self):
        """Test saving an array with compression."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        with mock.patch('satpy.writers.cf_writer.xr.Dataset') as xrdataset,\
                mock.patch('satpy.writers.cf_writer.make_time_bounds'):
            scn['test-array'] = xr.DataArray([1, 2, 3],
                                             attrs=dict(start_time=start_time,
                                                        end_time=end_time,
                                                        prerequisites=[DatasetID('hej')]))

            comp = {'zlib': True, 'complevel': 9}
            scn.save_datasets(filename='bla', writer='cf', compression=comp)
            ars, kws = xrdataset.call_args_list[1]
            self.assertDictEqual(ars[0]['test-array'].encoding, comp)

    def test_save_array_coords(self):
        """Test saving array with coordinates."""
        from satpy import Scene
        import xarray as xr
        import numpy as np
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        coords = {
            'x': np.arange(3),
            'y': np.arange(1),
        }
        if CRS is not None:
            proj_str = ('+proj=geos +lon_0=-95.0 +h=35786023.0 '
                        '+a=6378137.0 +b=6356752.31414 +sweep=x '
                        '+units=m +no_defs')
            coords['crs'] = CRS.from_string(proj_str)
        scn['test-array'] = xr.DataArray([[1, 2, 3]],
                                         dims=('y', 'x'),
                                         coords=coords,
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[DatasetID('hej')]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                self.assertTrue(np.all(f['test-array'][:] == [1, 2, 3]))
                self.assertTrue(np.all(f['x'][:] == [0, 1, 2]))
                self.assertTrue(np.all(f['y'][:] == [0]))
                self.assertNotIn('crs', f)
                self.assertNotIn('_FillValue', f['x'].attrs)
                self.assertNotIn('_FillValue', f['y'].attrs)
                expected_prereq = ("DatasetID(name='hej', wavelength=None, "
                                   "resolution=None, polarization=None, "
                                   "calibration=None, level=None, modifiers=())")
                self.assertEqual(f['test-array'].attrs['prerequisites'],
                                 expected_prereq)

    def test_groups(self):
        """Test creating a file with groups."""
        import xarray as xr
        from satpy import Scene

        tstart = datetime(2019, 4, 1, 12, 0)
        tend = datetime(2019, 4, 1, 12, 15)

        data_visir = [[1, 2], [3, 4]]
        y_visir = [1, 2]
        x_visir = [1, 2]
        time_vis006 = [1, 2]
        time_ir_108 = [3, 4]

        data_hrv = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        y_hrv = [1, 2, 3]
        x_hrv = [1, 2, 3]
        time_hrv = [1, 2, 3]

        scn = Scene()
        scn['VIS006'] = xr.DataArray(data_visir,
                                     dims=('y', 'x'),
                                     coords={'y': y_visir, 'x': x_visir, 'acq_time': ('y', time_vis006)},
                                     attrs={'name': 'VIS006', 'start_time': tstart, 'end_time': tend})
        scn['IR_108'] = xr.DataArray(data_visir,
                                     dims=('y', 'x'),
                                     coords={'y': y_visir, 'x': x_visir, 'acq_time': ('y', time_ir_108)},
                                     attrs={'name': 'IR_108', 'start_time': tstart, 'end_time': tend})
        scn['HRV'] = xr.DataArray(data_hrv,
                                  dims=('y', 'x'),
                                  coords={'y': y_hrv, 'x': x_hrv, 'acq_time': ('y', time_hrv)},
                                  attrs={'name': 'HRV', 'start_time': tstart, 'end_time': tend})

        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', groups={'visir': ['IR_108', 'VIS006'], 'hrv': ['HRV']},
                              pretty=True)

            nc_root = xr.open_dataset(filename)
            self.assertIn('history', nc_root.attrs)
            self.assertSetEqual(set(nc_root.variables.keys()), set())

            nc_visir = xr.open_dataset(filename, group='visir')
            nc_hrv = xr.open_dataset(filename, group='hrv')
            self.assertSetEqual(set(nc_visir.variables.keys()), {'VIS006', 'IR_108', 'y', 'x', 'VIS006_acq_time',
                                                                 'IR_108_acq_time'})
            self.assertSetEqual(set(nc_hrv.variables.keys()), {'HRV', 'y', 'x', 'acq_time'})
            for tst, ref in zip([nc_visir['VIS006'], nc_visir['IR_108'], nc_hrv['HRV']],
                                [scn['VIS006'], scn['IR_108'], scn['HRV']]):
                self.assertTrue(np.all(tst.data == ref.data))
            nc_root.close()
            nc_visir.close()
            nc_hrv.close()

        # Different projection coordinates in one group are not supported
        with TempFile() as filename:
            self.assertRaises(ValueError, scn.save_datasets, datasets=['VIS006', 'HRV'], filename=filename, writer='cf')

    def test_single_time_value(self):
        """Test setting a single time value."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]])
        scn['test-array'] = xr.DataArray(test_array,
                                         dims=['x', 'y'],
                                         coords={'time': np.datetime64('2018-05-30T10:05:00')},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename, decode_cf=False) as f:
                self.assertTrue(np.all(f['time_bnds'][:] == np.array([-300.,  600.])))

    def test_bounds(self):
        """Test setting time bounds."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        scn['test-array'] = xr.DataArray(test_array,
                                         dims=['x', 'y', 'time'],
                                         coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename, decode_cf=False) as f:
                self.assertTrue(np.all(f['time_bnds'][:] == np.array([-300.,  600.])))

    def test_bounds_minimum(self):
        """Test minimum bounds."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_timeA = datetime(2018, 5, 30, 10, 0)  # expected to be used
        end_timeA = datetime(2018, 5, 30, 10, 20)
        start_timeB = datetime(2018, 5, 30, 10, 3)
        end_timeB = datetime(2018, 5, 30, 10, 15)  # expected to be used
        test_arrayA = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        test_arrayB = np.array([[1, 2], [3, 5]]).reshape(2, 2, 1)
        scn['test-arrayA'] = xr.DataArray(test_arrayA,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                          attrs=dict(start_time=start_timeA,
                                                     end_time=end_timeA))
        scn['test-arrayB'] = xr.DataArray(test_arrayB,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                          attrs=dict(start_time=start_timeB,
                                                     end_time=end_timeB))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename, decode_cf=False) as f:
                self.assertTrue(np.all(f['time_bnds'][:] == np.array([-300.,  600.])))

    def test_bounds_missing_time_info(self):
        """Test time bounds generation in case of missing time."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_timeA = datetime(2018, 5, 30, 10, 0)
        end_timeA = datetime(2018, 5, 30, 10, 15)
        test_arrayA = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        test_arrayB = np.array([[1, 2], [3, 5]]).reshape(2, 2, 1)
        scn['test-arrayA'] = xr.DataArray(test_arrayA,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                          attrs=dict(start_time=start_timeA,
                                                     end_time=end_timeA))
        scn['test-arrayB'] = xr.DataArray(test_arrayB,
                                          dims=['x', 'y', 'time'],
                                          coords={'time': [np.datetime64('2018-05-30T10:05:00')]})
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename, decode_cf=False) as f:
                self.assertTrue(np.all(f['time_bnds'][:] == np.array([-300.,  600.])))

    def test_encoding_kwarg(self):
        """Test encoding of keyword arguments."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            encoding = {'test-array': {'dtype': 'int8',
                                       'scale_factor': 0.1,
                                       'add_offset': 0.0,
                                       '_FillValue': 3}}
            scn.save_datasets(filename=filename, encoding=encoding, writer='cf')
            with xr.open_dataset(filename, mask_and_scale=False) as f:
                self.assertTrue(np.all(f['test-array'][:] == [10, 20, 30]))
                self.assertTrue(f['test-array'].attrs['scale_factor'] == 0.1)
                self.assertTrue(f['test-array'].attrs['_FillValue'] == 3)
                # check that dtype behave as int8
                self.assertTrue(np.iinfo(f['test-array'][:].dtype).max == 127)

    def test_header_attrs(self):
        """Check master attributes are set."""
        from satpy import Scene
        import xarray as xr
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            header_attrs = {'sensor': 'SEVIRI',
                            'orbit': None}
            scn.save_datasets(filename=filename,
                              header_attrs=header_attrs,
                              writer='cf')
            with xr.open_dataset(filename) as f:
                self.assertTrue(f.attrs['sensor'] == 'SEVIRI')
                self.assertTrue('sensor' in f.attrs.keys())
                self.assertTrue('orbit' not in f.attrs.keys())

    def get_test_attrs(self):
        """Create some dataset attributes for testing purpose.

        Returns:
            Attributes, encoded attributes, encoded and flattened attributes

        """
        attrs = {'name': 'IR_108',
                 'start_time': datetime(2018, 1, 1, 0),
                 'end_time': datetime(2018, 1, 1, 0, 15),
                 'int': 1,
                 'float': 1.0,
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
                   'list': [1, 2, 3.0],
                   'nested_list': '["1", ["2", [3]]]',
                   'bool': 'True',
                   'array': np.array([1, 2, 3], dtype='uint8'),
                   'array_bool': ['True', 'False', 'True'],
                   'array_2d': '[[1, 2], [3, 4]]',
                   'array_3d': '[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]',
                   'dict': '{"a": 1, "b": 2}',
                   'nested_dict': '{"l1": {"l2": {"l3": [1, 2, 3]}}}',
                   'raw_metadata': '{"recarray": [[0, 0], [0, 0], [0, 0]], '
                                   '"flag": "True", "dict": {"a": 1, "b": [1, 2, 3]}}'}
        encoded_flat = {'name': 'IR_108',
                        'start_time': '2018-01-01 00:00:00',
                        'end_time': '2018-01-01 00:15:00',
                        'int': 1,
                        'float': 1.0,
                        'list': [1, 2, 3.0],
                        'nested_list': '["1", ["2", [3]]]',
                        'bool': 'True',
                        'array': np.array([1, 2, 3], dtype='uint8'),
                        'array_bool': ['True', 'False', 'True'],
                        'array_2d': '[[1, 2], [3, 4]]',
                        'array_3d': '[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]',
                        'dict_a': 1,
                        'dict_b': 2,
                        'nested_dict_l1_l2_l3': np.array([1, 2, 3], dtype='uint8'),
                        'raw_metadata_recarray': '[[0, 0], [0, 0], [0, 0]]',
                        'raw_metadata_flag': 'True',
                        'raw_metadata_dict_a': 1,
                        'raw_metadata_dict_b': np.array([1, 2, 3], dtype='uint8')}
        return attrs, encoded, encoded_flat

    def assertDictWithArraysEqual(self, d1, d2):
        """Check that dicts containing arrays are equal."""
        self.assertSetEqual(set(d1.keys()), set(d2.keys()))
        for key, val1 in d1.items():
            val2 = d2[key]
            if isinstance(val1, np.ndarray):
                self.assertTrue(np.all(val1 == val2))
                self.assertEqual(val1.dtype, val2.dtype)
            else:
                self.assertEqual(val1, val2)

    def test_encode_attrs_nc(self):
        """Test attributes encoding."""
        from satpy.writers.cf_writer import encode_attrs_nc
        import json

        attrs, expected, _ = self.get_test_attrs()

        # Test encoding
        encoded = encode_attrs_nc(attrs)
        self.assertDictWithArraysEqual(encoded, expected)

        # Test decoding of json-encoded attributes
        raw_md_roundtrip = {'recarray': [[0, 0], [0, 0], [0, 0]],
                            'flag': "True",
                            'dict': {'a': 1, 'b': [1, 2, 3]}}
        self.assertDictEqual(json.loads(encoded['raw_metadata']), raw_md_roundtrip)
        self.assertListEqual(json.loads(encoded['array_3d']), [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        self.assertDictEqual(json.loads(encoded['nested_dict']), {"l1": {"l2": {"l3": [1, 2, 3]}}})
        self.assertListEqual(json.loads(encoded['nested_list']), ["1", ["2", [3]]])

    def test_da2cf(self):
        """Test the conversion of a DataArray to a CF-compatible DataArray."""
        from satpy.writers.cf_writer import CFWriter
        import xarray as xr

        # Create set of test attributes
        attrs, attrs_expected, attrs_expected_flat = self.get_test_attrs()
        attrs['area'] = 'some_area'
        attrs['prerequisites'] = [DatasetID('hej')]

        # Adjust expected attributes
        expected_prereq = ("DatasetID(name='hej', wavelength=None, resolution=None, polarization=None, "
                           "calibration=None, level=None, modifiers=())")
        update = {'prerequisites': [expected_prereq], 'long_name': attrs['name']}

        attrs_expected.update(update)
        attrs_expected_flat.update(update)

        attrs_expected.pop('name')
        attrs_expected_flat.pop('name')

        # Create test data array
        arr = xr.DataArray(np.array([[1, 2], [3, 4]]), attrs=attrs, dims=('y', 'x'),
                           coords={'y': [0, 1], 'x': [1, 2], 'acq_time': ('y', [3, 4])})

        # Test conversion to something cf-compliant
        res = CFWriter.da2cf(arr)
        self.assertTrue(np.all(res['x'] == arr['x']))
        self.assertTrue(np.all(res['y'] == arr['y']))
        self.assertTrue(np.all(res['acq_time'] == arr['acq_time']))
        self.assertDictEqual(res['x'].attrs, {'units': 'm', 'standard_name': 'projection_x_coordinate'})
        self.assertDictEqual(res['y'].attrs, {'units': 'm', 'standard_name': 'projection_y_coordinate'})
        self.assertDictWithArraysEqual(res.attrs, attrs_expected)

        # Test attribute kwargs
        res_flat = CFWriter.da2cf(arr, flatten_attrs=True, exclude_attrs=['int'])
        attrs_expected_flat.pop('int')
        self.assertDictWithArraysEqual(res_flat.attrs, attrs_expected_flat)

    @mock.patch('satpy.writers.cf_writer.CFWriter.__init__', return_value=None)
    @mock.patch('satpy.writers.cf_writer.area2cf')
    @mock.patch('satpy.writers.cf_writer.CFWriter.da2cf')
    @mock.patch('satpy.writers.cf_writer.make_alt_coords_unique')
    @mock.patch('satpy.writers.cf_writer.assert_xy_unique')
    @mock.patch('satpy.writers.cf_writer.link_coords')
    def test_collect_datasets(self, link_coords, assert_xy_unique, make_alt_coords_unique, da2cf, area2cf, *mocks):
        """Test collecting CF datasets from a DataArray objects."""
        from satpy.writers.cf_writer import CFWriter
        import xarray as xr

        # Patch methods
        def identity(arg, **kwargs):
            return arg

        def raise_key_error(arg, **kwargs):
            raise KeyError

        da2cf.side_effect = identity
        area2cf.side_effect = raise_key_error
        make_alt_coords_unique.return_value = 'unique_coords'

        # Define test datasets
        data = [[1, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        time = [1, 2]
        tstart = datetime(2019, 4, 1, 12, 0)
        tend = datetime(2019, 4, 1, 12, 15)
        datasets = [xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                 attrs={'name': 'var1', 'start_time': tstart, 'end_time': tend}),
                    xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                 attrs={'name': 'var2'})]
        expected = {'var1': datasets[0], 'var2': datasets[1]}

        # Collect datasets
        writer = CFWriter()
        datas, start_times, end_times = writer._collect_datasets(datasets, include_lonlats=True)

        # Test results
        self.assertEqual(datas, 'unique_coords')
        self.assertListEqual(start_times, [tstart, None])
        self.assertListEqual(end_times, [tend, None])

        # Test method calls
        self.assertEqual(len(area2cf.call_args_list), 2)
        for call_args, ds in zip(area2cf.call_args_list, datasets):
            self.assertEqual(call_args, mock.call(ds, strict=True))

        for func in (assert_xy_unique, link_coords, make_alt_coords_unique):
            func.assert_called()
            call_arg = func.call_args[0][0]
            self.assertIsInstance(call_arg, dict)
            self.assertSetEqual(set(call_arg.keys()), {'var1', 'var2'})
            for key, ds in expected.items():
                self.assertTrue(call_arg[key].identical(ds))

    def test_assert_xy_unique(self):
        """Test that the x and y coordinates are unique."""
        import xarray as xr
        from satpy.writers.cf_writer import assert_xy_unique

        dummy = [[1, 2], [3, 4]]
        datas = {'a': xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]}),
                 'b': xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]}),
                 'n': xr.DataArray(data=dummy, dims=('v', 'w'), coords={'v': [1, 2], 'w': [3, 4]})}
        assert_xy_unique(datas)

        datas['c'] = xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 3], 'x': [3, 4]})
        self.assertRaises(ValueError, assert_xy_unique, datas)

    def test_link_coords(self):
        """Check that coordinates link has been established correctly."""
        import xarray as xr
        from satpy.writers.cf_writer import link_coords
        import numpy as np

        data = [[1, 2], [3, 4]]
        lon = np.zeros((2, 2))
        lat = np.ones((2, 2))
        datasets = {
            'var1': xr.DataArray(data=data, dims=('y', 'x'), attrs={'coordinates': 'lon lat'}),
            'var2': xr.DataArray(data=data, dims=('y', 'x')),
            'lon': xr.DataArray(data=lon, dims=('y', 'x')),
            'lat': xr.DataArray(data=lat, dims=('y', 'x'))
        }

        link_coords(datasets)

        # Check that link has been established correctly and 'coordinate' atrribute has been dropped
        self.assertIn('lon', datasets['var1'].coords)
        self.assertIn('lat', datasets['var1'].coords)
        self.assertTrue(np.all(datasets['var1']['lon'].data == lon))
        self.assertTrue(np.all(datasets['var1']['lat'].data == lat))
        self.assertNotIn('coordinates', datasets['var1'].attrs)

        # There should be no link if there was no 'coordinate' attribute
        self.assertNotIn('lon', datasets['var2'].coords)
        self.assertNotIn('lat', datasets['var2'].coords)

    def test_make_alt_coords_unique(self):
        """Test that created coordinate variables are unique."""
        import xarray as xr
        from satpy.writers.cf_writer import make_alt_coords_unique

        data = [[1, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        time1 = [1, 2]
        time2 = [3, 4]
        datasets = {'var1': xr.DataArray(data=data,
                                         dims=('y', 'x'),
                                         coords={'y': y, 'x': x, 'acq_time': ('y', time1)}),
                    'var2': xr.DataArray(data=data,
                                         dims=('y', 'x'),
                                         coords={'y': y, 'x': x, 'acq_time': ('y', time2)})}

        # Test that dataset names are prepended to alternative coordinates
        res = make_alt_coords_unique(datasets)
        self.assertTrue(np.all(res['var1']['var1_acq_time'] == time1))
        self.assertTrue(np.all(res['var2']['var2_acq_time'] == time2))
        self.assertNotIn('acq_time', res['var1'].coords)
        self.assertNotIn('acq_time', res['var2'].coords)

        # Make sure nothing else is modified
        self.assertTrue(np.all(res['var1']['x'] == x))
        self.assertTrue(np.all(res['var1']['y'] == y))
        self.assertTrue(np.all(res['var2']['x'] == x))
        self.assertTrue(np.all(res['var2']['y'] == y))

        # Coords not unique -> Dataset names must be prepended, even if pretty=True
        with mock.patch('satpy.writers.cf_writer.warnings.warn') as warn:
            res = make_alt_coords_unique(datasets, pretty=True)
            warn.assert_called()
            self.assertTrue(np.all(res['var1']['var1_acq_time'] == time1))
            self.assertTrue(np.all(res['var2']['var2_acq_time'] == time2))
            self.assertNotIn('acq_time', res['var1'].coords)
            self.assertNotIn('acq_time', res['var2'].coords)

        # Coords unique and pretty=True -> Don't modify coordinate names
        datasets['var2']['acq_time'] = ('y', time1)
        res = make_alt_coords_unique(datasets, pretty=True)
        self.assertTrue(np.all(res['var1']['acq_time'] == time1))
        self.assertTrue(np.all(res['var2']['acq_time'] == time1))
        self.assertNotIn('var1_acq_time', res['var1'].coords)
        self.assertNotIn('var2_acq_time', res['var2'].coords)

    @mock.patch('satpy.writers.cf_writer.area2lonlat')
    @mock.patch('satpy.writers.cf_writer.area2gridmapping')
    def test_area2cf(self, area2gridmapping, area2lonlat):
        """Test the conversion of an area to CF standards."""
        import xarray as xr
        import pyresample.geometry
        from satpy.writers.cf_writer import area2cf

        area2gridmapping.side_effect = lambda x: [1, 2, 3]
        area2lonlat.side_effect = lambda x: [4, 5, 6]
        ds_base = xr.DataArray(data=[[1, 2], [3, 4]], dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]},
                               attrs={'name': 'var1'})

        # a) Area Definition and strict=False
        geos = pyresample.geometry.AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': 35785831., 'a': 6378169., 'b': 6356583.8},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = geos

        res = area2cf(ds)
        self.assertEqual(len(res), 4)
        self.assertListEqual(res[0:3], [1, 2, 3])
        self.assertTrue(ds.identical(res[3]))

        # b) Area Definition and strict=False
        area2cf(ds, strict=True)
        area2lonlat.assert_called()

        # c) Swath Definition
        swath = pyresample.geometry.SwathDefinition(lons=[[1, 1], [2, 2]], lats=[[1, 2], [1, 2]])
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = swath

        res = area2cf(ds)
        self.assertEqual(len(res), 4)
        self.assertListEqual(res[0:3], [4, 5, 6])
        self.assertTrue(ds.identical(res[3]))

    def test_area2gridmapping(self):
        """Test the conversion from pyresample area object to CF grid mapping."""
        import xarray as xr
        import pyresample.geometry
        from satpy.writers.cf_writer import area2gridmapping

        ds_base = xr.DataArray(data=[[1, 2], [3, 4]], dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]},
                               attrs={'name': 'var1'})

        # a) Projection has a corresponding CF representation (e.g. geos)
        a = 6378169.
        b = 6356583.8
        h = 35785831.
        geos = pyresample.geometry.AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': h, 'a': a, 'b': b},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        geos_expected = xr.DataArray(data=0,
                                     attrs={'perspective_point_height': h,
                                            'latitude_of_projection_origin': None,
                                            'longitude_of_projection_origin': None,
                                            'grid_mapping_name': 'geostationary',
                                            'semi_major_axis': a,
                                            'semi_minor_axis': b,
                                            'sweep_axis': None,
                                            'name': 'geos'})

        ds = ds_base.copy()
        ds.attrs['area'] = geos
        res, grid_mapping = area2gridmapping(ds)

        self.assertEqual(res.attrs['grid_mapping'], 'geos')
        self.assertEqual(grid_mapping, geos_expected)

        # b) Projection does not have a corresponding CF representation (COSMO)
        cosmo7 = pyresample.geometry.AreaDefinition(
            area_id='cosmo7',
            description='cosmo7',
            proj_id='cosmo7',
            projection={'proj': 'ob_tran', 'ellps': 'WGS84', 'lat_0': 46, 'lon_0': 4.535,
                        'o_proj': 'stere', 'o_lat_p': 90, 'o_lon_p': -5.465},
            width=597, height=510,
            area_extent=[-1812933, -1003565, 814056, 1243448]
        )
        proj_str = '+proj=ob_tran +ellps=WGS84 +lat_0=46.0 +lon_0=4.535 +o_proj=stere +o_lat_p=90.0 +o_lon_p=-5.465'
        cosmo_expected = xr.DataArray(data=0, attrs={'name': 'proj4', 'proj4': proj_str})

        ds = ds_base.copy()
        ds.attrs['area'] = cosmo7

        with mock.patch('satpy.writers.cf_writer.warnings.warn') as warn:
            res, grid_mapping = area2gridmapping(ds)
            warn.assert_called()
            self.assertDictEqual(dict(pyresample.geometry.proj4_str_to_dict(res.attrs['grid_proj4'])),
                                 dict(pyresample.geometry.proj4_str_to_dict(proj_str)))
            self.assertEqual(grid_mapping, cosmo_expected)

    def test_area2lonlat(self):
        """Test the conversion from areas to lon/lat."""
        import pyresample.geometry
        import xarray as xr
        from satpy.writers.cf_writer import area2lonlat

        area = pyresample.geometry.AreaDefinition(
            'seviri',
            'Native SEVIRI grid',
            'geos',
            "+a=6378169.0 +h=35785831.0 +b=6356583.8 +lon_0=0 +proj=geos",
            2, 2,
            [-5570248.686685662, -5567248.28340708, 5567248.28340708, 5570248.686685662]
        )
        lons_ref, lats_ref = area.get_lonlats()
        dataarray = xr.DataArray(data=[[1, 2], [3, 4]], dims=('y', 'x'), attrs={'area': area})

        res = area2lonlat(dataarray)

        self.assertEqual(len(res), 1)
        self.assertEqual(set(res[0].coords), {'longitude', 'latitude'})
        lat = res[0]['latitude']
        lon = res[0]['longitude']
        self.assertTrue(np.all(lat.data == lats_ref))
        self.assertTrue(np.all(lon.data == lons_ref))
        self.assertDictContainsSubset({'name': 'latitude', 'standard_name': 'latitude', 'units': 'degrees_north'},
                                      lat.attrs)
        self.assertDictContainsSubset({'name': 'longitude', 'standard_name': 'longitude', 'units': 'degrees_east'},
                                      lon.attrs)


def suite():
    """Test suite for this writer's tests."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCFWriter))
    return mysuite


if __name__ == "__main__":
    unittest.main()
