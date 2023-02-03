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

import logging
import os
import tempfile
import unittest
from collections import OrderedDict
from datetime import datetime
from unittest import mock

import numpy as np

from satpy.tests.utils import make_dsq

try:
    from pyproj import CRS
except ImportError:
    CRS = None

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - caplog


class TempFile(object):
    """A temporary filename class."""

    def __init__(self, suffix=".nc"):
        """Initialize."""
        self.filename = None
        self.suffix = suffix

    def __enter__(self):
        """Enter."""
        self.handle, self.filename = tempfile.mkstemp(suffix=self.suffix)
        os.close(self.handle)
        return self.filename

    def __exit__(self, *args):
        """Exit."""
        os.remove(self.filename)


class TestCFWriter(unittest.TestCase):
    """Test case for CF writer."""

    def test_init(self):
        """Test initializing the CFWriter class."""
        from satpy.writers import configs_for_writer
        from satpy.writers.cf_writer import CFWriter
        CFWriter(config_files=list(configs_for_writer('cf'))[0])

    def test_save_array(self):
        """Test saving an array to netcdf/cf."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[make_dsq(name='hej')]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['test-array'][:], [1, 2, 3])
                expected_prereq = ("DataQuery(name='hej')")
                self.assertEqual(f['test-array'].attrs['prerequisites'],
                                 expected_prereq)

    def test_save_with_compression(self):
        """Test saving an array with compression."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        with mock.patch('satpy.writers.cf_writer.xr.Dataset') as xrdataset,\
                mock.patch('satpy.writers.cf_writer.make_time_bounds'):
            scn['test-array'] = xr.DataArray([1, 2, 3],
                                             attrs=dict(start_time=start_time,
                                                        end_time=end_time,
                                                        prerequisites=[make_dsq(name='hej')]))

            comp = {'zlib': True, 'complevel': 9}
            scn.save_datasets(filename='bla', writer='cf', compression=comp)
            ars, kws = xrdataset.call_args_list[1]
            self.assertDictEqual(ars[0]['test-array'].encoding, comp)

    def test_save_array_coords(self):
        """Test saving array with coordinates."""
        import numpy as np
        import xarray as xr

        from satpy import Scene
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
                                                    prerequisites=[make_dsq(name='hej')]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['test-array'][:], [[1, 2, 3]])
                np.testing.assert_array_equal(f['x'][:], [0, 1, 2])
                np.testing.assert_array_equal(f['y'][:], [0])
                self.assertNotIn('crs', f)
                self.assertNotIn('_FillValue', f['x'].attrs)
                self.assertNotIn('_FillValue', f['y'].attrs)
                expected_prereq = ("DataQuery(name='hej')")
                self.assertEqual(f['test-array'].attrs['prerequisites'],
                                 expected_prereq)

    def test_save_dataset_a_digit(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['CHANNEL_1'][:], [1, 2, 3])

    def test_save_dataset_a_digit_prefix(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', numeric_name_prefix='TEST')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['TEST1'][:], [1, 2, 3])

    def test_save_dataset_a_digit_prefix_include_attr(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix include orig name."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', include_orig_name=True, numeric_name_prefix='TEST')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['TEST1'][:], [1, 2, 3])
                self.assertEqual(f['TEST1'].attrs['original_name'], '1')

    def test_save_dataset_a_digit_no_prefix_include_attr(self):
        """Test saving an array to netcdf/cf dataset name starting with a digit with no prefix include orig name."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', include_orig_name=True, numeric_name_prefix='')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['1'][:], [1, 2, 3])
                self.assertNotIn('original_name', f['1'].attrs)

    def test_ancillary_variables(self):
        """Test ancillary_variables cited each other."""
        import xarray as xr

        from satpy import Scene
        from satpy.tests.utils import make_dataid
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        da = xr.DataArray([1, 2, 3],
                          attrs=dict(start_time=start_time,
                          end_time=end_time,
                          prerequisites=[make_dataid(name='hej')]))
        scn['test-array-1'] = da
        scn['test-array-2'] = da.copy()
        scn['test-array-1'].attrs['ancillary_variables'] = [scn['test-array-2']]
        scn['test-array-2'].attrs['ancillary_variables'] = [scn['test-array-1']]
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                self.assertEqual(f['test-array-1'].attrs['ancillary_variables'],
                                 'test-array-2')
                self.assertEqual(f['test-array-2'].attrs['ancillary_variables'],
                                 'test-array-1')

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
                np.testing.assert_array_equal(tst.data, ref.data)
            nc_root.close()
            nc_visir.close()
            nc_hrv.close()

        # Different projection coordinates in one group are not supported
        with TempFile() as filename:
            self.assertRaises(ValueError, scn.save_datasets, datasets=['VIS006', 'HRV'], filename=filename, writer='cf')

    def test_single_time_value(self):
        """Test setting a single time value."""
        import xarray as xr

        from satpy import Scene
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
            with xr.open_dataset(filename, decode_cf=True) as f:
                np.testing.assert_array_equal(f['time'], scn['test-array']['time'])
                bounds_exp = np.array([[start_time, end_time]], dtype='datetime64[m]')
                np.testing.assert_array_equal(f['time_bnds'], bounds_exp)

    def test_time_coordinate_on_a_swath(self):
        """Test that time dimension is not added on swath data with time already as a coordinate."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        test_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        times = np.array(['2018-05-30T10:05:00', '2018-05-30T10:05:01',
                          '2018-05-30T10:05:02', '2018-05-30T10:05:03'], dtype=np.datetime64)
        scn['test-array'] = xr.DataArray(test_array,
                                         dims=['y', 'x'],
                                         coords={'time': ('y', times)},
                                         attrs=dict(start_time=times[0], end_time=times[-1]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', pretty=True)
            with xr.open_dataset(filename, decode_cf=True) as f:
                np.testing.assert_array_equal(f['time'], scn['test-array']['time'])

    def test_bounds(self):
        """Test setting time bounds."""
        import xarray as xr

        from satpy import Scene
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
            # Check decoded time coordinates & bounds
            with xr.open_dataset(filename, decode_cf=True) as f:
                bounds_exp = np.array([[start_time, end_time]], dtype='datetime64[m]')
                np.testing.assert_array_equal(f['time_bnds'], bounds_exp)
                self.assertEqual(f['time'].attrs['bounds'], 'time_bnds')

            # Check raw time coordinates & bounds
            with xr.open_dataset(filename, decode_cf=False) as f:
                np.testing.assert_almost_equal(f['time_bnds'], [[-0.0034722, 0.0069444]])

        # User-specified time encoding should have preference
        with TempFile() as filename:
            time_units = 'seconds since 2018-01-01'
            scn.save_datasets(filename=filename, encoding={'time': {'units': time_units}},
                              writer='cf')
            with xr.open_dataset(filename, decode_cf=False) as f:
                np.testing.assert_array_equal(f['time_bnds'], [[12909600, 12910500]])

    def test_bounds_minimum(self):
        """Test minimum bounds."""
        import xarray as xr

        from satpy import Scene
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
            with xr.open_dataset(filename, decode_cf=True) as f:
                bounds_exp = np.array([[start_timeA, end_timeB]], dtype='datetime64[m]')
                np.testing.assert_array_equal(f['time_bnds'], bounds_exp)

    def test_bounds_missing_time_info(self):
        """Test time bounds generation in case of missing time."""
        import xarray as xr

        from satpy import Scene
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
            with xr.open_dataset(filename, decode_cf=True) as f:
                bounds_exp = np.array([[start_timeA, end_timeA]], dtype='datetime64[m]')
                np.testing.assert_array_equal(f['time_bnds'], bounds_exp)

    def test_encoding_kwarg(self):
        """Test 'encoding' keyword argument."""
        import xarray as xr

        from satpy import Scene
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
                np.testing.assert_array_equal(f['test-array'][:], [10, 20, 30])
                self.assertEqual(f['test-array'].attrs['scale_factor'], 0.1)
                self.assertEqual(f['test-array'].attrs['_FillValue'], 3)
                # check that dtype behave as int8
                self.assertEqual(np.iinfo(f['test-array'][:].dtype).max, 127)

    def test_unlimited_dims_kwarg(self):
        """Test specification of unlimited dimensions."""
        import xarray as xr

        from satpy import Scene
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
            scn.save_datasets(filename=filename, writer='cf', unlimited_dims=['time'])
            with xr.open_dataset(filename) as f:
                self.assertSetEqual(f.encoding['unlimited_dims'], {'time'})

    def test_header_attrs(self):
        """Check global attributes are set."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        with TempFile() as filename:
            header_attrs = {'sensor': 'SEVIRI',
                            'orbit': 99999,
                            'none': None,
                            'list': [1, 2, 3],
                            'set': {1, 2, 3},
                            'dict': {'a': 1, 'b': 2},
                            'nested': {'outer': {'inner1': 1, 'inner2': 2}},
                            'bool': True,
                            'bool_': np.bool_(True)}
            scn.save_datasets(filename=filename,
                              header_attrs=header_attrs,
                              flatten_attrs=True,
                              writer='cf')
            with xr.open_dataset(filename) as f:
                self.assertIn('history', f.attrs)
                self.assertEqual(f.attrs['sensor'], 'SEVIRI')
                self.assertEqual(f.attrs['orbit'], 99999)
                np.testing.assert_array_equal(f.attrs['list'], [1, 2, 3])
                self.assertEqual(f.attrs['set'], '{1, 2, 3}')
                self.assertEqual(f.attrs['dict_a'], 1)
                self.assertEqual(f.attrs['dict_b'], 2)
                self.assertEqual(f.attrs['nested_outer_inner1'], 1)
                self.assertEqual(f.attrs['nested_outer_inner2'], 2)
                self.assertEqual(f.attrs['bool'], 'true')
                self.assertEqual(f.attrs['bool_'], 'true')
                self.assertTrue('none' not in f.attrs.keys())

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
        self.assertSetEqual(set(d1.keys()), set(d2.keys()))
        for key, val1 in d1.items():
            val2 = d2[key]
            if isinstance(val1, np.ndarray):
                np.testing.assert_array_equal(val1, val2)
                self.assertEqual(val1.dtype, val2.dtype)
            else:
                self.assertEqual(val1, val2)
                if isinstance(val1, (np.floating, np.integer, np.bool_)):
                    self.assertTrue(isinstance(val2, np.generic))
                    self.assertEqual(val1.dtype, val2.dtype)

    def test_encode_attrs_nc(self):
        """Test attributes encoding."""
        import json

        from satpy.writers.cf_writer import encode_attrs_nc

        attrs, expected, _ = self.get_test_attrs()

        # Test encoding
        encoded = encode_attrs_nc(attrs)
        self.assertDictWithArraysEqual(expected, encoded)

        # Test decoding of json-encoded attributes
        raw_md_roundtrip = {'recarray': [[0, 0], [0, 0], [0, 0]],
                            'flag': 'true',
                            'dict': {'a': 1, 'b': [1, 2, 3]}}
        self.assertDictEqual(json.loads(encoded['raw_metadata']), raw_md_roundtrip)
        self.assertListEqual(json.loads(encoded['array_3d']), [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        self.assertDictEqual(json.loads(encoded['nested_dict']), {"l1": {"l2": {"l3": [1, 2, 3]}}})
        self.assertListEqual(json.loads(encoded['nested_list']), ["1", ["2", [3]]])

    def test_da2cf(self):
        """Test the conversion of a DataArray to a CF-compatible DataArray."""
        import xarray as xr

        from satpy.writers.cf_writer import CFWriter

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
        res = CFWriter.da2cf(arr)
        np.testing.assert_array_equal(res['x'], arr['x'])
        np.testing.assert_array_equal(res['y'], arr['y'])
        np.testing.assert_array_equal(res['acq_time'], arr['acq_time'])
        self.assertDictEqual(res['x'].attrs, {'units': 'm', 'standard_name': 'projection_x_coordinate'})
        self.assertDictEqual(res['y'].attrs, {'units': 'm', 'standard_name': 'projection_y_coordinate'})
        self.assertDictWithArraysEqual(res.attrs, attrs_expected)

        # Test attribute kwargs
        res_flat = CFWriter.da2cf(arr, flatten_attrs=True, exclude_attrs=['int'])
        attrs_expected_flat.pop('int')
        self.assertDictWithArraysEqual(res_flat.attrs, attrs_expected_flat)

    @mock.patch('satpy.writers.cf_writer.CFWriter.__init__', return_value=None)
    def test_collect_datasets(self, *mocks):
        """Test collecting CF datasets from a DataArray objects."""
        import pyresample.geometry
        import xarray as xr

        from satpy.writers.cf_writer import CFWriter
        geos = pyresample.geometry.AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': 35785831., 'a': 6378169., 'b': 6356583.8},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])

        # Define test datasets
        data = [[1, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        time = [1, 2]
        tstart = datetime(2019, 4, 1, 12, 0)
        tend = datetime(2019, 4, 1, 12, 15)
        datasets = [xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                 attrs={'name': 'var1', 'start_time': tstart, 'end_time': tend, 'area': geos}),
                    xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                 attrs={'name': 'var2', 'long_name': 'variable 2'})]

        # Collect datasets
        writer = CFWriter()
        datas, start_times, end_times = writer._collect_datasets(datasets, include_lonlats=True)

        # Test results
        self.assertEqual(len(datas), 3)
        self.assertEqual(set(datas.keys()), {'var1', 'var2', 'geos'})
        self.assertListEqual(start_times, [None, tstart, None])
        self.assertListEqual(end_times, [None, tend, None])
        var1 = datas['var1']
        var2 = datas['var2']
        self.assertEqual(var1.name, 'var1')
        self.assertEqual(var1.attrs['grid_mapping'], 'geos')
        self.assertEqual(var1.attrs['start_time'], '2019-04-01 12:00:00')
        self.assertEqual(var1.attrs['end_time'], '2019-04-01 12:15:00')
        self.assertEqual(var1.attrs['long_name'], 'var1')
        # variable 2
        self.assertNotIn('grid_mapping', var2.attrs)
        self.assertEqual(var2.attrs['long_name'], 'variable 2')

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
        import numpy as np
        import xarray as xr

        from satpy.writers.cf_writer import link_coords

        data = [[1, 2], [3, 4]]
        lon = np.zeros((2, 2))
        lon2 = np.zeros((1, 2, 2))
        lat = np.ones((2, 2))
        datasets = {
            'var1': xr.DataArray(data=data, dims=('y', 'x'), attrs={'coordinates': 'lon lat'}),
            'var2': xr.DataArray(data=data, dims=('y', 'x')),
            'var3': xr.DataArray(data=data, dims=('y', 'x'), attrs={'coordinates': 'lon2 lat'}),
            'var4': xr.DataArray(data=data, dims=('y', 'x'), attrs={'coordinates': 'not_exist lon lat'}),
            'lon': xr.DataArray(data=lon, dims=('y', 'x')),
            'lon2': xr.DataArray(data=lon2, dims=('time', 'y', 'x')),
            'lat': xr.DataArray(data=lat, dims=('y', 'x'))
        }

        link_coords(datasets)

        # Check that link has been established correctly and 'coordinate' atrribute has been dropped
        self.assertIn('lon', datasets['var1'].coords)
        self.assertIn('lat', datasets['var1'].coords)
        np.testing.assert_array_equal(datasets['var1']['lon'].data, lon)
        np.testing.assert_array_equal(datasets['var1']['lat'].data, lat)
        self.assertNotIn('coordinates', datasets['var1'].attrs)

        # There should be no link if there was no 'coordinate' attribute
        self.assertNotIn('lon', datasets['var2'].coords)
        self.assertNotIn('lat', datasets['var2'].coords)

        # The non-existant dimension or coordinate should be dropped
        self.assertNotIn('time', datasets['var3'].coords)
        self.assertNotIn('not_exist', datasets['var4'].coords)

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
        np.testing.assert_array_equal(res['var1']['var1_acq_time'], time1)
        np.testing.assert_array_equal(res['var2']['var2_acq_time'], time2)
        self.assertNotIn('acq_time', res['var1'].coords)
        self.assertNotIn('acq_time', res['var2'].coords)

        # Make sure nothing else is modified
        np.testing.assert_array_equal(res['var1']['x'], x)
        np.testing.assert_array_equal(res['var1']['y'], y)
        np.testing.assert_array_equal(res['var2']['x'], x)
        np.testing.assert_array_equal(res['var2']['y'], y)

        # Coords not unique -> Dataset names must be prepended, even if pretty=True
        with mock.patch('satpy.writers.cf_writer.warnings.warn') as warn:
            res = make_alt_coords_unique(datasets, pretty=True)
            warn.assert_called()
            np.testing.assert_array_equal(res['var1']['var1_acq_time'], time1)
            np.testing.assert_array_equal(res['var2']['var2_acq_time'], time2)
            self.assertNotIn('acq_time', res['var1'].coords)
            self.assertNotIn('acq_time', res['var2'].coords)

        # Coords unique and pretty=True -> Don't modify coordinate names
        datasets['var2']['acq_time'] = ('y', time1)
        res = make_alt_coords_unique(datasets, pretty=True)
        np.testing.assert_array_equal(res['var1']['acq_time'], time1)
        np.testing.assert_array_equal(res['var2']['acq_time'], time1)
        self.assertNotIn('var1_acq_time', res['var1'].coords)
        self.assertNotIn('var2_acq_time', res['var2'].coords)

    def test_area2cf(self):
        """Test the conversion of an area to CF standards."""
        import pyresample.geometry
        import xarray as xr

        from satpy.writers.cf_writer import area2cf

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
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].size, 1)  # grid mapping variable
        self.assertEqual(res[0].name, res[1].attrs['grid_mapping'])

        # b) Area Definition and strict=False
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = geos
        res = area2cf(ds, strict=True)
        # same as above
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].size, 1)  # grid mapping variable
        self.assertEqual(res[0].name, res[1].attrs['grid_mapping'])
        # but now also have the lon/lats
        self.assertIn('longitude', res[1].coords)
        self.assertIn('latitude', res[1].coords)

        # c) Swath Definition
        swath = pyresample.geometry.SwathDefinition(lons=[[1, 1], [2, 2]], lats=[[1, 2], [1, 2]])
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = swath

        res = area2cf(ds)
        self.assertEqual(len(res), 1)
        self.assertIn('longitude', res[0].coords)
        self.assertIn('latitude', res[0].coords)
        self.assertNotIn('grid_mapping', res[0].attrs)

    def test_area2gridmapping(self):
        """Test the conversion from pyresample area object to CF grid mapping."""
        import pyresample.geometry
        import xarray as xr

        from satpy.writers.cf_writer import area2gridmapping

        def _gm_matches(gmapping, expected):
            """Assert that all keys in ``expected`` match the values in ``gmapping``."""
            for attr_key, attr_val in expected.attrs.items():
                test_val = gmapping.attrs[attr_key]
                if attr_val is None or isinstance(attr_val, str):
                    self.assertEqual(test_val, attr_val)
                else:
                    np.testing.assert_almost_equal(test_val, attr_val, decimal=3)

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
            projection={'proj': 'geos', 'h': h, 'a': a, 'b': b,
                        'lat_0': 0, 'lon_0': 0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        geos_expected = xr.DataArray(data=0,
                                     attrs={'perspective_point_height': h,
                                            'latitude_of_projection_origin': 0,
                                            'longitude_of_projection_origin': 0,
                                            'grid_mapping_name': 'geostationary',
                                            'semi_major_axis': a,
                                            'semi_minor_axis': b,
                                            # 'sweep_angle_axis': None,
                                            })

        ds = ds_base.copy()
        ds.attrs['area'] = geos
        new_ds, grid_mapping = area2gridmapping(ds)
        if 'sweep_angle_axis' in grid_mapping.attrs:
            # older versions of pyproj might not include this
            self.assertEqual(grid_mapping.attrs['sweep_angle_axis'], 'y')

        self.assertEqual(new_ds.attrs['grid_mapping'], 'geos')
        _gm_matches(grid_mapping, geos_expected)
        # should not have been modified
        self.assertNotIn('grid_mapping', ds.attrs)

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

        ds = ds_base.copy()
        ds.attrs['area'] = cosmo7

        new_ds, grid_mapping = area2gridmapping(ds)
        self.assertIn('crs_wkt', grid_mapping.attrs)
        wkt = grid_mapping.attrs['crs_wkt']
        self.assertIn('ELLIPSOID["WGS 84"', wkt)
        self.assertIn('PARAMETER["lat_0",46', wkt)
        self.assertIn('PARAMETER["lon_0",4.535', wkt)
        self.assertIn('PARAMETER["o_lat_p",90', wkt)
        self.assertIn('PARAMETER["o_lon_p",-5.465', wkt)
        self.assertEqual(new_ds.attrs['grid_mapping'], 'cosmo7')

        # c) Projection Transverse Mercator
        lat_0 = 36.5
        lon_0 = 15.0

        tmerc = pyresample.geometry.AreaDefinition(
            area_id='tmerc',
            description='tmerc',
            proj_id='tmerc',
            projection={'proj': 'tmerc', 'ellps': 'WGS84', 'lat_0': 36.5, 'lon_0': 15.0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])

        tmerc_expected = xr.DataArray(data=0,
                                      attrs={'latitude_of_projection_origin': lat_0,
                                             'longitude_of_central_meridian': lon_0,
                                             'grid_mapping_name': 'transverse_mercator',
                                             'reference_ellipsoid_name': 'WGS 84',
                                             'false_easting': 0.,
                                             'false_northing': 0.,
                                             })

        ds = ds_base.copy()
        ds.attrs['area'] = tmerc
        new_ds, grid_mapping = area2gridmapping(ds)
        self.assertEqual(new_ds.attrs['grid_mapping'], 'tmerc')
        _gm_matches(grid_mapping, tmerc_expected)

        # d) Projection that has a representation but no explicit a/b
        h = 35785831.
        geos = pyresample.geometry.AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': h, 'datum': 'WGS84', 'ellps': 'GRS80',
                        'lat_0': 0, 'lon_0': 0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        geos_expected = xr.DataArray(data=0,
                                     attrs={'perspective_point_height': h,
                                            'latitude_of_projection_origin': 0,
                                            'longitude_of_projection_origin': 0,
                                            'grid_mapping_name': 'geostationary',
                                            # 'semi_major_axis': 6378137.0,
                                            # 'semi_minor_axis': 6356752.314,
                                            # 'sweep_angle_axis': None,
                                            })

        ds = ds_base.copy()
        ds.attrs['area'] = geos
        new_ds, grid_mapping = area2gridmapping(ds)

        self.assertEqual(new_ds.attrs['grid_mapping'], 'geos')
        _gm_matches(grid_mapping, geos_expected)

        # e) oblique Mercator
        area = pyresample.geometry.AreaDefinition(
            area_id='omerc_otf',
            description='On-the-fly omerc area',
            proj_id='omerc',
            projection={'alpha': '9.02638777018478', 'ellps': 'WGS84', 'gamma': '0', 'k': '1',
                        'lat_0': '-0.256794486098476', 'lonc': '13.7888658224205',
                        'proj': 'omerc', 'units': 'm'},
            width=2837,
            height=5940,
            area_extent=[-1460463.0893, 3455291.3877, 1538407.1158, 9615788.8787]
        )

        omerc_dict = {'azimuth_of_central_line': 9.02638777018478,
                      'false_easting': 0.,
                      'false_northing': 0.,
                      # 'gamma': 0,  # this is not CF compliant
                      'grid_mapping_name': "oblique_mercator",
                      'latitude_of_projection_origin': -0.256794486098476,
                      'longitude_of_projection_origin': 13.7888658224205,
                      # 'prime_meridian_name': "Greenwich",
                      'reference_ellipsoid_name': "WGS 84"}
        omerc_expected = xr.DataArray(data=0, attrs=omerc_dict)

        ds = ds_base.copy()
        ds.attrs['area'] = area
        new_ds, grid_mapping = area2gridmapping(ds)

        self.assertEqual(new_ds.attrs['grid_mapping'], 'omerc_otf')
        _gm_matches(grid_mapping, omerc_expected)

        # f) Projection that has a representation but no explicit a/b
        h = 35785831.
        geos = pyresample.geometry.AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': h, 'datum': 'WGS84', 'ellps': 'GRS80',
                        'lat_0': 0, 'lon_0': 0},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        geos_expected = xr.DataArray(data=0,
                                     attrs={'perspective_point_height': h,
                                            'latitude_of_projection_origin': 0,
                                            'longitude_of_projection_origin': 0,
                                            'grid_mapping_name': 'geostationary',
                                            'reference_ellipsoid_name': 'WGS 84',
                                            })

        ds = ds_base.copy()
        ds.attrs['area'] = geos
        new_ds, grid_mapping = area2gridmapping(ds)

        self.assertEqual(new_ds.attrs['grid_mapping'], 'geos')
        _gm_matches(grid_mapping, geos_expected)

    def test_area2lonlat(self):
        """Test the conversion from areas to lon/lat."""
        import dask.array as da
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

        # original should be unmodified
        self.assertNotIn('longitude', dataarray.coords)
        self.assertEqual(set(res.coords), {'longitude', 'latitude'})
        lat = res['latitude']
        lon = res['longitude']
        np.testing.assert_array_equal(lat.data, lats_ref)
        np.testing.assert_array_equal(lon.data, lons_ref)
        assert {'name': 'latitude', 'standard_name': 'latitude', 'units': 'degrees_north'}.items() <= lat.attrs.items()
        assert {'name': 'longitude', 'standard_name': 'longitude', 'units': 'degrees_east'}.items() <= lon.attrs.items()

        area = pyresample.geometry.AreaDefinition(
            'seviri',
            'Native SEVIRI grid',
            'geos',
            "+a=6378169.0 +h=35785831.0 +b=6356583.8 +lon_0=0 +proj=geos",
            10, 10,
            [-5570248.686685662, -5567248.28340708, 5567248.28340708, 5570248.686685662]
        )
        lons_ref, lats_ref = area.get_lonlats()
        dataarray = xr.DataArray(data=da.from_array(np.arange(3*10*10).reshape(3, 10, 10), chunks=(1, 5, 5)),
                                 dims=('bands', 'y', 'x'), attrs={'area': area})
        res = area2lonlat(dataarray)

        # original should be unmodified
        self.assertNotIn('longitude', dataarray.coords)
        self.assertEqual(set(res.coords), {'longitude', 'latitude'})
        lat = res['latitude']
        lon = res['longitude']
        np.testing.assert_array_equal(lat.data, lats_ref)
        np.testing.assert_array_equal(lon.data, lons_ref)
        assert {'name': 'latitude', 'standard_name': 'latitude', 'units': 'degrees_north'}.items() <= lat.attrs.items()
        assert {'name': 'longitude', 'standard_name': 'longitude', 'units': 'degrees_east'}.items() <= lon.attrs.items()

    def test_load_module_with_old_pyproj(self):
        """Test that cf_writer can still be loaded with pyproj 1.9.6."""
        import importlib
        import sys

        import pyproj  # noqa 401
        old_version = sys.modules['pyproj'].__version__
        sys.modules['pyproj'].__version__ = "1.9.6"
        try:
            importlib.reload(sys.modules['satpy.writers.cf_writer'])
        finally:
            # Tear down
            sys.modules['pyproj'].__version__ = old_version
            importlib.reload(sys.modules['satpy.writers.cf_writer'])

    def test_global_attr_default_history_and_Conventions(self):
        """Test saving global attributes history and Conventions."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([[1, 2, 3]],
                                         dims=('y', 'x'),
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[make_dsq(name='hej')]))
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                self.assertEqual(f.attrs['Conventions'], 'CF-1.7')
                self.assertIn('Created by pytroll/satpy on', f.attrs['history'])

    def test_global_attr_history_and_Conventions(self):
        """Test saving global attributes history and Conventions."""
        import xarray as xr

        from satpy import Scene
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([[1, 2, 3]],
                                         dims=('y', 'x'),
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[make_dsq(name='hej')]))
        header_attrs = {}
        header_attrs['history'] = ('TEST add history',)
        header_attrs['Conventions'] = 'CF-1.7, ACDD-1.3'
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', header_attrs=header_attrs)
            with xr.open_dataset(filename) as f:
                self.assertEqual(f.attrs['Conventions'], 'CF-1.7, ACDD-1.3')
                self.assertIn('TEST add history\n', f.attrs['history'])
                self.assertIn('Created by pytroll/satpy on', f.attrs['history'])


def test_lonlat_storage(tmp_path):
    """Test correct storage for area with lon/lat units."""
    import xarray as xr
    from pyresample import create_area_def

    from ..utils import make_fake_scene
    scn = make_fake_scene(
            {"ketolysis": np.arange(25).reshape(5, 5)},
            daskify=True,
            area=create_area_def("mavas", 4326, shape=(5, 5),
                                 center=(0, 0), resolution=(1, 1)))

    filename = os.fspath(tmp_path / "test.nc")
    scn.save_datasets(filename=filename, writer="cf", include_lonlats=False)
    with xr.open_dataset(filename) as ds:
        assert ds["ketolysis"].attrs["grid_mapping"] == "mavas"
        assert ds["mavas"].attrs["grid_mapping_name"] == "latitude_longitude"
        assert ds["x"].attrs["units"] == "degrees_east"
        assert ds["y"].attrs["units"] == "degrees_north"
        assert ds["mavas"].attrs["longitude_of_prime_meridian"] == 0.0
        np.testing.assert_allclose(ds["mavas"].attrs["semi_major_axis"], 6378137.0)
        np.testing.assert_allclose(ds["mavas"].attrs["inverse_flattening"], 298.257223563)


def test_da2cf_lonlat():
    """Test correct da2cf encoding for area with lon/lat units."""
    import xarray as xr
    from pyresample import create_area_def

    from satpy.resample import add_crs_xy_coords
    from satpy.writers.cf_writer import CFWriter

    area = create_area_def("mavas", 4326, shape=(5, 5),
                           center=(0, 0), resolution=(1, 1))
    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"),
        attrs={"area": area})
    da = add_crs_xy_coords(da, area)
    new_da = CFWriter.da2cf(da)
    assert new_da["x"].attrs["units"] == "degrees_east"
    assert new_da["y"].attrs["units"] == "degrees_north"


def test_is_projected(caplog):
    """Tests for private _is_projected function."""
    import xarray as xr

    from satpy.writers.cf_writer import CFWriter

    # test case with units but no area
    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"),
        coords={"x": xr.DataArray(np.arange(5), dims=("x",), attrs={"units": "m"}),
                "y": xr.DataArray(np.arange(5), dims=("y",), attrs={"units": "m"})})
    assert CFWriter._is_projected(da)

    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"),
        coords={"x": xr.DataArray(np.arange(5), dims=("x",), attrs={"units": "degrees_east"}),
                "y": xr.DataArray(np.arange(5), dims=("y",), attrs={"units": "degrees_north"})})
    assert not CFWriter._is_projected(da)

    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"))
    with caplog.at_level(logging.WARNING):
        assert CFWriter._is_projected(da)
    assert "Failed to tell if data are projected." in caplog.text


class TestCFWriterData(unittest.TestCase):
    """Test case for CF writer where data arrays are needed."""

    def setUp(self):
        """Create some test data."""
        import pyresample.geometry
        import xarray as xr
        data = [[75, 2], [3, 4]]
        y = [1, 2]
        x = [1, 2]
        geos = pyresample.geometry.AreaDefinition(
            area_id='geos',
            description='geos',
            proj_id='geos',
            projection={'proj': 'geos', 'h': 35785831., 'a': 6378169., 'b': 6356583.8},
            width=2, height=2,
            area_extent=[-1, -1, 1, 1])
        self.datasets = {'var1': xr.DataArray(data=data,
                                              dims=('y', 'x'),
                                              coords={'y': y, 'x': x}),
                         'var2': xr.DataArray(data=data,
                                              dims=('y', 'x'),
                                              coords={'y': y, 'x': x}),
                         'lat': xr.DataArray(data=data,
                                             dims=('y', 'x'),
                                             coords={'y': y, 'x': x}),
                         'lon': xr.DataArray(data=data,
                                             dims=('y', 'x'),
                                             coords={'y': y, 'x': x})}
        self.datasets['lat'].attrs['standard_name'] = 'latitude'
        self.datasets['var1'].attrs['standard_name'] = 'dummy'
        self.datasets['var2'].attrs['standard_name'] = 'dummy'
        self.datasets['var2'].attrs['area'] = geos
        self.datasets['var1'].attrs['area'] = geos
        self.datasets['lat'].attrs['name'] = 'lat'
        self.datasets['var1'].attrs['name'] = 'var1'
        self.datasets['var2'].attrs['name'] = 'var2'
        self.datasets['lon'].attrs['name'] = 'lon'

    def test_dataset_is_projection_coords(self):
        """Test the dataset_is_projection_coords function."""
        from satpy.writers.cf_writer import dataset_is_projection_coords
        self.assertTrue(dataset_is_projection_coords(self.datasets['lat']))
        self.assertFalse(dataset_is_projection_coords(self.datasets['var1']))

    def test_has_projection_coords(self):
        """Test the has_projection_coords function."""
        from satpy.writers.cf_writer import has_projection_coords
        self.assertTrue(has_projection_coords(self.datasets))
        self.datasets['lat'].attrs['standard_name'] = 'dummy'
        self.assertFalse(has_projection_coords(self.datasets))

    @mock.patch('satpy.writers.cf_writer.CFWriter.__init__', return_value=None)
    def test_collect_datasets_with_latitude_named_lat(self, *mocks):
        """Test collecting CF datasets with latitude named lat."""
        from operator import getitem

        from satpy.writers.cf_writer import CFWriter
        self.datasets_list = [self.datasets[key] for key in self.datasets]
        self.datasets_list_no_latlon = [self.datasets[key] for key in ['var1', 'var2']]

        # Collect datasets
        writer = CFWriter()
        datas, start_times, end_times = writer._collect_datasets(self.datasets_list, include_lonlats=True)
        datas2, start_times, end_times = writer._collect_datasets(self.datasets_list_no_latlon, include_lonlats=True)
        # Test results

        self.assertEqual(len(datas), 5)
        self.assertEqual(set(datas.keys()), {'var1', 'var2', 'lon', 'lat', 'geos'})
        self.assertRaises(KeyError, getitem, datas['var1'], 'latitude')
        self.assertRaises(KeyError, getitem, datas['var1'], 'longitude')
        self.assertEqual(datas2['var1']['latitude'].attrs['name'], 'latitude')
        self.assertEqual(datas2['var1']['longitude'].attrs['name'], 'longitude')


class EncodingUpdateTest(unittest.TestCase):
    """Test update of netCDF encoding."""

    def setUp(self):
        """Create fake data for testing."""
        import xarray as xr
        self.ds = xr.Dataset({'foo': (('y', 'x'), [[1, 2], [3, 4]]),
                              'bar': (('y', 'x'), [[3, 4], [5, 6]])},
                             coords={'y': [1, 2],
                                     'x': [3, 4],
                                     'lon': (('y', 'x'), [[7, 8], [9, 10]])})
        self.ds_digit = xr.Dataset({'CHANNEL_1': (('y', 'x'), [[1, 2], [3, 4]]),
                                    'CHANNEL_2': (('y', 'x'), [[3, 4], [5, 6]])},
                                   coords={'y': [1, 2],
                                           'x': [3, 4],
                                           'lon': (('y', 'x'), [[7, 8], [9, 10]])})

    def test_dataset_name_digit(self):
        """Test data with dataset name staring with a digit."""
        from satpy.writers.cf_writer import update_encoding

        # Dataset with name staring with digit
        ds = self.ds_digit
        kwargs = {'encoding': {'1': {'dtype': 'float32'},
                               '2': {'dtype': 'float32'}},
                  'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds, kwargs, numeric_name_prefix='CHANNEL_')
        self.assertDictEqual(enc, {'y': {'_FillValue': None},
                                   'x': {'_FillValue': None},
                                   'CHANNEL_1': {'dtype': 'float32'},
                                   'CHANNEL_2': {'dtype': 'float32'}})
        self.assertDictEqual(other_kwargs, {'other': 'kwargs'})

    def test_without_time(self):
        """Test data with no time dimension."""
        from satpy.writers.cf_writer import update_encoding

        # Without time dimension
        ds = self.ds.chunk(2)
        kwargs = {'encoding': {'bar': {'chunksizes': (1, 1)}},
                  'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds, kwargs)
        self.assertDictEqual(enc, {'y': {'_FillValue': None},
                                   'x': {'_FillValue': None},
                                   'lon': {'chunksizes': (2, 2)},
                                   'foo': {'chunksizes': (2, 2)},
                                   'bar': {'chunksizes': (1, 1)}})
        self.assertDictEqual(other_kwargs, {'other': 'kwargs'})

        # Chunksize may not exceed shape
        ds = self.ds.chunk(8)
        kwargs = {'encoding': {}, 'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds, kwargs)
        self.assertDictEqual(enc, {'y': {'_FillValue': None},
                                   'x': {'_FillValue': None},
                                   'lon': {'chunksizes': (2, 2)},
                                   'foo': {'chunksizes': (2, 2)},
                                   'bar': {'chunksizes': (2, 2)}})

    def test_with_time(self):
        """Test data with a time dimension."""
        from satpy.writers.cf_writer import update_encoding

        # With time dimension
        ds = self.ds.chunk(8).expand_dims({'time': [datetime(2009, 7, 1, 12, 15)]})
        kwargs = {'encoding': {'bar': {'chunksizes': (1, 1, 1)}},
                  'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds, kwargs)
        self.assertDictEqual(enc, {'y': {'_FillValue': None},
                                   'x': {'_FillValue': None},
                                   'lon': {'chunksizes': (2, 2)},
                                   'foo': {'chunksizes': (1, 2, 2)},
                                   'bar': {'chunksizes': (1, 1, 1)},
                                   'time': {'_FillValue': None,
                                            'calendar': 'proleptic_gregorian',
                                            'units': 'days since 2009-07-01 12:15:00'},
                                   'time_bnds': {'_FillValue': None,
                                                 'calendar': 'proleptic_gregorian',
                                                 'units': 'days since 2009-07-01 12:15:00'}})

        # User-defined encoding may not be altered
        self.assertDictEqual(kwargs['encoding'], {'bar': {'chunksizes': (1, 1, 1)}})
