#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2017 David Hoese
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for the CF writer.
"""
from collections import OrderedDict
import os
import sys
from datetime import datetime
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


class TestCFWriter(unittest.TestCase):
    def test_init(self):
        from satpy.writers.cf_writer import CFWriter
        import satpy.config
        CFWriter(config_files=[os.path.join(satpy.config.CONFIG_PATH,
                                            'writers', 'cf.yaml')])

    def test_save_array(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time,
                                                    prerequisites=[DatasetID('hej')]))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['test-array'][:] == [1, 2, 3]))
                expected_prereq = ("DatasetID(name='hej', wavelength=None, "
                                   "resolution=None, polarization=None, "
                                   "calibration=None, level=None, modifiers=())")
                self.assertEqual(f['test-array'].attrs['prerequisites'][0],
                                 expected_prereq)
        finally:
            os.remove(filename)

    def test_single_time_value(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]])
        scn['test-array'] = xr.DataArray(test_array,
                                         dims=['x', 'y'],
                                         coords={'time': np.datetime64('2018-05-30T10:05:00')},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_bounds(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        test_array = np.array([[1, 2], [3, 4]]).reshape(2, 2, 1)
        scn['test-array'] = xr.DataArray(test_array,
                                         dims=['x', 'y', 'time'],
                                         coords={'time': [np.datetime64('2018-05-30T10:05:00')]},
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_bounds_minimum(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
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
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_bounds_missing_time_info(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
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
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            scn.save_datasets(filename=filename, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['time_bnds'][:] == np.array([-300.,  600.])))
        finally:
            os.remove(filename)

    def test_encoding_kwarg(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            encoding = {'test-array': {'dtype': 'int8',
                                       'scale_factor': 0.1,
                                       'add_offset': 0.0,
                                       '_FillValue': 3}}
            scn.save_datasets(filename=filename, encoding=encoding, writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(all(f['test-array'][:] == [10, 20, 30]))
                self.assertTrue(f['test-array'].attrs['scale_factor'] == 0.1)
                self.assertTrue(f['test-array'].attrs['_FillValue'] == 3)
                # check that dtype behave as int8
                self.assertTrue(np.iinfo(f['test-array'][:].dtype).max == 127)
        finally:
            os.remove(filename)

    def test_header_attrs(self):
        from satpy import Scene
        import xarray as xr
        import tempfile
        scn = Scene()
        start_time = datetime(2018, 5, 30, 10, 0)
        end_time = datetime(2018, 5, 30, 10, 15)
        scn['test-array'] = xr.DataArray([1, 2, 3],
                                         attrs=dict(start_time=start_time,
                                                    end_time=end_time))
        try:
            handle, filename = tempfile.mkstemp()
            os.close(handle)
            header_attrs = {'sensor': 'SEVIRI',
                            'orbit': None}
            scn.save_datasets(filename=filename,
                              header_attrs=header_attrs,
                              writer='cf')
            import h5netcdf as nc4
            with nc4.File(filename) as f:
                self.assertTrue(f.attrs['sensor'] == 'SEVIRI')
                self.assertTrue('sensor' in f.attrs.keys())
                self.assertTrue('orbit' not in f.attrs.keys())
        finally:
            os.remove(filename)

    def get_test_attrs(self):
        """Create some dataset attributes for testing purpose

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
                 'array': np.array([1, 2, 3]),
                 'array_bool': np.array([True, False, True]),
                 'array_2d': np.array([[1, 2], [3, 4]]),
                 'array_3d': np.array([[[1, 2], [3, 4]], [[1, 2], [3, 4]]]),
                 'dict': {'a': 1, 'b': 2},
                 'nested_dict': {'l1': {'l2': {'l3': np.array([1, 2, 3])}}},
                 'raw_metadata': OrderedDict([
                      ('recarray', np.zeros(3, dtype=[('x', 'i4'), ('y', 'u1')])),
                      ('flag', np.bool_(True)),
                      ('dict', OrderedDict([('a', 1), ('b', np.array([1, 2, 3]))]))
                 ])}
        encoded = {'name': 'IR_108',
                   'start_time': '2018-01-01 00:00:00',
                   'end_time': '2018-01-01 00:15:00',
                   'int': 1,
                   'float': 1.0,
                   'list': [1, 2, 3.0],
                   'nested_list': '["1", ["2", [3]]]',
                   'bool': 'True',
                   'array': [1, 2, 3],
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
                        'array': [1, 2, 3],
                        'array_bool': ['True', 'False', 'True'],
                        'array_2d': '[[1, 2], [3, 4]]',
                        'array_3d': '[[[1, 2], [3, 4]], [[1, 2], [3, 4]]]',
                        'dict_a': 1,
                        'dict_b': 2,
                        'nested_dict_l1_l2_l3': [1, 2, 3],
                        'raw_metadata_recarray': '[[0, 0], [0, 0], [0, 0]]',
                        'raw_metadata_flag': 'True',
                        'raw_metadata_dict_a': 1,
                        'raw_metadata_dict_b': [1, 2, 3]}
        return attrs, encoded, encoded_flat

    def test_encode_attrs_nc(self):
        from satpy.writers.cf_writer import encode_attrs_nc
        import json

        attrs, expected, _ = self.get_test_attrs()

        # Test encoding
        encoded = encode_attrs_nc(attrs)
        self.assertDictEqual(encoded, expected)

        # Test decoding of json-encoded attributes
        raw_md_roundtrip = {'recarray': [[0, 0], [0, 0], [0, 0]],
                            'flag': "True",
                            'dict': {'a': 1, 'b': [1, 2, 3]}}
        self.assertDictEqual(json.loads(encoded['raw_metadata']), raw_md_roundtrip)
        self.assertListEqual(json.loads(encoded['array_3d']), [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])
        self.assertDictEqual(json.loads(encoded['nested_dict']), {"l1": {"l2": {"l3": [1, 2, 3]}}})
        self.assertListEqual(json.loads(encoded['nested_list']), ["1", ["2", [3]]])

    def test_da2cf(self):
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
        self.assertDictEqual(res.attrs, attrs_expected)

        # Test attribute kwargs
        res_flat = CFWriter.da2cf(arr, flatten_attrs=True, exclude_attrs=['int'])
        attrs_expected_flat.pop('int')
        self.assertDictEqual(res_flat.attrs, attrs_expected_flat)

    @mock.patch('satpy.writers.cf_writer.CFWriter.__init__', return_value=None)
    @mock.patch('satpy.writers.cf_writer.area2cf')
    @mock.patch('satpy.writers.cf_writer.CFWriter.da2cf')
    @mock.patch('satpy.writers.cf_writer.make_coords_unique')
    def test_collect_datasets(self, make_coords_unique, da2cf, area2cf, *mocks):
        from satpy.writers.cf_writer import CFWriter
        import xarray as xr

        # Patch methods
        def identity(arg, **kwargs):
            return arg

        def raise_key_error(arg, **kwargs):
            raise KeyError

        da2cf.side_effect = identity
        area2cf.side_effect = raise_key_error
        make_coords_unique.return_value = 'unique_coords'

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
        datas, start_times, end_times = writer._collect_datasets(datasets, latlon=True)

        # Test results
        self.assertEqual(datas, 'unique_coords')
        self.assertListEqual(start_times, [tstart, None])
        self.assertListEqual(end_times, [tend, None])

        # Test area2cf call
        self.assertEqual(len(area2cf.call_args_list), 2)
        for call_args, ds in zip(area2cf.call_args_list, datasets):
            self.assertEqual(call_args, mock.call(ds, strict=True))

        # Test make_coords_unique call
        make_coords_unique.assert_called()
        call_arg = make_coords_unique.call_args[0][0]
        self.assertIsInstance(call_arg, dict)
        self.assertSetEqual(set(call_arg.keys()), {'var1', 'var2'})
        for key, ds in expected.items():
            self.assertTrue(call_arg[key].identical(ds))

    def test_make_coords_unique(self):
        import xarray as xr
        from satpy.writers.cf_writer import make_coords_unique

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
        res = make_coords_unique(datasets)

        # Test that dataset names are prepended to alternative coordinates
        self.assertTrue(np.all(res['var1']['var1_acq_time'] == time1))
        self.assertTrue(np.all(res['var2']['var2_acq_time'] == time2))
        self.assertNotIn('acq_time', res['var1'].coords)
        self.assertNotIn('acq_time', res['var2'].coords)

        # Make sure nothing else is modified
        self.assertTrue(np.all(res['var1']['x'] == x))
        self.assertTrue(np.all(res['var1']['y'] == y))
        self.assertTrue(np.all(res['var2']['x'] == x))
        self.assertTrue(np.all(res['var2']['y'] == y))

    @mock.patch('satpy.writers.cf_writer.area2lonlat')
    @mock.patch('satpy.writers.cf_writer.area2gridmapping')
    def test_area2cf(self, area2gridmapping, area2lonlat):
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
            self.assertEqual(res.attrs['grid_proj4'], proj_str)
            self.assertEqual(grid_mapping, cosmo_expected)


def suite():
    """The test suite for this writer's tests.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCFWriter))
    return mysuite


if __name__ == "__main__":
    unittest.main()
