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

import json
import logging
import os
import tempfile
import warnings
from collections import OrderedDict
from datetime import datetime

import dask.array as da
import numpy as np
import pyresample.geometry
import pytest
import xarray as xr
from packaging.version import Version
from pyresample import create_area_def

from satpy import Scene
from satpy.tests.utils import make_dsq
from satpy.writers.cf_writer import _get_backend_versions

try:
    from pyproj import CRS
except ImportError:
    CRS = None

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - caplog
# - request


class TempFile:
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


def test_lonlat_storage(tmp_path):
    """Test correct storage for area with lon/lat units."""
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
    from satpy.writers.cf.crs import _is_projected

    # test case with units but no area
    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"),
        coords={"x": xr.DataArray(np.arange(5), dims=("x",), attrs={"units": "m"}),
                "y": xr.DataArray(np.arange(5), dims=("y",), attrs={"units": "m"})})
    assert _is_projected(da)

    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"),
        coords={"x": xr.DataArray(np.arange(5), dims=("x",), attrs={"units": "degrees_east"}),
                "y": xr.DataArray(np.arange(5), dims=("y",), attrs={"units": "degrees_north"})})
    assert not _is_projected(da)

    da = xr.DataArray(
        np.arange(25).reshape(5, 5),
        dims=("y", "x"))
    with caplog.at_level(logging.WARNING):
        assert _is_projected(da)
    assert "Failed to tell if data are projected." in caplog.text


def test_preprocess_dataarray_name():
    """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix include orig name."""
    from satpy import Scene
    from satpy.writers.cf_writer import _preprocess_dataarray_name

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


def test_add_time_cf_attrs():
    """Test addition of CF-compliant time attributes."""
    from satpy import Scene
    from satpy.writers.cf_writer import add_time_bounds_dimension

    scn = Scene()
    test_array = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    times = np.array(['2018-05-30T10:05:00', '2018-05-30T10:05:01',
                      '2018-05-30T10:05:02', '2018-05-30T10:05:03'], dtype=np.datetime64)
    scn['test-array'] = xr.DataArray(test_array,
                                     dims=['y', 'x'],
                                     coords={'time': ('y', times)},
                                     attrs=dict(start_time=times[0], end_time=times[-1]))
    ds = scn['test-array'].to_dataset(name='test-array')
    ds = add_time_bounds_dimension(ds)
    assert "bnds_1d" in ds.dims
    assert ds.dims['bnds_1d'] == 2
    assert "time_bnds" in list(ds.data_vars)
    assert "bounds" in ds["time"].attrs
    assert "standard_name" in ds["time"].attrs


def test_empty_collect_cf_datasets():
    """Test that if no DataArrays, collect_cf_datasets raise error."""
    from satpy.writers.cf_writer import collect_cf_datasets

    with pytest.raises(RuntimeError):
        collect_cf_datasets(list_dataarrays=[])


class TestCFWriter:
    """Test case for CF writer."""

    def test_init(self):
        """Test initializing the CFWriter class."""
        from satpy.writers import configs_for_writer
        from satpy.writers.cf_writer import CFWriter

        CFWriter(config_files=list(configs_for_writer('cf'))[0])

    def test_save_array(self):
        """Test saving an array to netcdf/cf."""
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
                assert f['test-array'].attrs['prerequisites'] == expected_prereq

    def test_save_array_coords(self):
        """Test saving array with coordinates."""
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
                assert 'crs' not in f
                assert '_FillValue' not in f['x'].attrs
                assert '_FillValue' not in f['y'].attrs
                expected_prereq = ("DataQuery(name='hej')")
                assert f['test-array'].attrs['prerequisites'] == expected_prereq

    def test_save_dataset_a_digit(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit."""
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['CHANNEL_1'][:], [1, 2, 3])

    def test_save_dataset_a_digit_prefix(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix."""
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', numeric_name_prefix='TEST')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['TEST1'][:], [1, 2, 3])

    def test_save_dataset_a_digit_prefix_include_attr(self):
        """Test saving an array to netcdf/cf where dataset name starting with a digit with prefix include orig name."""
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', include_orig_name=True, numeric_name_prefix='TEST')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['TEST1'][:], [1, 2, 3])
                assert f['TEST1'].attrs['original_name'] == '1'

    def test_save_dataset_a_digit_no_prefix_include_attr(self):
        """Test saving an array to netcdf/cf dataset name starting with a digit with no prefix include orig name."""
        scn = Scene()
        scn['1'] = xr.DataArray([1, 2, 3])
        with TempFile() as filename:
            scn.save_datasets(filename=filename, writer='cf', include_orig_name=True, numeric_name_prefix='')
            with xr.open_dataset(filename) as f:
                np.testing.assert_array_equal(f['1'][:], [1, 2, 3])
                assert 'original_name' not in f['1'].attrs

    def test_ancillary_variables(self):
        """Test ancillary_variables cited each other."""
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
                assert f['test-array-1'].attrs['ancillary_variables'] == 'test-array-2'
                assert f['test-array-2'].attrs['ancillary_variables'] == 'test-array-1'

    def test_groups(self):
        """Test creating a file with groups."""
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
            assert 'history' in nc_root.attrs
            assert set(nc_root.variables.keys()) == set()

            nc_visir = xr.open_dataset(filename, group='visir')
            nc_hrv = xr.open_dataset(filename, group='hrv')
            assert set(nc_visir.variables.keys()) == {'VIS006', 'IR_108',
                                                      'y', 'x', 'VIS006_acq_time', 'IR_108_acq_time'}
            assert set(nc_hrv.variables.keys()) == {'HRV', 'y', 'x', 'acq_time'}
            for tst, ref in zip([nc_visir['VIS006'], nc_visir['IR_108'], nc_hrv['HRV']],
                                [scn['VIS006'], scn['IR_108'], scn['HRV']]):
                np.testing.assert_array_equal(tst.data, ref.data)
            nc_root.close()
            nc_visir.close()
            nc_hrv.close()

        # Different projection coordinates in one group are not supported
        with TempFile() as filename:
            with pytest.raises(ValueError):
                scn.save_datasets(datasets=['VIS006', 'HRV'], filename=filename, writer='cf')

    def test_single_time_value(self):
        """Test setting a single time value."""
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
                assert f['time'].attrs['bounds'] == 'time_bnds'

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

    def test_unlimited_dims_kwarg(self):
        """Test specification of unlimited dimensions."""
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
                assert set(f.encoding['unlimited_dims']) == {'time'}

    def test_header_attrs(self):
        """Check global attributes are set."""
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
                assert 'history' in f.attrs
                assert f.attrs['sensor'] == 'SEVIRI'
                assert f.attrs['orbit'] == 99999
                np.testing.assert_array_equal(f.attrs['list'], [1, 2, 3])
                assert f.attrs['set'] == '{1, 2, 3}'
                assert f.attrs['dict_a'] == 1
                assert f.attrs['dict_b'] == 2
                assert f.attrs['nested_outer_inner1'] == 1
                assert f.attrs['nested_outer_inner2'] == 2
                assert f.attrs['bool'] == 'true'
                assert f.attrs['bool_'] == 'true'
                assert 'none' not in f.attrs.keys()

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
                 'numpy_string': np.str_('test'),
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

    def test_encode_attrs_nc(self):
        """Test attributes encoding."""
        from satpy.writers.cf_writer import encode_attrs_nc

        attrs, expected, _ = self.get_test_attrs()

        # Test encoding
        encoded = encode_attrs_nc(attrs)
        self.assertDictWithArraysEqual(expected, encoded)

        # Test decoding of json-encoded attributes
        raw_md_roundtrip = {'recarray': [[0, 0], [0, 0], [0, 0]],
                            'flag': 'true',
                            'dict': {'a': 1, 'b': [1, 2, 3]}}
        assert json.loads(encoded['raw_metadata']) == raw_md_roundtrip
        assert json.loads(encoded['array_3d']) == [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]
        assert json.loads(encoded['nested_dict']) == {"l1": {"l2": {"l3": [1, 2, 3]}}}
        assert json.loads(encoded['nested_list']) == ["1", ["2", [3]]]

    def test_da2cf(self):
        """Test the conversion of a DataArray to a CF-compatible DataArray."""
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
        assert res['x'].attrs == {'units': 'm', 'standard_name': 'projection_x_coordinate'}
        assert res['y'].attrs == {'units': 'm', 'standard_name': 'projection_y_coordinate'}
        self.assertDictWithArraysEqual(res.attrs, attrs_expected)

        # Test attribute kwargs
        res_flat = CFWriter.da2cf(arr, flatten_attrs=True, exclude_attrs=['int'])
        attrs_expected_flat.pop('int')
        self.assertDictWithArraysEqual(res_flat.attrs, attrs_expected_flat)

    def test_da2cf_one_dimensional_array(self):
        """Test the conversion of an 1d DataArray to a CF-compatible DataArray."""
        from satpy.writers.cf_writer import CFWriter

        arr = xr.DataArray(np.array([1, 2, 3, 4]), attrs={}, dims=('y',),
                           coords={'y': [0, 1, 2, 3], 'acq_time': ('y', [0, 1, 2, 3])})
        _ = CFWriter.da2cf(arr)

    def test_collect_cf_dataarrays(self):
        """Test collecting CF datasets from a DataArray objects."""
        from satpy.writers.cf_writer import _collect_cf_dataset

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
        list_dataarrays = [xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                        attrs={'name': 'var1', 'start_time': tstart, 'end_time': tend, 'area': geos}),
                           xr.DataArray(data=data, dims=('y', 'x'), coords={'y': y, 'x': x, 'acq_time': ('y', time)},
                                        attrs={'name': 'var2', 'long_name': 'variable 2'})]

        # Collect datasets
        ds = _collect_cf_dataset(list_dataarrays, include_lonlats=True)

        # Test results
        assert len(ds.keys()) == 3
        assert set(ds.keys()) == {'var1', 'var2', 'geos'}

        da_var1 = ds['var1']
        da_var2 = ds['var2']
        assert da_var1.name == 'var1'
        assert da_var1.attrs['grid_mapping'] == 'geos'
        assert da_var1.attrs['long_name'] == 'var1'
        # variable 2
        assert 'grid_mapping' not in da_var2.attrs
        assert da_var2.attrs['long_name'] == 'variable 2'

    def test_assert_xy_unique(self):
        """Test that the x and y coordinates are unique."""
        from satpy.writers.cf_writer import assert_xy_unique

        dummy = [[1, 2], [3, 4]]
        datas = {'a': xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]}),
                 'b': xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 2], 'x': [3, 4]}),
                 'n': xr.DataArray(data=dummy, dims=('v', 'w'), coords={'v': [1, 2], 'w': [3, 4]})}
        assert_xy_unique(datas)

        datas['c'] = xr.DataArray(data=dummy, dims=('y', 'x'), coords={'y': [1, 3], 'x': [3, 4]})
        with pytest.raises(ValueError):
            assert_xy_unique(datas)

    def test_link_coords(self):
        """Check that coordinates link has been established correctly."""
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
        assert 'lon' in datasets['var1'].coords
        assert 'lat' in datasets['var1'].coords
        np.testing.assert_array_equal(datasets['var1']['lon'].data, lon)
        np.testing.assert_array_equal(datasets['var1']['lat'].data, lat)
        assert 'coordinates' not in datasets['var1'].attrs

        # There should be no link if there was no 'coordinate' attribute
        assert 'lon' not in datasets['var2'].coords
        assert 'lat' not in datasets['var2'].coords

        # The non-existent dimension or coordinate should be dropped
        assert 'time' not in datasets['var3'].coords
        assert 'not_exist' not in datasets['var4'].coords

    def test_make_alt_coords_unique(self):
        """Test that created coordinate variables are unique."""
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
        assert 'acq_time' not in res['var1'].coords
        assert 'acq_time' not in res['var2'].coords

        # Make sure nothing else is modified
        np.testing.assert_array_equal(res['var1']['x'], x)
        np.testing.assert_array_equal(res['var1']['y'], y)
        np.testing.assert_array_equal(res['var2']['x'], x)
        np.testing.assert_array_equal(res['var2']['y'], y)

        # Coords not unique -> Dataset names must be prepended, even if pretty=True
        with pytest.warns(UserWarning, match='Cannot pretty-format "acq_time"'):
            res = make_alt_coords_unique(datasets, pretty=True)
        np.testing.assert_array_equal(res['var1']['var1_acq_time'], time1)
        np.testing.assert_array_equal(res['var2']['var2_acq_time'], time2)
        assert 'acq_time' not in res['var1'].coords
        assert 'acq_time' not in res['var2'].coords

        # Coords unique and pretty=True -> Don't modify coordinate names
        datasets['var2']['acq_time'] = ('y', time1)
        res = make_alt_coords_unique(datasets, pretty=True)
        np.testing.assert_array_equal(res['var1']['acq_time'], time1)
        np.testing.assert_array_equal(res['var2']['acq_time'], time1)
        assert 'var1_acq_time' not in res['var1'].coords
        assert 'var2_acq_time' not in res['var2'].coords

    def test_area2cf(self):
        """Test the conversion of an area to CF standards."""
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

        res = area2cf(ds, include_lonlats=False)
        assert len(res) == 2
        assert res[0].size == 1  # grid mapping variable
        assert res[0].name == res[1].attrs['grid_mapping']

        # b) Area Definition and include_lonlats=False
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = geos
        res = area2cf(ds, include_lonlats=True)
        # same as above
        assert len(res) == 2
        assert res[0].size == 1  # grid mapping variable
        assert res[0].name == res[1].attrs['grid_mapping']
        # but now also have the lon/lats
        assert 'longitude' in res[1].coords
        assert 'latitude' in res[1].coords

        # c) Swath Definition
        swath = pyresample.geometry.SwathDefinition(lons=[[1, 1], [2, 2]], lats=[[1, 2], [1, 2]])
        ds = ds_base.copy(deep=True)
        ds.attrs['area'] = swath

        res = area2cf(ds, include_lonlats=False)
        assert len(res) == 1
        assert 'longitude' in res[0].coords
        assert 'latitude' in res[0].coords
        assert 'grid_mapping' not in res[0].attrs

    def test__add_grid_mapping(self):
        """Test the conversion from pyresample area object to CF grid mapping."""
        from satpy.writers.cf_writer import _add_grid_mapping

        def _gm_matches(gmapping, expected):
            """Assert that all keys in ``expected`` match the values in ``gmapping``."""
            for attr_key, attr_val in expected.attrs.items():
                test_val = gmapping.attrs[attr_key]
                if attr_val is None or isinstance(attr_val, str):
                    assert test_val == attr_val
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
        new_ds, grid_mapping = _add_grid_mapping(ds)
        if 'sweep_angle_axis' in grid_mapping.attrs:
            # older versions of pyproj might not include this
            assert grid_mapping.attrs['sweep_angle_axis'] == 'y'

        assert new_ds.attrs['grid_mapping'] == 'geos'
        _gm_matches(grid_mapping, geos_expected)
        # should not have been modified
        assert 'grid_mapping' not in ds.attrs

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

        new_ds, grid_mapping = _add_grid_mapping(ds)
        assert 'crs_wkt' in grid_mapping.attrs
        wkt = grid_mapping.attrs['crs_wkt']
        assert 'ELLIPSOID["WGS 84"' in wkt
        assert 'PARAMETER["lat_0",46' in wkt
        assert 'PARAMETER["lon_0",4.535' in wkt
        assert 'PARAMETER["o_lat_p",90' in wkt
        assert 'PARAMETER["o_lon_p",-5.465' in wkt
        assert new_ds.attrs['grid_mapping'] == 'cosmo7'

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
        new_ds, grid_mapping = _add_grid_mapping(ds)
        assert new_ds.attrs['grid_mapping'] == 'tmerc'
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
        new_ds, grid_mapping = _add_grid_mapping(ds)

        assert new_ds.attrs['grid_mapping'] == 'geos'
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
        new_ds, grid_mapping = _add_grid_mapping(ds)

        assert new_ds.attrs['grid_mapping'] == 'omerc_otf'
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
        new_ds, grid_mapping = _add_grid_mapping(ds)

        assert new_ds.attrs['grid_mapping'] == 'geos'
        _gm_matches(grid_mapping, geos_expected)

    def test_add_lonlat_coords(self):
        """Test the conversion from areas to lon/lat."""
        from satpy.writers.cf_writer import add_lonlat_coords

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

        res = add_lonlat_coords(dataarray)

        # original should be unmodified
        assert 'longitude' not in dataarray.coords
        assert set(res.coords) == {'longitude', 'latitude'}
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
        dataarray = xr.DataArray(data=da.from_array(np.arange(3 * 10 * 10).reshape(3, 10, 10), chunks=(1, 5, 5)),
                                 dims=('bands', 'y', 'x'), attrs={'area': area})
        res = add_lonlat_coords(dataarray)

        # original should be unmodified
        assert 'longitude' not in dataarray.coords
        assert set(res.coords) == {'longitude', 'latitude'}
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
                assert f.attrs['Conventions'] == 'CF-1.7'
                assert 'Created by pytroll/satpy on' in f.attrs['history']

    def test_global_attr_history_and_Conventions(self):
        """Test saving global attributes history and Conventions."""
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
                assert f.attrs['Conventions'] == 'CF-1.7, ACDD-1.3'
                assert 'TEST add history\n' in f.attrs['history']
                assert 'Created by pytroll/satpy on' in f.attrs['history']


class TestCFWriterData:
    """Test case for CF writer where data arrays are needed."""

    @pytest.fixture
    def datasets(self):
        """Create test dataset."""
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
        datasets = {
            'var1': xr.DataArray(data=data,
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
        datasets['lat'].attrs['standard_name'] = 'latitude'
        datasets['var1'].attrs['standard_name'] = 'dummy'
        datasets['var2'].attrs['standard_name'] = 'dummy'
        datasets['var2'].attrs['area'] = geos
        datasets['var1'].attrs['area'] = geos
        datasets['lat'].attrs['name'] = 'lat'
        datasets['var1'].attrs['name'] = 'var1'
        datasets['var2'].attrs['name'] = 'var2'
        datasets['lon'].attrs['name'] = 'lon'
        return datasets

    def test_is_lon_or_lat_dataarray(self, datasets):
        """Test the is_lon_or_lat_dataarray function."""
        from satpy.writers.cf_writer import is_lon_or_lat_dataarray

        assert is_lon_or_lat_dataarray(datasets['lat'])
        assert not is_lon_or_lat_dataarray(datasets['var1'])

    def test_has_projection_coords(self, datasets):
        """Test the has_projection_coords function."""
        from satpy.writers.cf_writer import has_projection_coords

        assert has_projection_coords(datasets)
        datasets['lat'].attrs['standard_name'] = 'dummy'
        assert not has_projection_coords(datasets)

    def test_collect_cf_dataarrays_with_latitude_named_lat(self, datasets):
        """Test collecting CF datasets with latitude named lat."""
        from satpy.writers.cf_writer import _collect_cf_dataset

        datasets_list = [datasets[key] for key in datasets.keys()]
        datasets_list_no_latlon = [datasets[key] for key in ['var1', 'var2']]

        # Collect datasets
        ds = _collect_cf_dataset(datasets_list, include_lonlats=True)
        ds2 = _collect_cf_dataset(datasets_list_no_latlon, include_lonlats=True)

        # Test results
        assert len(ds.keys()) == 5
        assert set(ds.keys()) == {'var1', 'var2', 'lon', 'lat', 'geos'}
        with pytest.raises(KeyError):
            ds['var1'].attrs["latitude"]
        with pytest.raises(KeyError):
            ds['var1'].attrs["longitude"]
        assert ds2['var1']['latitude'].attrs['name'] == 'latitude'
        assert ds2['var1']['longitude'].attrs['name'] == 'longitude'


class EncodingUpdateTest:
    """Test update of netCDF encoding."""

    @pytest.fixture
    def fake_ds(self):
        """Create fake data for testing."""
        ds = xr.Dataset({'foo': (('y', 'x'), [[1, 2], [3, 4]]),
                         'bar': (('y', 'x'), [[3, 4], [5, 6]])},
                        coords={'y': [1, 2],
                                'x': [3, 4],
                                'lon': (('y', 'x'), [[7, 8], [9, 10]])})
        return ds

    @pytest.fixture
    def fake_ds_digit(self):
        """Create fake data for testing."""
        ds_digit = xr.Dataset({'CHANNEL_1': (('y', 'x'), [[1, 2], [3, 4]]),
                               'CHANNEL_2': (('y', 'x'), [[3, 4], [5, 6]])},
                              coords={'y': [1, 2],
                                      'x': [3, 4],
                                      'lon': (('y', 'x'), [[7, 8], [9, 10]])})
        return ds_digit

    def test_dataset_name_digit(self, fake_ds_digit):
        """Test data with dataset name staring with a digit."""
        from satpy.writers.cf_writer import update_encoding

        # Dataset with name staring with digit
        ds_digit = fake_ds_digit
        kwargs = {'encoding': {'1': {'dtype': 'float32'},
                               '2': {'dtype': 'float32'}},
                  'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds_digit, kwargs, numeric_name_prefix='CHANNEL_')
        expected_dict = {
            'y': {'_FillValue': None},
            'x': {'_FillValue': None},
            'CHANNEL_1': {'dtype': 'float32'},
            'CHANNEL_2': {'dtype': 'float32'}
        }
        assert enc == expected_dict
        assert other_kwargs == {'other': 'kwargs'}

    def test_without_time(self, fake_ds):
        """Test data with no time dimension."""
        from satpy.writers.cf_writer import update_encoding

        # Without time dimension
        ds = fake_ds.chunk(2)
        kwargs = {'encoding': {'bar': {'chunksizes': (1, 1)}},
                  'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds, kwargs)
        expected_dict = {
            'y': {'_FillValue': None},
            'x': {'_FillValue': None},
            'lon': {'chunksizes': (2, 2)},
            'foo': {'chunksizes': (2, 2)},
            'bar': {'chunksizes': (1, 1)}
        }
        assert enc == expected_dict
        assert other_kwargs == {'other': 'kwargs'}

        # Chunksize may not exceed shape
        ds = fake_ds.chunk(8)
        kwargs = {'encoding': {}, 'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds, kwargs)
        expected_dict = {
            'y': {'_FillValue': None},
            'x': {'_FillValue': None},
            'lon': {'chunksizes': (2, 2)},
            'foo': {'chunksizes': (2, 2)},
            'bar': {'chunksizes': (2, 2)}
        }
        assert enc == expected_dict

    def test_with_time(self, fake_ds):
        """Test data with a time dimension."""
        from satpy.writers.cf_writer import update_encoding

        # With time dimension
        ds = fake_ds.chunk(8).expand_dims({'time': [datetime(2009, 7, 1, 12, 15)]})
        kwargs = {'encoding': {'bar': {'chunksizes': (1, 1, 1)}},
                  'other': 'kwargs'}
        enc, other_kwargs = update_encoding(ds, kwargs)
        expected_dict = {
            'y': {'_FillValue': None},
            'x': {'_FillValue': None},
            'lon': {'chunksizes': (2, 2)},
            'foo': {'chunksizes': (1, 2, 2)},
            'bar': {'chunksizes': (1, 1, 1)},
            'time': {'_FillValue': None,
                     'calendar': 'proleptic_gregorian',
                     'units': 'days since 2009-07-01 12:15:00'},
            'time_bnds': {'_FillValue': None,
                          'calendar': 'proleptic_gregorian',
                          'units': 'days since 2009-07-01 12:15:00'}
        }
        assert enc == expected_dict
        # User-defined encoding may not be altered
        assert kwargs['encoding'] == {'bar': {'chunksizes': (1, 1, 1)}}


class TestEncodingKwarg:
    """Test CF writer with 'encoding' keyword argument."""

    @pytest.fixture
    def scene(self):
        """Create a fake scene."""
        scn = Scene()
        attrs = {
            "start_time": datetime(2018, 5, 30, 10, 0),
            "end_time": datetime(2018, 5, 30, 10, 15)
        }
        scn['test-array'] = xr.DataArray([1., 2, 3], attrs=attrs)
        return scn

    @pytest.fixture(params=[True, False])
    def compression_on(self, request):
        """Get compression options."""
        return request.param

    @pytest.fixture
    def encoding(self, compression_on):
        """Get encoding."""
        enc = {
            'test-array': {
                'dtype': 'int8',
                'scale_factor': 0.1,
                'add_offset': 0.0,
                '_FillValue': 3,
            }
        }
        if compression_on:
            comp_params = _get_compression_params(complevel=7)
            enc["test-array"].update(comp_params)
        return enc

    @pytest.fixture
    def filename(self, tmp_path):
        """Get output filename."""
        return str(tmp_path / "test.nc")

    @pytest.fixture
    def complevel_exp(self, compression_on):
        """Get expected compression level."""
        if compression_on:
            return 7
        return 0

    @pytest.fixture
    def expected(self, complevel_exp):
        """Get expectated file contents."""
        return {
            "data": [10, 20, 30],
            "scale_factor": 0.1,
            "fill_value": 3,
            "dtype": np.int8,
            "complevel": complevel_exp
        }

    def test_encoding_kwarg(self, scene, encoding, filename, expected):
        """Test 'encoding' keyword argument."""
        scene.save_datasets(filename=filename, encoding=encoding, writer='cf')
        self._assert_encoding_as_expected(filename, expected)

    def _assert_encoding_as_expected(self, filename, expected):
        with xr.open_dataset(filename, mask_and_scale=False) as f:
            np.testing.assert_array_equal(f['test-array'][:], expected["data"])
            assert f['test-array'].attrs['scale_factor'] == expected["scale_factor"]
            assert f['test-array'].attrs['_FillValue'] == expected["fill_value"]
            assert f['test-array'].dtype == expected["dtype"]
            assert f["test-array"].encoding["complevel"] == expected["complevel"]

    def test_warning_if_backends_dont_match(self, scene, filename, monkeypatch):
        """Test warning if backends don't match."""
        import netCDF4
        with monkeypatch.context() as m:
            m.setattr(netCDF4, "__version__", "1.5.0")
            m.setattr(netCDF4, "__netcdf4libversion__", "4.9.1")
            with pytest.warns(UserWarning, match=r"Backend version mismatch"):
                scene.save_datasets(filename=filename, writer="cf")

    def test_no_warning_if_backends_match(self, scene, filename, monkeypatch):
        """Make sure no warning is issued if backends match."""
        import netCDF4
        with monkeypatch.context() as m:
            m.setattr(netCDF4, "__version__", "1.6.0")
            m.setattr(netCDF4, "__netcdf4libversion__", "4.9.0")
            m.setattr(xr, "__version__", "2022.12.0")
            with warnings.catch_warnings():
                scene.save_datasets(filename=filename, writer="cf")
                warnings.simplefilter("error")


class TestEncodingAttribute(TestEncodingKwarg):
    """Test CF writer with 'encoding' dataset attribute."""

    @pytest.fixture
    def scene_with_encoding(self, scene, encoding):
        """Create scene with a dataset providing the 'encoding' attribute."""
        scene["test-array"].encoding = encoding["test-array"]
        return scene

    def test_encoding_attribute(self, scene_with_encoding, filename, expected):
        """Test 'encoding' dataset attribute."""
        scene_with_encoding.save_datasets(filename=filename, writer='cf')
        self._assert_encoding_as_expected(filename, expected)


def _get_compression_params(complevel):
    params = {"complevel": complevel}
    if _should_use_compression_keyword():
        params["compression"] = "zlib"
    else:
        params["zlib"] = True
    return params


def _should_use_compression_keyword():
    # xarray currently ignores the "compression" keyword, see
    # https://github.com/pydata/xarray/issues/7388. There's already an open
    # PR, so we assume that this will be fixed in the next minor release
    # (current release is 2023.02). If not, tests will fail and remind us.
    versions = _get_backend_versions()
    return (
        versions["libnetcdf"] >= Version("4.9.0") and
        versions["xarray"] >= Version("2023.10")
    )
