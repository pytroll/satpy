#!/usr/bin/python
# Copyright (c) 2016 Satpy developers
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
"""
"""

import unittest
import tempfile
import shutil
import os

try:
    from unittest import mock
except ImportError:
    import mock

try:
    from pyproj import CRS
except ImportError:
    CRS = None


def get_test_data(input_shape=(100, 50), output_shape=(200, 100), output_proj=None,
                  input_dims=('y', 'x')):
    """Get common data objects used in testing.

    Returns: tuple with the following elements
        input_data_on_area: DataArray with dimensions as if it is a gridded
            dataset.
        input_area_def: AreaDefinition of the above DataArray
        input_data_on_swath: DataArray with dimensions as if it is a swath.
        input_swath: SwathDefinition of the above DataArray
        target_area_def: AreaDefinition to be used as a target for resampling

    """
    from xarray import DataArray
    import dask.array as da
    from pyresample.geometry import AreaDefinition, SwathDefinition
    from pyresample.utils import proj4_str_to_dict
    ds1 = DataArray(da.zeros(input_shape, chunks=85),
                    dims=input_dims,
                    attrs={'name': 'test_data_name', 'test': 'test'})
    if input_dims and 'y' in input_dims:
        ds1 = ds1.assign_coords(y=da.arange(input_shape[-2], chunks=85))
    if input_dims and 'x' in input_dims:
        ds1 = ds1.assign_coords(x=da.arange(input_shape[-1], chunks=85))
    if input_dims and 'bands' in input_dims:
        ds1 = ds1.assign_coords(bands=list('RGBA'[:ds1.sizes['bands']]))

    input_proj_str = ('+proj=geos +lon_0=-95.0 +h=35786023.0 +a=6378137.0 '
                      '+b=6356752.31414 +sweep=x +units=m +no_defs')
    source = AreaDefinition(
        'test_target',
        'test_target',
        'test_target',
        proj4_str_to_dict(input_proj_str),
        input_shape[1],  # width
        input_shape[0],  # height
        (-1000., -1500., 1000., 1500.))
    ds1.attrs['area'] = source
    if CRS is not None:
        crs = CRS.from_string(input_proj_str)
        ds1 = ds1.assign_coords(crs=crs)

    ds2 = ds1.copy()
    input_area_shape = tuple(ds1.sizes[dim] for dim in ds1.dims
                             if dim in ['y', 'x'])
    geo_dims = ('y', 'x') if input_dims else None
    lons = da.random.random(input_area_shape, chunks=50)
    lats = da.random.random(input_area_shape, chunks=50)
    swath_def = SwathDefinition(
        DataArray(lons, dims=geo_dims),
        DataArray(lats, dims=geo_dims))
    ds2.attrs['area'] = swath_def
    if CRS is not None:
        crs = CRS.from_string('+proj=latlong +datum=WGS84 +ellps=WGS84')
        ds2 = ds2.assign_coords(crs=crs)

    # set up target definition
    output_proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                       '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
    output_proj_str = output_proj or output_proj_str
    target = AreaDefinition(
        'test_target',
        'test_target',
        'test_target',
        proj4_str_to_dict(output_proj_str),
        output_shape[1],  # width
        output_shape[0],  # height
        (-1000., -1500., 1000., 1500.),
    )
    return ds1, source, ds2, swath_def, target


class TestHLResample(unittest.TestCase):
    """Test the higher level resampling functions."""

    def test_type_preserve(self):
        """Check that the type of resampled datasets is preserved."""
        from satpy.resample import resample_dataset
        import xarray as xr
        import dask.array as da
        import numpy as np
        from pyresample.geometry import SwathDefinition
        source_area = SwathDefinition(xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)), dims=['y', 'x']),
                                      xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)), dims=['y', 'x']))
        dest_area = SwathDefinition(xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)) + .0001, dims=['y', 'x']),
                                    xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)) + .0001, dims=['y', 'x']))
        expected_gap = np.array([[1, 2], [3, 255]])
        data = xr.DataArray(da.from_array(expected_gap, chunks=5), dims=['y', 'x'])
        data.attrs['_FillValue'] = 255
        data.attrs['area'] = source_area
        res = resample_dataset(data, dest_area)
        self.assertEqual(res.dtype, data.dtype)
        self.assertTrue(np.all(res.values == expected_gap))

        expected_filled = np.array([[1, 2], [3, 3]])
        res = resample_dataset(data, dest_area, radius_of_influence=1000000)
        self.assertEqual(res.dtype, data.dtype)
        self.assertTrue(np.all(res.values == expected_filled))


class TestKDTreeResampler(unittest.TestCase):
    """Test the kd-tree resampler."""

    @mock.patch('satpy.resample.np.savez')
    @mock.patch('satpy.resample.np.load')
    @mock.patch('satpy.resample.KDTreeResampler._create_cache_filename')
    @mock.patch('satpy.resample.XArrayResamplerNN')
    def test_kd_resampling(self, resampler, create_filename, load, savez):
        """Test the kd resampler."""
        import numpy as np
        import dask.array as da
        from satpy.resample import KDTreeResampler
        data, source_area, swath_data, source_swath, target_area = get_test_data()

        resampler = KDTreeResampler(source_swath, target_area)
        resampler.precompute(
            mask=da.arange(5, chunks=5).astype(np.bool), cache_dir='.')
        resampler.resampler.get_neighbour_info.assert_called()
        # swath definitions should not be cached
        self.assertFalse(len(savez.mock_calls), 0)
        resampler.resampler.reset_mock()

        resampler = KDTreeResampler(source_area, target_area)
        resampler.precompute()
        resampler.resampler.get_neighbour_info.assert_called_with(mask=None)

        try:
            the_dir = tempfile.mkdtemp()
            resampler = KDTreeResampler(source_area, target_area)
            create_filename.return_value = os.path.join(the_dir, 'test_cache.npz')
            load.side_effect = IOError()
            resampler.precompute(cache_dir=the_dir)
            # assert data was saved to the on-disk cache
            self.assertEqual(len(savez.mock_calls), 1)
            # assert that load was called to try to load something from disk
            self.assertEqual(len(load.mock_calls), 1)
            # we should have cached things in-memory
            self.assertEqual(len(resampler._index_caches), 1)
            nbcalls = len(resampler.resampler.get_neighbour_info.mock_calls)
            # test reusing the resampler
            load.side_effect = None

            class FakeNPZ(dict):
                def close(self):
                    pass

            load.return_value = FakeNPZ(valid_input_index=1,
                                        valid_output_index=2,
                                        index_array=3,
                                        distance_array=4)
            resampler.precompute(cache_dir=the_dir)
            # we already have things cached in-memory, no need to save again
            self.assertEqual(len(savez.mock_calls), 1)
            # we already have things cached in-memory, don't need to load
            self.assertEqual(len(load.mock_calls), 1)
            # we should have cached things in-memory
            self.assertEqual(len(resampler._index_caches), 1)
            self.assertEqual(len(resampler.resampler.get_neighbour_info.mock_calls), nbcalls)

            # test loading saved resampler
            resampler = KDTreeResampler(source_area, target_area)
            resampler.precompute(cache_dir=the_dir)
            self.assertEqual(len(load.mock_calls), 2)
            self.assertEqual(len(resampler.resampler.get_neighbour_info.mock_calls), nbcalls)
            # we should have cached things in-memory now
            self.assertEqual(len(resampler._index_caches), 1)
        finally:
            shutil.rmtree(the_dir)

        fill_value = 8
        resampler.compute(data, fill_value=fill_value)
        resampler.resampler.get_sample_from_neighbour_info.assert_called_with(data, fill_value)


class TestEWAResampler(unittest.TestCase):
    """Test EWA resampler class."""

    @mock.patch('satpy.resample.fornav')
    @mock.patch('satpy.resample.ll2cr')
    @mock.patch('satpy.resample.SwathDefinition.get_lonlats')
    def test_2d_ewa(self, get_lonlats, ll2cr, fornav):
        """Test EWA with a 2D dataset."""
        import numpy as np
        import xarray as xr
        from satpy.resample import resample_dataset
        ll2cr.return_value = (100,
                              np.zeros((10, 10), dtype=np.float32),
                              np.zeros((10, 10), dtype=np.float32))
        fornav.return_value = (100 * 200,
                               np.zeros((200, 100), dtype=np.float32))
        _, _, swath_data, source_swath, target_area = get_test_data()
        get_lonlats.return_value = (source_swath.lons, source_swath.lats)
        swath_data.data = swath_data.data.astype(np.float32)
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])

        new_data = resample_dataset(swath_data, target_area, resampler='ewa')
        self.assertTupleEqual(new_data.shape, (200, 100))
        self.assertEqual(new_data.dtype, np.float32)
        self.assertEqual(new_data.attrs['test'], 'test')
        self.assertIs(new_data.attrs['area'], target_area)
        # make sure we can actually compute everything
        new_data.compute()
        lonlat_calls = get_lonlats.call_count
        ll2cr_calls = ll2cr.call_count

        # resample a different dataset and make sure cache is used
        data = xr.DataArray(
            swath_data.data,
            dims=('y', 'x'), attrs={'area': source_swath, 'test': 'test2',
                                    'name': 'test2'})
        new_data = resample_dataset(data, target_area, resampler='ewa')
        new_data.compute()
        # ll2cr will be called once more because of the computation
        self.assertEqual(ll2cr.call_count, ll2cr_calls + num_chunks)
        # but we should already have taken the lonlats from the SwathDefinition
        self.assertEqual(get_lonlats.call_count, lonlat_calls)
        self.assertIn('y', new_data.coords)
        self.assertIn('x', new_data.coords)
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())
            self.assertEqual(new_data.coords['y'].attrs['units'], 'meter')
            self.assertEqual(new_data.coords['x'].attrs['units'], 'meter')

    @mock.patch('satpy.resample.fornav')
    @mock.patch('satpy.resample.ll2cr')
    @mock.patch('satpy.resample.SwathDefinition.get_lonlats')
    def test_3d_ewa(self, get_lonlats, ll2cr, fornav):
        """Test EWA with a 3D dataset."""
        import numpy as np
        import xarray as xr
        from satpy.resample import resample_dataset
        _, _, swath_data, source_swath, target_area = get_test_data(
            input_shape=(3, 200, 100), input_dims=('bands', 'y', 'x'))
        swath_data.data = swath_data.data.astype(np.float32)
        ll2cr.return_value = (100,
                              np.zeros((10, 10), dtype=np.float32),
                              np.zeros((10, 10), dtype=np.float32))
        fornav.return_value = ([100 * 200] * 3,
                               [np.zeros((200, 100), dtype=np.float32)] * 3)
        get_lonlats.return_value = (source_swath.lons, source_swath.lats)
        num_chunks = len(source_swath.lons.chunks[0]) * len(source_swath.lons.chunks[1])

        new_data = resample_dataset(swath_data, target_area, resampler='ewa')
        self.assertTupleEqual(new_data.shape, (3, 200, 100))
        self.assertEqual(new_data.dtype, np.float32)
        self.assertEqual(new_data.attrs['test'], 'test')
        self.assertIs(new_data.attrs['area'], target_area)
        # make sure we can actually compute everything
        new_data.compute()
        lonlat_calls = get_lonlats.call_count
        ll2cr_calls = ll2cr.call_count

        # resample a different dataset and make sure cache is used
        swath_data = xr.DataArray(
            swath_data.data,
            dims=('bands', 'y', 'x'), coords={'bands': ['R', 'G', 'B']},
            attrs={'area': source_swath, 'test': 'test'})
        new_data = resample_dataset(swath_data, target_area, resampler='ewa')
        new_data.compute()
        # ll2cr will be called once more because of the computation
        self.assertEqual(ll2cr.call_count, ll2cr_calls + num_chunks)
        # but we should already have taken the lonlats from the SwathDefinition
        self.assertEqual(get_lonlats.call_count, lonlat_calls)
        self.assertIn('y', new_data.coords)
        self.assertIn('x', new_data.coords)
        self.assertIn('bands', new_data.coords)
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())
            self.assertEqual(new_data.coords['y'].attrs['units'], 'meter')
            self.assertEqual(new_data.coords['x'].attrs['units'], 'meter')
            np.testing.assert_equal(new_data.coords['bands'].values,
                                    ['R', 'G', 'B'])


class TestNativeResampler(unittest.TestCase):
    """Tests for the 'native' resampling method."""

    def test_expand_reduce(self):
        """Test class method 'expand_reduce' basics."""
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        d_arr = da.zeros((6, 20), chunks=4)
        new_data = NativeResampler.expand_reduce(d_arr, {0: 2., 1: 2.})
        self.assertEqual(new_data.shape, (12, 40))
        new_data = NativeResampler.expand_reduce(d_arr, {0: .5, 1: .5})
        self.assertEqual(new_data.shape, (3, 10))
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 1. / 3, 1: 1.})
        new_data = NativeResampler.expand_reduce(d_arr, {0: 1., 1: 1.})
        self.assertEqual(new_data.shape, (6, 20))
        self.assertIs(new_data, d_arr)
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 0.333323423, 1: 1.})
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 1.333323423, 1: 1.})

        n_arr = np.zeros((6, 20))
        new_data = NativeResampler.expand_reduce(n_arr, {0: 2., 1: 1.0})
        self.assertTrue(np.all(new_data.compute()[::2, :] == n_arr))

    def test_expand_dims(self):
        """Test expanding native resampling with 2D data."""
        from satpy.resample import NativeResampler
        import numpy as np
        ds1, source_area, _, _, target_area = get_test_data()
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        new_data = resampler.resample(ds1)
        self.assertEqual(new_data.shape, (200, 100))
        new_data2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_data == new_data2))
        self.assertIn('y', new_data.coords)
        self.assertIn('x', new_data.coords)
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())
            self.assertEqual(new_data.coords['y'].attrs['units'], 'meter')
            self.assertEqual(new_data.coords['x'].attrs['units'], 'meter')

    def test_expand_dims_3d(self):
        """Test expanding native resampling with 3D data."""
        from satpy.resample import NativeResampler
        import numpy as np
        ds1, source_area, _, _, target_area = get_test_data(
            input_shape=(3, 100, 50), input_dims=('bands', 'y', 'x'))
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        new_data = resampler.resample(ds1)
        self.assertEqual(new_data.shape, (3, 200, 100))
        new_data2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_data == new_data2))
        self.assertIn('y', new_data.coords)
        self.assertIn('x', new_data.coords)
        self.assertIn('bands', new_data.coords)
        np.testing.assert_equal(new_data.coords['bands'].values,
                                ['R', 'G', 'B'])
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())
            self.assertEqual(new_data.coords['y'].attrs['units'], 'meter')
            self.assertEqual(new_data.coords['x'].attrs['units'], 'meter')

    def test_expand_without_dims(self):
        """Test expanding native resampling with no dimensions specified."""
        from satpy.resample import NativeResampler
        import numpy as np
        ds1, source_area, _, _, target_area = get_test_data(input_dims=None)
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        new_data = resampler.resample(ds1)
        self.assertEqual(new_data.shape, (200, 100))
        new_data2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_data == new_data2))
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())

    def test_expand_without_dims_4D(self):
        """Test expanding native resampling with 4D data with no dimensions specified."""
        from satpy.resample import NativeResampler
        ds1, source_area, _, _, target_area = get_test_data(
            input_shape=(2, 3, 100, 50), input_dims=None)
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        self.assertRaises(ValueError, resampler.resample, ds1)


class TestBilinearResampler(unittest.TestCase):
    """Test the bilinear resampler."""

    @mock.patch('satpy.resample.np.savez')
    @mock.patch('satpy.resample.np.load')
    @mock.patch('satpy.resample.BilinearResampler._create_cache_filename')
    @mock.patch('satpy.resample.XArrayResamplerBilinear')
    def test_bil_resampling(self, resampler, create_filename, load, savez):
        """Test the bilinear resampler."""
        import numpy as np
        import dask.array as da
        import xarray as xr
        from satpy.resample import BilinearResampler
        data, source_area, swath_data, source_swath, target_area = get_test_data()

        # Test that bilinear resampling info calculation is called,
        # and the info is saved
        load.side_effect = IOError()
        resampler = BilinearResampler(source_swath, target_area)
        resampler.precompute(
            mask=da.arange(5, chunks=5).astype(np.bool))
        resampler.resampler.get_bil_info.assert_called()
        resampler.resampler.get_bil_info.assert_called_with()
        self.assertFalse(len(savez.mock_calls), 1)
        resampler.resampler.reset_mock()
        load.reset_mock()
        load.side_effect = None

        # Test that get_sample_from_bil_info is called properly
        fill_value = 8
        resampler.resampler.get_sample_from_bil_info.return_value = \
            xr.DataArray(da.zeros(target_area.shape), dims=('y', 'x'))
        new_data = resampler.compute(data, fill_value=fill_value)
        resampler.resampler.get_sample_from_bil_info.assert_called_with(
            data, fill_value=fill_value, output_shape=target_area.shape)
        self.assertIn('y', new_data.coords)
        self.assertIn('x', new_data.coords)
        if CRS is not None:
            self.assertIn('crs', new_data.coords)
            self.assertIsInstance(new_data.coords['crs'].item(), CRS)
            self.assertIn('lcc', new_data.coords['crs'].item().to_proj4())
            self.assertEqual(new_data.coords['y'].attrs['units'], 'meter')
            self.assertEqual(new_data.coords['x'].attrs['units'], 'meter')

        # Test that the resampling info is tried to read from the disk
        resampler = BilinearResampler(source_swath, target_area)
        resampler.precompute(cache_dir='.')
        load.assert_called()

        # Test caching the resampling info
        try:
            the_dir = tempfile.mkdtemp()
            resampler = BilinearResampler(source_area, target_area)
            create_filename.return_value = os.path.join(the_dir, 'test_cache.npz')
            load.reset_mock()
            load.side_effect = IOError()

            resampler.precompute(cache_dir=the_dir)
            savez.assert_called()
            # assert data was saved to the on-disk cache
            self.assertEqual(len(savez.mock_calls), 1)
            # assert that load was called to try to load something from disk
            self.assertEqual(len(load.mock_calls), 1)

            nbcalls = len(resampler.resampler.get_bil_info.mock_calls)
            # test reusing the resampler
            load.side_effect = None

            class FakeNPZ(dict):
                def close(self):
                    pass

            load.return_value = FakeNPZ(bilinear_s=1,
                                        bilinear_t=2,
                                        valid_input_index=3,
                                        index_array=4)
            resampler.precompute(cache_dir=the_dir)
            # we already have things cached in-memory, no need to save again
            self.assertEqual(len(savez.mock_calls), 1)
            # we already have things cached in-memory, don't need to load
            # self.assertEqual(len(load.mock_calls), 1)
            self.assertEqual(len(resampler.resampler.get_bil_info.mock_calls), nbcalls)

            # test loading saved resampler
            resampler = BilinearResampler(source_area, target_area)
            resampler.precompute(cache_dir=the_dir)
            self.assertEqual(len(load.mock_calls), 2)
            self.assertEqual(len(resampler.resampler.get_bil_info.mock_calls), nbcalls)
            # we should have cached things in-memory now
            # self.assertEqual(len(resampler._index_caches), 1)
        finally:
            shutil.rmtree(the_dir)


class TestCoordinateHelpers(unittest.TestCase):
    """Test various utility functions for working with coordinates."""

    def test_area_def_coordinates(self):
        """Test coordinates being added with an AreaDefinition."""
        import numpy as np
        import dask.array as da
        import xarray as xr
        from pyresample.geometry import AreaDefinition
        from satpy.resample import add_crs_xy_coords
        area_def = AreaDefinition(
            'test', 'test', 'test', {'proj': 'lcc', 'lat_1': 25, 'lat_0': 25},
            100, 200, [-100, -100, 100, 100]
        )
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={'area': area_def},
            dims=('y', 'x'),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        self.assertIn('y', new_data_arr.coords)
        self.assertIn('x', new_data_arr.coords)

        if CRS is not None:
            self.assertIn('units', new_data_arr.coords['y'].attrs)
            self.assertEqual(
                new_data_arr.coords['y'].attrs['units'], 'meter')
            self.assertIn('units', new_data_arr.coords['x'].attrs)
            self.assertEqual(
                new_data_arr.coords['x'].attrs['units'], 'meter')
            self.assertIn('crs', new_data_arr.coords)
            self.assertIsInstance(new_data_arr.coords['crs'].item(), CRS)

        # already has coords
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={'area': area_def},
            dims=('y', 'x'),
            coords={'y': np.arange(2, 202), 'x': np.arange(100)}
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        self.assertIn('y', new_data_arr.coords)
        self.assertNotIn('units', new_data_arr.coords['y'].attrs)
        self.assertIn('x', new_data_arr.coords)
        self.assertNotIn('units', new_data_arr.coords['x'].attrs)
        np.testing.assert_equal(new_data_arr.coords['y'], np.arange(2, 202))

        if CRS is not None:
            self.assertIn('crs', new_data_arr.coords)
            self.assertIsInstance(new_data_arr.coords['crs'].item(), CRS)

        # lat/lon area
        area_def = AreaDefinition(
            'test', 'test', 'test', {'proj': 'latlong'},
            100, 200, [-100, -100, 100, 100]
        )
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={'area': area_def},
            dims=('y', 'x'),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        self.assertIn('y', new_data_arr.coords)
        self.assertIn('x', new_data_arr.coords)

        if CRS is not None:
            self.assertIn('units', new_data_arr.coords['y'].attrs)
            self.assertEqual(
                new_data_arr.coords['y'].attrs['units'], 'degrees_north')
            self.assertIn('units', new_data_arr.coords['x'].attrs)
            self.assertEqual(
                new_data_arr.coords['x'].attrs['units'], 'degrees_east')
            self.assertIn('crs', new_data_arr.coords)
            self.assertIsInstance(new_data_arr.coords['crs'].item(), CRS)

    def test_swath_def_coordinates(self):
        """Test coordinates being added with an SwathDefinition."""
        import dask.array as da
        import xarray as xr
        from pyresample.geometry import SwathDefinition
        from satpy.resample import add_crs_xy_coords
        lons_data = da.random.random((200, 100), chunks=50)
        lats_data = da.random.random((200, 100), chunks=50)
        lons = xr.DataArray(lons_data, attrs={'units': 'degrees_east'},
                            dims=('y', 'x'))
        lats = xr.DataArray(lats_data, attrs={'units': 'degrees_north'},
                            dims=('y', 'x'))
        area_def = SwathDefinition(lons, lats)
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={'area': area_def},
            dims=('y', 'x'),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        # See https://github.com/pydata/xarray/issues/3068
        # self.assertIn('longitude', new_data_arr.coords)
        # self.assertIn('units', new_data_arr.coords['longitude'].attrs)
        # self.assertEqual(
        #     new_data_arr.coords['longitude'].attrs['units'], 'degrees_east')
        # self.assertIsInstance(new_data_arr.coords['longitude'].data, da.Array)
        # self.assertIn('latitude', new_data_arr.coords)
        # self.assertIn('units', new_data_arr.coords['latitude'].attrs)
        # self.assertEqual(
        #     new_data_arr.coords['latitude'].attrs['units'], 'degrees_north')
        # self.assertIsInstance(new_data_arr.coords['latitude'].data, da.Array)

        if CRS is not None:
            self.assertIn('crs', new_data_arr.coords)
            crs = new_data_arr.coords['crs'].item()
            self.assertIsInstance(crs, CRS)
            self.assertIn('longlat', crs.to_proj4())
            self.assertIsInstance(new_data_arr.coords['crs'].item(), CRS)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNativeResampler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestKDTreeResampler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestEWAResampler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestHLResample))
    mysuite.addTest(loader.loadTestsFromTestCase(TestBilinearResampler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestCoordinateHelpers))

    return mysuite


if __name__ == '__main__':
    unittest.main()
