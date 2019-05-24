#!/usr/bin/python
# Copyright (c) 2016.
#

# Author(s):
#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

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
        from pyresample.geometry import SwathDefinition
        source_area = mock.MagicMock()
        source_swath = SwathDefinition(
            da.arange(5, chunks=5), da.arange(5, chunks=5))
        target_area = mock.MagicMock()

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

        data = mock.MagicMock()
        data.name = 'hej'
        data.data = [1, 2, 3]
        fill_value = 8
        resampler.compute(data, fill_value=fill_value)
        resampler.resampler.get_sample_from_neighbour_info.assert_called_with(data, fill_value)


class TestEWAResampler(unittest.TestCase):
    """Test EWA resampler class."""

    @mock.patch('satpy.resample.fornav')
    @mock.patch('satpy.resample.ll2cr')
    def test_2d_ewa(self, ll2cr, fornav):
        """Test EWA with a 2D dataset."""
        import numpy as np
        import dask.array as da
        import xarray as xr
        from satpy.resample import resample_dataset
        from pyresample.geometry import SwathDefinition, AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        lons = xr.DataArray(da.zeros((10, 10), chunks=5))
        lats = xr.DataArray(da.zeros((10, 10), chunks=5))
        ll2cr.return_value = (100,
                              np.zeros((10, 10), dtype=np.float32),
                              np.zeros((10, 10), dtype=np.float32))
        fornav.return_value = (100 * 200,
                               np.zeros((200, 100), dtype=np.float32))
        sgd = SwathDefinition(lons, lats)
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        tgd = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        input_data = xr.DataArray(
            da.zeros((10, 10), chunks=5, dtype=np.float32),
            dims=('y', 'x'), attrs={'area': sgd, 'test': 'test'})

        new_data = resample_dataset(input_data, tgd, resampler='ewa')
        self.assertTupleEqual(new_data.shape, (200, 100))
        self.assertEqual(new_data.dtype, np.float32)
        self.assertEqual(new_data.attrs['test'], 'test')
        self.assertIs(new_data.attrs['area'], tgd)
        # make sure we can actually compute everything
        new_data.compute()
        previous_calls = ll2cr.call_count

        # resample a different dataset and make sure cache is used
        input_data = xr.DataArray(
            da.zeros((10, 10), chunks=5, dtype=np.float32),
            dims=('y', 'x'), attrs={'area': sgd, 'test': 'test'})
        new_data = resample_dataset(input_data, tgd, resampler='ewa')
        self.assertEqual(ll2cr.call_count, previous_calls)
        new_data.compute()

    @mock.patch('satpy.resample.fornav')
    @mock.patch('satpy.resample.ll2cr')
    def test_3d_ewa(self, ll2cr, fornav):
        """Test EWA with a 3D dataset."""
        import numpy as np
        import dask.array as da
        import xarray as xr
        from satpy.resample import resample_dataset
        from pyresample.geometry import SwathDefinition, AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        lons = xr.DataArray(da.zeros((10, 10), chunks=5))
        lats = xr.DataArray(da.zeros((10, 10), chunks=5))
        ll2cr.return_value = (100,
                              np.zeros((10, 10), dtype=np.float32),
                              np.zeros((10, 10), dtype=np.float32))
        fornav.return_value = ([100 * 200] * 3,
                               [np.zeros((200, 100), dtype=np.float32)] * 3)
        sgd = SwathDefinition(lons, lats)
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        tgd = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        input_data = xr.DataArray(
            da.zeros((3, 10, 10), chunks=5, dtype=np.float32),
            dims=('bands', 'y', 'x'), attrs={'area': sgd, 'test': 'test'})

        new_data = resample_dataset(input_data, tgd, resampler='ewa')
        self.assertTupleEqual(new_data.shape, (3, 200, 100))
        self.assertEqual(new_data.dtype, np.float32)
        self.assertEqual(new_data.attrs['test'], 'test')
        self.assertIs(new_data.attrs['area'], tgd)
        # make sure we can actually compute everything
        new_data.compute()
        previous_calls = ll2cr.call_count

        # resample a different dataset and make sure cache is used
        input_data = xr.DataArray(
            da.zeros((3, 10, 10), chunks=5, dtype=np.float32),
            dims=('bands', 'y', 'x'), attrs={'area': sgd, 'test': 'test'})
        new_data = resample_dataset(input_data, tgd, resampler='ewa')
        self.assertEqual(ll2cr.call_count, previous_calls)
        new_data.compute()


class TestNativeResampler(unittest.TestCase):
    """Tests for the 'native' resampling method."""

    def test_expand_reduce(self):
        """Test class method 'expand_reduce' basics."""
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        d_arr = da.zeros((6, 20), chunks=4)
        new_arr = NativeResampler.expand_reduce(d_arr, {0: 2., 1: 2.})
        self.assertEqual(new_arr.shape, (12, 40))
        new_arr = NativeResampler.expand_reduce(d_arr, {0: .5, 1: .5})
        self.assertEqual(new_arr.shape, (3, 10))
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 1. / 3, 1: 1.})
        new_arr = NativeResampler.expand_reduce(d_arr, {0: 1., 1: 1.})
        self.assertEqual(new_arr.shape, (6, 20))
        self.assertIs(new_arr, d_arr)
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 0.333323423, 1: 1.})
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 1.333323423, 1: 1.})

        n_arr = np.zeros((6, 20))
        new_arr = NativeResampler.expand_reduce(n_arr, {0: 2., 1: 1.0})
        self.assertTrue(np.all(new_arr.compute()[::2, :] == n_arr))

    def test_expand_dims(self):
        """Test expanding native resampling with 2D data."""
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        ds1 = DataArray(da.zeros((100, 50), chunks=85), dims=('y', 'x'),
                        coords={'y': da.arange(100, chunks=85),
                                'x': da.arange(50, chunks=85)})
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        target = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        # source geo def doesn't actually matter
        resampler = NativeResampler(None, target)
        new_arr = resampler.resample(ds1)
        self.assertEqual(new_arr.shape, (200, 100))
        new_arr2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_arr == new_arr2))

    def test_expand_dims_3d(self):
        """Test expanding native resampling with 3D data."""
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        ds1 = DataArray(da.zeros((3, 100, 50), chunks=85), dims=('bands', 'y', 'x'),
                        coords={'bands': ['R', 'G', 'B'],
                                'y': da.arange(100, chunks=85),
                                'x': da.arange(50, chunks=85)})
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        target = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        # source geo def doesn't actually matter
        resampler = NativeResampler(None, target)
        new_arr = resampler.resample(ds1)
        self.assertEqual(new_arr.shape, (3, 200, 100))
        new_arr2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_arr == new_arr2))

    def test_expand_without_dims(self):
        """Test expanding native resampling with no dimensions specified."""
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        ds1 = DataArray(da.zeros((100, 50), chunks=85))
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        target = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        # source geo def doesn't actually matter
        resampler = NativeResampler(None, target)
        new_arr = resampler.resample(ds1)
        self.assertEqual(new_arr.shape, (200, 100))
        new_arr2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_arr == new_arr2))

    def test_expand_without_dims_4D(self):
        """Test expanding native resampling with 4D data with no dimensions specified."""
        from satpy.resample import NativeResampler
        import dask.array as da
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        ds1 = DataArray(da.zeros((2, 3, 100, 50), chunks=85))
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        target = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        # source geo def doesn't actually matter
        resampler = NativeResampler(None, target)
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
        from satpy.resample import BilinearResampler
        from pyresample.geometry import SwathDefinition
        source_area = mock.MagicMock()
        source_swath = SwathDefinition(
            da.arange(5, chunks=5), da.arange(5, chunks=5))
        target_area = mock.MagicMock()

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
        data = mock.MagicMock()
        data.name = 'foo'
        data.data = [1, 2, 3]
        fill_value = 8
        resampler.compute(data, fill_value=fill_value)
        resampler.resampler.get_sample_from_bil_info.assert_called_with(
            data, fill_value=fill_value, output_shape=target_area.shape)

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


class TestBucketResampler(unittest.TestCase):
    """Test the bucket resampler."""

    def setUp(self):
        from satpy.resample import BucketResampler
        get_lonlats = mock.MagicMock()
        get_lonlats.return_value = (1, 2)
        self.source_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.target_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.bucket = BucketResampler(self.source_geo_def, self.target_geo_def)

    def test_init(self):
        """Test bucket resampler initialization"""
        self.assertIsNone(self.bucket.resampler)
        self.assertTrue(self.bucket.source_geo_def == self.source_geo_def)
        self.assertTrue(self.bucket.target_geo_def == self.target_geo_def)

    @mock.patch('pyresample.bucket.BucketResampler')
    def test_precompute(self, bucket):
        """Test bucket resampler precomputation"""
        bucket.return_value = True
        self.bucket.precompute()
        self.assertTrue(self.bucket.resampler)
        bucket.assert_called_once_with(self.target_geo_def, 1, 2)

    def test_compute(self):
        """Test bucket resampler computation."""
        import dask.array as da
        # 1D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((5,))
        self.bucket.resampler.get_average.return_value = data
        res = self.bucket.compute(data, fill_value=2)
        self.bucket.resampler.get_average.assert_called_once_with(data,
                                                                  fill_value=2)
        self.assertEqual(res.shape, (1, 5))
        # 2D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((5, 5))
        self.bucket.resampler.get_average.return_value = data
        res = self.bucket.compute(data, fill_value=2)
        self.bucket.resampler.get_average.assert_called_once_with(data,
                                                                  fill_value=2)
        self.assertEqual(res.shape, (1, 5, 5))
        # 3D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((3, 5, 5))
        self.bucket.resampler.get_average.return_value = data[0, :, :]
        res = self.bucket.compute(data, fill_value=2)
        self.assertEqual(res.shape, (3, 5, 5))

    @mock.patch('pyresample.bucket.BucketResampler')
    def test_resample(self, pyresample_bucket):
        """Test bucket resamplers resample method."""
        import xarray as xr
        import dask.array as da
        self.bucket.resampler = mock.MagicMock()
        self.bucket.precompute = mock.MagicMock()
        self.bucket.compute = mock.MagicMock()

        # 1D input data
        data = xr.DataArray(da.ones((5,)), dims=('foo'), attrs={'bar': 'baz'})
        self.bucket.compute.return_value = da.ones((5, 5))
        res = self.bucket.resample(data)
        self.bucket.precompute.assert_called_once()
        self.bucket.compute.assert_called_once()
        self.assertEqual(res.shape, (5, 5))
        self.assertEqual(res.dims, ('y', 'x'))
        self.assertTrue('bar' in res.attrs)
        self.assertEqual(res.attrs['bar'], 'baz')

        # 2D input data
        data = xr.DataArray(da.ones((5, 5)), dims=('foo', 'bar'))
        self.bucket.compute.return_value = da.ones((5, 5))
        res = self.bucket.resample(data)
        self.assertEqual(res.shape, (5, 5))
        self.assertEqual(res.dims, ('y', 'x'))

        # 3D input data with 'bands' dim
        data = xr.DataArray(da.ones((1, 5, 5)), dims=('bands', 'foo', 'bar'))
        self.bucket.compute.return_value = da.ones((1, 5, 5))
        res = self.bucket.resample(data)
        self.assertEqual(res.shape, (1, 5, 5))
        self.assertEqual(res.dims, ('bands', 'y', 'x'))

        # 3D input data with misc dim names
        data = xr.DataArray(da.ones((3, 5, 5)), dims=('foo', 'bar', 'baz'))
        self.bucket.compute.return_value = da.ones((3, 5, 5))
        res = self.bucket.resample(data)
        self.assertEqual(res.shape, (3, 5, 5))
        self.assertEqual(res.dims, ('foo', 'bar', 'baz'))


class TestBucketSum(unittest.TestCase):
    """Test the sum bucket resampler."""

    def setUp(self):
        from satpy.resample import BucketSum
        get_lonlats = mock.MagicMock()
        get_lonlats.return_value = (1, 2)
        self.source_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.target_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.bucket = BucketSum(self.source_geo_def, self.target_geo_def)

    def test_compute(self):
        """Test sum bucket resampler computation."""
        import dask.array as da
        # 1D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((5,))
        self.bucket.resampler.get_sum.return_value = data
        res = self.bucket.compute(data)
        self.bucket.resampler.get_sum.assert_called_once_with(data)
        self.assertEqual(res.shape, (1, 5))
        # 2D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((5, 5))
        self.bucket.resampler.get_sum.return_value = data
        res = self.bucket.compute(data)
        self.bucket.resampler.get_sum.assert_called_once_with(data)
        self.assertEqual(res.shape, (1, 5, 5))
        # 3D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((3, 5, 5))
        self.bucket.resampler.get_sum.return_value = data[0, :, :]
        res = self.bucket.compute(data)
        self.assertEqual(res.shape, (3, 5, 5))


class TestBucketCount(unittest.TestCase):
    """Test the sum bucket resampler."""

    def setUp(self):
        from satpy.resample import BucketCount
        get_lonlats = mock.MagicMock()
        get_lonlats.return_value = (1, 2)
        self.source_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.target_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.bucket = BucketCount(self.source_geo_def, self.target_geo_def)

    def test_compute(self):
        """Test sum bucket resampler computation."""
        import dask.array as da
        # 1D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((5,))
        self.bucket.resampler.get_count.return_value = data
        res = self.bucket.compute(data)
        self.bucket.resampler.get_count.assert_called_once_with()
        self.assertEqual(res.shape, (1, 5))
        # 2D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((5, 5))
        self.bucket.resampler.get_count.return_value = data
        res = self.bucket.compute(data)
        self.bucket.resampler.get_count.assert_called_once_with()
        self.assertEqual(res.shape, (1, 5, 5))
        # 3D data
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((3, 5, 5))
        self.bucket.resampler.get_count.return_value = data[0, :, :]
        res = self.bucket.compute(data)
        self.assertEqual(res.shape, (3, 5, 5))


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
    mysuite.addTest(loader.loadTestsFromTestCase(TestBucketResampler))
    mysuite.addTest(loader.loadTestsFromTestCase(TestBucketSum))
    mysuite.addTest(loader.loadTestsFromTestCase(TestBucketCount))

    return mysuite


if __name__ == '__main__':
    unittest.main()
