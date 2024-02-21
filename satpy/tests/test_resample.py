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
"""Unittests for resamplers."""

import os
import shutil
import tempfile
import unittest
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from satpy.resample import NativeResampler


def get_test_data(input_shape=(100, 50), output_shape=(200, 100), output_proj=None,
                  input_dims=("y", "x")):
    """Get common data objects used in testing.

    Returns:
        tuple:

        * input_data_on_area: DataArray with dimensions as if it is a gridded
          dataset.
        * input_area_def: AreaDefinition of the above DataArray
        * input_data_on_swath: DataArray with dimensions as if it is a swath.
        * input_swath: SwathDefinition of the above DataArray
        * target_area_def: AreaDefinition to be used as a target for resampling

    """
    import dask.array as da
    from pyresample.geometry import AreaDefinition, SwathDefinition
    from xarray import DataArray
    ds1 = DataArray(da.zeros(input_shape, chunks=85),
                    dims=input_dims,
                    attrs={"name": "test_data_name", "test": "test"})
    if input_dims and "y" in input_dims:
        ds1 = ds1.assign_coords(y=da.arange(input_shape[-2], chunks=85))
    if input_dims and "x" in input_dims:
        ds1 = ds1.assign_coords(x=da.arange(input_shape[-1], chunks=85))
    if input_dims and "bands" in input_dims:
        ds1 = ds1.assign_coords(bands=list("RGBA"[:ds1.sizes["bands"]]))

    input_proj_str = ("+proj=geos +lon_0=-95.0 +h=35786023.0 +a=6378137.0 "
                      "+b=6356752.31414 +sweep=x +units=m +no_defs")
    crs = CRS(input_proj_str)
    source = AreaDefinition(
        "test_target",
        "test_target",
        "test_target",
        crs,
        input_shape[1],  # width
        input_shape[0],  # height
        (-1000., -1500., 1000., 1500.))
    ds1.attrs["area"] = source
    ds1 = ds1.assign_coords(crs=crs)

    ds2 = ds1.copy()
    input_area_shape = tuple(ds1.sizes[dim] for dim in ds1.dims
                             if dim in ["y", "x"])
    geo_dims = ("y", "x") if input_dims else None
    lons = da.random.random(input_area_shape, chunks=50)
    lats = da.random.random(input_area_shape, chunks=50)
    swath_def = SwathDefinition(
        DataArray(lons, dims=geo_dims),
        DataArray(lats, dims=geo_dims))
    ds2.attrs["area"] = swath_def
    crs = CRS.from_string("+proj=latlong +datum=WGS84 +ellps=WGS84")
    ds2 = ds2.assign_coords(crs=crs)

    # set up target definition
    output_proj_str = ("+proj=lcc +datum=WGS84 +ellps=WGS84 "
                       "+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs")
    output_proj_str = output_proj or output_proj_str
    target = AreaDefinition(
        "test_target",
        "test_target",
        "test_target",
        CRS(output_proj_str),
        output_shape[1],  # width
        output_shape[0],  # height
        (-1000., -1500., 1000., 1500.),
    )
    return ds1, source, ds2, swath_def, target


class TestHLResample(unittest.TestCase):
    """Test the higher level resampling functions."""

    def test_type_preserve(self):
        """Check that the type of resampled datasets is preserved."""
        from pyresample.geometry import SwathDefinition

        from satpy.resample import resample_dataset
        source_area = SwathDefinition(xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)), dims=["y", "x"]),
                                      xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)), dims=["y", "x"]))
        dest_area = SwathDefinition(xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)) + .0001, dims=["y", "x"]),
                                    xr.DataArray(da.arange(4, chunks=5).reshape((2, 2)) + .0001, dims=["y", "x"]))
        expected_gap = np.array([[1, 2], [3, 255]])
        data = xr.DataArray(da.from_array(expected_gap, chunks=5), dims=["y", "x"])
        data.attrs["_FillValue"] = 255
        data.attrs["area"] = source_area
        res = resample_dataset(data, dest_area)
        assert res.dtype == data.dtype
        assert np.all(res.values == expected_gap)

        expected_filled = np.array([[1, 2], [3, 3]])
        res = resample_dataset(data, dest_area, radius_of_influence=1000000)
        assert res.dtype == data.dtype
        assert np.all(res.values == expected_filled)


class TestKDTreeResampler(unittest.TestCase):
    """Test the kd-tree resampler."""

    @mock.patch("satpy.resample.xr.Dataset")
    @mock.patch("satpy.resample.zarr.open")
    @mock.patch("satpy.resample.KDTreeResampler._create_cache_filename")
    @mock.patch("pyresample.kd_tree.XArrayResamplerNN")
    def test_kd_resampling(self, xr_resampler, create_filename, zarr_open,
                           xr_dset):
        """Test the kd resampler."""
        from satpy.resample import KDTreeResampler
        data, source_area, swath_data, source_swath, target_area = get_test_data()
        mock_dset = mock.MagicMock()
        xr_dset.return_value = mock_dset
        resampler = KDTreeResampler(source_swath, target_area)
        resampler.precompute(
            mask=da.arange(5, chunks=5).astype(bool), cache_dir=".")
        xr_resampler.assert_called_once()
        resampler.resampler.get_neighbour_info.assert_called()
        # swath definitions should not be cached
        assert len(mock_dset.to_zarr.mock_calls) == 0
        resampler.resampler.reset_mock()

        resampler = KDTreeResampler(source_area, target_area)
        resampler.precompute()
        resampler.resampler.get_neighbour_info.assert_called_with(mask=None)

        try:
            the_dir = tempfile.mkdtemp()
            resampler = KDTreeResampler(source_area, target_area)
            create_filename.return_value = os.path.join(the_dir, "test_cache.zarr")
            zarr_open.side_effect = ValueError()
            resampler.precompute(cache_dir=the_dir)
            # assert data was saved to the on-disk cache
            assert len(mock_dset.to_zarr.mock_calls) == 1
            # assert that zarr_open was called to try to zarr_open something from disk
            assert len(zarr_open.mock_calls) == 1
            # we should have cached things in-memory
            assert len(resampler._index_caches) == 1
            nbcalls = len(resampler.resampler.get_neighbour_info.mock_calls)
            # test reusing the resampler
            zarr_open.side_effect = None
            # The kdtree shouldn't be available after saving cache to disk
            assert resampler.resampler.delayed_kdtree is None

            class FakeZarr(dict):

                def close(self):
                    pass

                def astype(self, dtype):
                    pass

            zarr_open.return_value = FakeZarr(valid_input_index=1,
                                              valid_output_index=2,
                                              index_array=3,
                                              distance_array=4)
            resampler.precompute(cache_dir=the_dir)
            # we already have things cached in-memory, no need to save again
            assert len(mock_dset.to_zarr.mock_calls) == 1
            # we already have things cached in-memory, don't need to load
            assert len(zarr_open.mock_calls) == 1
            # we should have cached things in-memory
            assert len(resampler._index_caches) == 1
            assert len(resampler.resampler.get_neighbour_info.mock_calls) == nbcalls

            # test loading saved resampler
            resampler = KDTreeResampler(source_area, target_area)
            resampler.precompute(cache_dir=the_dir)
            assert len(zarr_open.mock_calls) == 4
            assert len(resampler.resampler.get_neighbour_info.mock_calls) == nbcalls
            # we should have cached things in-memory now
            assert len(resampler._index_caches) == 1
        finally:
            shutil.rmtree(the_dir)

        fill_value = 8
        resampler.compute(data, fill_value=fill_value)
        resampler.resampler.get_sample_from_neighbour_info.assert_called_with(data, fill_value)


class TestNativeResampler:
    """Tests for the 'native' resampling method."""

    def setup_method(self):
        """Create test data used by multiple tests."""
        self.d_arr = da.zeros((6, 20), chunks=4)

    def test_expand_reduce_replicate(self):
        """Test classmethod 'expand_reduce' to replicate by 2."""
        new_data = NativeResampler._expand_reduce(self.d_arr, {0: 2., 1: 2.})
        assert new_data.shape == (12, 40)

    def test_expand_reduce_aggregate(self):
        """Test classmethod 'expand_reduce' to aggregate by half."""
        new_data = NativeResampler._expand_reduce(self.d_arr, {0: .5, 1: .5})
        assert new_data.shape == (3, 10)

    def test_expand_reduce_aggregate_identity(self):
        """Test classmethod 'expand_reduce' returns the original dask array when factor is 1."""
        new_data = NativeResampler._expand_reduce(self.d_arr, {0: 1., 1: 1.})
        assert new_data.shape == (6, 20)
        assert new_data is self.d_arr

    @pytest.mark.parametrize("dim0_factor", [1. / 4, 0.333323423, 1.333323423])
    def test_expand_reduce_aggregate_invalid(self, dim0_factor):
        """Test classmethod 'expand_reduce' fails when factor does not divide evenly."""
        with pytest.raises(ValueError, match="[Aggregation, Expand] .*"):
            NativeResampler._expand_reduce(self.d_arr, {0: dim0_factor, 1: 1.})

    def test_expand_reduce_agg_rechunk(self):
        """Test that an incompatible factor for the chunk size is rechunked.

        This can happen when a user chunks their data that makes sense for
        the overall shape of the array and for their local machine's
        performance, but the resulting resampling factor does not divide evenly
        into that chunk size.

        """
        from satpy.utils import PerformanceWarning

        d_arr = da.zeros((6, 20), chunks=3)
        text = "Array chunk size is not divisible by aggregation factor. Re-chunking to continue native resampling."
        with pytest.warns(PerformanceWarning, match=text):
            new_data = NativeResampler._expand_reduce(d_arr, {0: 0.5, 1: 0.5})
        assert new_data.shape == (3, 10)

    def test_expand_reduce_numpy(self):
        """Test classmethod 'expand_reduce' converts numpy arrays to dask arrays."""
        n_arr = np.zeros((6, 20))
        new_data = NativeResampler._expand_reduce(n_arr, {0: 2., 1: 1.0})
        np.testing.assert_equal(new_data.compute()[::2, :], n_arr)

    def test_expand_dims(self):
        """Test expanding native resampling with 2D data."""
        ds1, source_area, _, _, target_area = get_test_data()
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        new_data = resampler.resample(ds1)
        assert new_data.shape == (200, 100)
        new_data2 = resampler.resample(ds1.compute())
        np.testing.assert_equal(new_data.compute().data, new_data2.compute().data)
        assert "y" in new_data.coords
        assert "x" in new_data.coords
        assert "crs" in new_data.coords
        assert isinstance(new_data.coords["crs"].item(), CRS)
        assert "lambert" in new_data.coords["crs"].item().coordinate_operation.method_name.lower()
        assert new_data.coords["y"].attrs["units"] == "meter"
        assert new_data.coords["x"].attrs["units"] == "meter"
        assert target_area.crs == new_data.coords["crs"].item()

    def test_expand_dims_3d(self):
        """Test expanding native resampling with 3D data."""
        ds1, source_area, _, _, target_area = get_test_data(
            input_shape=(3, 100, 50), input_dims=("bands", "y", "x"))
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        new_data = resampler.resample(ds1)
        assert new_data.shape == (3, 200, 100)
        new_data2 = resampler.resample(ds1.compute())
        np.testing.assert_equal(new_data.compute().data, new_data2.compute().data)
        assert "y" in new_data.coords
        assert "x" in new_data.coords
        assert "bands" in new_data.coords
        np.testing.assert_equal(new_data.coords["bands"].values, ["R", "G", "B"])
        assert "crs" in new_data.coords
        assert isinstance(new_data.coords["crs"].item(), CRS)
        assert "lambert" in new_data.coords["crs"].item().coordinate_operation.method_name.lower()
        assert new_data.coords["y"].attrs["units"] == "meter"
        assert new_data.coords["x"].attrs["units"] == "meter"
        assert target_area.crs == new_data.coords["crs"].item()

    def test_expand_without_dims(self):
        """Test expanding native resampling with no dimensions specified."""
        ds1, source_area, _, _, target_area = get_test_data(input_dims=None)
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        new_data = resampler.resample(ds1)
        assert new_data.shape == (200, 100)
        new_data2 = resampler.resample(ds1.compute())
        np.testing.assert_equal(new_data.compute().data, new_data2.compute().data)
        assert "crs" in new_data.coords
        assert isinstance(new_data.coords["crs"].item(), CRS)
        assert "lambert" in new_data.coords["crs"].item().coordinate_operation.method_name.lower()
        assert target_area.crs == new_data.coords["crs"].item()

    def test_expand_without_dims_4D(self):
        """Test expanding native resampling with 4D data with no dimensions specified."""
        ds1, source_area, _, _, target_area = get_test_data(
            input_shape=(2, 3, 100, 50), input_dims=None)
        # source geo def doesn't actually matter
        resampler = NativeResampler(source_area, target_area)
        with pytest.raises(ValueError, match="Can only handle 2D or 3D arrays without dimensions."):
            resampler.resample(ds1)


class TestBilinearResampler(unittest.TestCase):
    """Test the bilinear resampler."""

    @mock.patch("satpy.resample._move_existing_caches")
    @mock.patch("satpy.resample.BilinearResampler._create_cache_filename")
    @mock.patch("pyresample.bilinear.XArrayBilinearResampler")
    def test_bil_resampling(self, xr_resampler, create_filename,
                            move_existing_caches):
        """Test the bilinear resampler."""
        from satpy.resample import BilinearResampler
        data, source_area, swath_data, source_swath, target_area = get_test_data()

        # Test that bilinear resampling info calculation is called
        resampler = BilinearResampler(source_swath, target_area)
        resampler.precompute(
            mask=da.arange(5, chunks=5).astype(bool))
        resampler.resampler.load_resampling_info.assert_not_called()
        resampler.resampler.get_bil_info.assert_called_once()
        resampler.resampler.reset_mock()

        # Test that get_sample_from_bil_info is called properly
        fill_value = 8
        resampler.resampler.get_sample_from_bil_info.return_value = \
            xr.DataArray(da.zeros(target_area.shape), dims=("y", "x"))
        new_data = resampler.compute(data, fill_value=fill_value)
        resampler.resampler.get_sample_from_bil_info.assert_called_with(
            data, fill_value=fill_value, output_shape=target_area.shape)
        assert "y" in new_data.coords
        assert "x" in new_data.coords
        assert "crs" in new_data.coords
        assert isinstance(new_data.coords["crs"].item(), CRS)
        assert "lambert" in new_data.coords["crs"].item().coordinate_operation.method_name.lower()
        assert new_data.coords["y"].attrs["units"] == "meter"
        assert new_data.coords["x"].attrs["units"] == "meter"
        assert target_area.crs == new_data.coords["crs"].item()

        # Test that the resampling info is tried to read from the disk
        resampler = BilinearResampler(source_swath, target_area)
        resampler.precompute(cache_dir=".")
        resampler.resampler.load_resampling_info.assert_called()

        # Test caching the resampling info
        try:
            the_dir = tempfile.mkdtemp()
            resampler = BilinearResampler(source_area, target_area)
            create_filename.return_value = os.path.join(the_dir, "test_cache.zarr")
            xr_resampler.return_value.load_resampling_info.side_effect = IOError

            resampler.precompute(cache_dir=the_dir)
            resampler.resampler.save_resampling_info.assert_called()
            # assert data was saved to the on-disk cache
            resampler.resampler.save_resampling_info.assert_called_once()

            nbcalls = resampler.resampler.get_bil_info.call_count
            resampler.resampler.load_resampling_info.side_effect = None

            resampler.precompute(cache_dir=the_dir)
            # we already have things cached in-memory, no need to save again
            resampler.resampler.save_resampling_info.assert_called_once()
            # we already have things cached in-memory, don't need to load
            assert resampler.resampler.get_bil_info.call_count == nbcalls

            # test loading saved resampler
            resampler = BilinearResampler(source_area, target_area)
            resampler.precompute(cache_dir=the_dir)
            assert resampler.resampler.load_resampling_info.call_count == 3
            assert resampler.resampler.get_bil_info.call_count == nbcalls

            resampler = BilinearResampler(source_area, target_area)
            resampler.precompute(cache_dir=the_dir)
            resampler.save_bil_info(cache_dir=the_dir)
            zarr_file = os.path.join(the_dir, "test_cache.zarr")
            # Save again faking the cache file already exists
            with mock.patch("os.path.exists") as exists:
                exists.return_value = True
                resampler.save_bil_info(cache_dir=the_dir)
            move_existing_caches.assert_called_once_with(the_dir, zarr_file)

        finally:
            shutil.rmtree(the_dir)

    def test_move_existing_caches(self):
        """Test that existing caches are moved to a subdirectory."""
        try:
            the_dir = tempfile.mkdtemp()
            # Test that existing cache file is moved away
            zarr_file = os.path.join(the_dir, "test.zarr")
            with open(zarr_file, "w") as fid:
                fid.write("42")
            from satpy.resample import _move_existing_caches
            _move_existing_caches(the_dir, zarr_file)
            assert not os.path.exists(zarr_file)
            assert os.path.exists(os.path.join(the_dir, "moved_by_satpy", "test.zarr"))
            # Run again to see that the existing dir doesn't matter
            with open(zarr_file, "w") as fid:
                fid.write("42")
            _move_existing_caches(the_dir, zarr_file)
        finally:
            shutil.rmtree(the_dir)


class TestCoordinateHelpers(unittest.TestCase):
    """Test various utility functions for working with coordinates."""

    def test_area_def_coordinates(self):
        """Test coordinates being added with an AreaDefinition."""
        from pyresample.geometry import AreaDefinition

        from satpy.resample import add_crs_xy_coords
        area_def = AreaDefinition(
            "test", "test", "test", {"proj": "lcc", "lat_1": 25, "lat_0": 25},
            100, 200, [-100, -100, 100, 100]
        )
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        assert "y" in new_data_arr.coords
        assert "x" in new_data_arr.coords

        assert "units" in new_data_arr.coords["y"].attrs
        assert new_data_arr.coords["y"].attrs["units"] == "meter"
        assert "units" in new_data_arr.coords["x"].attrs
        assert new_data_arr.coords["x"].attrs["units"] == "meter"
        assert "crs" in new_data_arr.coords
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)
        assert area_def.crs == new_data_arr.coords["crs"].item()

        # already has coords
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
            coords={"y": np.arange(2, 202), "x": np.arange(100)}
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        assert "y" in new_data_arr.coords
        assert "units" not in new_data_arr.coords["y"].attrs
        assert "x" in new_data_arr.coords
        assert "units" not in new_data_arr.coords["x"].attrs
        np.testing.assert_equal(new_data_arr.coords["y"], np.arange(2, 202))

        assert "crs" in new_data_arr.coords
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)
        assert area_def.crs == new_data_arr.coords["crs"].item()

        # lat/lon area
        area_def = AreaDefinition(
            "test", "test", "test", {"proj": "latlong"},
            100, 200, [-100, -100, 100, 100]
        )
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
        )
        new_data_arr = add_crs_xy_coords(data_arr, area_def)
        assert "y" in new_data_arr.coords
        assert "x" in new_data_arr.coords

        assert "units" in new_data_arr.coords["y"].attrs
        assert new_data_arr.coords["y"].attrs["units"] == "degrees_north"
        assert "units" in new_data_arr.coords["x"].attrs
        assert new_data_arr.coords["x"].attrs["units"] == "degrees_east"
        assert "crs" in new_data_arr.coords
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)
        assert area_def.crs == new_data_arr.coords["crs"].item()

    def test_swath_def_coordinates(self):
        """Test coordinates being added with an SwathDefinition."""
        from pyresample.geometry import SwathDefinition

        from satpy.resample import add_crs_xy_coords
        lons_data = da.random.random((200, 100), chunks=50)
        lats_data = da.random.random((200, 100), chunks=50)
        lons = xr.DataArray(lons_data, attrs={"units": "degrees_east"},
                            dims=("y", "x"))
        lats = xr.DataArray(lats_data, attrs={"units": "degrees_north"},
                            dims=("y", "x"))
        area_def = SwathDefinition(lons, lats)
        data_arr = xr.DataArray(
            da.zeros((200, 100), chunks=50),
            attrs={"area": area_def},
            dims=("y", "x"),
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

        assert "crs" in new_data_arr.coords
        crs = new_data_arr.coords["crs"].item()
        assert isinstance(crs, CRS)
        assert crs.is_geographic
        assert isinstance(new_data_arr.coords["crs"].item(), CRS)


class TestBucketAvg(unittest.TestCase):
    """Test the bucket resampler."""

    def setUp(self):
        """Create fake area definitions and resampler to be tested."""
        from satpy.resample import BucketAvg
        get_lonlats = mock.MagicMock()
        get_lonlats.return_value = (1, 2)
        get_proj_vectors = mock.MagicMock()
        get_proj_vectors.return_value = ([1, 2, 3, 4, 5],  [1, 2, 3, 4, 5])
        self.source_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.target_geo_def = mock.MagicMock(get_lonlats=get_lonlats, crs=None, get_proj_vectors=get_proj_vectors)
        self.bucket = BucketAvg(self.source_geo_def, self.target_geo_def)

    def test_init(self):
        """Test bucket resampler initialization."""
        assert self.bucket.resampler is None
        assert self.bucket.source_geo_def == self.source_geo_def
        assert self.bucket.target_geo_def == self.target_geo_def

    @mock.patch("pyresample.bucket.BucketResampler")
    def test_precompute(self, bucket):
        """Test bucket resampler precomputation."""
        bucket.return_value = True
        self.bucket.precompute()
        assert self.bucket.resampler
        bucket.assert_called_once_with(self.target_geo_def, 1, 2)

    def _compute_mocked_bucket_avg(self, data, return_data=None, **kwargs):
        """Compute the mocked bucket average."""
        self.bucket.resampler = mock.MagicMock()
        if return_data is not None:
            self.bucket.resampler.get_average.return_value = return_data
        else:
            self.bucket.resampler.get_average.return_value = data
        res = self.bucket.compute(data, **kwargs)
        return res

    def test_compute(self):
        """Test bucket resampler computation."""
        # 1D data
        data = da.ones((5,))
        res = self._compute_mocked_bucket_avg(data, fill_value=2)
        assert res.shape == (1, 5)
        # 2D data
        data = da.ones((5, 5))
        res = self._compute_mocked_bucket_avg(data, fill_value=2)
        assert res.shape == (1, 5, 5)
        # 3D data
        data = da.ones((3, 5, 5))
        self.bucket.resampler.get_average.return_value = data[0, :, :]
        res = self._compute_mocked_bucket_avg(data, return_data=data[0, :, :], fill_value=2)
        assert res.shape == (3, 5, 5)

    def test_compute_and_use_skipna_handling(self):
        """Test bucket resampler computation and use skipna handling."""
        data = da.ones((5,))

        self._compute_mocked_bucket_avg(data, fill_value=2, skipna=False)
        self.bucket.resampler.get_average.assert_called_once_with(
            data,
            fill_value=2,
            skipna=False)

        self._compute_mocked_bucket_avg(data, fill_value=2)
        self.bucket.resampler.get_average.assert_called_once_with(
            data,
            fill_value=2,
            skipna=True)

    @mock.patch("pyresample.bucket.BucketResampler")
    def test_resample(self, pyresample_bucket):
        """Test bucket resamplers resample method."""
        self.bucket.resampler = mock.MagicMock()
        self.bucket.precompute = mock.MagicMock()
        self.bucket.compute = mock.MagicMock()

        # 1D input data
        data = xr.DataArray(da.ones((5,)), dims=("foo"), attrs={"bar": "baz"})
        self.bucket.compute.return_value = da.ones((5, 5))
        res = self.bucket.resample(data)
        self.bucket.precompute.assert_called_once()
        self.bucket.compute.assert_called_once()
        assert res.shape == (5, 5)
        assert res.dims == ("y", "x")
        assert "bar" in res.attrs
        assert res.attrs["bar"] == "baz"

        # 2D input data
        data = xr.DataArray(da.ones((5, 5)), dims=("foo", "bar"))
        self.bucket.compute.return_value = da.ones((5, 5))
        res = self.bucket.resample(data)
        assert res.shape == (5, 5)
        assert res.dims == ("y", "x")

        # 3D input data with 'bands' dim
        data = xr.DataArray(da.ones((1, 5, 5)), dims=("bands", "foo", "bar"),
                            coords={"bands": ["L"]})
        self.bucket.compute.return_value = da.ones((1, 5, 5))
        res = self.bucket.resample(data)
        assert res.shape == (1, 5, 5)
        assert res.dims == ("bands", "y", "x")
        assert res.coords["bands"] == ["L"]

        # 3D input data with misc dim names
        data = xr.DataArray(da.ones((3, 5, 5)), dims=("foo", "bar", "baz"))
        self.bucket.compute.return_value = da.ones((3, 5, 5))
        res = self.bucket.resample(data)
        assert res.shape == (3, 5, 5)
        assert res.dims == ("foo", "bar", "baz")


class TestBucketSum(unittest.TestCase):
    """Test the sum bucket resampler."""

    def setUp(self):
        """Create fake area definitions and resampler to be tested."""
        from satpy.resample import BucketSum
        get_lonlats = mock.MagicMock()
        get_lonlats.return_value = (1, 2)
        self.source_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.target_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.bucket = BucketSum(self.source_geo_def, self.target_geo_def)

    def _compute_mocked_bucket_sum(self, data, return_data=None, **kwargs):
        """Compute the mocked bucket sum."""
        self.bucket.resampler = mock.MagicMock()
        if return_data is not None:
            self.bucket.resampler.get_sum.return_value = return_data
        else:
            self.bucket.resampler.get_sum.return_value = data
        res = self.bucket.compute(data, **kwargs)
        return res

    def test_compute(self):
        """Test sum bucket resampler computation."""
        # 1D data
        data = da.ones((5,))
        res = self._compute_mocked_bucket_sum(data)
        assert res.shape == (1, 5)
        # 2D data
        data = da.ones((5, 5))
        res = self._compute_mocked_bucket_sum(data)
        assert res.shape == (1, 5, 5)
        # 3D data
        data = da.ones((3, 5, 5))
        res = self._compute_mocked_bucket_sum(data, return_data=data[0, :, :])
        assert res.shape == (3, 5, 5)

    def test_compute_and_use_skipna_handling(self):
        """Test bucket resampler computation and use skipna handling."""
        data = da.ones((5,))

        self._compute_mocked_bucket_sum(data, skipna=False)
        self.bucket.resampler.get_sum.assert_called_once_with(
            data,
            skipna=False)

        self._compute_mocked_bucket_sum(data)
        self.bucket.resampler.get_sum.assert_called_once_with(
            data,
            skipna=True)


class TestBucketCount(unittest.TestCase):
    """Test the count bucket resampler."""

    def setUp(self):
        """Create fake area definitions and resampler to be tested."""
        from satpy.resample import BucketCount
        get_lonlats = mock.MagicMock()
        get_lonlats.return_value = (1, 2)
        self.source_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.target_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.bucket = BucketCount(self.source_geo_def, self.target_geo_def)

    def _compute_mocked_bucket_count(self, data, return_data=None, **kwargs):
        """Compute the mocked bucket count."""
        self.bucket.resampler = mock.MagicMock()
        if return_data is not None:
            self.bucket.resampler.get_count.return_value = return_data
        else:
            self.bucket.resampler.get_count.return_value = data
        res = self.bucket.compute(data, **kwargs)
        return res

    def test_compute(self):
        """Test count bucket resampler computation."""
        # 1D data
        data = da.ones((5,))
        res = self._compute_mocked_bucket_count(data)
        self.bucket.resampler.get_count.assert_called_once_with()
        assert res.shape == (1, 5)
        # 2D data
        data = da.ones((5, 5))
        res = self._compute_mocked_bucket_count(data)
        self.bucket.resampler.get_count.assert_called_once_with()
        assert res.shape == (1, 5, 5)
        # 3D data
        data = da.ones((3, 5, 5))
        res = self._compute_mocked_bucket_count(data, return_data=data[0, :, :])
        assert res.shape == (3, 5, 5)


class TestBucketFraction(unittest.TestCase):
    """Test the fraction bucket resampler."""

    def setUp(self):
        """Create fake area definitions and resampler to be tested."""
        from satpy.resample import BucketFraction
        get_lonlats = mock.MagicMock()
        get_lonlats.return_value = (1, 2)
        get_proj_vectors = mock.MagicMock()
        get_proj_vectors.return_value = ([1, 2, 3, 4, 5],  [1, 2, 3, 4, 5])
        self.source_geo_def = mock.MagicMock(get_lonlats=get_lonlats)
        self.target_geo_def = mock.MagicMock(get_lonlats=get_lonlats, crs=None, get_proj_vectors=get_proj_vectors)
        self.bucket = BucketFraction(self.source_geo_def, self.target_geo_def)

    def test_compute(self):
        """Test fraction bucket resampler computation."""
        self.bucket.resampler = mock.MagicMock()
        data = da.ones((3, 3))

        # No kwargs given
        _ = self.bucket.compute(data)
        self.bucket.resampler.get_fractions.assert_called_with(
            data,
            categories=None,
            fill_value=np.nan)
        # Custom kwargs
        _ = self.bucket.compute(data, categories=[1, 2], fill_value=0)
        self.bucket.resampler.get_fractions.assert_called_with(
            data,
            categories=[1, 2],
            fill_value=0)

        # Too many dimensions
        data = da.ones((3, 5, 5))
        with pytest.raises(ValueError, match="BucketFraction not implemented for 3D datasets"):
            _ = self.bucket.compute(data)

    @mock.patch("pyresample.bucket.BucketResampler")
    def test_resample(self, pyresample_bucket):
        """Test fraction bucket resamplers resample method."""
        self.bucket.resampler = mock.MagicMock()
        self.bucket.precompute = mock.MagicMock()
        self.bucket.compute = mock.MagicMock()

        # Fractions return a dict
        data = xr.DataArray(da.ones((1, 5, 5)), dims=("bands", "y", "x"))
        arr = da.ones((5, 5))
        self.bucket.compute.return_value = {0: arr, 1: arr, 2: arr}
        res = self.bucket.resample(data)
        assert "categories" in res.coords
        assert "categories" in res.dims
        assert np.all(res.coords["categories"] == np.array([0, 1, 2]))
