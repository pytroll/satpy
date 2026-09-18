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

from satpy.resample.native import NativeResampler
from satpy.utils import PerformanceWarning


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

        from satpy.resample.base import resample_dataset
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

    @mock.patch("satpy.resample.kdtree.xr.Dataset")
    @mock.patch("satpy.resample.kdtree.zarr.open")
    @mock.patch("satpy.resample.kdtree.KDTreeResampler._create_cache_filename")
    @mock.patch("pyresample.kd_tree.XArrayResamplerNN")
    def test_kd_resampling(self, xr_resampler, create_filename, zarr_open,
                           xr_dset):
        """Test the kd resampler."""
        from satpy.resample.kdtree import KDTreeResampler
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

    def test_reduce_first_dim_rechunk_second_dim_good(self):
        """Test inconsistent chunks are handled consistently."""
        # See https://github.com/pytroll/satpy/issues/3304
        row_chunks = (6, 6, 7, 6, 6, 7, 8, 6, 4) * 5
        col_chunks = (280,)
        d_arr = da.zeros((280, 280), chunks=(row_chunks, col_chunks))
        text = "Array chunk size is not divisible by aggregation factor. Re-chunking to continue native resampling."
        with pytest.warns(PerformanceWarning, match=text):
            new_data = NativeResampler._expand_reduce(d_arr, {0: 0.5, 1: 0.5})
        assert new_data.shape == (140, 140)


class TestBilinearResampler(unittest.TestCase):
    """Test the bilinear resampler."""

    @mock.patch("satpy.resample.kdtree._move_existing_caches")
    @mock.patch("satpy.resample.kdtree.BilinearResampler._create_cache_filename")
    @mock.patch("pyresample.bilinear.XArrayBilinearResampler")
    def test_bil_resampling(self, xr_resampler, create_filename,
                            move_existing_caches):
        """Test the bilinear resampler."""
        from satpy.resample.kdtree import BilinearResampler
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
            from satpy.resample.kdtree import _move_existing_caches
            _move_existing_caches(the_dir, zarr_file)
            assert not os.path.exists(zarr_file)
            assert os.path.exists(os.path.join(the_dir, "moved_by_satpy", "test.zarr"))
            # Run again to see that the existing dir doesn't matter
            with open(zarr_file, "w") as fid:
                fid.write("42")
            _move_existing_caches(the_dir, zarr_file)
        finally:
            shutil.rmtree(the_dir)


class TestBucketAvg(unittest.TestCase):
    """Test the bucket resampler."""

    def setUp(self):
        """Create fake area definitions and resampler to be tested."""
        from satpy.resample.bucket import BucketAvg
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
        from satpy.resample.bucket import BucketSum
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
        from satpy.resample.bucket import BucketCount
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
        from satpy.resample.bucket import BucketFraction
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


PROJ_STR = "+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs"


def _fake_resample_dataset(dataset, destination_area, **kwargs):
    """Pretend to resample by only shallow-copying attrs, the way :func:`resample_dataset` does."""
    res = dataset.copy(deep=False)
    res.attrs = dataset.attrs.copy()
    res.attrs["area"] = destination_area
    return res


def _make_data_array(name, area, anc_vars=None):
    attrs = {"name": name}
    if area is not None:
        attrs["area"] = area
    if anc_vars is not None:
        attrs["ancillary_variables"] = anc_vars
    if area is None:
        return xr.DataArray(da.arange(5, dtype=np.float32), dims=("y",), attrs=attrs)
    shape = area.shape
    return xr.DataArray(da.arange(np.prod(shape), dtype=np.float32).reshape(shape), dims=("y", "x"), attrs=attrs)


class TestDatasetResampler:
    """Test the DatasetResampler helper that Scene.resample builds on."""

    @pytest.fixture
    def src_area(self):
        """Get a 10x10 source area."""
        from pyresample.geometry import AreaDefinition
        return AreaDefinition("src", "src", "src", PROJ_STR, 10, 10, (-1000., -1500., 1000., 1500.))

    @pytest.fixture
    def src_area2(self):
        """Get a second, distinct, 5x5 source area."""
        from pyresample.geometry import AreaDefinition
        return AreaDefinition("src2", "src2", "src2", PROJ_STR, 5, 5, (-1000., -1500., 1000., 1500.))

    @pytest.fixture
    def dst_area(self):
        """Get a destination area covering the lower-left quadrant of the source areas."""
        from pyresample.geometry import AreaDefinition
        return AreaDefinition("dst", "dst", "dst", PROJ_STR, 4, 4, (-1000., -1500., 0., 0.))

    @pytest.fixture
    def resample_dataset(self):
        """Replace resample_dataset with a fake that does no real resampling."""
        with mock.patch("satpy.resample.base.resample_dataset", side_effect=_fake_resample_dataset) as rs:
            yield rs

    def test_reduction_cached_per_source_area(self, src_area, src_area2, dst_area, resample_dataset):
        """Test that area slicing is computed once per source area and reused for other datasets."""
        from satpy.resample.base import DatasetResampler

        ds_resampler = DatasetResampler(dst_area)
        with mock.patch.object(src_area, "get_area_slices", wraps=src_area.get_area_slices) as gas, \
                mock.patch.object(src_area2, "get_area_slices", wraps=src_area2.get_area_slices) as gas2, \
                mock.patch.object(DatasetResampler, "_slice_data", wraps=DatasetResampler._slice_data) as slice_data:
            res1 = ds_resampler.resample(_make_data_array("ds1", src_area))
            res2 = ds_resampler.resample(_make_data_array("ds2", src_area))
            res3 = ds_resampler.resample(_make_data_array("ds3", src_area2))

        assert gas.call_count == 1
        assert gas2.call_count == 1
        assert slice_data.call_count == 3
        # cache is keyed by the original area and the reduced area is what gets resampled
        (slices, reduced_area) = ds_resampler._reductions[src_area]
        assert reduced_area != src_area
        assert reduced_area.shape < src_area.shape
        for call, exp_reduced in zip(resample_dataset.call_args_list,
                                     [reduced_area, reduced_area, ds_resampler._reductions[src_area2][1]]):
            sent_dataset = call.args[0]
            assert sent_dataset.attrs["area"] is exp_reduced
            assert sent_dataset.shape == exp_reduced.shape
        for res in (res1, res2, res3):
            assert res.attrs["area"] is dst_area

    def test_resampler_reused_per_source_area(self, src_area, src_area2, dst_area, resample_dataset):
        """Test that one resampler is created per source area and exposed by its cache key."""
        from satpy.resample.base import DatasetResampler, prepare_resampler, resamplers_cache

        ds_resampler = DatasetResampler(dst_area, resampler="nearest")
        with mock.patch("satpy.resample.base.prepare_resampler", wraps=prepare_resampler) as prep:
            ds_resampler.resample(_make_data_array("ds1", src_area))
            ds_resampler.resample(_make_data_array("ds2", src_area))
            ds_resampler.resample(_make_data_array("ds3", src_area2))

        assert prep.call_count == 2
        resamplers = ds_resampler.resamplers
        assert len(resamplers) == 2
        for key, resampler in resamplers.items():
            assert resamplers_cache[key] is resampler
        used = [call.kwargs["resampler"] for call in resample_dataset.call_args_list]
        assert used[0] is used[1]
        assert used[2] is not used[0]
        assert set(used) == set(resamplers.values())

    def test_reduce_data_disabled(self, src_area, dst_area, resample_dataset):
        """Test that no slicing happens when data reduction is disabled."""
        from satpy.resample.base import DatasetResampler

        ds_resampler = DatasetResampler(dst_area, reduce_data=False)
        data_arr = _make_data_array("ds1", src_area)
        with mock.patch.object(src_area, "get_area_slices") as gas:
            ds_resampler.resample(data_arr)
        gas.assert_not_called()
        assert resample_dataset.call_args.args[0] is data_arr
        assert ds_resampler._reductions == {}

    def test_swath_source_not_reduced(self, dst_area, resample_dataset):
        """Test that sources that can't be sliced (swaths) are resampled without reduction."""
        from pyresample.geometry import SwathDefinition

        from satpy.resample.base import DatasetResampler

        lons = xr.DataArray(da.linspace(-100., -90., 25, dtype=np.float32).reshape(5, 5), dims=("y", "x"))
        lats = xr.DataArray(da.linspace(20., 30., 25, dtype=np.float32).reshape(5, 5), dims=("y", "x"))
        swath = SwathDefinition(lons, lats)
        data_arr = _make_data_array("ds1", swath)

        ds_resampler = DatasetResampler(dst_area)
        res = ds_resampler.resample(data_arr)

        assert resample_dataset.call_args.args[0] is data_arr
        assert res.attrs["area"] is dst_area
        assert ds_resampler._reductions == {}

    @pytest.mark.parametrize(
        ("resample_kwargs", "exp_factor"),
        [
            ({}, None),
            ({"resampler": "nearest"}, None),
            ({"resampler": "gradient_search"}, 2),
            ({"resampler": "gradient_search", "shape_divisible_by": 4}, 4),
        ]
    )
    def test_shape_divisible_by(self, src_area, dst_area, resample_dataset, resample_kwargs, exp_factor):
        """Test that reduction slices are made divisible by a factor for the gradient search resampler."""
        from satpy.resample.base import DatasetResampler

        ds_resampler = DatasetResampler(dst_area, **resample_kwargs)
        with mock.patch.object(src_area, "get_area_slices", wraps=src_area.get_area_slices) as gas, \
                mock.patch("satpy.resample.base.prepare_resampler", return_value=("key", "resampler")):
            ds_resampler.resample(_make_data_array("ds1", src_area))
        gas.assert_called_once_with(dst_area, shape_divisible_by=exp_factor)

    def test_ancillary_variables(self, src_area, dst_area, resample_dataset):
        """Test that ancillary variables are resampled once and attached to every resampled parent."""
        from satpy.resample.base import DatasetResampler

        anc = _make_data_array("anc", src_area)
        anc_no_area = _make_data_array("anc_no_area", None)
        ds1 = _make_data_array("ds1", src_area, anc_vars=[anc, anc_no_area, "not_a_data_array"])
        ds2 = _make_data_array("ds2", src_area, anc_vars=[anc])
        src_anc_list = ds1.attrs["ancillary_variables"]

        ds_resampler = DatasetResampler(dst_area)
        res1 = ds_resampler.resample(ds1)
        res2 = ds_resampler.resample(ds2)
        res_anc = ds_resampler.resample(anc)

        assert resample_dataset.call_count == 3
        new_anc = res1.attrs["ancillary_variables"][0]
        assert new_anc is not anc
        assert new_anc.attrs["area"] is dst_area
        assert res2.attrs["ancillary_variables"][0] is new_anc
        assert res_anc is new_anc
        assert res1.attrs["ancillary_variables"][1] is anc_no_area
        assert res1.attrs["ancillary_variables"][2] == "not_a_data_array"
        # the inputs must not be modified
        assert ds1.attrs["ancillary_variables"] is src_anc_list
        assert src_anc_list[0] is anc
        assert ds2.attrs["ancillary_variables"][0] is anc

    def test_no_area_dataset_passthrough(self, src_area, dst_area, resample_dataset):
        """Test that datasets without an area are returned untouched, ancillary variables included."""
        from satpy.resample.base import DatasetResampler

        anc = _make_data_array("anc", src_area)
        data_arr = _make_data_array("no_area", None, anc_vars=[anc])

        ds_resampler = DatasetResampler(dst_area)
        res = ds_resampler.resample(data_arr)

        assert res is data_arr
        assert res.attrs["ancillary_variables"][0] is anc
        resample_dataset.assert_not_called()

    def test_slice_data_shape_mismatch(self, src_area, dst_area):
        """Test that slicing data that doesn't match its area raises an error."""
        from satpy.resample.base import DatasetResampler

        slices = (slice(0, 5), slice(0, 5))
        reduced_area = src_area[slices[1], slices[0]]
        data_arr = _make_data_array("ds1", src_area)
        assert DatasetResampler._slice_data(data_arr, slices, reduced_area).shape == (5, 5)
        with pytest.raises(RuntimeError):
            DatasetResampler._slice_data(data_arr, (slice(0, 4), slice(0, 5)), reduced_area)
        with pytest.raises(RuntimeError):
            DatasetResampler._slice_data(data_arr, (slice(0, 5), slice(0, 4)), reduced_area)


@pytest.mark.parametrize("name",
                         ["KDTreeResampler",
                          "BilinearResampler",
                          "NativeResampler",
                          "BucketResamplerBase",
                          "BucketAvg",
                          "BucketSum",
                          "BucketCount",
                          "BucketFraction",
                          "resample",
                          "prepare_resampler",
                          "resample_dataset",
                          "get_area_file",
                          "get_area_def",
                          "add_xy_coords",
                          "add_crs_xy_coords",
                          ]
                         )
def test_moved_import_warns(name):
    """Test that imports done directly from satpy.resample sub-package issue a warning."""
    import satpy.resample
    with pytest.warns(UserWarning, match=".*has been moved.*"):
        _ = getattr(satpy.resample, name)
