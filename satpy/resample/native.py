"""Native resampler."""

import warnings
from math import lcm  # type: ignore

import dask.array as da
import numpy as np
import xarray as xr
from pyresample.resampler import BaseResampler as PRBaseResampler

from satpy.resample.base import _update_resampled_coords
from satpy.utils import PerformanceWarning, get_legacy_chunk_size

CHUNK_SIZE = get_legacy_chunk_size()


class NativeResampler(PRBaseResampler):
    """Expand or reduce input datasets to be the same shape.

    If data is higher resolution (more pixels) than the destination area
    then data is averaged to match the destination resolution.

    If data is lower resolution (less pixels) than the destination area
    then data is repeated to match the destination resolution.

    This resampler does not perform any caching or masking due to the
    simplicity of the operations.

    """

    def resample(self, data, cache_dir=None, mask_area=False, **kwargs):
        """Run NativeResampler."""
        # use 'mask_area' with a default of False. It wouldn't do anything.
        return super(NativeResampler, self).resample(data,
                                                     cache_dir=cache_dir,
                                                     mask_area=mask_area,
                                                     **kwargs)

    @classmethod
    def _expand_reduce(cls, d_arr, repeats):
        """Expand reduce."""
        d_arr = _ensure_dask_array(d_arr)
        if all(x == 1 for x in repeats.values()):
            return d_arr
        if all(x >= 1 for x in repeats.values()):
            return _replicate(d_arr, repeats)
        if all(x <= 1 for x in repeats.values()):
            # reduce
            y_size = 1. / repeats[0]
            x_size = 1. / repeats[1]
            return _aggregate(d_arr, y_size, x_size)
        raise ValueError("Must either expand or reduce in both "
                         "directions")

    def compute(self, data, expand=True, **kwargs):
        """Resample data with NativeResampler."""
        if isinstance(self.target_geo_def, (list, tuple)):
            # find the highest/lowest area among the provided
            test_func = max if expand else min
            target_geo_def = test_func(self.target_geo_def,
                                       key=lambda x: x.shape)
        else:
            target_geo_def = self.target_geo_def

        # convert xarray backed with numpy array to dask array
        repeats = _get_repeats(target_geo_def, data)

        d_arr = self._expand_reduce(data.data, repeats)
        new_data = xr.DataArray(d_arr, dims=data.dims)
        return _update_resampled_coords(data, new_data, target_geo_def)


def _ensure_dask_array(d_arr):
    if not isinstance(d_arr, da.Array):
        d_arr = da.from_array(d_arr, chunks=CHUNK_SIZE)
    return d_arr


def _get_repeats(target_geo_def, data):
    y_axis, x_axis = _get_axes(data)
    out_shape = target_geo_def.shape
    in_shape = data.shape
    y_repeats = out_shape[0] / float(in_shape[y_axis])
    x_repeats = out_shape[1] / float(in_shape[x_axis])
    repeats = {axis_idx: 1. for axis_idx in range(data.ndim) if axis_idx not in [y_axis, x_axis]}
    repeats[y_axis] = y_repeats
    repeats[x_axis] = x_repeats

    return repeats


def _get_axes(data):
    if "x" not in data.dims or "y" not in data.dims:
        if data.ndim not in [2, 3]:
            raise ValueError("Can only handle 2D or 3D arrays without dimensions.")
        # assume rows is the second to last axis
        y_axis = data.ndim - 2
        x_axis = data.ndim - 1
    else:
        y_axis = data.dims.index("y")
        x_axis = data.dims.index("x")

    return y_axis, x_axis


def _aggregate(d, y_size, x_size):
    """Average every 4 elements (2x2) in a 2D array."""
    if d.ndim != 2:
        # we can't guarantee what blocks we are getting and how
        # it should be reshaped to do the averaging.
        raise ValueError("Can't aggregrate (reduce) data arrays with "
                         "more than 2 dimensions.")
    if not (x_size.is_integer() and y_size.is_integer()):
        raise ValueError("Aggregation factors are not integers")
    y_size = int(y_size)
    x_size = int(x_size)
    d = _rechunk_if_nonfactor_chunks(d, y_size, x_size)
    new_chunks = (tuple(int(x / y_size) for x in d.chunks[0]),
                  tuple(int(x / x_size) for x in d.chunks[1]))
    return da.core.map_blocks(_mean, d, y_size, x_size,
                              meta=np.array((), dtype=d.dtype),
                              dtype=d.dtype, chunks=new_chunks)


def _mean(data, y_size, x_size):
    rows, cols = data.shape
    new_shape = (int(rows / y_size), int(y_size),
                 int(cols / x_size), int(x_size))
    data_mean = np.nanmean(data.reshape(new_shape), axis=(1, 3))
    return data_mean


def _replicate(d_arr, repeats):
    """Repeat data pixels by the per-axis factors specified."""
    repeated_chunks = _get_replicated_chunk_sizes(d_arr, repeats)
    d_arr = d_arr.map_blocks(_repeat_by_factor,
                             meta=np.array((), dtype=d_arr.dtype),
                             dtype=d_arr.dtype,
                             chunks=repeated_chunks)
    return d_arr


def _get_replicated_chunk_sizes(d_arr, repeats):
    repeated_chunks = []
    for axis, axis_chunks in enumerate(d_arr.chunks):
        factor = repeats[axis]
        if not factor.is_integer():
            raise ValueError("Expand factor must be a whole number")
        repeated_chunks.append(tuple(x * int(factor) for x in axis_chunks))
    return tuple(repeated_chunks)


def _repeat_by_factor(data, block_info=None):
    if block_info is None:
        return data
    out_shape = block_info[None]["chunk-shape"]
    out_data = data
    for axis, axis_size in enumerate(out_shape):
        in_size = data.shape[axis]
        out_data = np.repeat(out_data, int(axis_size / in_size), axis=axis)
    return out_data


def _rechunk_if_nonfactor_chunks(dask_arr, y_size, x_size):
    new_chunks = list(dask_arr.chunks)
    for dim_idx, agg_size in enumerate([y_size, x_size]):
        if dask_arr.shape[dim_idx] % agg_size != 0:
            raise ValueError("Aggregation requires arrays with shapes divisible by the factor.")
        need_rechunk = _check_chunking(new_chunks, dask_arr, dim_idx, agg_size)
    if need_rechunk:
        warnings.warn(
            "Array chunk size is not divisible by aggregation factor. "
            "Re-chunking to continue native resampling.",
            PerformanceWarning,
            stacklevel=5
        )
        dask_arr = dask_arr.rechunk(tuple(new_chunks))
    return dask_arr


def _check_chunking(new_chunks, dask_arr, dim_idx, agg_size):
    need_rechunk = False
    for chunk_size in dask_arr.chunks[dim_idx]:
        if chunk_size % agg_size != 0:
            need_rechunk = True
            new_dim_chunk = lcm(chunk_size, agg_size)
            new_chunks[dim_idx] = new_dim_chunk
    return need_rechunk


def get_resampler_classes():
    """Get classes based on native resampler."""
    return {"native": NativeResampler}
