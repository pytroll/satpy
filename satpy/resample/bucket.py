"""Bucket resamplers."""

from logging import getLogger

import dask.array as da
import numpy as np
import xarray as xr
from pyresample.resampler import BaseResampler as PRBaseResampler

from satpy.resample.base import _update_resampled_coords
from satpy.utils import get_legacy_chunk_size

LOG = getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()


class BucketResamplerBase(PRBaseResampler):
    """Base class for bucket resampling which implements averaging."""

    def __init__(self, source_geo_def, target_geo_def):
        """Initialize bucket resampler."""
        super(BucketResamplerBase, self).__init__(source_geo_def, target_geo_def)
        self.resampler = None

    def precompute(self, **kwargs):
        """Create X and Y indices and store them for later use."""
        from pyresample import bucket

        LOG.debug("Initializing bucket resampler.")
        source_lons, source_lats = self.source_geo_def.get_lonlats(
            chunks=CHUNK_SIZE)
        self.resampler = bucket.BucketResampler(self.target_geo_def,
                                                source_lons,
                                                source_lats)

    def compute(self, data, **kwargs):
        """Call the resampling."""
        raise NotImplementedError("Use the sub-classes")

    def resample(self, data, **kwargs):  # noqa: D417
        """Resample `data` by calling `precompute` and `compute` methods.

        Args:
            data (xarray.DataArray): Data to be resampled

        Returns (xarray.DataArray): Data resampled to the target area

        """
        self.precompute(**kwargs)
        attrs = data.attrs.copy()
        data_arr = data.data
        dims = _get_dims(data)
        LOG.debug("Resampling %s", str(data.attrs.get("_satpy_id", "unknown")))
        result = self.compute(data_arr, **kwargs)
        coords, result, dims = _check_coords_results_dims(result, data, dims)

        self._adjust_attrs(attrs)

        result = xr.DataArray(result, dims=dims, coords=coords,
                              attrs=attrs)

        return _update_resampled_coords(data, result, self.target_geo_def)

    def _adjust_attrs(self, attrs):
        # Adjust some attributes
        if "BucketFraction" in str(self):
            attrs["units"] = ""
            attrs["calibration"] = ""
            attrs["standard_name"] = "area_fraction"
        elif "BucketCount" in str(self):
            attrs["units"] = ""
            attrs["calibration"] = ""
            attrs["standard_name"] = "number_of_observations"


def _get_dims(data):
    if data.ndim == 3 and data.dims[0] == "bands":
        dims = ("bands", "y", "x")
    # Both one and two dimensional input data results in 2D output
    elif data.ndim in (1, 2):
        dims = ("y", "x")
    else:
        dims = data.dims
    return dims


def _check_coords_results_dims(result, data, dims):
    coords = {}
    if "bands" in data.coords:
        coords["bands"] = data.coords["bands"]
    # Fractions are returned in a dict
    elif isinstance(result, dict):
        coords["categories"] = sorted(result.keys())
        dims = ("categories", "y", "x")
        new_result = []
        for cat in coords["categories"]:
            new_result.append(result[cat])
        result = da.stack(new_result)
    if result.ndim > len(dims):
        result = da.squeeze(result)

    return coords, result, dims


class BucketAvg(BucketResamplerBase):
    """Class for averaging bucket resampling.

    Bucket resampling calculates the average of all the values that
    are closest to each bin and inside the target area.

    Parameters
    ----------
    fill_value : (float) default: `np.nan`
        Fill value to mark missing/invalid values in the input data,
        as well as in the binned and averaged output data.
    skipna : (bool) default: `True`
        If True, skips missing values (as marked by NaN or `fill_value`) for the average calculation
        (similarly to Numpy's `nanmean`). Buckets containing only missing values are set to fill_value.
        If False, sets the bucket to fill_value if one or more missing values are present in the bucket
        (similarly to Numpy's `mean`).
        In both cases, empty buckets are set to `fill_value`.

    """

    def compute(self, data, fill_value=np.nan, skipna=True, **kwargs):  # noqa: D417
        """Call the resampling.

        Args:
            data (numpy.ndarray | dask.array.Array): Data to be resampled
            fill_value (float | int): fill_value. Defaults to numpy.nan
            skipna (bool): Skip NA's. Default `True`

        Returns:
            dask.array.Array
        """
        results = []
        if data.ndim == 3:
            for i in range(data.shape[0]):
                res = self.resampler.get_average(data[i, :, :],
                                                 fill_value=fill_value,
                                                 skipna=skipna,
                                                 **kwargs)
                results.append(res)
        else:
            res = self.resampler.get_average(data, fill_value=fill_value, skipna=skipna,
                                             **kwargs)
            results.append(res)

        return da.stack(results)


class BucketSum(BucketResamplerBase):
    """Class for bucket resampling which implements accumulation (sum).

    This resampler calculates the cumulative sum of all the values
    that are closest to each bin and inside the target area.

    Parameters
    ----------
    fill_value : (float) default: `np.nan`
        Fill value for missing data
    skipna : (bool) default: `True`
        If True, skips NaN values for the sum calculation
        (similarly to Numpy's `nansum`). Buckets containing only NaN are set to zero.
        If False, sets the bucket to NaN if one or more NaN values are present in the bucket
        (similarly to Numpy's `sum`).
        In both cases, empty buckets are set to 0.

    """

    def compute(self, data, skipna=True, **kwargs):
        """Call the resampling."""
        results = []
        if data.ndim == 3:
            for i in range(data.shape[0]):
                res = self.resampler.get_sum(data[i, :, :], skipna=skipna,
                                             **kwargs)
                results.append(res)
        else:
            res = self.resampler.get_sum(data, skipna=skipna, **kwargs)
            results.append(res)

        return da.stack(results)


class BucketCount(BucketResamplerBase):
    """Class for bucket resampling which implements hit-counting.

    This resampler calculates the number of occurences of the input
    data closest to each bin and inside the target area.

    """

    def compute(self, data, **kwargs):
        """Call the resampling."""
        results = []
        if data.ndim == 3:
            for _i in range(data.shape[0]):
                res = self.resampler.get_count()
                results.append(res)
        else:
            res = self.resampler.get_count()
            results.append(res)

        return da.stack(results)


class BucketFraction(BucketResamplerBase):
    """Class for bucket resampling to compute category fractions.

    This resampler calculates the fraction of occurences of the input
    data per category.

    """

    def compute(self, data, fill_value=np.nan, categories=None, **kwargs):
        """Call the resampling."""
        if data.ndim > 2:
            raise ValueError("BucketFraction not implemented for 3D datasets")

        result = self.resampler.get_fractions(data, categories=categories,
                                              fill_value=fill_value)

        return result


def get_resampler_classes():
    """Get bucket resampler classes."""
    return {
        "bucket_avg": BucketAvg,
        "bucket_sum": BucketSum,
        "bucket_count": BucketCount,
        "bucket_fraction": BucketFraction
    }
