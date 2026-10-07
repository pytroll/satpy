"""Base resampling functionality."""
from __future__ import annotations

import hashlib
import json
import typing
import warnings
from contextlib import suppress
from functools import lru_cache
from importlib import import_module
from logging import getLogger
from weakref import WeakValueDictionary

import numpy as np

from satpy.dataset import DataID
from satpy.utils import get_legacy_chunk_size

if typing.TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    import xarray as xr
    from pyresample.geometry import AreaDefinition, BaseDefinition
    from pyresample.resampler import BaseResampler as PRBaseResampler

LOG = getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

resamplers_cache: "WeakValueDictionary[tuple, object]" = WeakValueDictionary()

#: Default maximum number of entries kept in each :class:`DatasetResampler` cache.
CACHE_SIZE = 100


def _hash_dict(the_dict, the_hash=None):
    """Calculate a hash for a dictionary."""
    if the_hash is None:
        the_hash = hashlib.sha1()  # nosec
    the_hash.update(json.dumps(the_dict, sort_keys=True).encode("utf-8"))
    return the_hash


def _update_resampled_coords(old_data, new_data, new_area):
    """Add coordinate information to newly resampled DataArray.

    Args:
        old_data (xarray.DataArray): Old data before resampling.
        new_data (xarray.DataArray): New data after resampling.
        new_area (pyresample.geometry.BaseDefinition): Area definition
            for the newly resampled data.

    """
    from satpy.coords import add_crs_xy_coords

    # copy over other non-x/y coordinates
    # this *MUST* happen before we set 'crs' below otherwise any 'crs'
    # coordinate in the coordinate variables we are copying will overwrite the
    # 'crs' coordinate we just assigned to the data
    ignore_coords = ("y", "x", "crs")
    new_coords = {}
    for cname, cval in old_data.coords.items():
        # we don't want coordinates that depended on the old x/y dimensions
        has_ignored_dims = any(dim in cval.dims for dim in ignore_coords)
        if cname in ignore_coords or has_ignored_dims:
            continue
        new_coords[cname] = cval
    new_data = new_data.assign_coords(**new_coords)

    # add crs, x, and y coordinates
    new_data = add_crs_xy_coords(new_data, new_area)
    return new_data


# TODO: move these to pyresample.resampler
RESAMPLER_MODULES = [
    "satpy.resample.native",
    "satpy.resample.kdtree",
    "satpy.resample.bucket",
    "satpy.resample.ewa",
]

def _get_resampler_classes_from_module(import_path):
    with suppress(ImportError):
        mod = import_module(import_path)
        return mod.get_resampler_classes()
    return {}


def get_all_resampler_classes():
    """Get all available resampler classes."""
    resamplers = {}
    # Collect all available resampler classes
    for import_path in RESAMPLER_MODULES:
        res = _get_resampler_classes_from_module(import_path)
        resamplers.update(res)

    # Add gradient search, which is infact a factory function
    # TODO: add `get_resampler_classes()` function to pyresample.gradient
    with suppress(ImportError):
        from pyresample.gradient import create_gradient_search_resampler
        resamplers["gradient_search"] = create_gradient_search_resampler

    return resamplers


# TODO: move this to pyresample
def prepare_resampler(source_area, destination_area, resampler=None, **resample_kwargs):
    """Instantiate and return a resampler."""
    from pyresample.resampler import BaseResampler as PRBaseResampler

    if resampler is None:
        LOG.info("Using default KDTree resampler")
        resampler = "kd_tree"

    if isinstance(resampler, PRBaseResampler):
        raise ValueError("Trying to create a resampler when one already "
                         "exists.")
    if isinstance(resampler, str):
        resampler_class = get_all_resampler_classes().get(resampler, None)
        _check_resampler_class(resampler_class, resampler)
    else:
        resampler_class = resampler

    key = (resampler_class,
           source_area, destination_area,
           _hash_dict(resample_kwargs).hexdigest())
    try:
        resampler_instance = resamplers_cache[key]
    except KeyError:
        resampler_instance = resampler_class(source_area, destination_area)
        resamplers_cache[key] = resampler_instance
    return key, resampler_instance


def _check_resampler_class(resampler_class, resampler):
    if resampler_class is not None:
        return
    if resampler == "gradient_search":
        warnings.warn(
            "Gradient search resampler not available. Maybe missing `shapely`?",
            stacklevel=2
        )
    raise KeyError("Resampler '%s' not available" % resampler)


# TODO: move this to pyresample
def resample(source_area, data, destination_area,
             resampler=None, **kwargs):
    """Do the resampling."""
    from pyresample.resampler import BaseResampler as PRBaseResampler

    if not isinstance(resampler, PRBaseResampler):
        # we don't use the first argument (cache key)
        _, resampler_instance = prepare_resampler(source_area,
                                                  destination_area,
                                                  resampler)
    else:
        resampler_instance = resampler

    if isinstance(data, list):
        res = [resampler_instance.resample(ds, **kwargs) for ds in data]
    else:
        res = resampler_instance.resample(data, **kwargs)

    return res


def _get_fill_value(dataset):
    """Get the fill value of the *dataset*, defaulting to np.nan."""
    if np.issubdtype(dataset.dtype, np.integer):
        return dataset.attrs.get("_FillValue", np.nan)
    return np.nan


def resample_dataset(dataset, destination_area, **kwargs):
    """Resample *dataset* and return the resampled version.

    Args:
        dataset (xarray.DataArray): Data to be resampled.
        destination_area: The destination onto which to project the data,
          either a full blown area definition or a string corresponding to
          the name of the area as defined in the area file.
        **kwargs: The extra parameters to pass to the resampler objects.

    Returns:
        A resampled DataArray with updated ``.attrs["area"]`` field. The dtype
        of the array is preserved.

    """
    # call the projection stuff here
    try:
        source_area = dataset.attrs["area"]
    except KeyError:
        LOG.info("Cannot reproject dataset %s, missing area info",
                 dataset.attrs["name"])

        return dataset

    fill_value = kwargs.pop("fill_value", _get_fill_value(dataset))
    new_data = resample(source_area, dataset, destination_area, fill_value=fill_value, **kwargs)
    new_attrs = new_data.attrs
    new_data.attrs = dataset.attrs.copy()
    new_data.attrs.update(new_attrs)
    new_data.attrs.update(area=destination_area)

    return new_data


class DatasetResampler:
    """Resample many datasets to one destination area, reusing work between them.

    Datasets sharing a source area reuse the same data-reduction slices and the
    same resampler instance. Ancillary variables shared by datasets resampled in
    the same :meth:`resample_all` call are resampled once and re-attached to
    every resampled parent. Datasets without an ``area`` attribute are returned
    unchanged.

    Args:
        destination_area: The area to resample all datasets to.
        reduce_data: Slice source data to the part covering the destination
            area before resampling (default: True).
        cache_size: Maximum number of source areas to keep reduction slices and
            resampler instances for (default: :data:`CACHE_SIZE`). Resampler
            instances can hold large precomputed index arrays, so a long lived
            instance resampling from many source areas may want a lower value.
        resample_kwargs: Keyword arguments passed to :func:`prepare_resampler`
            and :func:`resample_dataset`, for example ``resampler="nearest"``.

    """

    def __init__(
            self,
            destination_area: BaseDefinition,
            reduce_data: bool = True,
            cache_size: int = CACHE_SIZE,
            **resample_kwargs: Any,
    ) -> None:
        """Set up caches for a resampling operation to *destination_area*."""
        self.destination_area = destination_area
        self.reduce_data = reduce_data
        self.resample_kwargs = resample_kwargs
        # source_area -> ((slice_x, slice_y), reduced_area)
        self._get_reduction = lru_cache(maxsize=cache_size)(self._get_reduction_uncached)
        # source_area -> resampler instance
        self._get_resampler = lru_cache(maxsize=cache_size)(self._get_resampler_uncached)

    def resample_all(self, datasets: Iterable[xr.DataArray]) -> list[xr.DataArray]:
        """Resample every dataset of *datasets* to the destination area.

        An ancillary variable shared by several of the datasets is resampled
        only once and the same resampled object is attached to each of its
        resampled parents. A dataset that is also an ancillary variable of
        another dataset in *datasets* is the same object in both places.
        """
        resampled: dict[DataID, xr.DataArray] = {}
        return [self._resample(dataset, resampled) for dataset in datasets]

    def resample(self, dataset: xr.DataArray) -> xr.DataArray:
        """Resample *dataset* and its ancillary variables.

        Use :meth:`resample_all` to share resampled ancillary variables between
        multiple datasets; nothing is remembered between separate calls to this
        method.
        """
        return self._resample(dataset, {})

    def _resample(self, dataset: xr.DataArray, resampled: dict[DataID, xr.DataArray]) -> xr.DataArray:
        """Resample *dataset*, reusing anything already in the *resampled* memo."""
        ds_id = DataID.from_dataarray(dataset)
        try:
            return resampled[ds_id]
        except KeyError:
            pass
        if dataset.attrs.get("area") is None:
            return dataset

        LOG.debug("Resampling %s", ds_id)
        res = self._reduce_and_resample(dataset)

        anc_vars = dataset.attrs.get("ancillary_variables")
        if anc_vars:
            # new list on the (already copied) attrs so the source dataset is untouched
            res.attrs["ancillary_variables"] = [self._resample_ancillary(anc, resampled) for anc in anc_vars]
        resampled[ds_id] = res
        return res

    def _reduce_and_resample(self, dataset: xr.DataArray) -> xr.DataArray:
        """Reduce *dataset* to the destination area and resample it (no memoization)."""
        reduced, source_area = self._reduce_data(dataset)
        kwargs = self.resample_kwargs.copy()
        kwargs["resampler"] = self._get_resampler(source_area)
        return resample_dataset(reduced, self.destination_area, **kwargs)

    def _resample_ancillary(self, anc: Any, resampled: dict[DataID, xr.DataArray]) -> Any:
        if not hasattr(anc, "attrs"):
            return anc
        return self._resample(anc, resampled)

    def _get_resampler_uncached(self, source_area: BaseDefinition) -> PRBaseResampler:
        """Create the resampler going from *source_area* to the destination area."""
        # we don't use the first argument (cache key)
        _, resampler = prepare_resampler(source_area, self.destination_area, **self.resample_kwargs)
        return resampler

    def _reduce_data(self, dataset: xr.DataArray) -> tuple[xr.DataArray, BaseDefinition]:
        """Slice *dataset* to the part of its area covering the destination area."""
        source_area = dataset.attrs["area"]
        if not self.reduce_data:
            LOG.debug("Data reduction disabled by the user")
            return dataset, source_area

        try:
            slices, reduced_area = self._get_reduction(source_area)
        except NotImplementedError:
            LOG.info("Not reducing data before resampling.")
            return dataset, source_area
        return self._slice_data(dataset, slices, reduced_area), reduced_area

    def _get_reduction_uncached(self, source_area: BaseDefinition) -> tuple[tuple[slice, slice], AreaDefinition]:
        """Compute the slices and the reduced version of *source_area*."""
        if self.resample_kwargs.get("resampler") == "gradient_search":
            factor = self.resample_kwargs.get("shape_divisible_by", 2)
        else:
            factor = None
        try:
            # only AreaDefinition accepts `shape_divisible_by`, hence the ignore
            slice_x, slice_y = source_area.get_area_slices(  # type: ignore[call-arg]
                self.destination_area, shape_divisible_by=factor)
        except TypeError:
            # BaseDefinition (e.g. SwathDefinition) does not accept shape_divisible_by
            # and only raises NotImplementedError when called without it
            slice_x, slice_y = source_area.get_area_slices(self.destination_area)

        return (slice_x, slice_y), source_area[slice_y, slice_x]

    @staticmethod
    def _slice_data(dataset: xr.DataArray, slices: tuple[slice, slice], reduced_area: AreaDefinition) -> xr.DataArray:
        """Slice the data to reduce it."""
        slice_x, slice_y = slices
        dataset = dataset.isel(x=slice_x, y=slice_y)
        if ("x", reduced_area.width) not in dataset.sizes.items():
            raise RuntimeError
        if ("y", reduced_area.height) not in dataset.sizes.items():
            raise RuntimeError
        dataset.attrs["area"] = reduced_area
        return dataset
