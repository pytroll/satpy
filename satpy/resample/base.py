"""Base resampling functionality."""
import hashlib
import json
import warnings
from contextlib import suppress
from importlib import import_module
from logging import getLogger
from weakref import WeakValueDictionary

import numpy as np

from satpy.dataset import DataID
from satpy.utils import get_legacy_chunk_size

LOG = getLogger(__name__)

CHUNK_SIZE = get_legacy_chunk_size()

resamplers_cache: "WeakValueDictionary[tuple, object]" = WeakValueDictionary()


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
    same resampler instance. Ancillary variables are resampled once (memoized by
    :class:`~satpy.dataset.dataid.DataID`) and re-attached to every resampled
    parent. Datasets without an ``area`` attribute are returned unchanged.

    Args:
        destination_area: The area to resample all datasets to.
        reduce_data: Slice source data to the part covering the destination
            area before resampling (default: True).
        resample_coords: Also resample coordinates that share all of the
            dataset's dimensions (for example a per-pixel ``time`` coordinate)
            and attach them to the resampled dataset. If False (default) such
            coordinates are dropped.
        resample_kwargs: Keyword arguments passed to :func:`prepare_resampler`
            and :func:`resample_dataset`, for example ``resampler="nearest"``.

    """

    def __init__(self, destination_area, reduce_data=True, resample_coords=False, **resample_kwargs):
        """Set up caches for a resampling operation to *destination_area*."""
        self.destination_area = destination_area
        self.reduce_data = reduce_data
        self.resample_coords = resample_coords
        self.resample_kwargs = resample_kwargs
        # source_area -> ((slice_x, slice_y), reduced_area)
        self._reductions = {}
        # source_area -> (resamplers_cache key, resampler instance)
        self._resamplers = {}
        # DataID -> resampled DataArray
        self._resampled = {}

    @property
    def resamplers(self):
        """Resampler instances created so far, keyed by their ``resamplers_cache`` key."""
        return dict(self._resamplers.values())

    def resample(self, dataset):
        """Resample *dataset* and its ancillary variables.

        Results are memoized by DataID, so a dataset (typically an ancillary
        variable) that is encountered multiple times is only resampled once and
        the same resampled object is returned each time.
        """
        ds_id = DataID.from_dataarray(dataset)
        try:
            return self._resampled[ds_id]
        except KeyError:
            pass
        if dataset.attrs.get("area") is None:
            return dataset

        LOG.debug("Resampling %s", ds_id)
        res = self._reduce_and_resample(dataset)
        if self.resample_coords:
            self._resample_coords(dataset, res)

        anc_vars = dataset.attrs.get("ancillary_variables")
        if anc_vars:
            # new list on the (already copied) attrs so the source dataset is untouched
            res.attrs["ancillary_variables"] = [self._resample_ancillary(anc) for anc in anc_vars]
        self._resampled[ds_id] = res
        return res

    def _reduce_and_resample(self, dataset):
        """Reduce *dataset* to the destination area and resample it (no memoization)."""
        reduced, source_area = self._reduce_data(dataset)
        kwargs = self.resample_kwargs.copy()
        kwargs["resampler"] = self._get_resampler(source_area)
        return resample_dataset(reduced, self.destination_area, **kwargs)

    def _resample_coords(self, orig_dataset, res):
        """Resample the coordinates of *orig_dataset* that span all of its dims and attach them to *res*."""
        for coord_name, coord in orig_dataset.coords.items():
            if coord.dims != orig_dataset.dims:
                continue
            LOG.debug("Resampling coordinate %s", coord_name)
            # shallow copy so the source dataset's coordinate attrs are untouched
            coord = coord.copy(deep=False)
            coord.attrs["area"] = orig_dataset.attrs["area"]
            res.coords[coord_name] = self._reduce_and_resample(coord)

    def _resample_ancillary(self, anc):
        if not hasattr(anc, "attrs"):
            return anc
        return self.resample(anc)

    def _get_resampler(self, source_area):
        """Get the resampler for *source_area*, creating it on first use."""
        try:
            return self._resamplers[source_area][1]
        except KeyError:
            key, resampler = prepare_resampler(source_area, self.destination_area, **self.resample_kwargs)
            self._resamplers[source_area] = (key, resampler)
            return resampler

    def _reduce_data(self, dataset):
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

    def _get_reduction(self, source_area):
        """Get the slices and reduced area for *source_area*, reusing previous results."""
        try:
            return self._reductions[source_area]
        except KeyError:
            pass

        if self.resample_kwargs.get("resampler") == "gradient_search":
            factor = self.resample_kwargs.get("shape_divisible_by", 2)
        else:
            factor = None
        try:
            slice_x, slice_y = source_area.get_area_slices(self.destination_area, shape_divisible_by=factor)
        except TypeError:
            # BaseDefinition (e.g. SwathDefinition) does not accept shape_divisible_by
            # and only raises NotImplementedError when called without it
            slice_x, slice_y = source_area.get_area_slices(self.destination_area)

        reduction = (slice_x, slice_y), source_area[slice_y, slice_x]
        self._reductions[source_area] = reduction
        return reduction

    @staticmethod
    def _slice_data(dataset, slices, reduced_area):
        """Slice the data to reduce it."""
        slice_x, slice_y = slices
        dataset = dataset.isel(x=slice_x, y=slice_y)
        if ("x", reduced_area.width) not in dataset.sizes.items():
            raise RuntimeError
        if ("y", reduced_area.height) not in dataset.sizes.items():
            raise RuntimeError
        dataset.attrs["area"] = reduced_area
        return dataset
