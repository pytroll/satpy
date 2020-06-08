#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2017 Satpy developers
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
"""Scene object to hold satellite data."""

import logging
import os

from satpy.composites import CompositorLoader, IncompatibleAreas
from satpy.config import get_environ_config_dir
from satpy.dataset import (DatasetID, MetadataObject, dataset_walker,
                           replace_anc, combine_metadata)
from satpy.node import DependencyTree
from satpy.readers import DatasetDict, load_readers
from satpy.resample import (resample_dataset,
                            prepare_resampler, get_area_def)
from satpy.writers import load_writer
from pyresample.geometry import AreaDefinition, BaseDefinition, SwathDefinition

import xarray as xr
from xarray import DataArray
import numpy as np

LOG = logging.getLogger(__name__)


class DelayedGeneration(KeyError):
    """Mark that a dataset can't be generated without further modification."""

    pass


class Scene(MetadataObject):
    """The Almighty Scene Class.

    Example usage::

        from satpy import Scene
        from glob import glob

        # create readers and open files
        scn = Scene(filenames=glob('/path/to/files/*'), reader='viirs_sdr')

        # load datasets from input files
        scn.load(['I01', 'I02'])

        # resample from satellite native geolocation to builtin 'eurol' Area
        new_scn = scn.resample('eurol')

        # save all resampled datasets to geotiff files in the current directory
        new_scn.save_datasets()

    """

    def __init__(self, filenames=None, reader=None, filter_parameters=None, reader_kwargs=None,
                 ppp_config_dir=None,
                 base_dir=None,
                 sensor=None,
                 start_time=None,
                 end_time=None,
                 area=None):
        """Initialize Scene with Reader and Compositor objects.

        To load data `filenames` and preferably `reader` must be specified. If `filenames` is provided without `reader`
        then the available readers will be searched for a Reader that can support the provided files. This can take
        a considerable amount of time so it is recommended that `reader` always be provided. Note without `filenames`
        the Scene is created with no Readers available requiring Datasets to be added manually::

            scn = Scene()
            scn['my_dataset'] = Dataset(my_data_array, **my_info)

        Args:
            filenames (iterable or dict): A sequence of files that will be used to load data from. A ``dict`` object
                                          should map reader names to a list of filenames for that reader.
            reader (str or list): The name of the reader to use for loading the data or a list of names.
            filter_parameters (dict): Specify loaded file filtering parameters.
                                      Shortcut for `reader_kwargs['filter_parameters']`.
            reader_kwargs (dict): Keyword arguments to pass to specific reader instances.
            ppp_config_dir (str): The directory containing the configuration files for satpy.
            base_dir (str): (DEPRECATED) The directory to search for files containing the
                            data to load. If *filenames* is also provided,
                            this is ignored.
            sensor (list or str): (DEPRECATED: Use `find_files_and_readers` function) Limit used files by provided
                                  sensors.
            area (AreaDefinition): (DEPRECATED: Use `filter_parameters`) Limit used files by geographic area.
            start_time (datetime): (DEPRECATED: Use `filter_parameters`) Limit used files by starting time.
            end_time (datetime): (DEPRECATED: Use `filter_parameters`) Limit used files by ending time.

        """
        super(Scene, self).__init__()
        if ppp_config_dir is None:
            ppp_config_dir = get_environ_config_dir()
        # Set the PPP_CONFIG_DIR in the environment in case it's used elsewhere in pytroll
        LOG.debug("Setting 'PPP_CONFIG_DIR' to '%s'", ppp_config_dir)
        os.environ["PPP_CONFIG_DIR"] = self.ppp_config_dir = ppp_config_dir

        if not filenames and (start_time or end_time or base_dir):
            import warnings
            warnings.warn(
                "Deprecated: Use " +
                "'from satpy import find_files_and_readers' to find files")
            from satpy import find_files_and_readers
            filenames = find_files_and_readers(
                start_time=start_time,
                end_time=end_time,
                base_dir=base_dir,
                reader=reader,
                sensor=sensor,
                ppp_config_dir=self.ppp_config_dir,
                reader_kwargs=reader_kwargs,
            )
        elif start_time or end_time or area:
            import warnings
            warnings.warn(
                "Deprecated: Use " +
                "'filter_parameters' to filter loaded files by 'start_time', " +
                "'end_time', or 'area'.")
            fp = filter_parameters if filter_parameters else {}
            fp.update({
                'start_time': start_time,
                'end_time': end_time,
                'area': area,
            })
            filter_parameters = fp
        if filter_parameters:
            if reader_kwargs is None:
                reader_kwargs = {}
            else:
                reader_kwargs = reader_kwargs.copy()
            reader_kwargs.setdefault('filter_parameters', {}).update(filter_parameters)

        if filenames and isinstance(filenames, str):
            raise ValueError("'filenames' must be a list of files: Scene(filenames=[filename])")

        self.readers = self.create_reader_instances(filenames=filenames,
                                                    reader=reader,
                                                    reader_kwargs=reader_kwargs)
        self.attrs.update(self._compute_metadata_from_readers())
        self.datasets = DatasetDict()
        self.cpl = CompositorLoader(self.ppp_config_dir)
        comps, mods = self.cpl.load_compositors(self.attrs['sensor'])
        self.wishlist = set()
        self.dep_tree = DependencyTree(self.readers, comps, mods)
        self.resamplers = {}

    def _ipython_key_completions_(self):
        return [x.name for x in self.datasets.keys()]

    def _compute_metadata_from_readers(self):
        """Determine pieces of metadata from the readers loaded."""
        mda = {'sensor': self._get_sensor_names()}

        # overwrite the request start/end times with actual loaded data limits
        if self.readers:
            mda['start_time'] = min(x.start_time
                                    for x in self.readers.values())
            mda['end_time'] = max(x.end_time
                                  for x in self.readers.values())
        return mda

    def _get_sensor_names(self):
        """Join the sensors from all loaded readers."""
        # if the user didn't tell us what sensors to work with, let's figure it
        # out
        if not self.attrs.get('sensor'):
            # reader finder could return multiple readers
            return set([sensor for reader_instance in self.readers.values()
                        for sensor in reader_instance.sensor_names])
        elif not isinstance(self.attrs['sensor'], (set, tuple, list)):
            return set([self.attrs['sensor']])
        else:
            return set(self.attrs['sensor'])

    def create_reader_instances(self,
                                filenames=None,
                                reader=None,
                                reader_kwargs=None):
        """Find readers and return their instances."""
        return load_readers(filenames=filenames,
                            reader=reader,
                            reader_kwargs=reader_kwargs,
                            ppp_config_dir=self.ppp_config_dir)

    @property
    def start_time(self):
        """Return the start time of the file."""
        return self.attrs['start_time']

    @property
    def end_time(self):
        """Return the end time of the file."""
        return self.attrs['end_time']

    @property
    def missing_datasets(self):
        """Set of DatasetIDs that have not been successfully loaded."""
        return set(self.wishlist) - set(self.datasets.keys())

    def _compare_areas(self, datasets=None, compare_func=max):
        """Compare areas for the provided datasets.

        Args:
            datasets (iterable): Datasets whose areas will be compared. Can
                                 be either `xarray.DataArray` objects or
                                 identifiers to get the DataArrays from the
                                 current Scene. Defaults to all datasets.
                                 This can also be a series of area objects,
                                 typically AreaDefinitions.
            compare_func (callable): `min` or `max` or other function used to
                                     compare the dataset's areas.

        """
        if datasets is None:
            datasets = list(self.values())

        areas = []
        for ds in datasets:
            if isinstance(ds, BaseDefinition):
                areas.append(ds)
                continue
            elif not isinstance(ds, DataArray):
                ds = self[ds]
            area = ds.attrs.get('area')
            areas.append(area)

        areas = [x for x in areas if x is not None]
        if not areas:
            raise ValueError("No dataset areas available")

        if not all(isinstance(x, type(areas[0]))
                   for x in areas[1:]):
            raise ValueError("Can't compare areas of different types")
        elif isinstance(areas[0], AreaDefinition):
            first_pstr = areas[0].proj_str
            if not all(ad.proj_str == first_pstr for ad in areas[1:]):
                raise ValueError("Can't compare areas with different "
                                 "projections.")

            def key_func(ds):
                return 1. / ds.pixel_size_x
        else:
            def key_func(ds):
                return ds.shape

        # find the highest/lowest area among the provided
        return compare_func(areas, key=key_func)

    def max_area(self, datasets=None):
        """Get highest resolution area for the provided datasets.

        Args:
            datasets (iterable): Datasets whose areas will be compared. Can
                                 be either `xarray.DataArray` objects or
                                 identifiers to get the DataArrays from the
                                 current Scene. Defaults to all datasets.

        """
        return self._compare_areas(datasets=datasets, compare_func=max)

    def min_area(self, datasets=None):
        """Get lowest resolution area for the provided datasets.

        Args:
            datasets (iterable): Datasets whose areas will be compared. Can
                                 be either `xarray.DataArray` objects or
                                 identifiers to get the DataArrays from the
                                 current Scene. Defaults to all datasets.

        """
        return self._compare_areas(datasets=datasets, compare_func=min)

    def available_dataset_ids(self, reader_name=None, composites=False):
        """Get DatasetIDs of loadable datasets.

        This can be for all readers loaded by this Scene or just for
        ``reader_name`` if specified.

        Available dataset names are determined by what each individual reader
        can load. This is normally determined by what files are needed to load
        a dataset and what files have been provided to the scene/reader.
        Some readers dynamically determine what is available based on the
        contents of the files provided.

        Returns: list of available dataset names

        """
        try:
            if reader_name:
                readers = [self.readers[reader_name]]
            else:
                readers = self.readers.values()
        except (AttributeError, KeyError):
            raise KeyError("No reader '%s' found in scene" % reader_name)

        available_datasets = sorted([dataset_id
                                     for reader in readers
                                     for dataset_id in reader.available_dataset_ids])
        if composites:
            available_datasets += sorted(self.available_composite_ids())
        return available_datasets

    def available_dataset_names(self, reader_name=None, composites=False):
        """Get the list of the names of the available datasets."""
        return sorted(set(x.name for x in self.available_dataset_ids(
            reader_name=reader_name, composites=composites)))

    def all_dataset_ids(self, reader_name=None, composites=False):
        """Get names of all datasets from loaded readers or `reader_name` if specified.

        Returns: list of all dataset names

        """
        try:
            if reader_name:
                readers = [self.readers[reader_name]]
            else:
                readers = self.readers.values()
        except (AttributeError, KeyError):
            raise KeyError("No reader '%s' found in scene" % reader_name)

        all_datasets = [dataset_id
                        for reader in readers
                        for dataset_id in reader.all_dataset_ids]
        if composites:
            all_datasets += self.all_composite_ids()
        return all_datasets

    def all_dataset_names(self, reader_name=None, composites=False):
        """Get all known dataset names configured for the loaded readers.

        Note that some readers dynamically determine what datasets are known
        by reading the contents of the files they are provided. This means
        that the list of datasets returned by this method may change depending
        on what files are provided even if a product/dataset is a "standard"
        product for a particular reader.

        """
        return sorted(set(x.name for x in self.all_dataset_ids(
            reader_name=reader_name, composites=composites)))

    def _check_known_composites(self, available_only=False):
        """Create new dependency tree and check what composites we know about."""
        # Note if we get compositors from the dep tree then it will include
        # modified composites which we don't want
        sensor_comps, mods = self.cpl.load_compositors(self.attrs['sensor'])
        # recreate the dependency tree so it doesn't interfere with the user's
        # wishlist from self.dep_tree
        dep_tree = DependencyTree(self.readers, sensor_comps, mods, available_only=True)
        # ignore inline compositor dependencies starting with '_'
        comps = (comp for comp_dict in sensor_comps.values()
                 for comp in comp_dict.keys() if not comp.name.startswith('_'))
        # make sure that these composites are even create-able by these readers
        all_comps = set(comps)
        # find_dependencies will update the all_comps set with DatasetIDs
        dep_tree.find_dependencies(all_comps)
        available_comps = set(x.name for x in dep_tree.trunk())
        # get rid of modified composites that are in the trunk
        return sorted(available_comps & set(all_comps))

    def available_composite_ids(self):
        """Get names of composites that can be generated from the available datasets."""
        return self._check_known_composites(available_only=True)

    def available_composite_names(self):
        """All configured composites known to this Scene."""
        return sorted(set(x.name for x in self.available_composite_ids()))

    def all_composite_ids(self):
        """Get all IDs for configured composites."""
        return self._check_known_composites()

    def all_composite_names(self):
        """Get all names for all configured composites."""
        return sorted(set(x.name for x in self.all_composite_ids()))

    def all_modifier_names(self):
        """Get names of configured modifier objects."""
        return sorted(self.dep_tree.modifiers.keys())

    def __str__(self):
        """Generate a nice print out for the scene."""
        res = (str(proj) for proj in self.datasets.values())
        return "\n".join(res)

    def __iter__(self):
        """Iterate over the datasets."""
        for x in self.datasets.values():
            yield x

    def iter_by_area(self):
        """Generate datasets grouped by Area.

        :return: generator of (area_obj, list of dataset objects)
        """
        datasets_by_area = {}
        for ds in self:
            a = ds.attrs.get('area')
            datasets_by_area.setdefault(a, []).append(
                DatasetID.from_dict(ds.attrs))

        return datasets_by_area.items()

    def keys(self, **kwargs):
        """Get DatasetID keys for the underlying data container."""
        return self.datasets.keys(**kwargs)

    def values(self):
        """Get values for the underlying data container."""
        return self.datasets.values()

    def copy(self, datasets=None):
        """Create a copy of the Scene including dependency information.

        Args:
            datasets (list, tuple): `DatasetID` objects for the datasets
                                    to include in the new Scene object.

        """
        new_scn = self.__class__()
        new_scn.attrs = self.attrs.copy()
        new_scn.dep_tree = self.dep_tree.copy()

        for ds_id in (datasets or self.keys()):
            # NOTE: Must use `.datasets` or side effects of `__setitem__`
            #       could hurt us with regards to the wishlist
            new_scn.datasets[ds_id] = self[ds_id]

        if not datasets:
            new_scn.wishlist = self.wishlist.copy()
        else:
            new_scn.wishlist = set([DatasetID.from_dict(ds.attrs)
                                    for ds in new_scn])
        return new_scn

    @property
    def all_same_area(self):
        """All contained data arrays are on the same area."""
        all_areas = [x.attrs.get('area', None) for x in self.values()]
        all_areas = [x for x in all_areas if x is not None]
        return all(all_areas[0] == x for x in all_areas[1:])

    @property
    def all_same_proj(self):
        """All contained data array are in the same projection."""
        all_areas = [x.attrs.get('area', None) for x in self.values()]
        all_areas = [x for x in all_areas if x is not None]
        return all(all_areas[0].proj_str == x.proj_str for x in all_areas[1:])

    def _slice_area_from_bbox(self, src_area, dst_area, ll_bbox=None,
                              xy_bbox=None):
        """Slice the provided area using the bounds provided."""
        if ll_bbox is not None:
            dst_area = AreaDefinition(
                'crop_area', 'crop_area', 'crop_latlong',
                {'proj': 'latlong'}, 100, 100, ll_bbox)
        elif xy_bbox is not None:
            crs = src_area.crs if hasattr(src_area, 'crs') else src_area.proj_dict
            dst_area = AreaDefinition(
                'crop_area', 'crop_area', 'crop_xy',
                crs, src_area.x_size, src_area.y_size,
                xy_bbox)
        x_slice, y_slice = src_area.get_area_slices(dst_area)
        return src_area[y_slice, x_slice], y_slice, x_slice

    def _slice_datasets(self, dataset_ids, slice_key, new_area, area_only=True):
        """Slice scene in-place for the datasets specified."""
        new_datasets = {}
        datasets = (self[ds_id] for ds_id in dataset_ids)
        for ds, parent_ds in dataset_walker(datasets):
            ds_id = DatasetID.from_dict(ds.attrs)
            # handle ancillary variables
            pres = None
            if parent_ds is not None:
                pres = new_datasets[DatasetID.from_dict(parent_ds.attrs)]
            if ds_id in new_datasets:
                replace_anc(ds, pres)
                continue
            if area_only and ds.attrs.get('area') is None:
                new_datasets[ds_id] = ds
                replace_anc(ds, pres)
                continue

            if not isinstance(slice_key, dict):
                # match dimension name to slice object
                key = dict(zip(ds.dims, slice_key))
            else:
                key = slice_key
            new_ds = ds.isel(**key)
            if new_area is not None:
                new_ds.attrs['area'] = new_area

            new_datasets[ds_id] = new_ds
            if parent_ds is None:
                # don't use `__setitem__` because we don't want this to
                # affect the existing wishlist/dep tree
                self.datasets[ds_id] = new_ds
            else:
                replace_anc(new_ds, pres)

    def slice(self, key):
        """Slice Scene by dataset index.

        .. note::

            DataArrays that do not have an ``area`` attribute will not be
            sliced.

        """
        if not self.all_same_area:
            raise RuntimeError("'Scene' has different areas and cannot "
                               "be usefully sliced.")
        # slice
        new_scn = self.copy()
        new_scn.wishlist = self.wishlist
        for area, dataset_ids in self.iter_by_area():
            if area is not None:
                # assume dimensions for area are y and x
                one_ds = self[dataset_ids[0]]
                area_key = tuple(sl for dim, sl in zip(one_ds.dims, key) if dim in ['y', 'x'])
                new_area = area[area_key]
            else:
                new_area = None
            new_scn._slice_datasets(dataset_ids, key, new_area)
        return new_scn

    def crop(self, area=None, ll_bbox=None, xy_bbox=None, dataset_ids=None):
        """Crop Scene to a specific Area boundary or bounding box.

        Args:
            area (AreaDefinition): Area to crop the current Scene to
            ll_bbox (tuple, list): 4-element tuple where values are in
                                   lon/lat degrees. Elements are
                                   ``(xmin, ymin, xmax, ymax)`` where X is
                                   longitude and Y is latitude.
            xy_bbox (tuple, list): Same as `ll_bbox` but elements are in
                                   projection units.
            dataset_ids (iterable): DatasetIDs to include in the returned
                                 `Scene`. Defaults to all datasets.

        This method will attempt to intelligently slice the data to preserve
        relationships between datasets. For example, if we are cropping two
        DataArrays of 500m and 1000m pixel resolution then this method will
        assume that exactly 4 pixels of the 500m array cover the same
        geographic area as a single 1000m pixel. It handles these cases based
        on the shapes of the input arrays and adjusting slicing indexes
        accordingly. This method will have trouble handling cases where data
        arrays seem related but don't cover the same geographic area or if the
        coarsest resolution data is not related to the other arrays which are
        related.

        It can be useful to follow cropping with a call to the native
        resampler to resolve all datasets to the same resolution and compute
        any composites that could not be generated previously::

        >>> cropped_scn = scn.crop(ll_bbox=(-105., 40., -95., 50.))
        >>> remapped_scn = cropped_scn.resample(resampler='native')

        .. note::

            The `resample` method automatically crops input data before
            resampling to save time/memory.

        """
        if len([x for x in [area, ll_bbox, xy_bbox] if x is not None]) != 1:
            raise ValueError("One and only one of 'area', 'll_bbox', "
                             "or 'xy_bbox' can be specified.")

        new_scn = self.copy(datasets=dataset_ids)
        if not new_scn.all_same_proj and xy_bbox is not None:
            raise ValueError("Can't crop when dataset_ids are not all on the "
                             "same projection.")

        # get the lowest resolution area, use it as the base of the slice
        # this makes sure that the other areas *should* be a consistent factor
        min_area = new_scn.min_area()
        if isinstance(area, str):
            area = get_area_def(area)
        new_min_area, min_y_slice, min_x_slice = self._slice_area_from_bbox(
            min_area, area, ll_bbox, xy_bbox)
        new_target_areas = {}
        for src_area, dataset_ids in new_scn.iter_by_area():
            if src_area is None:
                for ds_id in dataset_ids:
                    new_scn.datasets[ds_id] = self[ds_id]
                continue

            y_factor, y_remainder = np.divmod(float(src_area.shape[0]),
                                              min_area.shape[0])
            x_factor, x_remainder = np.divmod(float(src_area.shape[1]),
                                              min_area.shape[1])
            y_factor = int(y_factor)
            x_factor = int(x_factor)
            if y_remainder == 0 and x_remainder == 0:
                y_slice = slice(min_y_slice.start * y_factor,
                                min_y_slice.stop * y_factor)
                x_slice = slice(min_x_slice.start * x_factor,
                                min_x_slice.stop * x_factor)
                new_area = src_area[y_slice, x_slice]
                slice_key = {'y': y_slice, 'x': x_slice}
                new_scn._slice_datasets(dataset_ids, slice_key, new_area)
            else:
                new_target_areas[src_area] = self._slice_area_from_bbox(
                    src_area, area, ll_bbox, xy_bbox
                )

        return new_scn

    def aggregate(self, dataset_ids=None, boundary='exact', side='left', func='mean', **dim_kwargs):
        """Create an aggregated version of the Scene.

        Args:
            dataset_ids (iterable): DatasetIDs to include in the returned
                                    `Scene`. Defaults to all datasets.
            func (string): Function to apply on each aggregation window. One of
                           'mean', 'sum', 'min', 'max', 'median', 'argmin',
                           'argmax', 'prod', 'std', 'var'.
                           'mean' is the default.
            boundary: Not implemented.
            side: Not implemented.
            dim_kwargs: the size of the windows to aggregate.

        Returns:
            A new aggregated scene

        See also:
            xarray.DataArray.coarsen

        Example:
            `scn.aggregate(func='min', x=2, y=2)` will aggregate 2x2 pixels by
            applying the `min` function.

        """
        new_scn = self.copy(datasets=dataset_ids)

        for src_area, ds_ids in new_scn.iter_by_area():
            if src_area is None:
                for ds_id in ds_ids:
                    new_scn.datasets[ds_id] = self[ds_id]
                continue

            if boundary != 'exact':
                raise NotImplementedError("boundary modes appart from 'exact' are not implemented yet.")
            target_area = src_area.aggregate(**dim_kwargs)
            try:
                resolution = max(target_area.pixel_size_x, target_area.pixel_size_y)
            except AttributeError:
                resolution = max(target_area.lats.resolution, target_area.lons.resolution)
            for ds_id in ds_ids:
                res = self[ds_id].coarsen(boundary=boundary, side=side, **dim_kwargs)

                new_scn.datasets[ds_id] = getattr(res, func)()
                new_scn.datasets[ds_id].attrs['area'] = target_area
                new_scn.datasets[ds_id].attrs['resolution'] = resolution

        return new_scn

    def get(self, key, default=None):
        """Return value from DatasetDict with optional default."""
        return self.datasets.get(key, default)

    def __getitem__(self, key):
        """Get a dataset or create a new 'slice' of the Scene."""
        if isinstance(key, tuple) and not isinstance(key, DatasetID):
            return self.slice(key)
        return self.datasets[key]

    def __setitem__(self, key, value):
        """Add the item to the scene."""
        self.datasets[key] = value
        # this could raise a KeyError but never should in this case
        ds_id = self.datasets.get_key(key)
        self.wishlist.add(ds_id)
        self.dep_tree.add_leaf(ds_id)

    def __delitem__(self, key):
        """Remove the item from the scene."""
        k = self.datasets.get_key(key)
        self.wishlist.discard(k)
        del self.datasets[k]

    def __contains__(self, name):
        """Check if the dataset is in the scene."""
        return name in self.datasets

    def _read_datasets(self, dataset_nodes, **kwargs):
        """Read the given datasets from file."""
        # Sort requested datasets by reader
        reader_datasets = {}

        for node in dataset_nodes:
            ds_id = node.name
            # if we already have this node loaded or the node was assigned
            # by the user (node data is None) then don't try to load from a
            # reader
            if ds_id in self.datasets or not isinstance(node.data, dict):
                continue
            reader_name = node.data.get('reader_name')
            if reader_name is None:
                # This shouldn't be possible
                raise RuntimeError("Dependency tree has a corrupt node.")
            reader_datasets.setdefault(reader_name, set()).add(ds_id)

        # load all datasets for one reader at a time
        loaded_datasets = DatasetDict()
        for reader_name, ds_ids in reader_datasets.items():
            reader_instance = self.readers[reader_name]
            new_datasets = reader_instance.load(ds_ids, **kwargs)
            loaded_datasets.update(new_datasets)
        self.datasets.update(loaded_datasets)
        return loaded_datasets

    def _get_prereq_datasets(self, comp_id, prereq_nodes, keepables, skip=False):
        """Get a composite's prerequisites, generating them if needed.

        Args:
            comp_id (DatasetID): DatasetID for the composite whose
                                 prerequisites are being collected.
            prereq_nodes (sequence of Nodes): Prerequisites to collect
            keepables (set): `set` to update if any prerequisites can't
                             be loaded at this time (see
                             `_generate_composite`).
            skip (bool): If True, consider prerequisites as optional and
                         only log when they are missing. If False,
                         prerequisites are considered required and will
                         raise an exception and log a warning if they can't
                         be collected. Defaults to False.

        Raises:
            KeyError: If required (skip=False) prerequisite can't be collected.

        """
        prereq_datasets = []
        delayed_gen = False
        for prereq_node in prereq_nodes:
            prereq_id = prereq_node.name
            if prereq_id not in self.datasets and prereq_id not in keepables \
                    and not prereq_node.is_leaf:
                self._generate_composite(prereq_node, keepables)

            if prereq_node is self.dep_tree.empty_node:
                # empty sentinel node - no need to load it
                continue
            elif prereq_id in self.datasets:
                prereq_datasets.append(self.datasets[prereq_id])
            elif not prereq_node.is_leaf and prereq_id in keepables:
                delayed_gen = True
                continue
            elif not skip:
                LOG.debug("Missing prerequisite for '{}': '{}'".format(
                    comp_id, prereq_id))
                raise KeyError("Missing composite prerequisite for"
                               " '{}': '{}'".format(comp_id, prereq_id))
            else:
                LOG.debug("Missing optional prerequisite for {}: {}".format(comp_id, prereq_id))

        if delayed_gen:
            keepables.add(comp_id)
            keepables.update([x.name for x in prereq_nodes])
            LOG.debug("Delaying generation of %s because of dependency's delayed generation: %s", comp_id, prereq_id)
            if not skip:
                LOG.debug("Delayed prerequisite for '{}': '{}'".format(comp_id, prereq_id))
                raise DelayedGeneration(
                    "Delayed composite prerequisite for "
                    "'{}': '{}'".format(comp_id, prereq_id))
            else:
                LOG.debug("Delayed optional prerequisite for {}: {}".format(comp_id, prereq_id))

        return prereq_datasets

    def _generate_composite(self, comp_node, keepables):
        """Collect all composite prereqs and create the specified composite.

        Args:
            comp_node (Node): Composite Node to generate a Dataset for
            keepables (set): `set` to update if any datasets are needed
                             when generation is continued later. This can
                             happen if generation is delayed to incompatible
                             areas which would require resampling first.

        """
        if comp_node.name in self.datasets:
            # already loaded
            return
        compositor, prereqs, optional_prereqs = comp_node.data

        try:
            delayed_prereq = False
            prereq_datasets = self._get_prereq_datasets(
                comp_node.name,
                prereqs,
                keepables,
            )
        except DelayedGeneration:
            # if we are missing a required dependency that could be generated
            # later then we need to wait to return until after we've also
            # processed the optional dependencies
            delayed_prereq = True
        except KeyError:
            # we are missing a hard requirement that will never be available
            # there is no need to "keep" optional dependencies
            return

        optional_datasets = self._get_prereq_datasets(
            comp_node.name,
            optional_prereqs,
            keepables,
            skip=True
        )

        # we are missing some prerequisites
        # in the future we may be able to generate this composite (delayed)
        # so we need to hold on to successfully loaded prerequisites and
        # optional prerequisites
        if delayed_prereq:
            preservable_datasets = set(self.datasets.keys())
            prereq_ids = set(p.name for p in prereqs)
            opt_prereq_ids = set(p.name for p in optional_prereqs)
            keepables |= preservable_datasets & (prereq_ids | opt_prereq_ids)
            return

        try:
            composite = compositor(prereq_datasets,
                                   optional_datasets=optional_datasets,
                                   **self.attrs)

            cid = DatasetID.from_dict(composite.attrs)

            self.datasets[cid] = composite
            # update the node with the computed DatasetID
            if comp_node.name in self.wishlist:
                self.wishlist.remove(comp_node.name)
                self.wishlist.add(cid)
            comp_node.name = cid
        except IncompatibleAreas:
            LOG.debug("Delaying generation of %s because of incompatible areas", str(compositor.id))
            preservable_datasets = set(self.datasets.keys())
            prereq_ids = set(p.name for p in prereqs)
            opt_prereq_ids = set(p.name for p in optional_prereqs)
            keepables |= preservable_datasets & (prereq_ids | opt_prereq_ids)
            # even though it wasn't generated keep a list of what
            # might be needed in other compositors
            keepables.add(comp_node.name)
            return

    def _read_composites(self, compositor_nodes):
        """Read (generate) composites."""
        keepables = set()
        for item in compositor_nodes:
            self._generate_composite(item, keepables)
        return keepables

    def read(self, nodes=None, **kwargs):
        """Load datasets from the necessary reader.

        Args:
            nodes (iterable): DependencyTree Node objects
            **kwargs: Keyword arguments to pass to the reader's `load` method.

        Returns:
            DatasetDict of loaded datasets

        """
        if nodes is None:
            required_nodes = self.wishlist - set(self.datasets.keys())
            nodes = self.dep_tree.leaves(nodes=required_nodes)
        return self._read_datasets(nodes, **kwargs)

    def generate_composites(self, nodes=None):
        """Compute all the composites contained in `requirements`."""
        if nodes is None:
            required_nodes = self.wishlist - set(self.datasets.keys())
            nodes = set(self.dep_tree.trunk(nodes=required_nodes)) - \
                set(self.datasets.keys())
        return self._read_composites(nodes)

    def _remove_failed_datasets(self, keepables):
        keepables = keepables or set()
        # remove reader datasets that couldn't be loaded so they aren't
        # attempted again later
        for n in self.missing_datasets:
            if n not in keepables:
                self.wishlist.discard(n)

    def unload(self, keepables=None):
        """Unload all unneeded datasets.

        Datasets are considered unneeded if they weren't directly requested
        or added to the Scene by the user or they are no longer needed to
        generate composites that have yet to be generated.

        Args:
            keepables (iterable): DatasetIDs to keep whether they are needed
                                  or not.

        """
        to_del = [ds_id for ds_id, projectable in self.datasets.items()
                  if ds_id not in self.wishlist and (not keepables or ds_id
                                                     not in keepables)]
        for ds_id in to_del:
            LOG.debug("Unloading dataset: %r", ds_id)
            del self.datasets[ds_id]

    def load(self, wishlist, calibration=None, resolution=None,
             polarization=None, level=None, generate=True, unload=True,
             **kwargs):
        """Read and generate requested datasets.

        When the `wishlist` contains `DatasetID` objects they can either be
        fully-specified `DatasetID` objects with every parameter specified
        or they can not provide certain parameters and the "best" parameter
        will be chosen. For example, if a dataset is available in multiple
        resolutions and no resolution is specified in the wishlist's DatasetID
        then the highest (smallest number) resolution will be chosen.

        Loaded `DataArray` objects are created and stored in the Scene object.

        Args:
            wishlist (iterable): List of names (str), wavelengths (float), or
                                 DatasetID objects of the requested datasets
                                 to load. See `available_dataset_ids()` for
                                 what datasets are available.
            calibration (list, str): Calibration levels to limit available
                                     datasets. This is a shortcut to
                                     having to list each DatasetID in
                                     `wishlist`.
            resolution (list | float): Resolution to limit available datasets.
                                       This is a shortcut similar to
                                       calibration.
            polarization (list | str): Polarization ('V', 'H') to limit
                                       available datasets. This is a shortcut
                                       similar to calibration.
            level (list | str): Pressure level to limit available datasets.
                                Pressure should be in hPa or mb. If an
                                altitude is used it should be specified in
                                inverse meters (1/m). The units of this
                                parameter ultimately depend on the reader.
            generate (bool): Generate composites from the loaded datasets
                             (default: True)
            unload (bool): Unload datasets that were required to generate
                           the requested datasets (composite dependencies)
                           but are no longer needed.

        """
        if isinstance(wishlist, str):
            raise TypeError("'load' expects a list of datasets, got a string.")
        dataset_keys = set(wishlist)
        needed_datasets = (self.wishlist | dataset_keys) - \
            set(self.datasets.keys())
        unknown = self.dep_tree.find_dependencies(needed_datasets,
                                                  calibration=calibration,
                                                  polarization=polarization,
                                                  resolution=resolution,
                                                  level=level)
        self.wishlist |= needed_datasets
        if unknown:
            unknown_str = ", ".join(map(str, unknown))
            raise KeyError("Unknown datasets: {}".format(unknown_str))

        self.read(**kwargs)
        if generate:
            keepables = self.generate_composites()
        else:
            # don't lose datasets we loaded to try to generate composites
            keepables = set(self.datasets.keys()) | self.wishlist
        if self.missing_datasets:
            # copy the set of missing datasets because they won't be valid
            # after they are removed in the next line
            missing = self.missing_datasets.copy()
            self._remove_failed_datasets(keepables)
            missing_str = ", ".join(str(x) for x in missing)
            LOG.warning("The following datasets were not created and may require "
                        "resampling to be generated: {}".format(missing_str))
        if unload:
            self.unload(keepables=keepables)

    def _slice_data(self, source_area, slices, dataset):
        """Slice the data to reduce it."""
        slice_x, slice_y = slices
        dataset = dataset.isel(x=slice_x, y=slice_y)
        assert ('x', source_area.x_size) in dataset.sizes.items()
        assert ('y', source_area.y_size) in dataset.sizes.items()
        dataset.attrs['area'] = source_area

        return dataset

    def _resampled_scene(self, new_scn, destination_area, reduce_data=True,
                         **resample_kwargs):
        """Resample `datasets` to the `destination` area.

        If data reduction is enabled, some local caching is perfomed in order to
        avoid recomputation of area intersections.
        """
        new_datasets = {}
        datasets = list(new_scn.datasets.values())
        if isinstance(destination_area, str):
            destination_area = get_area_def(destination_area)
        if hasattr(destination_area, 'freeze'):
            try:
                max_area = new_scn.max_area()
                destination_area = destination_area.freeze(max_area)
            except ValueError:
                raise ValueError("No dataset areas available to freeze "
                                 "DynamicAreaDefinition.")

        resamplers = {}
        reductions = {}
        for dataset, parent_dataset in dataset_walker(datasets):
            ds_id = DatasetID.from_dict(dataset.attrs)
            pres = None
            if parent_dataset is not None:
                pres = new_datasets[DatasetID.from_dict(parent_dataset.attrs)]
            if ds_id in new_datasets:
                replace_anc(new_datasets[ds_id], pres)
                if ds_id in new_scn.datasets:
                    new_scn.datasets[ds_id] = new_datasets[ds_id]
                continue
            if dataset.attrs.get('area') is None:
                if parent_dataset is None:
                    new_scn.datasets[ds_id] = dataset
                else:
                    replace_anc(dataset, pres)
                continue
            LOG.debug("Resampling %s", ds_id)
            source_area = dataset.attrs['area']
            try:
                if reduce_data:
                    key = source_area
                    try:
                        (slice_x, slice_y), source_area = reductions[key]
                    except KeyError:
                        if resample_kwargs.get('resampler') == 'gradient_search':
                            factor = resample_kwargs.get('shape_divisible_by', 2)
                        else:
                            factor = None
                        try:
                            slice_x, slice_y = source_area.get_area_slices(
                                destination_area, shape_divisible_by=factor)
                        except TypeError:
                            slice_x, slice_y = source_area.get_area_slices(
                                destination_area)
                        source_area = source_area[slice_y, slice_x]
                        reductions[key] = (slice_x, slice_y), source_area
                    dataset = self._slice_data(source_area, (slice_x, slice_y), dataset)
                else:
                    LOG.debug("Data reduction disabled by the user")
            except NotImplementedError:
                LOG.info("Not reducing data before resampling.")
            if source_area not in resamplers:
                key, resampler = prepare_resampler(
                    source_area, destination_area, **resample_kwargs)
                resamplers[source_area] = resampler
                self.resamplers[key] = resampler
            kwargs = resample_kwargs.copy()
            kwargs['resampler'] = resamplers[source_area]
            res = resample_dataset(dataset, destination_area, **kwargs)
            new_datasets[ds_id] = res
            if ds_id in new_scn.datasets:
                new_scn.datasets[ds_id] = res
            if parent_dataset is not None:
                replace_anc(res, pres)

    def resample(self, destination=None, datasets=None, generate=True,
                 unload=True, resampler=None, reduce_data=True,
                 **resample_kwargs):
        """Resample datasets and return a new scene.

        Args:
            destination (AreaDefinition, GridDefinition): area definition to
                resample to. If not specified then the area returned by
                `Scene.max_area()` will be used.
            datasets (list): Limit datasets to resample to these specified
                `DatasetID` objects . By default all currently loaded
                datasets are resampled.
            generate (bool): Generate any requested composites that could not
                be previously due to incompatible areas (default: True).
            unload (bool): Remove any datasets no longer needed after
                requested composites have been generated (default: True).
            resampler (str): Name of resampling method to use. By default,
                this is a nearest neighbor KDTree-based resampling
                ('nearest'). Other possible values include 'native', 'ewa',
                etc. See the :mod:`~satpy.resample` documentation for more
                information.
            reduce_data (bool): Reduce data by matching the input and output
                areas and slicing the data arrays (default: True)
            resample_kwargs: Remaining keyword arguments to pass to individual
                resampler classes. See the individual resampler class
                documentation :mod:`here <satpy.resample>` for available
                arguments.

        """
        to_resample_ids = [dsid for (dsid, dataset) in self.datasets.items()
                           if (not datasets) or dsid in datasets]

        if destination is None:
            destination = self.max_area(to_resample_ids)
        new_scn = self.copy(datasets=to_resample_ids)
        # we may have some datasets we asked for but don't exist yet
        new_scn.wishlist = self.wishlist.copy()
        self._resampled_scene(new_scn, destination, resampler=resampler,
                              reduce_data=reduce_data, **resample_kwargs)

        # regenerate anything from the wishlist that needs it (combining
        # multiple resolutions, etc.)
        if generate:
            keepables = new_scn.generate_composites()
        else:
            # don't lose datasets that we may need later for generating
            # composites
            keepables = set(new_scn.datasets.keys()) | new_scn.wishlist

        if new_scn.missing_datasets:
            # copy the set of missing datasets because they won't be valid
            # after they are removed in the next line
            missing = new_scn.missing_datasets.copy()
            new_scn._remove_failed_datasets(keepables)
            missing_str = ", ".join(str(x) for x in missing)
            LOG.warning(
                "The following datasets "
                "were not created: {}".format(missing_str))
        if unload:
            new_scn.unload(keepables)

        return new_scn

    def show(self, dataset_id, overlay=None):
        """Show the *dataset* on screen as an image.

        Show dataset on screen as an image, possibly with an overlay.

        Args:
            dataset_id (DatasetID or str):
                Either a DatasetID or a string representing a DatasetID, that
                has been previously loaded using Scene.load.
            overlay (dict, optional):
                Add an overlay before showing the image.  The keys/values for
                this dictionary are as the arguments for
                :meth:`~satpy.writers.add_overlay`.  The dictionary should
                contain at least the key ``"coast_dir"``, which should refer
                to a top-level directory containing shapefiles.  See the
                pycoast_ package documentation for coastline shapefile
                installation instructions.

        .. _pycoast: https://pycoast.readthedocs.io/

        """
        from satpy.writers import get_enhanced_image
        from satpy.utils import in_ipynb
        img = get_enhanced_image(self[dataset_id].squeeze(), overlay=overlay)
        if not in_ipynb():
            img.show()
        return img

    def to_geoviews(self, gvtype=None, datasets=None, kdims=None, vdims=None, dynamic=False):
        """Convert satpy Scene to geoviews.

        Args:
            gvtype (gv plot type):
                One of gv.Image, gv.LineContours, gv.FilledContours, gv.Points
                Default to :class:`geoviews.Image`.
                See Geoviews documentation for details.
            datasets (list): Limit included products to these datasets
            kdims (list of str):
                Key dimensions. See geoviews documentation for more information.
            vdims : list of str, optional
                Value dimensions. See geoviews documentation for more information.
                If not given defaults to first data variable
            dynamic : boolean, optional, default False

        Returns: geoviews object

        Todo:
            * better handling of projection information in datasets which are
              to be passed to geoviews

        """
        import geoviews as gv
        from cartopy import crs  # noqa
        if gvtype is None:
            gvtype = gv.Image

        ds = self.to_xarray_dataset(datasets)

        if vdims is None:
            # by default select first data variable as display variable
            vdims = ds.data_vars[list(ds.data_vars.keys())[0]].name

        if hasattr(ds, "area") and hasattr(ds.area, 'to_cartopy_crs'):
            dscrs = ds.area.to_cartopy_crs()
            gvds = gv.Dataset(ds, crs=dscrs)
        else:
            gvds = gv.Dataset(ds)

        if "latitude" in ds.coords.keys():
            gview = gvds.to(gv.QuadMesh, kdims=["longitude", "latitude"], vdims=vdims, dynamic=dynamic)
        else:
            gview = gvds.to(gvtype, kdims=["x", "y"], vdims=vdims, dynamic=dynamic)

        return gview

    def to_xarray_dataset(self, datasets=None):
        """Merge all xr.DataArrays of a scene to a xr.DataSet.

        Parameters:
            datasets (list):
                List of products to include in the :class:`xarray.Dataset`

        Returns: :class:`xarray.Dataset`

        """
        if datasets is not None:
            datasets = [self[ds] for ds in datasets]
        else:
            datasets = [self.datasets.get(ds) for ds in self.wishlist]
            datasets = [ds for ds in datasets if ds is not None]

        ds_dict = {i.attrs['name']: i.rename(i.attrs['name']) for i in datasets if i.attrs.get('area') is not None}
        mdata = combine_metadata(*tuple(i.attrs for i in datasets))
        if mdata.get('area') is None or not isinstance(mdata['area'], SwathDefinition):
            # either don't know what the area is or we have an AreaDefinition
            ds = xr.merge(ds_dict.values())
        else:
            # we have a swath definition and should use lon/lat values
            lons, lats = mdata['area'].get_lonlats()
            if not isinstance(lons, DataArray):
                lons = DataArray(lons, dims=('y', 'x'))
                lats = DataArray(lats, dims=('y', 'x'))
            # ds_dict['longitude'] = lons
            # ds_dict['latitude'] = lats
            ds = xr.Dataset(ds_dict, coords={"latitude": (["y", "x"], lats),
                                             "longitude": (["y", "x"], lons)})

        ds.attrs = mdata
        return ds

    def images(self):
        """Generate images for all the datasets from the scene."""
        for ds_id, projectable in self.datasets.items():
            if ds_id in self.wishlist:
                yield projectable.to_image()

    def save_dataset(self, dataset_id, filename=None, writer=None,
                     overlay=None, decorate=None, compute=True, **kwargs):
        """Save the ``dataset_id`` to file using ``writer``.

        Args:
            dataset_id (str or Number or DatasetID): Identifier for the
                dataset to save to disk.
            filename (str): Optionally specify the filename to save this
                            dataset to. It may include string formatting
                            patterns that will be filled in by dataset
                            attributes.
            writer (str): Name of writer to use when writing data to disk.
                Default to ``"geotiff"``. If not provided, but ``filename`` is
                provided then the filename's extension is used to determine
                the best writer to use. See :meth:`Scene.get_writer_by_ext`
                for details.
            overlay (dict): See :func:`satpy.writers.add_overlay`. Only valid
                for "image" writers like `geotiff` or `simple_image`.
            decorate (dict): See :func:`satpy.writers.add_decorate`. Only valid
                for "image" writers like `geotiff` or `simple_image`.
            compute (bool): If `True` (default), compute all of the saves to
                disk. If `False` then the return value is either a
                :doc:`dask:delayed` object or two lists to be passed to
                a `dask.array.store` call. See return values below for more
                details.
            kwargs: Additional writer arguments. See :doc:`../writers` for more
                information.

        Returns:
            Value returned depends on `compute`. If `compute` is `True` then
            the return value is the result of computing a
            :doc:`dask:delayed` object or running :func:`dask.array.store`.
            If `compute` is `False` then the returned value is either a
            :doc:`dask:delayed` object that can be computed using
            `delayed.compute()` or a tuple of (source, target) that should be
            passed to :func:`dask.array.store`. If target is provided the the
            caller is responsible for calling `target.close()` if the target
            has this method.

        """
        if writer is None and filename is None:
            writer = 'geotiff'
        elif writer is None:
            writer = self.get_writer_by_ext(os.path.splitext(filename)[1])

        writer, save_kwargs = load_writer(writer,
                                          ppp_config_dir=self.ppp_config_dir,
                                          filename=filename,
                                          **kwargs)
        return writer.save_dataset(self[dataset_id],
                                   overlay=overlay, decorate=decorate,
                                   compute=compute, **save_kwargs)

    def save_datasets(self, writer=None, filename=None, datasets=None, compute=True,
                      **kwargs):
        """Save all the datasets present in a scene to disk using ``writer``.

        Args:
            writer (str): Name of writer to use when writing data to disk.
                Default to ``"geotiff"``. If not provided, but ``filename`` is
                provided then the filename's extension is used to determine
                the best writer to use. See :meth:`Scene.get_writer_by_ext`
                for details.
            filename (str): Optionally specify the filename to save this
                            dataset to. It may include string formatting
                            patterns that will be filled in by dataset
                            attributes.
            datasets (iterable): Limit written products to these datasets
            compute (bool): If `True` (default), compute all of the saves to
                disk. If `False` then the return value is either a
                :doc:`dask:delayed` object or two lists to be passed to
                a `dask.array.store` call. See return values below for more
                details.
            kwargs: Additional writer arguments. See :doc:`../writers` for more
                information.

        Returns:
            Value returned depends on `compute` keyword argument. If
            `compute` is `True` the value is the result of a either a
            `dask.array.store` operation or a :doc:`dask:delayed`
            compute, typically this is `None`. If `compute` is `False` then the
            result is either a :doc:`dask:delayed` object that can be
            computed with `delayed.compute()` or a two element tuple of
            sources and targets to be passed to :func:`dask.array.store`. If
            `targets` is provided then it is the caller's responsibility to
            close any objects that have a "close" method.

        """
        if datasets is not None:
            datasets = [self[ds] for ds in datasets]
        else:
            datasets = [self.datasets.get(ds) for ds in self.wishlist]
            datasets = [ds for ds in datasets if ds is not None]
        if not datasets:
            raise RuntimeError("None of the requested datasets have been "
                               "generated or could not be loaded. Requested "
                               "composite inputs may need to have matching "
                               "dimensions (eg. through resampling).")
        if writer is None and filename is None:
            writer = 'geotiff'
        elif writer is None:
            writer = self.get_writer_by_ext(os.path.splitext(filename)[1])
        writer, save_kwargs = load_writer(writer,
                                          ppp_config_dir=self.ppp_config_dir,
                                          filename=filename,
                                          **kwargs)
        return writer.save_datasets(datasets, compute=compute, **save_kwargs)

    @classmethod
    def get_writer_by_ext(cls, extension):
        """Find the writer matching the ``extension``.

        Defaults to "simple_image".

        Example Mapping:

            - geotiff: .tif, .tiff
            - cf: .nc
            - mitiff: .mitiff
            - simple_image: .png, .jpeg, .jpg, ...

        Args:
            extension (str): Filename extension starting with
                "." (ex. ".png").

        Returns:
            str: The name of the writer to use for this extension.

        """
        mapping = {".tiff": "geotiff", ".tif": "geotiff", ".nc": "cf",
                   ".mitiff": "mitiff"}
        return mapping.get(extension.lower(), 'simple_image')
