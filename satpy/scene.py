#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2017
#
# Author(s):
#
#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>
#   Esben S. Nielsen <esn@dmi.dk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Scene objects to hold satellite data.
"""

import logging
import os

from satpy.composites import CompositorLoader, IncompatibleAreas
from satpy.config import get_environ_config_dir
from satpy.dataset import (DatasetID, MetadataObject, dataset_walker,
                           replace_anc)
from satpy.node import DependencyTree
from satpy.readers import DatasetDict, load_readers
from satpy.resample import (resample_dataset, get_frozen_area,
                            prepare_resampler)
from satpy.writers import load_writer
from pyresample.geometry import AreaDefinition
from xarray import DataArray
import numpy as np

try:
    import configparser
except ImportError:
    from six.moves import configparser  # noqa: F401

LOG = logging.getLogger(__name__)


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
                 ppp_config_dir=get_environ_config_dir(),
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
        """DatasetIDs that have not been loaded."""
        return set(self.wishlist) - set(self.datasets.keys())

    def _compare_areas(self, datasets=None, compare_func=max):
        """Get  for the provided datasets.

        Args:
            datasets (iterable): Datasets whose areas will be compared. Can
                                 be either `xarray.DataArray` objects or
                                 identifiers to get the DataArrays from the
                                 current Scene. Defaults to all datasets.
            compare_func (callable): `min` or `max` or other function used to
                                     compare the dataset's areas.

        """
        if datasets is None:
            check_datasets = list(self.values())
        else:
            check_datasets = []
            for ds in datasets:
                if not isinstance(ds, DataArray):
                    ds = self[ds]
                check_datasets.append(ds)

        if not check_datasets:
            raise ValueError("No dataset areas available")

        areas = [x.attrs['area'] for x in check_datasets]
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
        """Get names of available datasets, globally or just for *reader_name*
        if specified, that can be loaded.

        Available dataset names are determined by what each individual reader
        can load. This is normally determined by what files are needed to load
        a dataset and what files have been provided to the scene/reader.

        :return: list of available dataset names
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
            available_datasets += sorted(self.available_composite_ids(
                available_datasets))
        return available_datasets

    def available_dataset_names(self, reader_name=None, composites=False):
        """Get the list of the names of the available datasets."""
        return sorted(set(x.name for x in self.available_dataset_ids(
            reader_name=reader_name, composites=composites)))

    def all_dataset_ids(self, reader_name=None, composites=False):
        """Get names of all datasets from loaded readers or `reader_name` if
        specified..

        :return: list of all dataset names
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
        return sorted(set(x.name for x in self.all_dataset_ids(
            reader_name=reader_name, composites=composites)))

    def available_composite_ids(self, available_datasets=None):
        """Get names of compositors that can be generated from the available
        datasets.

        :return: generator of available compositor's names
        """
        if available_datasets is None:
            available_datasets = self.available_dataset_ids(composites=False)
        else:
            if not all(isinstance(ds_id, DatasetID) for ds_id in available_datasets):
                raise ValueError(
                    "'available_datasets' must all be DatasetID objects")

        all_comps = self.all_composite_ids()
        # recreate the dependency tree so it doesn't interfere with the user's
        # wishlist
        comps, mods = self.cpl.load_compositors(self.attrs['sensor'])
        dep_tree = DependencyTree(self.readers, comps, mods)
        dep_tree.find_dependencies(set(available_datasets + all_comps))
        available_comps = set(x.name for x in dep_tree.trunk())
        # get rid of modified composites that are in the trunk
        return sorted(available_comps & set(all_comps))

    def available_composite_names(self, available_datasets=None):
        return sorted(set(x.name for x in self.available_composite_ids(
            available_datasets=available_datasets)))

    def all_composite_ids(self, sensor_names=None):
        """Get all composite IDs that are configured.

        :return: generator of configured composite names
        """
        if sensor_names is None:
            sensor_names = self.attrs['sensor']
        compositors = []
        # Note if we get compositors from the dep tree then it will include
        # modified composites which we don't want
        for sensor_name in sensor_names:
            compositors.extend(
                self.cpl.compositors.get(sensor_name, {}).keys())
        return sorted(set(compositors))

    def all_composite_names(self, sensor_names=None):
        return sorted(set(x.name for x in self.all_composite_ids(sensor_names=sensor_names)))

    def all_modifier_names(self):
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
            a_str = str(a) if a is not None else None
            datasets_by_area.setdefault(
                a_str, (a, []))
            datasets_by_area[a_str][1].append(DatasetID.from_dict(ds.attrs))

        for area_name, (area_obj, ds_list) in datasets_by_area.items():
            yield area_obj, ds_list

    def keys(self, **kwargs):
        return self.datasets.keys(**kwargs)

    def values(self):
        return self.datasets.values()

    def __getitem__(self, key):
        """Get a dataset."""
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

            if prereq_id in self.datasets:
                prereq_datasets.append(self.datasets[prereq_id])
            elif not prereq_node.is_leaf and prereq_id in keepables:
                delayed_gen = True
                continue
            elif not skip:
                LOG.warning("Missing prerequisite for '{}': '{}'".format(
                    comp_id, prereq_id))
                raise KeyError("Missing composite prerequisite")
            else:
                LOG.debug("Missing optional prerequisite for {}: {}".format(
                    comp_id, prereq_id))

        if delayed_gen:
            keepables.add(comp_id)
            keepables.update([x.name for x in prereq_nodes])
            LOG.warning("Delaying generation of %s "
                        "because of dependency's delayed generation: %s",
                        comp_id, prereq_id)
            if not skip:
                LOG.warning("Missing prerequisite for '{}': '{}'".format(
                    comp_id, prereq_id))
                raise KeyError("Missing composite prerequisite")
            else:
                LOG.debug("Missing optional prerequisite for {}: {}".format(
                    comp_id, prereq_id))

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
            prereq_datasets = self._get_prereq_datasets(
                comp_node.name,
                prereqs,
                keepables,
            )
        except KeyError:
            return

        optional_datasets = self._get_prereq_datasets(
            comp_node.name,
            optional_prereqs,
            keepables,
            skip=True
        )

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
            LOG.warning("Delaying generation of %s "
                        "because of incompatible areas",
                        str(compositor.id))
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
        """Compute all the composites contained in `requirements`.
        """
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

    def load(self,
             wishlist,
             calibration=None,
             resolution=None,
             polarization=None,
             generate=True,
             unload=True,
             **kwargs):
        """Read, generate, and unload.
        """
        dataset_keys = set(wishlist)
        needed_datasets = (self.wishlist | dataset_keys) - \
            set(self.datasets.keys())
        unknown = self.dep_tree.find_dependencies(needed_datasets,
                                                  calibration=calibration,
                                                  polarization=polarization,
                                                  resolution=resolution)
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
            LOG.warning(
                "The following datasets were not created: {}".format(missing_str))
        if unload:
            self.unload(keepables=keepables)

    def _resampled_scene(self, datasets, destination, **resample_kwargs):
        """Generate a new scene with resampled *datasets*."""
        new_scn = self.__class__()
        new_datasets = {}
        destination_area = None
        resamplers = {}
        resampler = resample_kwargs.get('resampler')
        for dataset, parent_dataset in dataset_walker(datasets):
            ds_id = DatasetID.from_dict(dataset.attrs)
            pres = None
            if parent_dataset is not None:
                pres = new_datasets[DatasetID.from_dict(parent_dataset.attrs)]
            if ds_id in new_datasets:
                replace_anc(dataset, pres)
                continue
            if dataset.attrs.get('area') is None:
                if parent_dataset is None:
                    new_scn[ds_id] = dataset
                else:
                    replace_anc(dataset, pres)
                continue
            if destination_area is None:
                # FIXME: We should allow users to freeze based with specific
                #        dataset
                destination_area = get_frozen_area(destination,
                                                   dataset.attrs['area'])
            LOG.debug("Resampling %s", ds_id)
            source_area = dataset.attrs['area']
            try:
                slice_x, slice_y = source_area.get_area_slices(destination_area)
                source_area = source_area[slice_y, slice_x]
                dataset.data = dataset.data.rechunk(1024)
                dataset = dataset.isel(x=slice_x, y=slice_y)
                dataset.attrs['area'] = source_area
            except (NotImplementedError, np.ma.MaskError):
                LOG.info("Not reducing data before resampling.")
            if source_area not in resamplers:
                key, resampler = prepare_resampler(
                        source_area, destination_area, **resample_kwargs)
                resamplers[source_area] = resampler
                self.resamplers[key] = resampler
            kwargs = resample_kwargs.copy()
            kwargs['resampler'] = resamplers[source_area]
            res = resample_dataset(dataset, destination_area,
                                   **kwargs)
            new_datasets[ds_id] = res
            if parent_dataset is None:
                new_scn[ds_id] = res
            else:
                replace_anc(res, pres)

        return new_scn, destination_area

    def resample(self,
                 destination=None,
                 datasets=None,
                 generate=True,
                 unload=True,
                 **resample_kwargs):
        """Resample the datasets and return a new scene."""
        to_resample = [dataset for (dsid, dataset) in self.datasets.items()
                       if (not datasets) or dsid in datasets]

        if destination is None:
            destination = self.max_area(to_resample)
        new_scn, destination_area = self._resampled_scene(to_resample, destination,
                                                          **resample_kwargs)

        new_scn.attrs = self.attrs.copy()
        new_scn.dep_tree = self.dep_tree.copy()

        # MUST set this after assigning the resampled datasets otherwise
        # composite prereqs that were resampled will be considered "wishlisted"
        if datasets is None:
            new_scn.wishlist = self.wishlist.copy()
        else:
            new_scn.wishlist = set([DatasetID.from_dict(ds.attrs)
                                    for ds in new_scn])

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
        """Show the *dataset* on screen as an image."""
        from satpy.writers import get_enhanced_image
        img = get_enhanced_image(self[dataset_id].squeeze(), overlay=overlay)
        img.show()
        return img

    def images(self):
        """Generate images for all the datasets from the scene."""
        for ds_id, projectable in self.datasets.items():
            if ds_id in self.wishlist:
                yield projectable.to_image()

    def save_dataset(self, dataset_id, filename=None, writer=None,
                     overlay=None, compute=True, **kwargs):
        """Save the *dataset_id* to file using *writer* (default: geotiff)."""
        if writer is None and filename is None:
            writer = 'geotiff'
        elif writer is None:
            writer = self.get_writer_by_ext(os.path.splitext(filename)[1])

        writer, save_kwargs = load_writer(writer,
                                          ppp_config_dir=self.ppp_config_dir)
        return writer.save_dataset(self[dataset_id], filename=filename,
                                   overlay=overlay, compute=compute,
                                   **save_kwargs)

    def save_datasets(self, writer="geotiff", datasets=None, compute=True,
                      **kwargs):
        """Save all the datasets present in a scene to disk using *writer*."""
        if datasets is not None:
            datasets = [self[ds] for ds in datasets]
        else:
            datasets = self.datasets.values()
        writer, save_kwargs = load_writer(writer,
                                          ppp_config_dir=self.ppp_config_dir,
                                          **kwargs)
        return writer.save_datasets(datasets, compute=compute, **save_kwargs)

    @classmethod
    def get_writer_by_ext(cls, extension):
        """Find the writer matching the *extension*."""
        mapping = {".tiff": "geotiff", ".tif": "geotiff", ".nc": "cf"}
        return mapping.get(extension.lower(), 'simple_image')
