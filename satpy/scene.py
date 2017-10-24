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
import yaml

from satpy.composites import CompositorLoader, IncompatibleAreas
from satpy.config import (config_search_paths, get_environ_config_dir,
                          runtime_import, recursive_dict_update)
from satpy.dataset import Dataset, DatasetID, InfoObject
from satpy.node import DependencyTree
from satpy.readers import DatasetDict, ReaderFinder

try:
    import configparser
except ImportError:
    from six.moves import configparser

LOG = logging.getLogger(__name__)


class Scene(InfoObject):

    """The almighty scene class."""

    def __init__(self,
                 filenames=None,
                 ppp_config_dir=get_environ_config_dir(),
                 reader=None,
                 base_dir=None,
                 sensor=None,
                 start_time=None,
                 end_time=None,
                 area=None,
                 reader_kwargs=None,
                 **metadata):
        """The Scene object constructor.

        Note to load data either `filenames`, `reader`, or a 'base_dir' must
        be specified. If `filenames` is not specified then `reader` must be
        specified to search the current directory or `base_dir` if specified.
        If neither `filenames` nor `reader` is specified then `base_dir` will
        be used to find any files matching the file patterns for any
        configured readers. Otherwise the Scene is created with no Readers
        available meaning Datasets must be added manually:

            scn = Scene(sensor='viirs', start_time=start_time)
            scn['my_dataset'] = Dataset(my_data_array, **my_info)

        Args:
            filenames (iterable): A sequence of files that will be used to load
                                  data from.
            ppp_config_dir (str): The directory containing the configuration
                                  files for satpy.
            reader: The name of the reader to use for loading the data.
            base_dir (str): The directory to search for files containing the
                            data to load. If *filenames* is also provided,
                            this is ignored.
            sensor (list or str): Limit used files by provided sensors.
            area (AreaDefinition): Limit used files by geographic area.
            start_time (datetime): Limit used files by starting time.
            end_time (datetime): Limit used files by ending time.
            reader_kwargs (dict): Keyword arguments to pass to specific reader
                                  instances.
            metadata: Other metadata to assign to the Scene's ``.info``.
        """
        InfoObject.__init__(self, sensor=sensor or set(), area=area,
                            start_time=start_time, end_time=end_time,
                            **metadata)
        # Set the PPP_CONFIG_DIR in the environment in case it's used elsewhere
        # in pytroll
        LOG.debug("Setting 'PPP_CONFIG_DIR' to '%s'", ppp_config_dir)
        os.environ["PPP_CONFIG_DIR"] = self.ppp_config_dir = ppp_config_dir

        self.readers = self.create_reader_instances(filenames=filenames,
                                                    base_dir=base_dir,
                                                    reader=reader,
                                                    reader_kwargs=reader_kwargs,
                                                    metadata=metadata)
        self.info.update(self._compute_metadata_from_readers())
        self.datasets = DatasetDict()
        self.cpl = CompositorLoader(self.ppp_config_dir)
        comps, mods = self.cpl.load_compositors(self.info['sensor'])
        self.wishlist = set()
        self.dep_tree = DependencyTree(self.readers, comps, mods)

    def _compute_metadata_from_readers(self):
        mda = {}
        mda['sensor'] = self._get_sensor_names()

        # overwrite the request start/end times with actual loaded data limits
        if self.readers:
            mda['start_time'] = min(x.start_time
                                    for x in self.readers.values())
            mda['end_time'] = max(x.end_time
                                  for x in self.readers.values())
        return mda

    def _get_sensor_names(self):
        # if the user didn't tell us what sensors to work with, let's figure it
        # out
        if not self.info.get('sensor'):
            # reader finder could return multiple readers
            return set([sensor for reader_instance in self.readers.values()
                        for sensor in reader_instance.sensor_names])
        elif not isinstance(self.info['sensor'], (set, tuple, list)):
            return set([self.info['sensor']])
        else:
            return set(self.info['sensor'])

    def create_reader_instances(self,
                                filenames=None,
                                base_dir=None,
                                reader=None,
                                reader_kwargs=None,
                                metadata=None):
        """Find readers and return their instanciations."""
        finder = ReaderFinder(ppp_config_dir=self.ppp_config_dir,
                              base_dir=base_dir,
                              start_time=self.info.get('start_time'),
                              end_time=self.info.get('end_time'),
                              area=self.info.get('area'), )
        return finder(reader=reader,
                      sensor=self.info.get("sensor"),
                      filenames=filenames,
                      reader_kwargs=reader_kwargs,
                      metadata=metadata)

    @property
    def start_time(self):
        """Return the start time of the file."""
        return self.info['start_time']

    @property
    def end_time(self):
        """Return the end time of the file."""
        return self.info['end_time']

    @property
    def missing_datasets(self):
        return set(self.wishlist) - set(self.datasets.keys())

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
        comps, mods = self.cpl.load_compositors(self.info['sensor'])
        dep_tree = DependencyTree(self.readers, comps, mods)
        unknowns = dep_tree.find_dependencies(
            set(available_datasets + all_comps))
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
            sensor_names = self.info['sensor']
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
            a = ds.info.get('area')
            a_str = str(a) if a is not None else None
            datasets_by_area.setdefault(
                a_str, (a, []))
            datasets_by_area[a_str][1].append(ds.id)

        for area_name, (area_obj, ds_list) in datasets_by_area.items():
            yield area_obj, ds_list

    def keys(self, **kwargs):
        return self.datasets.keys(**kwargs)

    def __getitem__(self, key):
        """Get a dataset."""
        return self.datasets[key]

    def __setitem__(self, key, value):
        """Add the item to the scene."""
        if not isinstance(value, Dataset):
            raise ValueError("Only 'Dataset' objects can be assigned")
        self.datasets[key] = value
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

    def read_datasets(self, dataset_nodes, **kwargs):
        """Read the given datasets from file."""
        # Sort requested datasets by reader
        reader_datasets = {}

        for node in dataset_nodes:
            ds_id = node.name
            if ds_id in self.datasets and self.datasets[ds_id].is_loaded():
                continue
            reader_name = node.data['reader_name']
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
        for prereq_node in prereq_nodes:
            prereq_id = prereq_node.name
            if prereq_id not in self.datasets and prereq_id not in keepables \
                    and not prereq_node.is_leaf:
                self._generate_composite(prereq_node, keepables)

            if prereq_id in self.datasets:
                prereq_datasets.append(self.datasets[prereq_id])
            else:
                if not prereq_node.is_leaf and prereq_id in keepables:
                    keepables.add(comp_id)
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
                                   **self.info)
            self.datasets[composite.id] = composite
            if comp_node.name in self.wishlist:
                self.wishlist.remove(comp_node.name)
                self.wishlist.add(composite.id)
            # update the node with the computed DatasetID
            comp_node.name = composite.id
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

    def read_composites(self, compositor_nodes):
        """Read (generate) composites.
        """
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
        return self.read_datasets(nodes, **kwargs)

    def compute(self, nodes=None):
        """Compute all the composites contained in `requirements`.
        """
        if nodes is None:
            required_nodes = self.wishlist - set(self.datasets.keys())
            nodes = set(self.dep_tree.trunk(nodes=required_nodes)) - \
                set(self.datasets.keys())
        return self.read_composites(nodes)

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
        compute composites that have yet to be computed.

        Args:
            keepables (iterable): DatasetIDs to keep whether they are needed
                                  or not.

        """
        to_del = [ds_id for ds_id, projectable in self.datasets.items()
                  if ds_id not in self.wishlist and (not keepables or ds_id
                                                     not in keepables)]
        for ds_id in to_del:
            del self.datasets[ds_id]

    def load(self,
             wishlist,
             calibration=None,
             resolution=None,
             polarization=None,
             compute=True,
             unload=True,
             **kwargs):
        """Read, compute and unload.
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
        keepables = None
        if compute:
            keepables = self.compute()
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

    def resample(self,
                 destination,
                 datasets=None,
                 compute=True,
                 unload=True,
                 **resample_kwargs):
        """Resample the datasets and return a new scene.
        """
        new_scn = Scene()
        new_scn.info = self.info.copy()
        # new_scn.cpl = self.cpl
        new_scn.dep_tree = self.dep_tree.copy()
        for ds_id, projectable in self.datasets.items():
            LOG.debug("Resampling %s", ds_id)
            if datasets and ds_id not in datasets:
                continue
            new_scn[ds_id] = projectable.resample(destination,
                                                  **resample_kwargs)
        # MUST set this after assigning the resampled datasets otherwise
        # composite prereqs that were resampled will be considered "wishlisted"
        if datasets is None:
            new_scn.wishlist = self.wishlist
        else:
            new_scn.wishlist = set([ds.id for ds in new_scn])

        # recompute anything from the wishlist that needs it (combining multiple
        # resolutions, etc.)
        keepables = None
        if compute:
            nodes = [self.dep_tree[i]
                     for i in new_scn.wishlist if not self.dep_tree[i].is_leaf]
            keepables = new_scn.compute(nodes=nodes)
        if new_scn.missing_datasets:
            # copy the set of missing datasets because they won't be valid
            # after they are removed in the next line
            missing = new_scn.missing_datasets.copy()
            new_scn._remove_failed_datasets(keepables)
            missing_str = ", ".join(str(x) for x in missing)
            LOG.warning(
                "The following datasets were not created: {}".format(missing_str))
        if unload:
            new_scn.unload(keepables)

        return new_scn

    def show(self, dataset_id, overlay=None):
        """Show the *dataset* on screen as an image.
        """

        from satpy.writers import get_enhanced_image
        get_enhanced_image(self[dataset_id], overlay=overlay).show()

    def images(self):
        """Generate images for all the datasets from the scene.
        """
        for ds_id, projectable in self.datasets.items():
            if ds_id in self.wishlist:
                yield projectable.to_image()

    def load_writer_config(self, config_files, **kwargs):
        conf = {}
        for conf_fn in config_files:
            with open(conf_fn) as fd:
                conf = recursive_dict_update(conf, yaml.load(fd))
        writer_class = conf['writer']['writer']
        writer = writer_class(ppp_config_dir=self.ppp_config_dir,
                              config_files=config_files,
                              **kwargs)
        return writer

    def save_dataset(self, dataset_id, filename=None, writer=None, overlay=None, **kwargs):
        """Save the *dataset_id* to file using *writer* (geotiff by default).
        """
        if writer is None:
            if filename is None:
                writer = self.get_writer("geotiff", **kwargs)
            else:
                writer = self.get_writer_by_ext(
                    os.path.splitext(filename)[1], **kwargs)
        else:
            writer = self.get_writer(writer, **kwargs)
        writer.save_dataset(self[dataset_id],
                            filename=filename,
                            overlay=overlay, **kwargs)

    def save_datasets(self, writer="geotiff", datasets=None, **kwargs):
        """Save all the datasets present in a scene to disk using *writer*.
        """
        if datasets is not None:
            datasets = [self[ds] for ds in datasets]
        else:
            datasets = self.datasets.values()
        writer = self.get_writer(writer, **kwargs)
        writer.save_datasets(datasets, **kwargs)

    def get_writer(self, writer="geotiff", **kwargs):
        config_fn = writer + ".yaml" if "." not in writer else writer
        config_files = config_search_paths(
            os.path.join("writers", config_fn), self.ppp_config_dir)
        kwargs.setdefault("config_files", config_files)
        return self.load_writer_config(**kwargs)

    def get_writer_by_ext(self, extension, **kwargs):
        mapping = {".tiff": "geotiff", ".tif": "geotiff", ".nc": "cf"}
        return self.get_writer(
            mapping.get(extension.lower(), "simple_image"), **kwargs)
