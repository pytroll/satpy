#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2016
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
from satpy.config import (config_search_paths, get_environ_config_dir,
                          runtime_import)
from satpy.node import Node, DependencyTree
from satpy.projectable import Dataset, InfoObject, Projectable
from satpy.readers import (DatasetDict,
                           DatasetID,
                           ReaderFinder,
                           DATASET_KEYS)

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
                 **metadata):
        """The Scene object constructor.

        Args:
            filenames: A sequence of files that will be used to load data from.
            ppp_config_dir: The directory containing the configuration files for
                satpy.
            reader: The name of the reader to use for loading the data.
            base_dir: The directory to search for files containing the data to
                load. If *filenames* is also provided, this is ignored.
            metadata: Free metadata information.
        """
        InfoObject.__init__(self, **metadata)
        # Set the PPP_CONFIG_DIR in the environment in case it's used elsewhere
        # in pytroll
        LOG.debug("Setting 'PPP_CONFIG_DIR' to '%s'", ppp_config_dir)
        os.environ["PPP_CONFIG_DIR"] = self.ppp_config_dir = ppp_config_dir

        self.readers = self.create_reader_instances(filenames=filenames,
                                                    base_dir=base_dir,
                                                    reader=reader)
        self.info.update(self._compute_metadata_from_readers())
        self.datasets = DatasetDict()
        self.cpl = CompositorLoader(self.ppp_config_dir)
        self.wishlist = set()
        self.dep_tree = DependencyTree()

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
                                reader=None):
        """Find readers and return their instanciations."""
        finder = ReaderFinder(ppp_config_dir=self.ppp_config_dir,
                              base_dir=base_dir,
                              start_time=self.info.get('start_time'),
                              end_time=self.info.get('end_time'),
                              area=self.info.get('area'), )
        try:
            return finder(reader=reader,
                          sensor=self.info.get("sensor"),
                          filenames=filenames)
        except ValueError as err:
            if filenames is None and base_dir is None:
                LOG.info('Neither filenames nor base_dir provided, '
                         'creating an empty scene (error was %s)', str(err))
                return {}
            else:
                raise

    @property
    def start_time(self):
        """Return the start time of the file."""
        return self.info['start_time']

    @property
    def end_time(self):
        """Return the end time of the file."""
        return self.info['end_time']

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

        available_datasets = [dataset_id
                              for reader in readers
                              for dataset_id in reader.available_dataset_ids]
        if composites:
            available_datasets += list(self.available_composite_ids(
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
            all_datasets += self.all_composites()
        return all_datasets

    def all_dataset_names(self, reader_name=None, composites=False):
        return sorted(set([x.name for x in self.all_dataset_ids(
            reader_name=reader_name, composites=composites)]))

    def _compositor_list(self, tree, modified=False):
        # all trunk nodes are composites of some sort
        composites = []
        for node in tree.trunk():
            # check if node is just a "modified" leaf node
            if modified:
                composites.append(node.name)
                continue

            for child_node in node.children:
                # NOTE: This depends on modifiers being the last identifying
                # element in DatasetID
                if node.name[:-1] == child_node.name[:-1]:
                    break
            else:
                composites.append(node.name)

        return sorted(set(composites))

    def available_composite_ids(self, available_datasets=None, modified=False):
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

        tree, unknowns = self.create_deptree(available_datasets + self.all_composite_ids())
        return self._compositor_list(tree, modified=modified)

    def available_composite_names(self, modified=False):
        return sorted(set([x.name for x in
                           self.available_composite_ids(modified=modified)]))

    def all_composite_ids(self, modified=False):
        """Get all composite IDs that are configured.

        :return: generator of configured composite names
        """
        comp_ids = self.all_composite_objects().keys()
        return sorted(set([x for x in comp_ids if modified or not x.modifiers]))

    def all_composite_names(self, modified=False):
        return sorted(set([x.name for x in
                           self.all_composite_ids(modified=modified)]))

    def all_composite_objects(self):
        """Get all compositor objects that are configured.

        :return: dictionary of composite name to compositor object
        """
        return self.cpl.load_compositors(sensor_names=self.info["sensor"])

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
            datasets_by_area.setdefault(
                str(ds.info["area"]), (ds.info["area"], []))
            datasets_by_area[str(ds.info["area"])][1].append(ds.info["id"])

        for area_name, (area_obj, ds_list) in datasets_by_area.items():
            yield area_obj, ds_list

    def __getitem__(self, key):
        """Get a dataset."""
        return self.datasets[key]

    def __setitem__(self, key, value):
        """Add the item to the scene."""
        if not isinstance(value, Projectable):
            raise ValueError("Only 'Projectable' objects can be assigned")
        self.datasets[key] = value
        self.wishlist.add(self.datasets.get_key(key))

    def __delitem__(self, key):
        """Remove the item from the scene."""
        k = self.datasets.get_key(key)
        self.wishlist.remove(k)
        del self.datasets[k]

    def __contains__(self, name):
        """Check if the dataset is in the scene."""
        return name in self.datasets

    def _find_reader_dataset(self,
                             dataset_key,
                             calibration=None,
                             polarization=None,
                             resolution=None):
        for reader_name, reader_instance in self.readers.items():
            try:
                ds_id = reader_instance.get_dataset_key(dataset_key,
                                                        calibration=calibration,
                                                        polarization=polarization,
                                                        resolution=resolution)
            except KeyError as err:
                # LOG.debug("Can't find dataset %s in reader %s",
                #           str(dataset_key), reader_name)
                pass
            else:
                try:
                    self.wishlist.remove(dataset_key)
                except KeyError:
                    pass
                else:
                    self.wishlist.add(ds_id)
                # LOG.debug("Found {} in reader {}".format(str(ds_id), reader_name))
                return Node(ds_id, {'reader_name': reader_name})

    def _merge_modified_id(self, parent_id, child_id):
        # resolve modifier-based IDs
        if isinstance(child_id, DatasetID) and \
                isinstance(parent_id, DatasetID) and \
                (DatasetID.name_match(child_id, parent_id.name) or
                     DatasetID.wavelength_match(child_id.wavelength, parent_id.wavelength)):
            new_id = []
            for elem_name, comp_elem, req_elem in zip(DATASET_KEYS, child_id, parent_id):
                if elem_name == 'modifiers':
                    comp_elem = comp_elem or tuple()
                    req_elem = req_elem or tuple()
                    new_id.append(comp_elem + tuple(m for m in req_elem if m not in comp_elem))
                else:
                    new_id.append(comp_elem or req_elem)
            parent_id = DatasetID(*new_id)
        return parent_id

    def _get_compositor_prereqs(self, prereq_names, skip=False, **kwargs):
        prereq_ids = []
        unknowns = set()
        for prereq in prereq_names:
            n, u = self._find_dependencies(prereq, **kwargs)
            if u:
                unknowns.update(u)
                if skip:
                    u_str = ", ".join([str(x) for x in u])
                    LOG.debug('Skipping optional %s: Unknown dataset %s',
                              str(prereq), u_str)
            else:
                prereq_ids.append(n)
        return prereq_ids, unknowns

    def _find_compositor(self, dataset_key, **kwargs):
        # NOTE: This function can not find a modifier that performs one or more modifications
        # if it has modifiers see if we can find the unmodified version first
        orig_key = dataset_key
        src_node = None
        if isinstance(dataset_key, DatasetID) and dataset_key.modifiers:
            new_prereq = DatasetID(*dataset_key[:-1] + (dataset_key.modifiers[:-1],))
            src_node, u = self._find_dependencies(new_prereq, **kwargs)
            if u:
                return None, u
            dataset_key = DatasetID(*src_node.name[:-1] + (dataset_key.modifiers,))

        try:
            compositor = self.cpl.load_compositor(
                dataset_key, self.info['sensor'])
        except KeyError:
            raise KeyError("Can't find anything called %s" %
                           (str(dataset_key),))

        dataset_key = compositor.info['id']
        # 2.1 get the prerequisites
        prereqs, unknowns = self._get_compositor_prereqs(
            compositor.info['prerequisites'],
            **kwargs)
        if unknowns:
            return None, unknowns

        optional_prereqs, _ = self._get_compositor_prereqs(
            compositor.info['optional_prerequisites'],
            skip=True,
            **kwargs)

        # Is this the right place for that?
        if src_node is not None:
            prereqs = [src_node] + prereqs
        root = Node(dataset_key, data=(compositor, prereqs, optional_prereqs))
        # LOG.debug("Found composite {}".format(str(dataset_key)))
        for prereq in prereqs + optional_prereqs:
            if prereq is not None:
                root.add_child(prereq)

        try:
            self.wishlist.remove(orig_key)
        except KeyError:
            pass
        else:
            self.wishlist.add(dataset_key)

        return root, set()

    def _find_dependencies(self,
                           dataset_key,
                           calibration=None,
                           polarization=None,
                           resolution=None,
                           **kwargs):
        """Find the dependencies for *dataset_key*."""
        # 0 check if the dataset is already loaded
        if dataset_key in self.datasets:
            return Node(dataset_key, None), set()

        # 1 try to get dataset from reader
        node = self._find_reader_dataset(dataset_key,
                                         calibration=calibration,
                                         polarization=polarization,
                                         resolution=resolution)
        if node is not None:
            return node, set()

        # 2 try to find a composite that matches
        try:
            node, unknowns = self._find_compositor(dataset_key, **kwargs)
        except KeyError:
            node = None
            unknowns = set([dataset_key])

        return node, unknowns

    def create_deptree(self,
                       dataset_keys,
                       calibration=None,
                       polarization=None,
                       resolution=None):
        """Create the dependency tree.

        Args:
            dataset_keys (iterable): Strings or DatasetIDs to find dependencies for
            calibration (iterable or None):

        Returns:
            (Node, set): Root node of the dependency tree and a set of unknown datasets

        """
        unknown_datasets = set()
        tree = DependencyTree()
        for key in dataset_keys:
            if key in self.datasets:
                n = Node(self.datasets[key].info['id'])
            else:
                n, unknowns = self._find_dependencies(key,
                                                      calibration=calibration,
                                                      polarization=polarization,
                                                      resolution=resolution)
                if unknowns:
                    unknown_datasets.update(unknowns)
                    continue
            tree.add_child(n)

        return tree, unknown_datasets

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
        loaded_datasets = {}
        for reader_name, ds_ids in reader_datasets.items():
            reader_instance = self.readers[reader_name]
            new_datasets = reader_instance.load(ds_ids, **kwargs)
            loaded_datasets.update(new_datasets)
        self.datasets.update(loaded_datasets)
        return new_datasets

    def _get_prereq_datasets(self, comp_id, prereq_nodes, wishlist, keepables, skip=False):
        prereq_datasets = []
        for prereq_node in prereq_nodes:
            prereq_id = prereq_node.name
            if prereq_id in self.datasets:
                prereq_datasets.append(self.datasets[prereq_id])
            elif prereq_id in keepables:
                keepables.add(comp_id)
                LOG.warning("Delaying generation of %s "
                            "because of dependency's delayed generation: %s",
                            comp_id, prereq_id)
            elif not skip:
                LOG.warning("Missing prerequisite for '{}': '{}'".format(
                    comp_id, prereq_id))
                if comp_id in wishlist:
                    wishlist.remove(comp_id)
                raise KeyError("Missing composite prerequisite")
            else:
                LOG.debug("Missing optional prerequisite for {}: {}".format(
                    comp_id, prereq_id))

        return prereq_datasets

    def read_composites(self, compositor_nodes):
        """Read (generate) composites.
        """
        keepables = set()
        # We need to do comparison with __eq__ which doesn't happen in a 'set'
        # but it does happen in a list. Now we can do `wishlist.remove('M05')`
        # and remove datasets
        wishlist = list(self.wishlist)

        for item in reversed(compositor_nodes):
            compositor, prereqs, optional_prereqs = item.data
            if item.name in self.datasets:
                # already loaded
                continue

            try:
                prereq_datasets = self._get_prereq_datasets(
                    item.name,
                    prereqs,
                    wishlist,
                    keepables,
                )
            except KeyError:
                continue

            optional_datasets = self._get_prereq_datasets(
                item.name,
                optional_prereqs,
                wishlist,
                keepables,
                skip=True
            )

            try:
                composite = compositor(prereq_datasets,
                                       optional_datasets=optional_datasets,
                                       **self.info)

            except IncompatibleAreas:
                LOG.warning("Delaying generation of %s "
                            "because of incompatible areas",
                            compositor.info['name'])
                preservable_datasets = set(self.datasets.keys())
                keepables |= preservable_datasets & set(prereqs + optional_prereqs)
                # even though it wasn't generated keep a list of what
                # might be needed in other compositors
                keepables.add(item.name)
                continue

            self.datasets[composite.info['id']] = composite
        self.wishlist = set(wishlist)
        return keepables

    def read(self, nodes=None, **kwargs):
        if nodes is None:
            nodes = self.dep_tree.leaves()
        return self.read_datasets(nodes, **kwargs)

    def compute(self, nodes=None):
        """Compute all the composites contained in `requirements`.
        """
        if nodes is None:
            nodes = self.dep_tree.trunk()
        return self.read_composites(nodes)

    def unload(self, keepables=None):
        """Unload all loaded composites.
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
        self.wishlist |= dataset_keys

        tree, unknown = self.create_deptree(dataset_keys,
                                            calibration=calibration,
                                            polarization=polarization,
                                            resolution=resolution)
        if unknown:
            unknown_str = ", ".join([str(x) for x in unknown])
            raise KeyError("Unknown datasets: {}".format(unknown_str))

        self.read(nodes=tree.leaves(), **kwargs)
        keepables = None
        if compute:
            keepables = self.compute(tree.trunk(), **kwargs)
        if unload:
            self.unload(keepables=keepables)

        self.dep_tree.update(tree)

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
        new_scn.cpl = self.cpl
        new_scn.dep_tree = self.dep_tree
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
            new_scn.wishlist = set([ds.info["id"] for ds in new_scn])

        # recompute anything from the wishlist that needs it (combining multiple
        # resolutions, etc.)
        keepables = None
        if compute:
            keepables = new_scn.compute()
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
        conf = configparser.RawConfigParser()
        successes = conf.read(config_files)
        if not successes:
            raise IOError("Writer configuration files do not exist: %s" %
                          (config_files, ))

        for section_name in conf.sections():
            if section_name.startswith("writer:"):
                options = dict(conf.items(section_name))
                writer_class_name = options["writer"]
                writer_class = runtime_import(writer_class_name)
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
                            overlay=overlay)

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
        config_fn = writer + ".cfg" if "." not in writer else writer
        config_files = config_search_paths(
            os.path.join("writers", config_fn), self.ppp_config_dir)
        kwargs.setdefault("config_files", config_files)
        return self.load_writer_config(**kwargs)

    def get_writer_by_ext(self, extension, **kwargs):
        mapping = {".tiff": "geotiff", ".tif": "geotiff", ".nc": "cf"}
        return self.get_writer(
            mapping.get(extension.lower(), "simple_image"), **kwargs)
