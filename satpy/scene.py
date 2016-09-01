#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010-2016

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>
#   Esben S. Nielsen <esn@dmi.dk>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Scene objects to hold satellite data.
"""

import logging
import os

from satpy.composites import IncompatibleAreas, load_compositors
from satpy.config import (config_search_paths, get_environ_config_dir,
                          runtime_import)
from satpy.projectable import InfoObject, Projectable
from satpy.readers import DatasetDict, DatasetID, ReaderFinder

try:
    import configparser
except ImportError:
    from six.moves import configparser


LOG = logging.getLogger(__name__)


class Node(object):
    """A node object."""

    def __init__(self, data):
        """Init the node object."""
        self.data = data
        self.children = []
        self.parents = []

    def add_child(self, obj):
        """Add a child to the node."""
        self.children.append(obj)
        obj.parents.append(self)

    def __str__(self):
        """Display the node."""
        return self.display()

    def display(self, previous=0):
        """Display the node."""
        return ((" +" * previous) + str(self.data) + '\n' +
                ''.join([child.display(previous + 1) for child in self.children]))

    def leaves(self):
        """Get the leaves of the tree starting at this root."""
        if not self.children:
            return [self]
        else:
            res = list()
            for child in self.children:
                res.extend(child.leaves())
            return res

    def trunk(self):
        """Get the trunk of the tree starting at this root."""
        res = []
        if self.children:
            if self.data is not None:
                res.append(self)
            for child in self.children:
                res.extend(child.trunk())
        return res


class Scene(InfoObject):
    """The almighty scene class."""

    def __init__(self, filenames=None, ppp_config_dir=None, reader=None, base_dir=None, **info):
        """The Scene object constructor.
        """
        # Get PPP_CONFIG_DIR
        self.ppp_config_dir = ppp_config_dir or get_environ_config_dir()
        # Set the PPP_CONFIG_DIR in the environment in case it's used else where in pytroll
        LOG.debug("Setting 'PPP_CONFIG_DIR' to '%s'", self.ppp_config_dir)
        os.environ["PPP_CONFIG_DIR"] = self.ppp_config_dir

        InfoObject.__init__(self, **info)
        self.readers = {}
        self.datasets = DatasetDict()
        self.compositors = {}
        self.wishlist = set()

        if filenames is not None and not filenames:
            raise ValueError("Filenames are specified but empty")

        finder = ReaderFinder(ppp_config_dir=self.ppp_config_dir, base_dir=base_dir,
                              start_time=self.info.get('start_time'),
                              end_time=self.info.get('end_time'),
                              area=self.info.get('area'),)
        reader_instances = finder(reader=reader, sensor=self.info.get("sensor"), filenames=filenames)
        # reader finder could return multiple readers
        sensors = []
        for reader_instance in reader_instances:
            if reader_instance:
                self.readers[reader_instance.name] = reader_instance
                sensors.extend(reader_instance.sensor_names)
        # if the user didn't tell us what sensors to work with, let's figure it out
        if not self.info.get("sensor"):
            self.info["sensor"] = sensors

    @property
    def start_time(self):
        return min(x.start_time for x in self.readers.values())

    @property
    def end_time(self):
        return max(x.end_time for x in self.readers.values())

    def available_datasets(self, reader_name=None, composites=False):
        """Get names of available datasets, globally or just for *reader_name* if specified, that can be loaded.

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

        available_datasets = [dataset_name for reader in readers for dataset_name in reader.available_datasets]
        if composites:
            available_datasets += list(self.available_composites(available_datasets))
        return available_datasets

    def all_datasets(self, reader_name=None, composites=False):
        """Get names of all datasets from loaded readers or `reader_name` if specified..

        :return: list of all dataset names
        """
        try:
            if reader_name:
                readers = [self.readers[reader_name]]
            else:
                readers = self.readers.values()
        except (AttributeError, KeyError):
            raise KeyError("No reader '%s' found in scene" % reader_name)

        all_datasets = [dataset_name for reader in readers for dataset_name in reader.dataset_names]
        if composites:
            all_datasets += self.all_composites()
        return all_datasets

    def available_composites(self, available_datasets=None):
        """Get names of compositors that can be generated from the available datasets.

        :return: generator of available compositor's names
        """
        if available_datasets is None:
            available_datasets = self.available_datasets(composites=False)

        available_datasets = set(available_datasets)
        # composite_objects = self.all_composites_objects()
        for composite_name, composite_obj in self.all_composites_objects().items():
            ###
            for reader_name, reader_instance in self.readers.items():
                try:
                    # overwrite any semi-qualified IDs with the fully qualified ID
                    prereqs = [reader_instance.get_dataset_key(prereq).name
                               for prereq in composite_obj.info["prerequisites"]]
                    if not (set(prereqs) - available_datasets):
                        # we have all the prereqs in the available datasets
                        yield composite_name
                except KeyError:
                    continue

    def all_composites(self):
        """Get all composite names that are configured.

        :return: generator of configured composite names
        """
        return (c.info["name"] for c in self.all_composites_objects().values())

    def all_composites_objects(self):
        """Get all compositors that are configured.

        :return: dictionary of composite name to compositor object
        """
        return load_compositors(sensor_names=self.info["sensor"], ppp_config_dir=self.ppp_config_dir)

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
            datasets_by_area.setdefault(str(ds.info["area"]), (ds.info["area"], []))
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

    def _find_dependencies(self, dataset_key, **kwargs):
        """Find the dependencies for *dataset_key*."""
        sensor_names = set()

        # 0 check if the dataset is already loaded

        if dataset_key in self.datasets:
            return Node(self.datasets[dataset_key])

        # 1 try to get dataset from reader
        for reader_name, reader_instance in self.readers.items():
            sensor_names |= set(reader_instance.sensor_names)
            try:
                ds_id = reader_instance.get_dataset_key(dataset_key)
            except KeyError:
                LOG.debug("Can't find dataset %s in reader %s", str(dataset_key), reader_name)
            else:
                try:
                    self.wishlist.remove(dataset_key)
                except KeyError:
                    pass
                else:
                    self.wishlist.add(ds_id)
                return Node(ds_id)

        # 2 try to find a composite that matches
        try:
            self.compositors.update(load_compositors([dataset_key], sensor_names,
                                                     ppp_config_dir=self.ppp_config_dir,
                                                     **kwargs))
            compositor = self.compositors[dataset_key]

        except KeyError:
            raise KeyError("Can't find anything called %s" % str(dataset_key))

        # 2.1 get the prerequisites
        prereqs = [self._find_dependencies(prereq, **kwargs)
                   for prereq in compositor.info["prerequisites"]]
        optional_prereqs = [self._find_dependencies(prereq, **kwargs)
                            for prereq in compositor.info["optional_prerequisites"]]

        root = Node((compositor, prereqs, optional_prereqs))

        for prereq in prereqs + optional_prereqs:
            if prereq is not None:
                root.add_child(prereq)

        return root

    def create_deptree(self, dataset_keys):
        """Create the dependency tree."""
        tree = Node(None)
        for key in dataset_keys:
            tree.add_child(self._find_dependencies(key))
        return tree

    def read_datasets(self, dataset_nodes, calibration=None, polarization=None, resolution=None, **kwargs):
        """Read the given datasets from file."""
        # TODO: handle optional datasets.
        sensor_names = set()
        loaded_datasets = {}
        # get datasets from readers

        if calibration is not None and not isinstance(calibration, (list, tuple)):
            calibration = [calibration]
        if resolution is not None and not isinstance(resolution, (list, tuple)):
            resolution = [resolution]
        if polarization is not None and not isinstance(polarization, (list, tuple)):
            polarization = [polarization]

        for reader_name, reader_instance in self.readers.items():

            sensor_names |= set(reader_instance.sensor_names)
            ds_ids = set()
            for node in dataset_nodes:
                dataset_key = node.data
                try:
                    ds_id = reader_instance.get_dataset_key(dataset_key,
                                                            calibration=calibration,
                                                            resolution=resolution,
                                                            polarization=polarization)
                except KeyError:
                    LOG.debug("Can't find dataset %s in reader %s", str(dataset_key), reader_name)
                try:
                    self.wishlist.remove(dataset_key)
                except KeyError:
                    pass
                else:
                    self.wishlist.add(ds_id)

                # if we haven't loaded this projectable then add it to the list to be loaded
                if ds_id not in self.datasets or not self.datasets[ds_id].is_loaded():
                    ds_ids.add(ds_id)
            new_datasets = reader_instance.load(ds_ids,
                                                start_time=kwargs.get('start_time'),
                                                end_time=kwargs.get('end_time'),
                                                area=kwargs.get('area'))
            loaded_datasets.update(new_datasets)
        self.datasets.update(loaded_datasets)
        return new_datasets

# TODO: unload unneeded stuff

    def read_composites(self, compositor_nodes, **kwargs):
        # TODO: does this work ?
        composites = []
        for item in reversed(compositor_nodes):
            compositor, prereqs, optional_prereqs = item.data
            if compositor.info['name'] not in self.datasets:
                prereqs = [self.datasets[prereq.data] for prereq in prereqs]
                optional_prereqs = [self.datasets[prereq.data] for prereq in optional_prereqs]
                composite = compositor(prereqs, optional_prereqs=optional_prereqs, **self.info)
                composite.info['name'] = compositor.info['name']

                self.datasets[compositor.info['name']] = composite
                try:
                    self.wishlist.remove(composite.info['name'])
                except KeyError:
                    pass
                else:
                    self.wishlist.add(composite.info['id'])

            composites.append(self.datasets[compositor.info['name']])
        return composites

    def read_from_deptree(self, dataset_keys, **kwargs):
        """Read the data by generating a dependency tree."""
        # TODO: handle wishlist (keepables)
        dataset_keys = set(dataset_keys)
        self.wishlist |= dataset_keys

        tree = self.create_deptree(dataset_keys)
        datasets = self.read_datasets(tree.leaves(), **kwargs)
        composites = self.read_composites(tree.trunk(), **kwargs)

    def read(self, dataset_keys, **kwargs):
        return self.read_from_deptree(dataset_keys, **kwargs)
        # return self.read_old(dataset_keys, **kwargs)

    def read_old(self, dataset_keys, calibration=None, resolution=None, polarization=None, metadata=None, **kwargs):
        """Read the composites called *dataset_keys* or their prerequisites.
        """
        # FIXME: Should this be a set?
        dataset_keys = set(dataset_keys)
        self.wishlist |= dataset_keys
        if calibration is not None and not isinstance(calibration, (list, tuple)):
            calibration = [calibration]
        if resolution is not None and not isinstance(resolution, (list, tuple)):
            resolution = [resolution]
        if polarization is not None and not isinstance(polarization, (list, tuple)):
            polarization = [polarization]

        dataset_ids = set()
        unknown_keys = set()

        # Check with all known readers to see which ones know about the requested datasets
        for reader_name, reader_instance in self.readers.items():
            for key in dataset_keys:
                try:
                    ds_id = reader_instance.get_dataset_key(key,
                                                            calibration=calibration,
                                                            resolution=resolution,
                                                            polarization=polarization)
                    # if the request wasn't a dataset id (wavelength, etc) then replace the request
                    # with the fully qualified id
                    if key != ds_id:
                        self.wishlist.remove(key)
                        self.wishlist.add(ds_id)
                    # if we haven't loaded this projectable then add it to the list to be loaded
                    if ds_id not in self.datasets or not self.datasets[ds_id].is_loaded():
                        dataset_ids.add(ds_id)
                except KeyError:
                    unknown_keys.add(key)
                    LOG.debug("Can't find dataset %s in reader %s", str(key), reader_name)

        # Get set of all names that can't be satisfied by the readers we've loaded
        # Get the list of keys that *none* of the readers knew about (assume names only for compositors)
        dataset_names = set(x.name for x in dataset_ids)
        composite_names = unknown_keys - dataset_names
        # composite_names = set(x.name for x in dataset_ids)
        sensor_names = set()
        unknown_names = set()
        # Look for compositors configurations specific to the sensors that our readers support
        for reader_instance in self.readers.values():
            sensor_names |= set(reader_instance.sensor_names)

        # If we have any composites that need to be made, then let's create the composite objects
        if composite_names:
            self.compositors.update(load_compositors(composite_names.copy(), sensor_names,
                                                     ppp_config_dir=self.ppp_config_dir,
                                                     **kwargs))
            unknown_names = set(composite_names) - set(self.compositors.keys())

        for unknown_name in unknown_names:
            LOG.warning("Unknown dataset or compositor: %s", unknown_name)

        # Don't include any of the 'unknown' projectable names
        composites_needed = set(composite for composite in self.compositors.keys()
                                if composite not in self.datasets
                                or not self[composite].is_loaded()) & composite_names
        # collect the metadata names that we will need to load to satisfy composites
        needed_metadata = set(metadata) if metadata is not None else set()
        for comp_name in composites_needed:
            needed_metadata |= set(self.compositors[comp_name].info["metadata_requirements"])

        for reader_name, reader_instance in self.readers.items():
            all_reader_datasets = set(reader_instance.datasets.keys())

            # compute the dependencies to load from file
            needed_bands = set(dataset_ids)
            while composites_needed:
                for band in composites_needed.copy():
                    # overwrite any semi-qualified IDs with the fully qualified ID
                    try:
                        compositor = self.compositors[band]
                    except KeyError:
                        self.compositors.update(load_compositors([band], sensor_names,
                                                                 ppp_config_dir=self.ppp_config_dir,
                                                                 **kwargs))
                        compositor = self.compositors[band]

                    prereqs = list()
                    for prereq in compositor.info["prerequisites"]:
                        try:
                            prereqs.append(reader_instance.get_dataset_key(prereq))
                        except KeyError:
                            composites_needed.add(prereq.name)
                            prereqs.append(prereq)
                    compositor.info["prerequisites"] = prereqs
                    needed_bands |= set(prereqs)

                    prereqs = list()
                    for prereq in compositor.info["optional_prerequisites"]:
                        try:
                            prereqs.append(reader_instance.get_dataset_key(prereq))
                        except KeyError:
                            composites_needed.add(prereq.name)
                            prereqs.append(prereq)
                    compositor.info["optional_prerequisites"] = prereqs
                    needed_bands |= set(prereqs)
                    composites_needed.remove(band)

            # A composite might use a product from another reader, so only pass along the ones we know about
            needed_bands &= all_reader_datasets
            needed_bands = set(band for band in needed_bands
                               if band not in self.datasets or not self[band].is_loaded())

            # Create datasets in reader and update the scenes datasets
            needed_bands = sorted(needed_bands)
            LOG.debug("Asking reader '%s' for the following datasets %s", reader_name, str(needed_bands))
            self.datasets.update(reader_instance.load(needed_bands, metadata=needed_metadata, **kwargs))

        # Update the scene with information contained in the files
        if not self.datasets:
            LOG.debug("No datasets loaded, can't set overall scene metadata")
            return

        # FIXME: should this really be in the scene ?
        self.info["start_time"] = min([p.info["start_time"] for p in self.datasets.values()])
        try:
            self.info["end_time"] = max([p.info["end_time"] for p in self.datasets.values()])
        except KeyError:
            pass
            # TODO: comments and history

    def compute(self, *requirements):
        """Compute all the composites contained in `requirements`.
        """
        if not requirements:
            requirements = self.wishlist.copy()
        keepables = set()
        for requirement in requirements:
            if isinstance(requirement, DatasetID) and requirement.name not in self.compositors:
                continue
            elif not isinstance(requirement, DatasetID) and requirement not in self.compositors:
                continue
            if requirement in self.datasets:
                continue
            if isinstance(requirement, DatasetID):
                requirement_name = requirement.name
            else:
                requirement_name = requirement

            compositor = self.compositors[requirement_name]
            # Compute any composites that this one depends on
            keepables |= self.compute(*compositor.info["prerequisites"])
            # Compute any composites that this composite might optionally depend on
            if compositor.info["optional_prerequisites"]:
                keepables |= self.compute(*compositor.info["optional_prerequisites"])

            # Resolve the simple name of a prereq to the fully qualified
            prereq_datasets = [self[prereq] for prereq in compositor.info["prerequisites"]]
            optional_datasets = [self[prereq] for prereq in compositor.info["optional_prerequisites"]
                                 if prereq in self]
            compositor.info["prerequisites"] = [ds.info["id"] for ds in prereq_datasets]
            compositor.info["optional_prerequisites"] = [ds.info["id"] for ds in optional_datasets]
            try:
                comp_projectable = compositor(prereq_datasets, optional_datasets=optional_datasets, **self.info)

                # validate the composite projectable
                assert("name" in comp_projectable.info)
                comp_projectable.info.setdefault("resolution", None)
                comp_projectable.info.setdefault("wavelength_range", None)
                comp_projectable.info.setdefault("polarization", None)
                comp_projectable.info.setdefault("calibration", None)
                # FIXME: Should this be a requirement of anything creating a Dataset? Special handling by .info?
                band_id = DatasetID(
                    name=comp_projectable.info["name"],
                    resolution=comp_projectable.info["resolution"],
                    wavelength=comp_projectable.info["wavelength_range"],
                    polarization=comp_projectable.info["polarization"],
                    calibration=comp_projectable.info["calibration"],
                )
                comp_projectable.info["id"] = band_id
                self.datasets[band_id] = comp_projectable

                # update the wishlist with this new dataset id
                if requirement_name in self.wishlist:
                    self.wishlist.remove(requirement_name)
                    self.wishlist.add(band_id)
            except IncompatibleAreas:
                LOG.debug("Composite '%s' could not be created because of incompatible areas", requirement_name)
                # FIXME: If a composite depends on this composite we need to notify the previous call
                preservable_datasets = set(compositor.info["prerequisites"] + compositor.info["optional_prerequisites"])
                for ds_id, projectable in self.datasets.items():
                    # FIXME: Can compositors use wavelengths or only names?
                    if ds_id in preservable_datasets:
                        keepables.add(ds_id)
        return keepables

    def unload(self, keepables=None):
        """Unload all loaded composites.
        """
        to_del = [ds_id for ds_id, projectable in self.datasets.items()
                  if ds_id not in self.wishlist and
                  (not keepables or ds_id not in keepables)]
        for ds_id in to_del:
            del self.datasets[ds_id]

    def load(self, wishlist, calibration=None, resolution=None, polarization=None, metadata=None, **kwargs):
        """Read, compute and unload.
        """
        self.read(wishlist, calibration=calibration, resolution=resolution,
                  polarization=polarization, metadata=metadata, **kwargs)
        keepables = None
        if kwargs.get("compute", True):
            keepables = self.compute()
        if kwargs.get("unload", True):
            self.unload(keepables=keepables)

    def resample(self, destination, datasets=None, compute=True, unload=True, **kwargs):
        """Resample the datasets and return a new scene.
        """
        new_scn = Scene()
        new_scn.info = self.info.copy()
        new_scn.compositors = self.compositors.copy()
        for ds_id, projectable in self.datasets.items():
            LOG.debug("Resampling %s", ds_id)
            if datasets and ds_id not in datasets:
                continue
            new_scn[ds_id] = projectable.resample(destination, **kwargs)
        # MUST set this after assigning the resampled datasets otherwise
        # composite prereqs that were resampled will be considered "wishlisted"
        if datasets is None:
            new_scn.wishlist = self.wishlist
        else:
            new_scn.wishlist = set([ds.info["id"] for ds in new_scn])

        # recompute anything from the wishlist that needs it (combining multiple resolutions, etc.)
        keepables = None
        if compute:
            keepables = new_scn.compute()
        if unload:
            new_scn.unload(keepables)

        return new_scn

    def show(self, dataset_id):
        """Show the *dataset* on screen as an image.
        """

        from satpy.writers import get_enhanced_image
        get_enhanced_image(self[dataset_id]).show()

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
            raise IOError("Writer configuration files do not exist: %s" % (config_files,))

        for section_name in conf.sections():
            if section_name.startswith("writer:"):
                options = dict(conf.items(section_name))
                writer_class_name = options["writer"]
                writer_class = runtime_import(writer_class_name)
                writer = writer_class(ppp_config_dir=self.ppp_config_dir, config_file=config_files, **kwargs)
                return writer

    def save_dataset(self, dataset_id, filename=None, writer=None, **kwargs):
        """Save the *dataset_id* to file using *writer* (geotiff by default).
        """
        if writer is None:
            if filename is None:
                writer = self.get_writer("geotiff", **kwargs)
            else:
                writer = self.get_writer_by_ext(os.path.splitext(filename)[1], **kwargs)
        else:
            writer = self.get_writer(writer, **kwargs)
        writer.save_dataset(self[dataset_id], filename=filename)

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
        config_files = config_search_paths(os.path.join("writers", config_fn), self.ppp_config_dir)
        kwargs.setdefault("config_files", config_files)
        return self.load_writer_config(**kwargs)

    def get_writer_by_ext(self, extension, **kwargs):
        mapping = {".tiff": "geotiff",
                   ".tif":  "geotiff",
                   }
        return self.get_writer(mapping.get(extension.lower(), "simple_image"), **kwargs)
