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
from satpy.node import Node
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
        self.compositors = {}
        self.wishlist = set()

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
            available_datasets += list(self.available_composites(
                available_datasets))
        return available_datasets

    def available_dataset_names(self, reader_name=None, composites=False):
        """Get the list of the names of the available datasets."""
        return list(set(x.name if isinstance(x, DatasetID) else x
                        for x in self.available_dataset_ids(reader_name=reader_name,
                                                            composites=composites)))

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
        return [x.name if isinstance(x, DatasetID) else x
                for x in self.all_dataset_ids(reader_name=reader_name, composites=composites)]

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

        return list(set(composites))

    def available_composites(self, available_datasets=None, modified=False):
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

        tree, unknowns = self.create_deptree(available_datasets + self.all_composites())
        return self._compositor_list(tree, modified=modified)

    def all_composites(self):
        """Get all composite IDs that are configured.

        :return: generator of configured composite names
        """
        return self.all_composite_objects().keys()

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
                return Node(ds_id)

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
        try:
            self.compositors[dataset_key] = self.cpl.load_compositor(
                dataset_key, self.info['sensor'])
            compositor = self.compositors[dataset_key]
        except KeyError:
            raise KeyError("Can't find anything called %s" %
                           (str(dataset_key),))

        # 2.1 get the prerequisites
        prereqs, unknowns = self._get_compositor_prereqs(
            compositor.info['prerequisites'],
            **kwargs)
        if unknowns:
            return None, unknowns

        dataset_key = self._merge_modified_id(dataset_key, prereqs[0].name)

        optional_prereqs, _ = self._get_compositor_prereqs(
            compositor.info['optional_prerequisites'],
            skip=True,
            **kwargs)

        # Is this the right place for that?
        root = Node(dataset_key, data=(compositor, prereqs, optional_prereqs))
        # LOG.debug("Found composite {}".format(str(dataset_key)))
        for prereq in prereqs + optional_prereqs:
            if prereq is not None:
                root.add_child(prereq)

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
            return Node(dataset_key, self.datasets[dataset_key])

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
        tree = Node(None)
        for key in dataset_keys:
            n, unknowns = self._find_dependencies(key,
                                                  calibration=calibration,
                                                  polarization=polarization,
                                                  resolution=resolution)
            if unknowns:
                unknown_datasets.update(unknowns)
                continue
            tree.add_child(n)

        return tree, unknown_datasets

    def read_datasets(self,
                      dataset_nodes,
                      calibration=None,
                      polarization=None,
                      resolution=None,
                      **kwargs):
        """Read the given datasets from file."""
        loaded_datasets = {}
        # get datasets from readers

        if calibration is not None and not isinstance(calibration,
                                                      (list, tuple)):
            calibration = [calibration]
        if resolution is not None and not isinstance(resolution,
                                                     (list, tuple)):
            resolution = [resolution]
        if polarization is not None and not isinstance(polarization,
                                                       (list, tuple)):
            polarization = [polarization]

        for reader_name, reader_instance in self.readers.items():
            ds_ids = set()
            for node in dataset_nodes:
                dataset_key = node.name
                if node.data is not None:
                    # we already loaded this in a previous call to `.load`
                    continue

                try:
                    ds_id = reader_instance.get_dataset_key(
                        dataset_key,
                        calibration=calibration,
                        resolution=resolution,
                        polarization=polarization)
                except KeyError:
                    LOG.debug("Can't find dataset %s in reader %s",
                              str(dataset_key), reader_name)
                    ds_id = dataset_key
                try:  # FIXME: Does this need to exist now that Nodes are fully resolved? Yes? In case the loading better resolved the ID?
                    self.wishlist.remove(dataset_key)
                except KeyError:
                    pass
                else:
                    self.wishlist.add(ds_id)

                # if we haven't loaded this projectable then add it to the list
                # to be loaded
                if (ds_id not in self.datasets or
                        not self.datasets[ds_id].is_loaded()):
                    ds_ids.add(ds_id)
            new_datasets = reader_instance.load(ds_ids, **kwargs)
            loaded_datasets.update(new_datasets)
        self.datasets.update(loaded_datasets)
        return new_datasets

# TODO: unload unneeded stuff

    def read_composites(self, compositor_nodes, **kwargs):
        """Read (generate) composites.
        """
        composites = []
        # We need to do comparison with __eq__ which doesn't happen in a 'set'
        # but it does happen in a list. Now we can do `wishlist.remove('M05')`
        # and remove datasets
        wishlist = list(self.wishlist)

        for item in reversed(compositor_nodes):
            compositor, prereqs, optional_prereqs = item.data
            if item.name not in self.datasets:
                new_prereqs = [prereq.name for prereq in prereqs]
                new_opt_prereqs = [prereq.name for prereq in optional_prereqs]

                try:
                    prereqs = [self.datasets[prereq] for prereq in new_prereqs]
                except KeyError as e:
                    LOG.warning("Missing composite '{}' prerequisite: {}".format(
                        compositor.info['name'], e.message))
                    wishlist.remove(compositor.info['name'])
                    continue

                # Any missing optional prerequisites are replaced with 'None'
                optional_prereqs = [self.datasets.get(prereq)
                                    for prereq in new_opt_prereqs]

                try:
                    composite = compositor(prereqs,
                                           optional_datasets=optional_prereqs,
                                           **self.info)

                except IncompatibleAreas:
                    LOG.warning("Delaying generation of %s "
                                "because of incompatible areas",
                                compositor.info['name'])
                    continue

                # could be requesting by compositor name or DatasetID
                if composite.info.get('id') is None:
                    raise ValueError("Compositor '%s' did not assign an 'id'" % (compositor.info['name']))
                elif isinstance(composite.info['id'], DatasetID):
                    for idx, id_field in enumerate(DATASET_KEYS):
                        # don't check for existence, force them
                        composite.info[id_field] = composite.info['id'][idx]
                self.datasets[composite.info['id']] = composite
                # overwrite the node in case the ID was changed during creation
                item.name = composite.info['id']

                try:
                    wishlist.remove(composite.info['name'])
                except ValueError:
                    pass
                else:
                    wishlist.append(composite.info['id'])

            composites.append(self.datasets[composite.info['id']])
        self.wishlist = set(wishlist)
        return composites

    def read_from_deptree(self, dataset_keys, **kwargs):
        """Read the data by generating a dependency tree."""
        # TODO: handle wishlist (keepables)
        dataset_keys = set(dataset_keys)
        self.wishlist |= dataset_keys

        tree, unknown = self.create_deptree(dataset_keys,
                                            calibration=kwargs.get('calibration'),
                                            polarization=kwargs.get('polarization'),
                                            resolution=kwargs.get('resolution'))
        if unknown:
            unknown_str = ", ".join([str(x) for x in unknown])
            raise KeyError("Unknown datasets: {}".format(unknown_str))

        datasets = self.read_datasets(tree.leaves(), **kwargs)
        composites = self.read_composites(tree.trunk(), **kwargs)

    def read(self, dataset_keys, **kwargs):
        return self.read_from_deptree(dataset_keys, **kwargs)

    def compute(self, *requirements):
        """Compute all the composites contained in `requirements`.
        """
        if not requirements:
            requirements = self.wishlist.copy()
        keepables = set()
        for requirement in requirements:
            if isinstance(
                    requirement,
                    DatasetID) and requirement.name not in self.compositors:
                continue
            elif not isinstance(
                    requirement,
                    DatasetID) and requirement not in self.compositors:
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
            # Compute any composites that this composite might optionally
            # depend on
            if compositor.info["optional_prerequisites"]:
                keepables |= self.compute(
                    *compositor.info["optional_prerequisites"])

            # Resolve the simple name of a prereq to the fully qualified
            prereq_datasets = [self[prereq]
                               for prereq in compositor.info["prerequisites"]]
            optional_datasets = [
                self[prereq]
                for prereq in compositor.info["optional_prerequisites"]
                if prereq in self
            ]
            compositor.info["prerequisites"] = [ds.info["id"]
                                                for ds in prereq_datasets]
            compositor.info["optional_prerequisites"] = [
                ds.info["id"] for ds in optional_datasets
            ]
            try:
                comp_projectable = compositor(
                    prereq_datasets,
                    optional_datasets=optional_datasets,
                    **self.info)

                # validate the composite projectable
                assert ("name" in comp_projectable.info)
                comp_projectable.info.setdefault("resolution", None)
                comp_projectable.info.setdefault("wavelength_range", None)
                comp_projectable.info.setdefault("polarization", None)
                comp_projectable.info.setdefault("calibration", None)
                comp_projectable.info.setdefault("modifiers", None)

                # FIXME: Should this be a requirement of anything creating a
                # Dataset? Special handling by .info?
                band_id = DatasetID(
                    name=comp_projectable.info["name"],
                    resolution=comp_projectable.info["resolution"],
                    wavelength=comp_projectable.info["wavelength_range"],
                    polarization=comp_projectable.info["polarization"],
                    calibration=comp_projectable.info["calibration"],
                    modifiers=comp_projectable.info["modifiers"])
                comp_projectable.info["id"] = band_id
                self.datasets[band_id] = comp_projectable

                # update the wishlist with this new dataset id
                if requirement_name in self.wishlist:
                    self.wishlist.remove(requirement_name)
                    self.wishlist.add(band_id)
            except IncompatibleAreas:
                LOG.debug("Composite '%s' could not be created because of"
                          " incompatible areas", requirement_name)
                # FIXME: If a composite depends on this composite we need to
                # notify the previous call
                preservable_datasets = set(compositor.info["prerequisites"] +
                                           compositor.info[
                                               "optional_prerequisites"])
                for ds_id, projectable in self.datasets.items():
                    # FIXME: Can compositors use wavelengths or only names?
                    if ds_id in preservable_datasets:
                        keepables.add(ds_id)
        return keepables

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
        self.read(wishlist,
                  calibration=calibration,
                  resolution=resolution,
                  polarization=polarization,
                  **kwargs)
        keepables = None
        if compute:
            keepables = self.compute()
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
        new_scn.compositors = self.compositors.copy()
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
