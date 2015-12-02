#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2010, 2011, 2012, 2013, 2014, 2015

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

try:
    import configparser
except:
    from six.moves import configparser
import os
import logging

from mpop import runtime_import, config_search_paths, get_environ_config_dir
from mpop.projectable import Projectable, InfoObject
from mpop import PACKAGE_CONFIG_PATH
from mpop.readers import ReaderFinder, DatasetDict, DatasetID

from mpop.utils import debug_on
debug_on()
LOG = logging.getLogger(__name__)


class IncompatibleAreas(Exception):
    """
    Error raised upon compositing things of different shapes.
    """
    pass


class Scene(InfoObject):
    """
    The almighty scene class.
    """

    def __init__(self, filenames=None, ppp_config_dir=None, reader_name=None, base_dir=None, **info):
        """platform_name=None, sensor=None, start_time=None, end_time=None,
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
        self._composite_configs = set()
        if filenames is not None and not filenames:
            raise ValueError("Filenames are specified but empty")

        finder = ReaderFinder(ppp_config_dir=self.ppp_config_dir, base_dir=base_dir, **self.info)
        reader_instances = finder(reader_name=reader_name, sensor=self.info.get("sensor"), filenames=filenames)
        # reader finder could return multiple readers
        for reader_instance in reader_instances:
            if reader_instance:
                self.readers[reader_instance.name] = reader_instance

    def read_composites_config(self, composite_configs=None, sensor=None, names=None, **kwargs):
        """Read the (generic) *composite_config* for *sensor* and *names*.
        """
        if not composite_configs:
            composite_configs = config_search_paths(os.path.join("composites", "generic.cfg"), self.ppp_config_dir)

        conf = configparser.ConfigParser()
        conf.read(composite_configs)
        compositors = {}
        for section_name in conf.sections():
            if section_name.startswith("composite:"):
                options = dict(conf.items(section_name))
                options["sensor"] = options.setdefault("sensor", None)
                if options["sensor"]:
                    options["sensor"] = set(options["sensor"].split(","))
                    if len(options["sensor"]) == 1:
                        # FIXME: Finalize how multiple sensors and platforms work
                        options["sensor"] = options["sensor"].pop()
                comp_cls = options.pop("compositor", None)
                if not comp_cls:
                    raise ValueError("'compositor' missing or empty in config files: %s" % (composite_configs,))

                # Check if the caller only wants composites for a certain sensor
                if sensor is not None and sensor not in options["sensor"]:
                    continue
                # Check if the caller only wants composites with certain names
                if names and options["name"] not in names:
                    continue

                if options["name"] in self.compositors:
                    LOG.debug("Duplicate composite found, previous composite '%s' will be overwritten",
                                options["name"])

                # Get other identifiers that could be used to filter the prerequisites
                other_identifiers = {}
                for o_id in ["resolution", "calibration", "polarization"]:
                    if o_id in options:
                        other_identifiers[o_id] = options[o_id].split(",")
                optional_other_identifiers = {}
                for o_id in ["resolution", "calibration", "polarization"]:
                    if "optional_" + o_id in options:
                        optional_other_identifiers[o_id] = options["optional_" + o_id].split(",")

                def _normalize_prereqs(prereqs, other_identifiers):
                    # Pull out prerequisites
                    prerequisites = options["prerequisites"].split(",")
                    prereqs = []
                    for idx, prerequisite in enumerate(prerequisites):
                        ds_id = {"name": None, "wavelength": None}
                        # convert the prerequisite
                        try:
                            # prereqs can be wavelengths
                            ds_id["wavelength"] = float(prerequisite)
                        except ValueError:
                            # or names
                            ds_id["name"] = prerequisite

                        # add further filtering to the prerequisites via the DatasetID namedtuple
                        for o_id, vals in other_identifiers.items():
                            ds_id[o_id] = vals[idx]

                        prereqs.append(DatasetID(**ds_id))

                    return prereqs

                if "prerequisites" in options:
                    options["prerequisites"] = _normalize_prereqs(options["prerequisites"], other_identifiers)

                if "optional_prerequisites" in options:
                    options["optional_prerequisites"] = _normalize_prereqs(options["optional_prerequisites"],
                                                                           optional_other_identifiers)

                if "metadata_requirements" in options:
                    options["metadata_requirements"] = options["metadata_requirements"].split(",")

                try:
                    loader = runtime_import(comp_cls)
                except ImportError:
                    LOG.warning("Could not import composite class '%s' for"
                                " compositor '%s'", comp_cls, options["name"])
                    continue

                options.update(**kwargs)
                comp = loader(**options)
                compositors[options["name"]] = comp
        return compositors

    def available_datasets(self, reader_name=None):
        """Return the available datasets, globally or just for *reader_name* if specified.
        """
        try:
            if reader_name:
                readers = [getattr(self, reader_name)]
            else:
                readers = self.readers.values()
        except (AttributeError, KeyError):
            raise KeyError("No reader '%s' found in scene" % reader_name)

        return [dataset_name for reader in readers for dataset_name in reader.dataset_names]

    def __str__(self):
        res = (str(proj) for proj in self.datasets.values())
        return "\n".join(res)

    def __iter__(self):
        return self.datasets.itervalues()

    def __getitem__(self, key):
        return self.datasets[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Projectable):
            raise ValueError("Only 'Projectable' objects can be assigned")
        self.datasets[key] = value

    def __delitem__(self, key):
        k = self.datasets.get_key(key)
        self.wishlist.remove(k)
        del self.datasets[k]


    def __contains__(self, name):
        return name in self.datasets

    def load_compositors(self, composite_names, sensor_names, **kwargs):
        """Load the compositors for *composite_names* for the given *sensor_names*
        """
        # Don't look for any composites that we may have loaded before
        composite_names -= set(self.compositors.keys())
        if not composite_names:
            LOG.debug("Already loaded needed composites")
            return composite_names

        # Load generic composites first
        # we haven't found them all yet, let's check the global composites config
        config_fn = "generic.cfg"
        if config_fn in self._composite_configs:
            LOG.debug("Generic composites already loaded, won't reload: %s", config_fn)
        else:
            composite_configs = config_search_paths(os.path.join("composites", config_fn), self.ppp_config_dir)
            if composite_configs:
                global_compositors = self.read_composites_config(composite_configs, names=composite_names, **kwargs)
                # Update the set of configs we've read already
                self._composite_configs.add(config_fn)
                # Update the list of composites the scene knows about
                self.compositors.update(global_compositors)
                # Remove the names we know how to create now
                composite_names -= set(global_compositors.keys())
            else:
                LOG.warning("No global composites/generic.cfg file found in config directory")

        # Check the composites for each particular sensor (may overwrite generic composites so load them all)
        for sensor_name in sensor_names:
            config_fn = sensor_name + ".cfg"
            if config_fn in self._composite_configs:
                LOG.debug("Sensor composites already loaded, won't reload: %s", config_fn)
                continue
            sensor_composite_configs = config_search_paths(os.path.join("composites", config_fn), self.ppp_config_dir)
            if not sensor_composite_configs:
                LOG.debug("No sensor composite config found for %s", config_fn)
                continue

            # Load all the compositors for this sensor for the needed names from the specified config
            sensor_compositors = self.read_composites_config(sensor_composite_configs, sensor_name, composite_names,
                                                             **kwargs)
            # Update the set of configs we've read already
            self._composite_configs.add(config_fn)
            # Update the list of composites the scene knows about
            self.compositors.update(sensor_compositors)
            # Remove the names we know how to create now
            composite_names -= set(sensor_compositors.keys())

        return composite_names

    def read(self, dataset_keys, calibration=None, resolution=None, polarization=None, metadata=None, **kwargs):
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
            unknown_names = self.load_compositors(composite_names.copy(), sensor_names, **kwargs)

        for unknown_name in unknown_names:
            LOG.warning("Unknown dataset or compositor: %s", unknown_name)

        # Don't include any of the 'unknown' projectable names
        composites_needed = set(composite for composite in self.compositors.keys()
                                if composite not in self.datasets or not self[
            composite].is_loaded()) & composite_names
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
                    new_bands = set(reader_instance.get_dataset_key(prereq)
                                        for prereq in self.compositors[band].prerequisites)
                    needed_bands |= new_bands
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
            # Compute any composites that this one depends on
            self.compute(*self.compositors[requirement_name].prerequisites)

            # TODO: Get non-projectable dependencies like moon illumination fraction
            prereq_projectables = [self[prereq] for prereq in self.compositors[requirement_name].prerequisites]
            try:
                comp_projectable = self.compositors[requirement_name](prereq_projectables, **self.info)

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
                for ds_id, projectable in self.datasets.iteritems():
                    # FIXME: Can compositors use wavelengths or only names?
                    if ds_id.name in self.compositors[requirement_name].prerequisites:
                        projectable.info["keep"] = True

    def unload(self):
        """Unload all loaded composites.
        """
        to_del = [ds_id for ds_id, projectable in self.datasets.items()
                  if ds_id not in self.wishlist and
                  not projectable.info.get("keep", False)]
        for ds_id in to_del:
            del self.datasets[ds_id]

    def load(self, wishlist, calibration=None, resolution=None, polarization=None, metadata=None, **kwargs):
        """Read, compute and unload.
        """
        self.read(wishlist, calibration=calibration, resolution=resolution, polarization=polarization, metadata=metadata, **kwargs)
        if kwargs.get("compute", True):
            self.compute()
        if kwargs.get("unload", True):
            self.unload()

    def resample(self, destination, datasets=None, **kwargs):
        """Resample the datasets and return a new scene.
        """
        new_scn = Scene()
        new_scn.info = self.info.copy()
        new_scn.wishlist = self.wishlist
        for ds_id, projectable in self.datasets.items():
            LOG.debug("Resampling %s", ds_id)
            if datasets and ds_id.name not in datasets:
                continue
            new_scn[ds_id] = projectable.resample(destination, **kwargs)

        # recompute anything from the wishlist that needs it (combining multiple resolutions, etc.)
        new_scn.compute()

        return new_scn

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

    def save_images(self, writer="geotiff", **kwargs):
        writer = self.get_writer(writer, **kwargs)
        for projectable in self.datasets.values():
            writer.save_dataset(projectable, **kwargs)

    def get_writer(self, writer="geotiff", **kwargs):
        config_fn = writer + ".cfg" if "." not in writer else writer
        config_files = config_search_paths(os.path.join("writers", config_fn), self.ppp_config_dir)
        kwargs.setdefault("config_files", config_files)
        return self.load_writer_config(**kwargs)