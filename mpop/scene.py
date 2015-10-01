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

import numbers
import ConfigParser
import os
import trollsift
import glob
import fnmatch
from mpop.utils import debug_on
debug_on()
from mpop.projectable import Projectable, InfoObject
from mpop import PACKAGE_CONFIG_PATH
from datetime import datetime, timedelta
import logging

LOG = logging.getLogger(__name__)


class IncompatibleAreas(StandardError):
    """
    Error raised upon compositing things of different shapes.
    """
    pass


class Scene(InfoObject):
    """
    The almighty scene class.
    """
    def __init__(self, filenames=None, ppp_config_dir=None, reader=None, **info):
        """platform_name=None, sensor=None, start_time=None, end_time=None,
        """
        # Get PPP_CONFIG_DIR
        self.ppp_config_dir = ppp_config_dir or os.environ.get("PPP_CONFIG_DIR", PACKAGE_CONFIG_PATH)
        # Set the PPP_CONFIG_DIR in the environment in case it's used else where in pytroll
        LOG.debug("Setting 'PPP_CONFIG_DIR' to '%s'", self.ppp_config_dir)
        os.environ["PPP_CONFIG_DIR"] = self.ppp_config_dir

        InfoObject.__init__(self, **info)
        self.readers = {}
        self.projectables = {}
        self.compositors = {}
        self.wishlist = []
        self._composite_configs = set()
        if filenames is not None and not filenames:
            raise ValueError("Filenames are specified but empty")

        if reader is not None:
            self._find_reader(reader, filenames)
        elif "sensor" in self.info:
            self._find_sensors_readers(self.info["sensor"], filenames)
        elif filenames is not None:
            self._find_files_readers(*filenames)

    def _find_sensors_readers(self, sensor, filenames):
        """Find the readers for the given *sensor* and *filenames*
        """
        if isinstance(sensor, (str, unicode)):
            sensor_set = set([sensor])
        else:
            sensor_set = set(sensor)
        for config_file in glob.glob(os.path.join(self.ppp_config_dir, "readers", "*.cfg")):
            try:
                reader_info = self._read_reader_config(config_file)
                LOG.debug("Successfully read reader config: %s", config_file)
            except ValueError:
                LOG.debug("Invalid reader config found: %s", config_file)
                continue

            if "sensor" in reader_info and (set(reader_info["sensor"]) & sensor_set):
                # we want this reader
                if filenames:
                    # returns a copy of the filenames remaining to be matched
                    filenames = self.assign_matching_files(reader_info, *filenames)
                    if filenames:
                        raise IOError("Don't know how to open the following files: %s" % str(filenames))
                else:
                    # find the files for this reader based on its file patterns
                    reader_info["filenames"] = self.get_filenames(reader_info)
                    if not reader_info["filenames"]:
                        LOG.warning("No filenames found for reader: %s", reader_info["name"])
                        continue
                self._load_reader(reader_info)

    def _find_reader(self, reader, filenames):
        """Find and get info for the *reader* for *filenames*
        """
        config_file = reader
        # were we given a path to a config file?
        if not os.path.exists(config_file):
            # no, we were given a name of a reader
            config_fn = reader + ".cfg"  # assumes no extension was given on the reader name
            config_file = os.path.join(self.ppp_config_dir, "readers", config_fn)
            if not os.path.exists(config_file):
                raise ValueError("Can't find config file for reader: %s" % (reader,))

        reader_info = self._read_reader_config(config_file)
        if filenames:
            filenames = self.assign_matching_files(reader_info, *filenames)
            if filenames:
                raise IOError("Don't know how to open the following files: %s" % str(filenames))
        else:
            reader_info["filenames"] = self.get_filenames(reader_info)
            if not reader_info["filenames"]:
                raise RuntimeError("No filenames found for reader: %s" % (reader_info["name"],))

        self._load_reader(reader_info)

    def _find_files_readers(self, *files):
        """Find the reader info for the provided *files*.
        """
        for config_file in glob.glob(os.path.join(self.ppp_config_dir, "readers", "*.cfg")):
            try:
                reader_info = self._read_reader_config(config_file)
                LOG.debug("Successfully read reader config: %s", config_file)
            except ValueError:
                LOG.debug("Invalid reader config found: %s", config_file)
                continue

            files = self.assign_matching_files(reader_info, *files)

            if reader_info["filenames"]:
                # we have some files for this reader so let's create it
                self._load_reader(reader_info)

            if not files:
                break
        if files:
            raise IOError("Don't know how to open the following files: %s" % str(files))

    def get_filenames(self, reader_info):
        """Get the filenames from disk given the patterns in *reader_info*.
        This assumes that the scene info contains start_time at least (possibly end_time too).
        """

        filenames = []
        info = self.info.copy()
        for key in info.keys():
            if key.endswith("_time"):
                info.pop(key, None)

        reader_start = reader_info["start_time"]
        reader_end = reader_info.get("end_time")
        if reader_start is None:
            raise ValueError("'start_time' keyword required with 'sensor' and 'reader' keyword arguments")

        for pattern in reader_info["file_patterns"]:
            parser = trollsift.parser.Parser(pattern)
            # FIXME: what if we are browsing a huge archive ?
            for filename in glob.iglob(parser.globify(info.copy())):
                try:
                    metadata = parser.parse(filename)
                except ValueError:
                    LOG.info("Can't get any metadata from filename: %s from %s", pattern, filename)
                    metadata = {}
                if "end_time" in metadata and metadata["start_time"] > metadata["end_time"]:
                    mdate = metadata["start_time"].date()
                    mtime = metadata["end_time"].time()
                    if mtime < metadata["start_time"].time():
                        mdate += timedelta(days=1)
                    metadata["end_time"] = datetime.combine(mdate, mtime)
                meta_start = metadata.get("start_time", metadata.get("nominal_time", None))
                meta_end = metadata.get("end_time", datetime(1950, 1, 1))
                if reader_end:
                    # get the data within the time interval
                    if ((reader_start <= meta_start <= reader_end) or
                            (reader_start <= meta_end <= reader_end)):
                        filenames.append(filename)
                else:
                    # get the data containing start_time
                    if "end_time" in metadata and meta_start <= reader_start <= meta_end:
                        filenames.append(filename)
                    elif meta_start == reader_start:
                        filenames.append(filename)
        return sorted(filenames)

    def read_composites_config(self, composite_config=None, sensor=None, names=None, **kwargs):
        """Read the (generic) *composite_config* for *sensor* and *names*.
        """
        if composite_config is None:
            composite_config = os.path.join(self.ppp_config_dir, "composites", "generic.cfg")

        conf = ConfigParser.ConfigParser()
        conf.read(composite_config)
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
                comp_cls = options.get("compositor", None)
                if not comp_cls:
                    raise ValueError("'compositor' missing or empty in config file: %s" % (composite_config,))

                # Check if the caller only wants composites for a certain sensor
                if sensor is not None and sensor not in options["sensor"]:
                    continue
                # Check if the caller only wants composites with certain names
                if not names and options["name"] not in names:
                    continue

                # FIXME: this warns also when rereading a composite.
                if options["name"] in self.compositors:
                    LOG.warning("Duplicate composite found, previous composite '%s' will be overwritten",
                                options["name"])

                try:
                    loader = self._runtime_import(comp_cls)
                except ImportError:
                    LOG.warning("Could not import composite class '%s' for"
                                " compositor '%s'", comp_cls, options["name"])
                    continue

                options.update(**kwargs)
                comp = loader(**options)
                compositors[options["name"]] = comp
        return compositors

    def _read_reader_config(self, cfg_file):
        """Read the reader *cfg_file* and return the info extracted.
        """
        if not os.path.exists(cfg_file):
            raise IOError("No such file: " + cfg_file)

        conf = ConfigParser.RawConfigParser()

        conf.read(cfg_file)
        file_patterns = []
        sensors = set()
        reader_name = None
        reader_class = None
        reader_info = None
        # Only one reader: section per config file
        for section in conf.sections():
            if section.startswith("reader:"):
                reader_info = dict(conf.items(section))
                reader_info["file_patterns"] = reader_info.setdefault("file_patterns", "").split(",")
                reader_info["sensor"] = reader_info.setdefault("sensor", "").split(",")
                # XXX: Readers can have separate start/end times from the
                # rest fo the scene...might be a bad idea?
                reader_info.setdefault("start_time", self.info.get("start_time", None))
                reader_info.setdefault("end_time", self.info.get("end_time", None))
                reader_info.setdefault("area", self.info.get("area", None))
                try:
                    reader_class = reader_info["reader"]
                    reader_name = reader_info["name"]
                except KeyError:
                    break
                file_patterns.extend(reader_info["file_patterns"])

                if reader_info["sensor"]:
                    sensors |= set(reader_info["sensor"])
            else:
                try:
                    file_patterns.extend(conf.get(section, "file_patterns").split(","))
                except ConfigParser.NoOptionError:
                    pass

                try:
                    sensors |= set(conf.get(section, "sensor").split(","))
                except ConfigParser.NoOptionError:
                    pass

        if reader_class is None:
            raise ValueError("Malformed config file %s: missing reader 'reader'" % cfg_file)
        if reader_name is None:
            raise ValueError("Malformed config file %s: missing reader 'name'" % cfg_file)
        reader_info["file_patterns"] = file_patterns
        reader_info["config_file"] = cfg_file
        reader_info["filenames"] = []
        reader_info["sensor"] = tuple(sensors)

        return reader_info

    def _load_reader(self, reader_info):
        """Import and setup the reader from *reader_info*
        """
        try:
            loader = self._runtime_import(reader_info["reader"])
        except ImportError:
            raise ImportError("Could not import reader class '%s' for reader '%s'" % (reader_info["reader"],
                                                                                      reader_info["name"]))

        reader_instance = loader(**reader_info)
        self.readers[reader_info["name"]] = reader_instance
        return reader_instance

    def available_datasets(self, reader_name=None):
        """Return the available datasets, globally or just for *reader_name* if specified.
        """
        try:
            if reader_name:
                readers = [getattr(self, reader_name)]
            else:
                readers = self.readers
        except (AttributeError, KeyError):
            raise KeyError("No reader '%s' found in scene")

        return [dataset_name for reader_name in readers for dataset_name in reader_name.dataset_names]

    def __str__(self):
        res = (str(proj) for proj in self.projectables.values())
        return "\n".join(res)

    def __iter__(self):
        return self.projectables.itervalues()

    def __getitem__(self, key):
        # get by wavelength
        if isinstance(key, numbers.Number):
            datasets = [ds for ds in self.projectables.values()
                        if("wavelength_range" in ds.info and
                           ds.info["wavelength_range"][0] <= key <= ds.info["wavelength_range"][2])]
            datasets = sorted(datasets,
                              lambda ch1, ch2:
                              cmp(abs(ch1.info["wavelength_range"][1] - key),
                                  abs(ch2.info["wavelength_range"][1] - key)))

            if not datasets:
                raise KeyError("Can't find any projectable at %gum" % key)
            return datasets[0]
        # get by name
        else:
            for bid, ds in self.projectables.items():
                try:
                    if key == bid or bid.name == key:
                        return ds
                except AttributeError:
                    pass

            raise KeyError("Can't find any projectable called '%s'" % key)

    def __setitem__(self, key, value):
        if not isinstance(value, Projectable):
            raise ValueError("Only 'Projectable' objects can be assigned")
        self.projectables[key] = value
        value.info["name"] = key

    def __delitem__(self, key):
        projectable = self[key]
        del self.projectables[projectable.info["name"]]

    def __contains__(self, name):
        return name in self.projectables

    @staticmethod
    def assign_matching_files(reader_info, *files):
        """Assign *files* to the *reader_info*
        """
        files = list(files)
        for file_pattern in reader_info["file_patterns"]:
            pattern = trollsift.globify(file_pattern)
            for filename in list(files):
                if fnmatch.fnmatch(os.path.basename(filename), os.path.basename(pattern)):
                    reader_info["filenames"].append(filename)
                    files.remove(filename)

        # return remaining/unmatched files
        return files

    @staticmethod
    def _runtime_import(object_path):
        """Import at runtime
        """
        obj_module, obj_element = object_path.rsplit(".", 1)
        loader = __import__(obj_module, globals(), locals(), [obj_element])
        return getattr(loader, obj_element)

    def load_compositors(self, composite_names, sensor_names, **kwargs):
        """Load the compositors for *composite_names* for the given *sensor_names*
        """
        # Don't look for any composites that we may have loaded before
        composite_names -= set(self.compositors.keys())
        if not composite_names:
            LOG.debug("Already loaded needed composites")
            return composite_names

        # Check the composites for each particular sensor first
        for sensor_name in sensor_names:
            sensor_composite_config = os.path.join(self.ppp_config_dir, "composites", sensor_name + ".cfg")
            if sensor_composite_config in self._composite_configs:
                LOG.debug("Sensor composites already loaded, won't reload: %s", sensor_composite_config)
                continue
            if not os.path.isfile(sensor_composite_config):
                LOG.debug("No sensor composite config found at %s", sensor_composite_config)
                continue

            # Load all the compositors for this sensor for the needed names from the specified config
            sensor_compositors = self.read_composites_config(sensor_composite_config, sensor_name, composite_names,
                                                             **kwargs)
            # Update the set of configs we've read already
            self._composite_configs.add(sensor_composite_config)
            # Update the list of composites the scene knows about
            self.compositors.update(sensor_compositors)
            # Remove the names we know how to create now
            composite_names -= set(sensor_compositors.keys())

            if not composite_names:
                # we found them all!
                break
        else:
            # we haven't found them all yet, let's check the global composites config
            composite_config = os.path.join(self.ppp_config_dir, "composites", "generic.cfg")
            if composite_config in self._composite_configs:
                LOG.debug("Generic composites already loaded, won't reload: %s", composite_config)
            elif os.path.isfile(composite_config):
                global_compositors = self.read_composites_config(composite_config, names=composite_names, **kwargs)
                # Update the set of configs we've read already
                self._composite_configs.add(composite_config)
                # Update the list of composites the scene knows about
                self.compositors.update(global_compositors)
                # Remove the names we know how to create now
                composite_names -= set(global_compositors.keys())
            else:
                LOG.warning("No global composites/generic.cfg file found in config directory")

        return composite_names

    def read(self, projectable_keys, **kwargs):
        """Read the composites called *projectable_keys* or their prerequisites.
        """
        self.wishlist = list(projectable_keys)

        projectable_names = set()

        for reader_name, reader_instance in self.readers.items():
            for key in projectable_keys:
                try:
                    ds_info = reader_instance.get_dataset(key)
                    projectable_name = ds_info["name"]
                    if key != projectable_name:
                        self.wishlist.remove(key)
                        self.wishlist.append(projectable_name)
                    if projectable_name not in self.projectables or not self.projectables[projectable_name].is_loaded():
                        projectable_names.add(projectable_name)
                except KeyError:
                    projectable_names.add(key)
                    LOG.debug("Can't find dataset %s in reader %s", str(key), reader_name)

        # Get set of all projectable names that can't be satisfied by the readers we've loaded
        composite_names = set(projectable_names)
        sensor_names = set()
        unknown_names = set()
        for reader_instance in self.readers.values():
            composite_names -= set(reader_instance.dataset_names)
            sensor_names |= set(reader_instance.sensor_names)

        # If we have any composites that need to be made, then let's create the composite objects
        if composite_names:
            unknown_names = self.load_compositors(composite_names, sensor_names, **kwargs)

        for unknown_name in unknown_names:
            LOG.warning("Unknown dataset or compositor: %s", unknown_name)

        # Don't include any of the 'unknown' projectable names
        projectable_names = set(projectable_names) - unknown_names
        composites_needed = set(composite for composite in self.compositors.keys()
                                if composite not in self.projectables or not self[composite].is_loaded()) & projectable_names

        for reader_name, reader_instance in self.readers.items():
            all_reader_datasets = set(reader_instance.dataset_names)

            # compute the dependencies to load from file
            needed_bands = all_reader_datasets & projectable_names
            needed_bands = set(band for band in needed_bands
                               if band not in self.projectables or not  self[band].is_loaded())
            while composites_needed:
                for band in composites_needed.copy():
                    needed_bands |= set(reader_instance.get_dataset(prereq)["name"]
                                        for prereq in self.compositors[band].prerequisites)
                    composites_needed.remove(band)

            # A composite might use a product from another reader, so only pass along the ones we know about
            needed_bands &= all_reader_datasets

            # Create projectables in reader and update the scenes projectables
            needed_bands = sorted(needed_bands)
            LOG.debug("Asking reader '%s' for the following datasets %s", reader_name, str(needed_bands))
            self.projectables.update(reader_instance.load(needed_bands, **kwargs))

        # Update the scene with information contained in the files
        # FIXME: should this really be in the scene ?
        self.info["start_time"] = min([p.info["start_time"] for p in self.projectables.values()])
        try:
            self.info["end_time"] = max([p.info["end_time"] for p in self.projectables.values()])
        except KeyError:
            pass
        # TODO: comments and history

    def compute(self, *requirements):
        """Compute all the composites from *requirements*
        """
        if not requirements:
            requirements = self.wishlist
        for requirement in requirements:
            if requirement not in self.compositors:
                continue
            if requirement in self.projectables:
                continue
            self.compute(*self.compositors[requirement].prerequisites)

            # TODO: Get non-projectable dependencies like moon illumination fraction
            prereq_projectables = [self[prereq] for prereq in self.compositors[requirement].prerequisites]
            try:
                self.projectables[requirement] = self.compositors[requirement](prereq_projectables, **self.info)
            except IncompatibleAreas:
                for name, projectable in self.projectables.iteritems():
                    if name in self.compositors[requirement].prerequisites:
                        projectable.info["keep"] = True

    def unload(self):
        """Unload all loaded composites.
        """
        to_del = [name for name, projectable in self.projectables.items()
                  if name not in self.wishlist and
                  not projectable.info.get("keep", False)]
        for name in to_del:
            del self.projectables[name]

    def load(self, wishlist, **kwargs):
        """Read, compute and unload.
        """
        self.read(wishlist, **kwargs)
        if kwargs.get("compute", True):
            self.compute()
        if kwargs.get("unload", True):
            self.unload()

    def resample(self, destination, datasets=None, **kwargs):
        """Resample the projectables and return a new scene.
        """
        new_scn = Scene()
        new_scn.info = self.info.copy()
        for name, projectable in self.projectables.items():
            LOG.debug("Resampling %s", name)
            if datasets and name not in datasets:
                continue
            new_scn[name] = projectable.resample(destination, **kwargs)
        return new_scn

    def images(self):
        """Generate images for all the composites from the scene.
        """
        for name, projectable in self.projectables.items():
            if name in self.wishlist:
                yield projectable.to_image()

    def load_writer_config(self, config_file, **kwargs):
        if not os.path.isfile(config_file):
            raise IOError("Writer configuration file does not exist: %s" % (config_file,))

        conf = ConfigParser.ConfigParser()
        conf.read(config_file)
        for section_name in conf.sections():
            if section_name.startswith("writer:"):
                options = dict(conf.items(section_name))
                writer_class_name = options["writer"]
                writer_class = self._runtime_import(writer_class_name)
                writer = writer_class(ppp_config_dir=self.ppp_config_dir, config_file=config_file, **kwargs)
                return writer

    def save_images(self, writer="geotiff", **kwargs):
        kwargs.setdefault("config_file", os.path.join(self.ppp_config_dir, "writers", writer + ".cfg"))
        writer = self.load_writer_config(**kwargs)
        for projectable in self.projectables.values():
            writer.save_dataset(projectable, **kwargs)

