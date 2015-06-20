#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

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
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class IncompatibleAreas(StandardError):
    pass


class Scene(InfoObject):

    def __init__(self, filenames=None, ppp_config_dir=None, reader=None, **info):
        """platform_name=None, sensor=None, start_time=None, end_time=None,
        """
        # Get PPP_CONFIG_DIR
        self.ppp_config_dir = ppp_config_dir or os.environ.get("PPP_CONFIG_DIR", '.')
        # Set the PPP_CONFIG_DIR in the environment in case it's used else where in pytroll
        logger.debug("Setting 'PPP_CONFIG_DIR' to '%s'", self.ppp_config_dir)
        os.environ["PPP_CONFIG_DIR"] = self.ppp_config_dir

        InfoObject.__init__(self, **info)
        self.readers = {}
        self.projectables = {}
        self.products = {}
        self.wishlist = []

        if filenames is not None and not filenames:
            raise ValueError("Filenames are specified but empty")

        if reader is not None:
            self._find_reader(reader, filenames)
        elif "sensor" in self.info:
            self._find_sensors_readers(self.info["sensor"], filenames)
        elif filenames is not None:
            self._find_files_readers(*filenames)

    def _find_sensors_readers(self, sensor, filenames):
        sensor_set = set([sensor]) if isinstance(sensor, str) else set(sensor)
        for config_file in glob.glob(os.path.join(self.ppp_config_dir, "readers", "*.cfg")):
            try:
                reader_info = self._read_config(config_file)
                logger.debug("Successfully read reader config: %s", config_file)
            except ValueError:
                logger.debug("Invalid reader config found: %s", config_file)
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
                        logger.warning("No filenames found for reader: %s", reader_info["name"])
                        continue
                self._load_reader(reader_info)

    def _find_reader(self, reader, filenames):
        config_file = reader
        # were we given a path to a config file?
        if not os.path.exists(config_file):
            # no, we were given a name of a reader
            config_fn = reader + ".cfg"  # assumes no extension was given on the reader name
            config_file = os.path.join(self.ppp_config_dir, "readers", config_fn)
            if not os.path.exists(config_file):
                raise ValueError("Can't find config file for reader: %s" % (reader,))

        reader_info = self._read_config(config_file)
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
                reader_info = self._read_config(config_file)
                logger.debug("Successfully read reader config: %s", config_file)
            except ValueError:
                logger.debug("Invalid reader config found: %s", config_file)
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
        epoch = datetime(1950, 1, 1)
        filenames = []
        info = self.info.copy()
        for key in info.keys():
            if key.endswith("_time"):
                info.pop(key, None)

        reader_start = reader_info["start_time"]
        reader_end = reader_info["end_time"]
        if reader_start is None:
            raise ValueError("'start_time' keyword required with 'sensor' and 'reader' keyword arguments")

        for pattern in reader_info["file_patterns"]:
            parser = trollsift.parser.Parser(pattern)
            # FIXME: what if we are browsing a huge archive ?
            globber = parser.globify(info.copy())
            for filename in glob.iglob(globber):
                metadata = parser.parse(filename)
                if "end_time" in metadata and metadata["start_time"] > metadata["end_time"]:
                    mdate = metadata["start_time"].date()
                    mtime = metadata["end_time"].time()
                    if mtime < metadata["start_time"].time():
                        mdate += timedelta(days=1)
                    metadata["end_time"] = datetime.combine(mdate, mtime)
                meta_start = metadata.get("start_time", metadata.get("nominal_time", None))
                meta_end = metadata.get("end_time", epoch)
                if reader_end:
                    # get the data within the time interval
                    if ((reader_start <= meta_start <= reader_end) or
                            (reader_start <=  meta_end <= reader_end)):
                        filenames.append(filename)
                else:
                    # get the data containing start_time
                    if "end_time" in metadata and meta_start <= reader_start <= meta_end:
                        filenames.append(filename)
                    elif meta_start == reader_start:
                        filenames.append(filename)
        return filenames

    def add_product(self, uid, obj):
        self.products[uid] = obj

    def read_composites_config(self, composite_config=None, sensor=None, uids=None, **kwargs):
        if composite_config is None:
            composite_config = os.path.join(self.ppp_config_dir, "composites", "generic.cfg")

        conf = ConfigParser.ConfigParser()
        conf.read(composite_config)
        compositors = {}
        for section_name in conf.sections():
            if section_name.startswith("composite:"):
                options = dict(conf.items(section_name))
                options["sensor"] = options.setdefault("sensor", "").split(",")
                comp_cls = options["format"]

                # Check if the caller only wants composites for a certain sensor
                if sensor is not None and sensor not in options["sensor"]:
                    continue
                # Check if the caller only wants composites with certain uids
                if not uids and options["uid"] not in uids:
                    continue

                if options["uid"] in self.products:
                    logger.warning("Duplicate composite found, previous composite '%s' will be overwritten",
                                   options["uid"])

                try:
                    loader = self._runtime_import(comp_cls)
                except ImportError:
                    logger.warning("Could not import composite class '%s' for compositor '%s'" % (comp_cls,
                                                                                                  options["uid"]))
                    continue

                options.update(**kwargs)
                comp = loader(**options)
                compositors[options["uid"]] = comp
        return compositors

    def _read_config(self, cfg_file):
        if not os.path.exists(cfg_file):
            raise IOError("No such file: " + cfg_file)

        conf = ConfigParser.RawConfigParser()

        conf.read(cfg_file)
        file_patterns = []
        sensors = set()
        reader_name = None
        reader_format = None
        reader_info = None
        # Only one reader: section per config file
        for section in conf.sections():
            if section.startswith("reader:"):
                reader_info = dict(conf.items(section))
                reader_info["file_patterns"] = reader_info.setdefault("file_patterns", "").split(",")
                # XXX: Readers can have separate start/end times from the rest fo the scene...might be a bad idea?
                reader_info.setdefault("start_time", self.info.get("start_time", None))
                reader_info.setdefault("end_time", self.info.get("end_time", None))
                reader_info.setdefault("area", self.info.get("area", None))
                try:
                    reader_format = reader_info["format"]
                    reader_name = reader_info["name"]
                except KeyError:
                    break
                file_patterns.extend(reader_info["file_patterns"])
            else:
                try:
                    file_patterns.extend(conf.get(section, "file_patterns").split(","))
                except ConfigParser.NoOptionError:
                    pass

                try:
                    sensors |= set(conf.get(section, "sensor").split(","))
                except ConfigParser.NoOptionError:
                    pass

        if reader_format is None:
            raise ValueError("Malformed config file %s: missing reader format" % cfg_file)
        if reader_name is None:
            raise ValueError("Malformed config file %s: missing reader name" % cfg_file)
        reader_info["file_patterns"] = file_patterns
        reader_info["config_file"] = cfg_file
        reader_info["filenames"] = []
        reader_info["sensor"] = tuple(sensors)

        return reader_info

    def _load_reader(self, reader_info):
        try:
            loader = self._runtime_import(reader_info["format"])
        except ImportError:
            raise ImportError("Could not import reader class '%s' for reader '%s'" % (reader_info["format"],
                                                                                      reader_info["name"]))

        reader_instance = loader(**reader_info)
        # setattr(self, reader_info["name"], reader_instance)
        self.readers[reader_info["name"]] = reader_instance
        return reader_instance

    def available_channels(self, reader_name=None):
        try:
            if reader_name:
                readers = [getattr(self, reader_name)]
            else:
                readers = self.readers
        except (AttributeError, KeyError):
            raise KeyError("No reader '%s' found in scene")

        return [channel_name for reader_name in readers for channel_name in reader_name.channel_names]

    def __str__(self):
        res = []
        for reader in self.readers:
            res.append(reader.info["name"] + ":")
            for channel_name in reader.channel_names:
                res.append("\t%s" % (str(self.projectables.get(channel_name, "%s: Not loaded" % (channel_name,))),))

        return "\n".join(res)

    def __iter__(self):
        return iter(self.projectables.values())

    def __getitem__(self, key):
        # get by wavelength
        if isinstance(key, numbers.Number):
            channels = [chn for chn in self.projectables.values()
                        if("wavelength_range" in chn.info and
                           chn.info["wavelength_range"][0] <= key <= chn.info["wavelength_range"][2])]
            channels = sorted(channels,
                              lambda ch1, ch2:
                              cmp(abs(ch1.info["wavelength_range"][1] - key),
                                  abs(ch2.info["wavelength_range"][1] - key)))

            if not channels:
                raise KeyError("Can't find any projectable at %gum" % key)
            return channels[0]
        # get by name
        else:
            return self.projectables[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Projectable):
            raise ValueError("Only 'Projectable' objects can be assigned")
        self.projectables[key] = value
        value.info["uid"] = key

    def __delitem__(self, key):
        # TODO: Delete item from projectables dictionary(!)
        raise NotImplementedError()

    def __contains__(self, uid):
        return uid in self.projectables

    def assign_matching_files(self, reader_info, *files):
        files = list(files)
        for file_pattern in reader_info["file_patterns"]:
            pattern = trollsift.globify(file_pattern)
            for filename in list(files):
                if fnmatch.fnmatch(os.path.basename(filename), os.path.basename(pattern)):
                    reader_info["filenames"].append(filename)
                    files.remove(filename)

        # return remaining/unmatched files
        return files

    def _runtime_import(self, object_path):
        obj_module, obj_element = object_path.rsplit(".", 1)
        loader = __import__(obj_module, globals(), locals(), [obj_element])
        return getattr(loader, obj_element)

    def load_compositors(self, composite_names, sensor_names, **kwargs):
        # Check the composites for each particular sensor first
        for sensor_name in sensor_names:
            sensor_composite_config = os.path.join(self.ppp_config_dir, "composites", sensor_name + ".cfg")
            if not os.path.isfile(sensor_composite_config):
                logger.debug("No sensor composite config found at %s", sensor_composite_config)
                continue

            # Load all the compositors for this sensor for the needed names from the specified config
            sensor_compositors = self.read_composites_config(sensor_composite_config, sensor_name, composite_names,
                                                             **kwargs)
            # Update the list of composites the scene knows about
            self.products.update(sensor_compositors)
            # Remove the names we know how to create now
            composite_names -= set(sensor_compositors.keys())

            if not composite_names:
                # we found them all!
                break
        else:
            # we haven't found them all yet, let's check the global composites config
            composite_config = os.path.join(self.ppp_config_dir, "composites.cfg")
            if os.path.isfile(composite_config):
                global_compositors = self.read_composites_config(composite_config, uids=composite_names, **kwargs)
                self.products.update(global_compositors)
                composite_names -= set(global_compositors.keys())
            else:
                logger.warning("No global composites.cfg file found in config directory")

        return composite_names

    def read(self, *projectable_keys, **kwargs):
        self.wishlist = projectable_keys

        projectable_names = set()

        for reader_name, reader_instance in self.readers.items():
            for key in projectable_keys:
                try:
                    projectable_names.add(reader_instance.get_channel(key)["uid"])
                except KeyError:
                    projectable_names.add(key)
                    logger.debug("Can't find channel %s in reader %s", str(key), reader_name)

        # Get set of all projectable names that can't be satisfied by the readers we've loaded
        composite_names = set(projectable_names)
        sensor_names = set()
        for reader_instance in self.readers.values():
            composite_names -= set(reader_instance.channel_names)
            sensor_names |= set(reader_instance.sensor_names)

        # If we have any composites that need to be made, then let's create the composite objects
        if composite_names:
            composite_names = self.load_compositors(composite_names, sensor_names, **kwargs)

        for composite_name in composite_names:
            logger.warning("Unknown channel or compositor: %s", composite_name)

        # Don't include any of the 'unknown' projectable names
        projectable_names = set(projectable_names) - composite_names
        composites_needed = set(self.products.keys())

        for reader_name, reader_instance in self.readers.items():
            all_reader_channels = set(reader_instance.channel_names)

            # compute the depencies to load from file
            needed_bands = all_reader_channels & projectable_names
            while composites_needed:
                for band in composites_needed.copy():
                    needed_bands |= set([reader_instance.get_channel(prereq)["uid"] for prereq in self.products[band].prerequisites])
                    composites_needed.remove(band)

            # A composite might use a product from another reader, so only pass along the ones we know about
            needed_bands &= all_reader_channels

            # Create projectables in reader and update the scenes projectables
            needed_bands = sorted(needed_bands)
            logger.debug("Asking reader '%s' for the following channels %s", reader_name, str(needed_bands))
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
        if not requirements:
            requirements = self.wishlist
        for requirement in requirements:
            if requirement not in self.products:
                continue
            if requirement in self.projectables:
                continue
            self.compute(*self.products[requirement].prerequisites)

            # TODO: Get nonprojectable dependencies like moon illumination fraction
            prereq_projectables = [self[prereq] for prereq in self.products[requirement].prerequisites]
            try:
                self.projectables[requirement] = self.products[requirement](prereq_projectables, **self.info)
            except IncompatibleAreas:
                for uid, projectable in self.projectables.item():
                    if uid in self.products[requirement].prerequisites:
                        projectable.info["keep"] = True

    def unload(self):
        to_del = [uid for uid, projectable in self.projectables.items()
                  if uid not in self.wishlist and
                  not projectable.info.get("keep", False)]
        for uid in to_del:
            del self.projectables[uid]

    def load(self, *wishlist, **kwargs):
        self.read(*wishlist, **kwargs)
        if kwargs.get("compute", True):
            self.compute()
        if kwargs.get("unload", True):
            self.unload()

    def resample(self, destination, channels=None, **kwargs):
        """Resample the projectables and return a new scene.
        """
        new_scn = Scene()
        new_scn.info = self.info.copy()
        for uid, projectable in self.projectables.items():
            logger.debug("Resampling %s", uid)
            if channels and uid not in channels:
                continue
            new_scn[uid] = projectable.resample(destination, **kwargs)
        return new_scn

    def images(self):
        for uid, projectable in self.projectables.items():
            if uid in self.wishlist:
                yield projectable.to_image()



import unittest


class TestScene(unittest.TestCase):

    def test_config_reader(self):
        "Check config reading"
        scn = Scene()
        scn._read_config(
            "/home/a001673/usr/src/newconfig/Suomi-NPP.cfg")
        self.assertTrue("DNB" in scn)

    def test_channel_get(self):
        scn = Scene()
        scn._read_config(
            "/home/a001673/usr/src/newconfig/Suomi-NPP.cfg")
        self.assertEqual(scn[0.67], scn["M05"])

    def test_metadata(self):
        scn = Scene()
        scn._read_config(
            "/home/a001673/usr/src/newconfig/Suomi-NPP.cfg")
        self.assertEqual(scn.info["platform_name"], "Suomi-NPP")

    def test_open(self):
        scn = Scene()
        scn._find_readers(
            "/home/a001673/data/satellite/Suomi-NPP/viirs/lvl1b/2015/04/20/SDR/SVM02_npp_d20150420_t0536333_e0537575_b18015_c20150420054512262557_cspp_dev.h5")

        self.assertEqual(scn.info["platform_name"], "Suomi-NPP")

        self.assertRaises(IOError, scn._find_readers, "bla")


class TestProjectable(unittest.TestCase):
    pass
