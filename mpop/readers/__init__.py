#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Shared objects of the various reader classes.

"""

import logging
import numbers
import os
import numpy as np
import six
from abc import abstractmethod, ABCMeta
from itertools import izip
from fnmatch import fnmatch
from collections import namedtuple
from datetime import datetime, timedelta
from trollsift.parser import globify, Parser

from mpop.plugin_base import Plugin
from mpop.projectable import Projectable
from mpop import runtime_import, get_config, glob_config, config_search_paths

try:
    import configparser
except ImportError:
    from six.moves import configparser
import glob

LOG = logging.getLogger(__name__)

DatasetID = namedtuple("Band", "name wavelength resolution polarization calibration")
DatasetID.__new__.__defaults__ = (None, None, None, None, None)


class DatasetDict(dict):
    """Special dictionary object that can handle dict operations based on dataset name, wavelength, or DatasetID

    Note: Internal dictionary keys are `DatasetID` objects.
    """
    def __init__(self, *args, **kwargs):
        super(DatasetDict, self).__init__(*args, **kwargs)

    def keys(self, names=False, wavelengths=False):
        keys = super(DatasetDict, self).keys()
        if names:
            return (k.name for k in keys)
        elif wavelengths:
            return (k.wavelength for k in keys)
        else:
            return keys

    def _name_match(self, a, b):
        return a == b

    def _wl_match(self, a, b):
        if type(a) == type(b):
            return a == b
        elif isinstance(a, (list, tuple)) and len(a) == 3:
            return a[0] <= b <= a[2]
        elif isinstance(b, (list, tuple)) and len(b) == 3:
            return b[0] <= a <= b[2]
        else:
            raise ValueError("Can only compare wavelengths of length 1 or 3")

    def get_key(self, key):
        if isinstance(key, DatasetID):
            return key
        # get by wavelength
        elif isinstance(key, numbers.Number):
            for k in self.keys():
                if k.wavelength is not None and self._wl_match(k.wavelength, key):
                    return k
        # get by name
        else:
            for k in self.keys():
                if self._name_match(k.name, key):
                    return k

    def get_keys(self, name_or_wl, resolution=None, polarization=None, calibration=None):
        # Get things that match at least the name_or_wl
        if isinstance(name_or_wl, numbers.Number):
            keys = [k for k in self.keys() if self._wl_match(k.wavelength, name_or_wl)]
        elif isinstance(name_or_wl, (str, unicode)):
            keys = [k for k in self.keys() if self._name_match(k.name, name_or_wl)]
        else:
            raise TypeError("First argument must be a wavelength or name")

        if resolution is not None:
            if not isinstance(resolution, (list, tuple)):
                resolution = (resolution,)
            keys = [k for k in keys if k.resolution is not None and k.resolution in resolution]
        if polarization is not None:
            if not isinstance(polarization, (list, tuple)):
                polarization = (polarization,)
            keys = [k for k in keys if k.polarization is not None and k.polarization in polarization]
        if calibration is not None:
            if not isinstance(calibration, (list, tuple)):
                calibration = (calibration,)
            keys = [k for k in keys if k.calibration is not None and k.calibration in calibration]

        return keys

    def get_item(self, name_or_wl, resolution=None, polarization=None, calibration=None):
        keys = self.get_keys(name_or_wl, resolution=resolution, polarization=polarization, calibration=calibration)
        if len(keys) == 0:
            raise KeyError("No keys found matching provided filters")

        return self[keys[0]]

    def __getitem__(self, item):
        key = self.get_key(item)
        if key is None:
            raise KeyError("No dataset matching '%s' found" % (str(item),))
        return super(DatasetDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Support assigning 'Projectable' objects or dictionaries of metadata.
        """
        d = value.info if isinstance(value, Projectable) else value
        if not isinstance(key, DatasetID):
            key = self.get_key(key)
            if key is None:
                # this is a new key and it's not a full DatasetID tuple
                key = DatasetID(
                    name=d["name"],
                    resolution=d["resolution"],
                    wavelength=d["wavelength_range"],
                    polarization=d["polarization"],
                    calibration=d["calibration"],
                )

        # update the 'value' with the information contained in the key
        d["name"] = key.name
        # XXX: What should users be allowed to modify?
        d["resolution"] = key.resolution
        d["calibration"] = key.calibration
        d["polarization"] = key.polarization
        # you can't change the wavelength of a dataset, that doesn't make sense
        assert(d["wavelength_range"] == key.wavelength)

        return super(DatasetDict, self).__setitem__(key, value)

    def __contains__(self, item):
        key = self.get_key(item)
        return super(DatasetDict, self).__contains__(key)

    def __delitem__(self, key):
        key = self.get_key(key)
        return super(DatasetDict, self).__delitem__(key)


class ReaderFinder(object):
    """Finds readers given a scene, filenames, sensors, and/or a reader_name
    """

    def __init__(self, ppp_config_dir=None, base_dir=None, **info):
        self.info = info
        self.ppp_config_dir = ppp_config_dir
        self.base_dir = base_dir

    def __call__(self, filenames=None, sensor=None, reader_name=None):
        if reader_name is not None:
            return [self._find_reader(reader_name, filenames)]
        elif sensor is not None:
            return list(self._find_sensors_readers(sensor, filenames))
        elif filenames is not None:
            return list(self._find_files_readers(filenames))
        return []

    def _find_sensors_readers(self, sensor, filenames):
        """Find the readers for the given *sensor* and *filenames*
        """
        if isinstance(sensor, (str, six.text_type)):
            sensor_set = set([sensor])
        else:
            sensor_set = set(sensor)

        reader_names = set()
        for config_file in glob_config(os.path.join("readers", "*.cfg"), self.ppp_config_dir):
            # This is just used to find the individual reader configurations, not necessarily the individual files
            config_fn = os.path.basename(config_file)
            if config_fn in reader_names:
                # we've already loaded this reader (even if we found it through another environment)
                continue

            try:
                config_files = config_search_paths(os.path.join("readers", config_fn), self.ppp_config_dir)
                reader_info = self._read_reader_config(config_files)
                LOG.debug("Successfully read reader config: %s", config_fn)
                reader_names.add(config_fn)
            except ValueError:
                LOG.debug("Invalid reader config found: %s", config_fn, exc_info=True)
                continue

            if "sensor" in reader_info and (set(reader_info["sensor"]) & sensor_set):
                # we want this reader
                if filenames:
                    # returns a copy of the filenames remaining to be matched
                    filenames = self.assign_matching_files(reader_info, *filenames, base_dir=self.base_dir)
                    if filenames:
                        raise IOError("Don't know how to open the following files: %s" % str(filenames))
                else:
                    # find the files for this reader based on its file patterns
                    reader_info["filenames"] = self.get_filenames(reader_info, self.base_dir)
                    if not reader_info["filenames"]:
                        LOG.warning("No filenames found for reader: %s", reader_info["name"])
                        continue
                yield self._load_reader(reader_info)

    def _find_reader(self, reader, filenames):
        """Find and get info for the *reader* for *filenames*
        """
        # were we given a path to a config file?
        if not os.path.exists(reader):
            # no, we were given a name of a reader
            config_fn = reader + ".cfg" if "." not in reader else reader
            config_files = config_search_paths(os.path.join("readers", config_fn), self.ppp_config_dir)
            if not config_files:
                raise ValueError("Can't find config file for reader: %s" % (reader,))
        else:
            # we may have been given a dependent config file (depends on builtin configuration)
            # so we need to find the others
            config_fn = os.path.basename(reader)
            config_files = config_search_paths(os.path.join("readers", config_fn), self.ppp_config_dir)
            config_files = [reader] + config_files

        reader_info = self._read_reader_config(config_files)
        if filenames:
            filenames = self.assign_matching_files(reader_info, *filenames, base_dir=self.base_dir)
            if filenames:
                raise IOError("Don't know how to open the following files: %s" % str(filenames))
        else:
            reader_info["filenames"] = self.get_filenames(reader_info, base_dir=self.base_dir)
            if not reader_info["filenames"]:
                raise RuntimeError("No filenames found for reader: %s" % (reader_info["name"],))

        return self._load_reader(reader_info)

    def _find_files_readers(self, files):
        """Find the reader info for the provided *files*.
        """
        reader_names = set()
        for config_file in glob_config(os.path.join("readers", "*.cfg"), self.ppp_config_dir):
            # This is just used to find the individual reader configurations, not necessarily the individual files
            config_fn = os.path.basename(config_file)
            if config_fn in reader_names:
                # we've already loaded this reader (even if we found it through another environment)
                continue

            try:
                config_files = config_search_paths(os.path.join("readers", config_fn), self.ppp_config_dir)
                reader_info = self._read_reader_config(config_files)
                LOG.debug("Successfully read reader config: %s", config_fn)
                reader_names.add(config_fn)
            except ValueError:
                LOG.debug("Invalid reader config found: %s", config_fn, exc_info=True)
                continue

            files = self.assign_matching_files(reader_info, *files, base_dir=self.base_dir)

            if reader_info["filenames"]:
                # we have some files for this reader so let's create it
                yield self._load_reader(reader_info)

            if not files:
                break
        if files:
            raise IOError("Don't know how to open the following files: %s" % str(files))

    def get_filenames(self, reader_info, base_dir=None):
        """Get the filenames from disk given the patterns in *reader_info*.
        This assumes that the scene info contains start_time at least (possibly end_time too).
        """

        filenames = []
        info = self.info.copy()
        for key in self.info.keys():
            if key.endswith("_time"):
                info.pop(key, None)

        reader_start = reader_info["start_time"]
        reader_end = reader_info.get("end_time")
        if reader_start is None:
            raise ValueError("'start_time' keyword required with 'sensor' and 'reader' keyword arguments")

        for pattern in reader_info["file_patterns"]:
            if base_dir:
                pattern = os.path.join(base_dir, pattern)
            parser = Parser(str(pattern))
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

    def _read_reader_config(self, config_files):
        """Read the reader *cfg_file* and return the info extracted.
        """
        conf = configparser.RawConfigParser()
        successes = conf.read(config_files)
        if not successes:
            raise ValueError("No valid configuration files found named: %s" % (config_files,))
        LOG.debug("Read config from %s", str(successes))

        file_patterns = []
        sensors = set()
        reader_name = None
        reader_class = None
        reader_info = None
        # Only one reader: section per config file
        for section in conf.sections():
            if section.startswith("reader:"):
                reader_info = dict(conf.items(section))
                reader_info["file_patterns"] = filter(None, reader_info.setdefault("file_patterns", "").split(","))
                reader_info["sensor"] = filter(None, reader_info.setdefault("sensor", "").split(","))
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
                except configparser.NoOptionError:
                    pass

                try:
                    sensors |= set(conf.get(section, "sensor").split(","))
                except configparser.NoOptionError:
                    pass

        if reader_class is None:
            raise ValueError("Malformed config file %s: missing reader 'reader'" % (config_files,))
        if reader_name is None:
            raise ValueError("Malformed config file %s: missing reader 'name'" % (config_files,))
        reader_info["file_patterns"] = file_patterns
        reader_info["config_files"] = config_files
        reader_info["filenames"] = []
        reader_info["sensor"] = tuple(sensors)

        return reader_info

    @staticmethod
    def _load_reader(reader_info):
        """Import and setup the reader from *reader_info*
        """
        try:
            loader = runtime_import(reader_info["reader"])
        except ImportError:
            raise ImportError("Could not import reader class '%s' for reader '%s'" % (reader_info["reader"],
                                                                                      reader_info["name"]))

        reader_instance = loader(**reader_info)
        # fixme: put this in the calling function
        # self.readers[reader_info["name"]] = reader_instance
        return reader_instance

    @staticmethod
    def assign_matching_files(reader_info, *files, **kwargs):
        """Assign *files* to the *reader_info*
        """
        files = list(files)
        for file_pattern in reader_info["file_patterns"]:
            if kwargs.get("base_dir", None):
                file_pattern = os.path.join(kwargs["base_dir"], file_pattern)
            pattern = globify(file_pattern)
            for filename in list(files):
                if fnmatch(os.path.basename(filename), os.path.basename(pattern)):
                    reader_info["filenames"].append(filename)
                    files.remove(filename)

        # return remaining/unmatched files
        return files


class Reader(Plugin):
    """Reader plugins. They should have a *pformat* attribute, and implement
    the *load* method. This is an abstract class to be inherited.
    """
    splittable_dataset_options = ["file_patterns", "navigation", "standard_name", "units"]

    def __init__(self, name=None,
                 file_patterns=None,
                 filenames=None,
                 description="",
                 start_time=None,
                 end_time=None,
                 area=None,
                 sensor=None,
                 **kwargs):
        """The reader plugin takes as input a satellite scene to fill in.

        Arguments:
        - `scene`: the scene to fill.
        """
        # Hold information about datasets
        self.datasets = DatasetDict()

        # Load the config
        super(Reader, self).__init__(**kwargs)

        # Use options from the config file if they weren't passed as arguments
        self.name = self.config_options.get("name", None) if name is None else name
        self.file_patterns = self.config_options.get("file_patterns", None) if file_patterns is None else file_patterns
        self.filenames = self.config_options.get("filenames", []) if filenames is None else filenames
        self.description = self.config_options.get("description", None) if description is None else description
        self.sensor = self.config_options.get("sensor", "").split(",") if sensor is None else set(sensor)

        # These can't be provided by a configuration file
        self.start_time = start_time
        self.end_time = end_time
        self.area = area

        if self.name is None:
            raise ValueError("Reader 'name' not provided")

    def add_filenames(self, *filenames):
        self.filenames |= set(filenames)

    @property
    def dataset_names(self):
        """Names of all datasets configured for this reader.
        """
        return self.datasets.keys(names=True)

    @property
    def sensor_names(self):
        """Sensors supported by this reader.
        """
        sensors = set()
        for ds_info in self.datasets.values():
            if "sensor" in ds_info:
                sensors |= set(ds_info["sensor"].split(","))
        return sensors | self.sensor

    def load_section_reader(self, section_name, section_options):
        del section_name
        self.config_options = section_options

    def load_section_dataset(self, section_name, section_options):
        # required for Dataset identification
        section_options["resolution"] = tuple(float(res) for res in section_options.get("resolution").split(','))
        num_permutations = len(section_options["resolution"])

        # optional or not applicable for all datasets for Dataset identification
        if "wavelength_range" in section_options:
            section_options["wavelength_range"] = tuple(float(wvl) for wvl in section_options.get("wavelength_range").split(','))
        else:
            section_options["wavelength_range"] = None

        if "calibration" in section_options:
            section_options["calibration"] = tuple(section_options.get("calibration").split(','))
        else:
            section_options["calibration"] = [None] * num_permutations

        if "polarization" in section_options:
            section_options["polarization"] = tuple(section_options.get("polarization").split(','))
        else:
            section_options["polarization"] = [None] * num_permutations

        # Sanity checks
        assert "name" in section_options
        assert section_options["wavelength_range"] is None or (len(section_options["wavelength_range"]) == 3)
        assert num_permutations == len(section_options["calibration"])
        assert num_permutations == len(section_options["polarization"])

        # Add other options that are based on permutations
        for k in self.splittable_dataset_options:
            if k in section_options:
                section_options[k] = section_options[k].split(",")
            else:
                section_options[k] = [None]

        for k in self.splittable_dataset_options + ["calibration", "polarization"]:
            if len(section_options[k]) == 1:
                # if a single value is used for all permutations, repeat it
                section_options[k] *= num_permutations
            else:
                assert(num_permutations == len(section_options[k]))

        # Add each possible permutation of this dataset to the datasets list for later use
        for idx, (res, cal, pol) in enumerate(izip(
                section_options["resolution"], section_options["calibration"], section_options["polarization"]
        )):
            bid = DatasetID(
                name=section_options["name"],
                wavelength=section_options["wavelength_range"],
                resolution=res,
                calibration=cal,
                polarization=pol,
            )

            opts = section_options.copy()
            # get only the specific permutations value that we want
            opts["id"] = bid
            for k in self.splittable_dataset_options + ["resolution", "calibration", "polarization"]:
                opts[k] = opts[k][idx]
            self.datasets[bid] = opts

    def get_dataset_key(self, key, calibration=None, resolution=None, polarization=None, aslist=False):
        """Get the fully qualified dataset corresponding to *key*, either by name or centerwavelength.

        If `key` is a `DatasetID` object its name is searched if it exists, otherwise its wavelength is used.
        """
        # get by wavelength
        if isinstance(key, numbers.Number):
            datasets = [ds for ds in self.datasets.keys() if (ds.wavelength[0] <= key <= ds.wavelength[2])]
            datasets = sorted(datasets, key=lambda ch: abs(ch.wavelength[1] - key))

            if not datasets:
                raise KeyError("Can't find any projectable at %gum" % key)
        elif isinstance(key, DatasetID):
            if key.name is not None:
                datasets = self.get_dataset_key(key.name, aslist=True)
            elif key.wavelength is not None:
                datasets = self.get_dataset_key(key.wavelength, aslist=True)
            else:
                raise KeyError("Can't find any projectable '%s'" % key)

            if calibration is None:
                calibration = [key.calibration]
            if resolution is None:
                resolution = [key.resolution]
            if polarization is None:
                polarization = [key.polarization]
        # get by name
        else:
            datasets = [ds_id for ds_id in self.datasets.keys() if ds_id.name == key]
            if len(datasets) == 0:
                raise KeyError("Can't find any projectable called '%s'" % key)

        # default calibration choices
        if calibration is None:
            calibration = ["bt", "reflectance"]

        if resolution is not None:
            datasets = [ds_id for ds_id in datasets if ds_id.resolution in resolution]
        if calibration is not None:
            # order calibration from highest level to lowest level
            calibration = [x for x in ["bt", "reflectance", "radiance", "counts"] if x in calibration]
            datasets = [ds_id for ds_id in datasets if ds_id.calibration is None or ds_id.calibration in calibration]
        if polarization is not None:
            datasets = [ds_id for ds_id in datasets if ds_id.polarization in polarization]

        if not datasets:
            raise KeyError("Can't find any projectable matching '%s'" % str(key))
        if aslist:
            return datasets
        else:
            return datasets[0]

    def load(self, datasets_to_load):
        """Loads the *datasets_to_load* into the scene object.
        """
        raise NotImplementedError


class FileKey(namedtuple("FileKey", ["name", "variable_name", "scaling_factors", "dtype", "standard_name", "units",
                                     "file_units", "kwargs"])):
    def __new__(cls, name, variable_name,
                scaling_factors=None, dtype=np.float32, standard_name=None, units=None, file_units=None, **kwargs):
        if isinstance(dtype, (str, six.text_type)):
            # get the data type from numpy
            dtype = getattr(np, dtype)
        return super(FileKey, cls).__new__(cls, name, variable_name, scaling_factors, dtype, standard_name, units,
                                           file_units, kwargs)


class ConfigBasedReader(Reader):
    splittable_dataset_options = Reader.splittable_dataset_options + ["file_type", "file_key"]

    def __init__(self, default_file_reader=None, **kwargs):
        self.file_types = {}
        self.file_readers = {}
        self.file_keys = {}
        self.navigations = {}
        self.calibrations = {}

        # Load the configuration file and other defaults
        super(ConfigBasedReader, self).__init__(**kwargs)

        # Set up the default class for reading individual files
        self.default_file_reader = self.config_options.get("default_file_reader", None) if default_file_reader is None else default_file_reader
        if isinstance(self.default_file_reader, (str, unicode)):
            self.default_file_reader = self._runtime_import(self.default_file_reader)
        if self.default_file_reader is None:
            raise RuntimeError("'default_file_reader' is a required argument")

        # Determine what we know about the files provided and create file readers to read them
        file_types = self.identify_file_types(self.filenames)
        # TODO: Add ability to discover files when none are provided
        if not file_types:
            raise ValueError("No input files found matching the configured file types")

        num_files = 0
        for file_type_name, file_type_files in file_types.items():
            file_type_files = self._get_swathsegment(file_type_files)
            LOG.debug("File type %s has %d files after segment selection", file_type_name, len(file_type_files))

            if len(file_type_files) == 0:
                raise IOError("No files matching!: " +
                              "Start time = " + str(self.start_time) +
                              "  End time = " + str(self.end_time))
            elif num_files and len(file_type_files) != num_files:
                raise IOError("Varying numbers of files found", file_type_name)
            else:
                num_files = len(file_type_files)

            file_reader = MultiFileReader(file_type_name, file_types[file_type_name], self.file_keys)
            self.file_readers[file_type_name] = file_reader

    def _get_swathsegment(self, file_readers):
        if self.area is not None:
            from trollsched.spherical import SphPolygon
            from trollsched.boundary import AreaBoundary

            lons, lats = self.area.get_boundary_lonlats()
            area_boundary = AreaBoundary((lons.side1, lats.side1),
                                         (lons.side2, lats.side2),
                                         (lons.side3, lats.side3),
                                         (lons.side4, lats.side4))
            area_boundary.decimate(500)
            contour_poly = area_boundary.contour_poly

        segment_readers = []
        for file_reader in file_readers:
            file_start = file_reader.start_time
            file_end = file_reader.end_time

            # Search for multiple granules using an area
            if self.area is not None:
                coords = np.vstack(file_reader.get_ring_lonlats())
                poly = SphPolygon(np.deg2rad(coords))
                if poly.intersection(contour_poly) is not None:
                    segment_readers.append(file_reader)
                continue

            if self.start_time is None:
                # if no start_time, assume no time filtering
                segment_readers.append(file_reader)
                continue

            # Search for single granule using time start
            if self.end_time is None:
                if file_start <= self.start_time <= file_end:
                    segment_readers.append(file_reader)
                    continue
            else:
                # search for multiple granules
                # check that granule start time is inside interval
                if self.start_time <= file_start <= self.end_time:
                    segment_readers.append(file_reader)
                    continue

                # check that granule end time is inside interval
                if self.start_time <= file_end <= self.end_time:
                    segment_readers.append(file_reader)
                    continue

        return sorted(segment_readers, key=lambda x: x.start_time)

    def _interpolate_navigation(self, lon, lat):
        return lon, lat

    def _load_navigation(self, nav_name, extra_mask=None, dep_file_type=None):
        """Load the `nav_name` navigation.

        :param dep_file_type: file type of dataset using this navigation. Useful for subclasses to implement relative
                              navigation file loading
        """
        nav_info = self.navigations[nav_name]
        lon_key = nav_info["longitude_key"]
        lat_key = nav_info["latitude_key"]
        file_type = nav_info["file_type"]

        file_reader = self.file_readers[file_type]

        gross_lon_data = file_reader.get_swath_data(lon_key)
        gross_lat_data = file_reader.get_swath_data(lat_key)

        lon_data, lat_data = self._interpolate_navigation(gross_lon_data, gross_lat_data)
        if extra_mask is not None:
            lon_data = np.ma.masked_where(extra_mask, lon_data)
            lat_data = np.ma.masked_where(extra_mask, lat_data)

        # FIXME: Is this really needed/does it belong here? Can we have a dummy/simple object?
        from pyresample import geometry
        area = geometry.SwathDefinition(lons=lon_data, lats=lat_data)
        area_name = ("swath_" +
                     file_reader.start_time.isoformat() + "_" +
                     file_reader.end_time.isoformat() + "_" +
                     str(lon_data.shape[0]) + "_" + str(lon_data.shape[1]))
        # FIXME: Which one is used now:
        area.area_id = area_name
        area.name = area_name

        return area

    def identify_file_types(self, filenames, default_file_reader=None):
        """Identify the type of a file by its filename or by its contents.

        Uses previously loaded information from the configuration file.
        """
        file_types = {}
        # remaining_filenames = [os.path.basename(fn) for fn in filenames]
        remaining_filenames = filenames[:]
        for file_type_name, file_type_info in self.file_types.items():
            file_types[file_type_name] = []

            if default_file_reader is None:
                file_reader_class = file_type_info.get("file_reader", self.default_file_reader)
            else:
                file_reader_class = default_file_reader

            if isinstance(file_reader_class, (str, unicode)):
                file_reader_class = self._runtime_import(file_reader_class)
            for file_pattern in file_type_info["file_patterns"]:
                tmp_remaining = []
                tmp_matching = []
                for fn in remaining_filenames:
                    # Add a wildcard to the front for path information
                    # FIXME: Is there a better way to generalize this besides removing the path every time
                    if fnmatch(fn, "*" + globify(file_pattern)):
                        reader = file_reader_class(file_type_name, fn, self.file_keys, **file_type_info)
                        tmp_matching.append(reader)
                    else:
                        tmp_remaining.append(fn)

                file_types[file_type_name].extend(tmp_matching)
                remaining_filenames = tmp_remaining

            if not file_types[file_type_name]:
                del file_types[file_type_name]

            if not remaining_filenames:
                break

        for remaining_filename in remaining_filenames:
            LOG.warning("Unidentified file: %s", remaining_filename)

        return file_types

    def load_section_file_type(self, section_name, section_options):
        name = section_name.split(":")[-1]
        section_options["file_patterns"] = section_options["file_patterns"].split(",")
        # Don't create the file reader object yet
        self.file_types[name] = section_options

    def load_section_file_key(self, section_name, section_options):
        name = section_name.split(":")[-1]
        self.file_keys[name] = FileKey(name=name, **section_options)

    def load_section_navigation(self, section_name, section_options):
        name = section_name.split(":")[-1]
        self.navigations[name] = section_options

    def load_section_calibration(self, section_name, section_options):
        name = section_name.split(":")[-1]
        self.calibrations[name] = section_options

    def _get_dataset_info(self, ds_id, calibration):
        dataset_info = self.datasets[ds_id].copy()

        if not dataset_info.get("calibration", None):
            LOG.debug("No calibration set for '%s'", ds_id)
            dataset_info["file_type"] = dataset_info["file_type"][0]
            dataset_info["file_key"] = dataset_info["file_key"][0]
            dataset_info["navigation"] = dataset_info["navigation"][0]
            return dataset_info

        # Remove any file types and associated calibration, file_key, navigation if file_type is not loaded
        for k in ["file_type", "file_key", "calibration", "navigation"]:
            dataset_info[k] = []
        for idx, ft in enumerate(self.datasets[ds_id]["file_type"]):
            if ft in self.file_readers:
                for k in ["file_type", "file_key", "calibration", "navigation"]:
                    dataset_info[k].append(self.datasets[ds_id][k][idx])

        # By default do the first calibration for a dataset
        cal_index = 0
        cal_name = dataset_info["calibration"][0]
        for idx, cname in enumerate(dataset_info["calibration"]):
            # is this the calibration we want for this channel?
            if cname in calibration:
                cal_index = idx
                cal_name = cname
                LOG.debug("Using calibration '%s' for dataset '%s'", cal_name, ds_id)
                break
        else:
            LOG.debug("Using default calibration '%s' for dataset '%s'", cal_name, ds_id)

        # Load metadata and calibration information for this dataset
        try:
            cal_info = self.calibrations.get(cal_name, None)
            for k, info_dict in [("file_type", self.file_types),
                                 ("file_key", self.file_keys),
                                 ("navigation", self.navigations),
                                 ("calibration", self.calibrations)]:
                val = dataset_info[k][cal_index]
                if cal_info is not None:
                    val = cal_info.get(k, val)

                if val not in info_dict and k != "calibration":
                    # We don't care if the calibration has its own section
                    raise RuntimeError("Unknown '%s': %s" % (k, val,))
                dataset_info[k] = val

                if k == "file_key":
                    # collect any other metadata
                    dataset_info["standard_name"] = self.file_keys[val].standard_name
                    # dataset_info["file_units"] = self.file_keys[val].file_units
                    # dataset_info["units"] = self.file_keys[val].units
        except (IndexError, KeyError):
            raise RuntimeError("Could not get information to perform calibration '%s'" % (cal_name,))

        return dataset_info

    def load(self, datasets_to_load, **dataset_info):
        if dataset_info:
            LOG.warning("Unsupported options for viirs reader: %s", str(dataset_info))

        datasets_loaded = DatasetDict()
        datasets_to_load = set(datasets_to_load) & set(self.datasets.keys())
        if len(datasets_to_load) == 0:
            LOG.debug("No datasets to load from this reader")
            # XXX: Is None really the best thing that can be returned here?
            return datasets_loaded

        LOG.debug("Channels to load: " + str(datasets_to_load))

        # Sanity check and get the navigation sets being used
        areas = {}
        for ds_id in datasets_to_load:
            dataset_info = self.datasets[ds_id]
            calibration = dataset_info["calibration"]

            # if there is a calibration section in the config, use that for more information
            # FIXME: We also need to get units and other information...and make it easier to do that, per attribute method?
            # Or maybe a post-configuration load method...that's probably best
            if calibration in self.calibrations:
                cal_info = self.calibrations[calibration]
                file_type = cal_info["file_type"]
                file_key = cal_info["file_key"]
                nav_name = cal_info["navigation"]
            else:
                file_type = dataset_info["file_type"]
                file_key = dataset_info["file_key"]
                nav_name = dataset_info["navigation"]
            file_reader = self.file_readers[file_type]

            # Get the swath data (fully scaled and in the correct data type)
            data = file_reader.get_swath_data(file_key, dataset_id=ds_id)

            # Load the navigation information first
            if nav_name not in areas:
                # FIXME: This ignores the possibility that data masks are different between bands
                areas[nav_name] = area = self._load_navigation(nav_name, extra_mask=data.mask, dep_file_type=file_type)
            else:
                area = areas[nav_name]

            # Create a projectable from info from the file data and the config file
            # FIXME: Remove metadata that is reader only
            dataset_info.setdefault("units", file_reader.get_units(file_key))
            dataset_info.setdefault("platform", file_reader.get_platform_name())
            dataset_info.setdefault("sensor", file_reader.get_sensor_name())
            dataset_info.setdefault("start_orbit", file_reader.get_begin_orbit_number())
            dataset_info.setdefault("end_orbit", file_reader.get_end_orbit_number())
            if "rows_per_scan" in self.navigations[nav_name]:
                dataset_info.setdefault("rows_per_scan", self.navigations[nav_name]["rows_per_scan"])
            projectable = Projectable(data=data,
                                      start_time=file_reader.start_time,
                                      end_time=file_reader.end_time,
                                      **dataset_info)
            projectable.info["area"] = area

            datasets_loaded[projectable.info["id"]] = projectable
        return datasets_loaded


class MultiFileReader(object):
    # FIXME: file_type isn't used here. Do we really need it ?
    def __init__(self, file_type, file_readers, file_keys, **kwargs):
        """
        :param file_type:
        :param file_readers: is a list of the reader instances to use.
        :param file_keys:
        :param kwargs:
        :return:
        """
        self.file_type = file_type
        self.file_readers = file_readers
        self.file_keys = file_keys

    @property
    def filenames(self):
        return [fr.filename for fr in self.file_readers]

    @property
    def start_time(self):
        return self.file_readers[0].start_time

    @property
    def end_time(self):
        return self.file_readers[-1].end_time

    def get_begin_orbit_number(self):
        return self.file_readers[0].get_begin_orbit_number()

    def get_end_orbit_number(self):
        return self.file_readers[-1].get_end_orbit_number()

    def get_platform_name(self):
        return self.file_readers[0].get_platform_name()

    def get_sensor_name(self):
        return self.file_readers[0].get_sensor_name()

    @property
    def geo_filenames(self):
        return [fr.get_geofilename() for fr in self.file_readers]

    def get_units(self, item):
        return self.file_readers[0].get_units(item)

    def get_swath_data(self, item, filename=None, dataset_id=None):
        var_info = self.file_keys[item]
        granule_shapes = [x.get_shape(item) for x in self.file_readers]
        num_rows = sum([x[0] for x in granule_shapes])
        num_cols = granule_shapes[0][1]

        if filename:
            raise NotImplementedError("Saving data arrays to disk is not supported yet")
            # data = np.memmap(filename, dtype=var_info.dtype, mode='w', shape=(num_rows, num_cols))
        else:
            data = np.empty((num_rows, num_cols), dtype=var_info.dtype)
            mask = np.zeros_like(data, dtype=np.bool)

        idx = 0
        for granule_shape, file_reader in zip(granule_shapes, self.file_readers):
            # Get the data from each individual file reader (assumes it gets the data with the right data type)
            file_reader.get_swath_data(item,
                                       data_out=data[idx: idx + granule_shape[0]],
                                       mask_out=mask[idx: idx + granule_shape[0]],
                                       dataset_id=dataset_id)
            idx += granule_shape[0]

        # FIXME: This may get ugly when using memmaps, maybe move projectable creation here instead
        return np.ma.array(data, mask=mask, copy=False)


class GenericFileReader(object):
    __metaclass__ = ABCMeta

    def __init__(self, file_type, filename, file_keys, **kwargs):
        self.file_type = file_type
        self.file_keys = file_keys
        self.file_info = kwargs
        self.filename, self.file_handle = self.create_file_handle(filename, **kwargs)

        self.start_time = self.get_begin_time()
        self.end_time = self.get_end_time()
        # FIXME: Rename the no argument methods in to properties

    @abstractmethod
    def create_file_handle(self, filename, **kwargs):
        # return tuple (filename, file_handle)
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def get_ring_lonlats(self):
        raise NotImplementedError

    @abstractmethod
    def get_begin_time(self):
        raise NotImplementedError

    @abstractmethod
    def get_end_time(self):
        raise NotImplementedError

    @abstractmethod
    def get_begin_orbit_number(self):
        raise NotImplementedError

    @abstractmethod
    def get_end_orbit_number(self):
        raise NotImplementedError

    @abstractmethod
    def get_platform_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_sensor_name(self):
        raise NotImplementedError

    @abstractmethod
    def get_geofilename(self):
        raise NotImplementedError

    @abstractmethod
    def get_shape(self, item):
        raise NotImplementedError

    @abstractmethod
    def get_file_units(self, item):
        raise NotImplementedError

    def get_units(self, item):
        units = self.file_keys[item].units
        file_units = self.get_file_units(item)
        # What units does the user want
        if units is None:
            # if the units in the file information
            return file_units
        return units

    @abstractmethod
    def get_swath_data(self, item, data_out=None, mask_out=None, dataset_id=None):
        raise NotImplementedError

