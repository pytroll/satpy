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
from fnmatch import fnmatch
from collections import namedtuple
import os
from datetime import datetime, timedelta
import numpy as np
import six
from trollsift.parser import globify, Parser

from mpop.plugin_base import Plugin
from mpop.projectable import Projectable
from mpop import runtime_import

try:
    import configparser
except ImportError:
    from six.moves import configparser
import glob

LOG = logging.getLogger(__name__)

BandID = namedtuple("Band", "name resolution wavelength polarization")
BandID.__new__.__defaults__ = (None, None, None, None)


class DatasetDict(dict):
    """Special dictionary object that can handle dict operations based on dataset name, wavelength, or BandID

    Note: Internal dictionary keys are `BandID` objects.
    """
    def __init__(self, *args, **kwargs):
        super(DatasetDict, self).__init__(*args, **kwargs)

        # self._name_dict = {}
        # self._wl_dict = {}
        # for ds_key in self.keys():
        #     self._name_dict

    def keys(self, names=False, wavelengths=False):
        keys = super(DatasetDict, self).keys()
        if names:
            return (k.name for k in keys)
        elif wavelengths:
            return (k.wavelength for k in keys)
        else:
            return keys

    def get_key(self, key):
        if isinstance(key, BandID):
            return key
        # get by wavelength
        elif isinstance(key, numbers.Number):
            for k in self.keys():
                if k.wavelength == key:
                    return k
        # get by name
        else:
            for k in self.keys():
                if k.name == key:
                    return k

    def __getitem__(self, item):
        key = self.get_key(item)
        return super(DatasetDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Support assigning 'Projectable' objects or dictionaries of metadata.
        """
        d = value.info if isinstance(value, Projectable) else value
        if not isinstance(key, BandID):
            key = self.get_key(key)
            if key is None:
                # this is a new key and it's not a full BandID tuple
                key = BandID(
                    name=d["name"],
                    resolution=d["resolution"],
                    wavelength=d["wavelength"],
                    polarization=d["polarization"]
                )

        # update the 'value' with the information contained in the key
        d["name"] = key.name
        d["resolution"] = key.resolution
        d["wavelength"] = key.wavelength
        d["polarization"] = key.polarization

        return super(DatasetDict, self).__setitem__(key, value)

    def __delitem__(self, key):
        key = self.get_key(key)
        return super(DatasetDict, self).__delitem__(key)


class ReaderFinder(object):
    """Finds readers given a scene, filenames, sensors, and/or a reader_name
    """

    def __init__(self, scene):
        # fixme: we could just pass the info and the ppp_config_dir
        self.info = scene.info.copy()
        self.ppp_config_dir = scene.ppp_config_dir

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
                yield self._load_reader(reader_info)

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

        return self._load_reader(reader_info)

    def _find_files_readers(self, files):
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
                yield self._load_reader(reader_info)

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
        for key in self.info.keys():
            if key.endswith("_time"):
                info.pop(key, None)

        reader_start = reader_info["start_time"]
        reader_end = reader_info.get("end_time")
        if reader_start is None:
            raise ValueError("'start_time' keyword required with 'sensor' and 'reader' keyword arguments")

        for pattern in reader_info["file_patterns"]:
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

    def _read_reader_config(self, cfg_file):
        """Read the reader *cfg_file* and return the info extracted.
        """
        if not os.path.exists(cfg_file):
            raise IOError("No such file: " + cfg_file)

        conf = configparser.RawConfigParser()

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
                except configparser.NoOptionError:
                    pass

                try:
                    sensors |= set(conf.get(section, "sensor").split(","))
                except configparser.NoOptionError:
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
    def assign_matching_files(reader_info, *files):
        """Assign *files* to the *reader_info*
        """
        files = list(files)
        for file_pattern in reader_info["file_patterns"]:
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
        name = section_options.get("name", section_name.split(":")[-1])
        resolution = float(section_options.get("resolution"))
        wavelength = tuple(float(wvl) for wvl in section_options.get("wavelength_range").split(','))
        bid = BandID(name=name, resolution=resolution, wavelength=wavelength)
        section_options["id"] = bid
        section_options["name"] = name

        # Allow subclasses to make up their own rules about datasets, but this is a good starting point
        for k in ["file_patterns", "file_type", "file_key", "navigation", "calibration"]:
            if k in section_options:
                section_options[k] = section_options[k].split(",")
        if "wavelength_range" in section_options:
            section_options["wavelength_range"] = [float(wl) for wl in section_options["wavelength_range"].split(",")]

        self.datasets[bid] = section_options

    def get_dataset(self, key, aslist=False):
        """Get the dataset corresponding to *key*, either by name or centerwavelength.
        """
        # get by wavelength
        if isinstance(key, numbers.Number):
            datasets = [ds for ds in self.datasets.values()
                        if ("wavelength_range" in ds and
                            ds["wavelength_range"][0] <= key <= ds["wavelength_range"][2])]
            datasets = sorted(datasets,
                              key=lambda ch: abs(ch["wavelength_range"][1] - key))

            if not datasets:
                raise KeyError("Can't find any projectable at %gum" % key)
            if aslist:
                return datasets
            else:
                return datasets[0]
        elif isinstance(key, BandID):
            if key.name is not None:
                datasets = self.get_dataset(key.name, aslist=True)
            elif key.wavelength is not None:
                datasets = self.get_dataset(key.wavelength, aslist=True)
            else:
                raise KeyError("Can't find any projectable '%s'" % key)
            if key.resolution is not None:
                datasets = [ds for ds in datasets if ds["resolution"] == key.resolution]
            if key.polarization is not None:
                datasets = [ds for ds in datasets if ds["polarization"] == key.polarization]
            if not datasets:
                raise KeyError("Can't find any projectable matching '%s'" % str(key))
            if aslist:
                return datasets
            else:
                return datasets[0]

        # get by name
        else:
            datasets = []
            for bid, ds in self.datasets.items():
                if key == bid or bid.name == key:
                    datasets.append(ds)
            if len(datasets) == 0:
                raise KeyError("Can't find any projectable called '%s'" % key)
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

    def _get_dataset_info(self, name, calibration):
        dataset_info = self.datasets[name].copy()

        if not dataset_info.get("calibration", None):
            LOG.debug("No calibration set for '%s'", name)
            dataset_info["file_type"] = dataset_info["file_type"][0]
            dataset_info["file_key"] = dataset_info["file_key"][0]
            dataset_info["navigation"] = dataset_info["navigation"][0]
            return dataset_info

        # Remove any file types and associated calibration, file_key, navigation if file_type is not loaded
        for k in ["file_type", "file_key", "calibration", "navigation"]:
            dataset_info[k] = []
        for idx, ft in enumerate(self.datasets[name]["file_type"]):
            if ft in self.file_readers:
                for k in ["file_type", "file_key", "calibration", "navigation"]:
                    dataset_info[k].append(self.datasets[name][k][idx])

        # By default do the first calibration for a dataset
        cal_index = 0
        cal_name = dataset_info["calibration"][0]
        for idx, cname in enumerate(dataset_info["calibration"]):
            # is this the calibration we want for this channel?
            if cname in calibration:
                cal_index = idx
                cal_name = cname
                LOG.debug("Using calibration '%s' for dataset '%s'", cal_name, name)
                break
        else:
            LOG.debug("Using default calibration '%s' for dataset '%s'", cal_name, name)

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

    def load(self, datasets_to_load, calibration=None, **dataset_info):
        if dataset_info:
            LOG.warning("Unsupported options for viirs reader: %s", str(dataset_info))
        if calibration is None:
            calibration = getattr(self, "calibration", ["bt", "reflectance"])
        # order calibration from highest level to lowest level
        calibration = [x for x in ["bt", "reflectance", "radiance", "counts"] if x in calibration]

        datasets_to_load = set(datasets_to_load) & set(self.dataset_names)
        if len(datasets_to_load) == 0:
            LOG.debug("No datasets to load from this reader")
            # XXX: Is None really the best thing that can be returned here?
            return

        LOG.debug("Channels to load: " + str(datasets_to_load))

        # Sanity check and get the navigation sets being used
        areas = {}
        datasets_loaded = DatasetDict()
        for ds in datasets_to_load:
            dataset_info = self._get_dataset_info(ds, calibration=calibration)
            file_type = dataset_info["file_type"]
            file_key = dataset_info["file_key"]
            file_reader = self.file_readers[file_type]
            nav_name = dataset_info["navigation"]

            # Get the swath data (fully scaled and in the correct data type)
            data = file_reader.get_swath_data(file_key, dataset_name=ds)

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

    def get_swath_data(self, item, filename=None, dataset_name=None):
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
                                       dataset_name=dataset_name)
            idx += granule_shape[0]

        # FIXME: This may get ugly when using memmaps, maybe move projectable creation here instead
        return np.ma.array(data, mask=mask, copy=False)


class GenericFileReader(object):
    def get_swath_data(self, item, dataset_name=None, data_out=None, mask_out=None):
        # FIXME: What is dataset_name supposed to be used for?
        if item in ["longitude", "latitude"]:
            # TODO: compute the lon lat here from tle and sensor geometry (pyorbital)
            return
        raise NotImplementedError

    def get_shape(self, item):
        raise NotImplementedError
