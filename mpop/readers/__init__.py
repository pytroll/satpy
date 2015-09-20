#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

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

from mpop.plugin_base import Plugin
import logging
import numbers
import numpy as np

LOG = logging.getLogger(__name__)


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
        self.datasets = {}

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
        return sorted(self.datasets.keys())

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
        self.config_options = section_options

    def load_section_dataset(self, section_name, section_options):
        name = section_options.get("name", section_name.split(":")[-1])
        section_options["name"] = name

        # Allow subclasses to make up their own rules about datasets, but this is a good starting point
        for k in ["file_patterns", "file_type", "file_key", "navigation", "calibration"]:
            if k in section_options:
                section_options[k] = section_options[k].split(",")
        if "wavelength_range" in section_options:
            section_options["wavelength_range"] = [float(wl) for wl in section_options["wavelength_range"].split(",")]

        self.datasets[name] = section_options

    def get_dataset(self, key):
        """Get the dataset corresponding to *key*, either by name or centerwavelength.
        """
        # get by wavelength
        if isinstance(key, numbers.Number):
            datasets = [ds for ds in self.datasets.values()
                        if("wavelength_range" in ds and
                           ds["wavelength_range"][0] <= key <= ds["wavelength_range"][2])]
            datasets = sorted(datasets,
                              lambda ch1, ch2:
                              cmp(abs(ch1["wavelength_range"][1] - key),
                                  abs(ch2["wavelength_range"][1] - key)))

            if not datasets:
                raise KeyError("Can't find any projectable at %gum" % key)
            return datasets[0]
        # get by name
        else:
            return self.datasets[key]

    def load(self, datasets_to_load):
        """Loads the *datasets_to_load* into the scene object.
        """
        raise NotImplementedError

from fnmatch import fnmatch
from trollsift.parser import globify
from mpop.projectable import Projectable
from collections import namedtuple

class FileKey(namedtuple("FileKey", ["name", "variable_name", "scaling_factors", "dtype", "standard_name", "units",
                                     "file_units", "kwargs"])):
    def __new__(cls, name, variable_name,
                scaling_factors=None, dtype=np.float32, standard_name=None, units=None, file_units=None, **kwargs):
        if isinstance(dtype, (str, unicode)):
            # get the data type from numpy
            dtype = getattr(np, dtype)
        return super(FileKey, cls).__new__(cls, name, variable_name, scaling_factors, dtype, standard_name, units,
                                           file_units, kwargs)


class ConfigBasedReader(Reader):
    def __init__(self, **kwargs):
        self.file_types = {}
        self.file_readers = {}
        self.file_keys = {}
        self.navigations = {}
        self.calibrations = {}

        #kwargs.setdefault("default_config_filename", "readers/viirs_sdr.cfg")

        # Load the configuration file and other defaults
        super(ConfigBasedReader, self).__init__(**kwargs)

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

    def _load_navigation(self, nav_name, dep_file_type, extra_mask=None):
        """Load the `nav_name` navigation.
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


    def identify_file_types(self, filenames, default_file_reader="mpop.readers.eps_l1b.EPSAVHRRL1BFileReader"):
        """Identify the type of a file by its filename or by its contents.

        Uses previously loaded information from the configuration file.
        """
        file_types = {}
        # remaining_filenames = [os.path.basename(fn) for fn in filenames]
        remaining_filenames = filenames[:]
        print self.file_types
        for file_type_name, file_type_info in self.file_types.items():
            file_types[file_type_name] = []
            file_reader_class = self._runtime_import(file_type_info.get("file_reader", default_file_reader))
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
            for k, info_dict in [
                ("file_type", self.file_types),
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
        datasets_loaded = {}
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
                areas[nav_name] = area = self._load_navigation(nav_name, file_type, extra_mask=data.mask)
            else:
                area = areas[nav_name]

            # Create a projectable from info from the file data and the config file
            # FIXME: Remove metadata that is reader only
            dataset_info.setdefault("units", file_reader.get_units(file_key))
            dataset_info.setdefault("platform", file_reader.get_platform_name())
            dataset_info.setdefault("sensor", file_reader.get_sensor_name())
            dataset_info.setdefault("start_orbit", file_reader.get_begin_orbit_number())
            dataset_info.setdefault("end_orbit", file_reader.get_end_orbit_number())
            #dataset_info.setdefault("rows_per_scan", self.navigations[nav_name]["rows_per_scan"])
            projectable = Projectable(data=data,
                                      start_time=file_reader.start_time,
                                      end_time=file_reader.end_time,
                                      **dataset_info)
            projectable.info["area"] = area

            datasets_loaded[ds] = projectable
        return datasets_loaded

    def _load_navigation_old(self, nav_name, dep_file_type, extra_mask=None):
        """Load the `nav_name` navigation.

        For VIIRS, if we haven't loaded the geolocation file read the `dep_file_type` header
        to figure out where it is.
        """
        nav_info = self.navigations[nav_name]
        lon_key = nav_info["longitude_key"]
        lat_key = nav_info["latitude_key"]
        file_type = nav_info["file_type"]

        if file_type in self.file_readers:
            file_reader = self.file_readers[file_type]
        else:
            LOG.debug("Geolocation files were not provided, will search band file header...")
            dataset_file_reader = self.file_readers[dep_file_type]
            base_dirs = [os.path.dirname(fn) for fn in dataset_file_reader.filenames]
            geo_filenames = dataset_file_reader.geo_filenames
            geo_filepaths = [os.path.join(bd, gf) for bd, gf in zip(base_dirs, geo_filenames)]

            file_types = self.identify_file_types(geo_filepaths)
            if file_type not in file_types:
                raise RuntimeError("The geolocation files from the header (ex. %s)"
                                   " do not match the configured geolocation (%s)" % (geo_filepaths[0], file_type))
            file_reader = MultiFileReader(file_type, file_types[file_type], self.file_keys)

        lon_data = file_reader.get_swath_data(lon_key, extra_mask=extra_mask)
        lat_data = file_reader.get_swath_data(lat_key, extra_mask=extra_mask)

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

def GenericFileReader(object):
    def get_swath_data(self, item, dataset_name=None, data_out=None, mask_out=None):
        if item in ["longitude", "latitude"]:
            # TODO: compute the lon lat here from tle (pyorbital)
            return
        raise NotImplementedError

    def get_shape(self, item):
        raise NotImplementedError

