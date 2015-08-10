#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2011, 2012, 2013, 2014, 2015.

# Author(s):

#
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Kristian Rune Larsen <krl@dmi.dk>
#   Lars Ørum Rasmussen <ras@dmi.dk>
#   Martin Raspaud <martin.raspaud@smhi.se>
#

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

"""Interface to VIIRS SDR format

Format documentation:
http://npp.gsfc.nasa.gov/science/sciencedocuments/082012/474-00001-03_CDFCBVolIII_RevC.pdf

"""
import os.path
from datetime import datetime, timedelta

import numpy as np
import h5py
import hashlib
import logging
from collections import namedtuple

from mpop.projectable import Projectable
from mpop.readers import Reader
from fnmatch import fnmatch


NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)

VIIRS_IR_BANDS = ('M16', 'M15', 'M14', 'M13', 'M12', 'I5', 'I4')
VIIRS_VIS_BANDS = ('M1', 'M2', 'M3', 'M4', 'M5', 'M6',
                   'M7', 'M8', 'M9', 'M10', 'M11',
                   'I1', 'I2', 'I3')
VIIRS_DNB_BANDS = ('DNB', )

LOG = logging.getLogger(__name__)


# FIXME: Add more information
class FileKey(namedtuple("FileKey", ["name", "variable_name", "scaling_factors", "dtype", "kwargs"])):
    def __new__(cls, name, variable_name, scaling_factors=None, dtype=np.float32, **kwargs):
        if isinstance(dtype, (str, unicode)):
            # get the data type from numpy
            dtype = getattr(np, dtype)
        return super(FileKey, cls).__new__(cls, name, variable_name, scaling_factors, dtype, kwargs)


class HDF5MetaData(object):
    """Small class for inspecting a HDF5 file and retrieve its metadata/header data.
    """
    def __init__(self, filename, **kwargs):
        self.metadata = {}
        self.filename = filename
        if not os.path.exists(filename):
            raise IOError("File %s does not exist!" % filename)
        file_handle = h5py.File(self.filename, 'r')
        file_handle.visititems(self.collect_metadata)
        self._collect_attrs('/', file_handle.attrs)
        file_handle.close()

    def _collect_attrs(self, name, attrs):
        for key, value in attrs.iteritems():
            value = np.squeeze(value)
            if issubclass(value.dtype.type, str):
                self.metadata["%s/attr/%s" % (name, key)] = str(value)
            else:
                self.metadata["%s/attr/%s" % (name, key)] = value

    def collect_metadata(self, name, obj):
        if isinstance(obj, h5py.Dataset):
            self.metadata[name] = obj
            self.metadata[name + "/shape"] = obj.shape
        self._collect_attrs(name, obj.attrs)

    def __getitem__(self, key):
        val = self.metadata[key]
        if isinstance(val, h5py.Dataset):
            # these datasets are closed and inaccessible when the file is closed, need to reopen
            return h5py.File(self.filename, 'r')[key].value
        return val

    def keys(self):
        return self.metadata.keys()

    def get_data_keys(self):

        data_keys = []
        for key in self.metadata.keys():
            if key.endswith("/shape"):
                data_key = key.split("/shape")[0]
                data_keys.append(data_key)
        return data_keys


class SDRFileReader(HDF5MetaData):
    """VIIRS HDF5 File Reader
    """
    def __init__(self, file_type, filename, file_keys=None, **kwargs):
        super(SDRFileReader, self).__init__(filename, **kwargs)
        self.file_type = file_type

        self.file_keys = file_keys
        self.file_info = kwargs

        self.start_time = self.get_begin_time()
        self.end_time = self.get_end_time()

    def __getitem__(self, item):
        if item.endswith("/shape") and item[:-6] in self.file_keys:
            item = self.file_keys[item[:-6]].variable_name.format(**self.file_info) + "/shape"
        elif item in self.file_keys:
            item = self.file_keys[item].variable_name.format(**self.file_info)

        return super(SDRFileReader, self).__getitem__(item)

    def _parse_npp_datatime(self, datestr, timestr):
        time_val = datetime.strptime(datestr + timestr, '%Y%m%d%H%M%S.%fZ')
        if abs(time_val - NO_DATE) < EPSILON_TIME:
            raise ValueError("Datetime invalid %s " % time_val)
        return time_val

    def get_ring_lonlats(self):
        return self["gring_longitude"], self["gring_latitude"]

    def get_begin_time(self):
        return self._parse_npp_datatime(self['beginning_date'], self['beginning_time'])

    def get_end_time(self):
        return self._parse_npp_datatime(self['ending_date'], self['ending_time'])

    def get_begin_orbit_number(self):
        return int(self['AggregateBeginningOrbitNumber'])

    def get_end_orbit_number(self):
        return int(self['AggregateEndingOrbitNumber'])

    def get_geofilname(self):
        return self['N_GEO_Ref']

    def get_unit(self, calibrate=1):

        band = self['Band_ID']
        if calibrate == 2 and band not in VIIRS_DNB_BANDS:
            return "W m-2 um-1 sr-1"

        if band in VIIRS_IR_BANDS:
            return "K"
        elif band in VIIRS_VIS_BANDS:
            return '%'
        elif band in VIIRS_DNB_BANDS:
            return 'W m-2 sr-1'

        return None

    def scale_swath_data(self, data, mask, scaling_factors):
        """Scale swath data using scaling factors and offsets.

        Multi-granule (a.k.a. aggregated) files will have more than the usual two values.
        """
        num_grans = len(scaling_factors)/2
        gran_size = data.shape[0]/num_grans
        for i in range(num_grans):
            start_idx = i * gran_size
            end_idx = start_idx + gran_size
            m = scaling_factors[i*2]
            b = scaling_factors[i*2 + 1]
            # in rare cases the scaling factors are actually fill values
            if m <= -999 or b <= -999:
                mask[start_idx:end_idx] = 1
            else:
                data[start_idx:end_idx] *= m
                data[start_idx:end_idx] += b

        return data, mask

    def get_swath_data(self, item, filename=False, data_out=None, mask_out=None):
        """Get swath data, apply proper scalings, and apply proper masks.
        """
        if filename:
            raise NotImplementedError("Saving data arrays to disk is not supported yet")

        if not self.file_keys:
            raise RuntimeError("Can not get swath data when no file key information was found")

        # Can't guarantee proper file info until we get the data first
        var_info = self.file_keys[item]
        data = self[item]
        is_floating = np.issubdtype(data_out.dtype, np.floating)
        if data_out is not None:
            # This assumes that we are promoting the dtypes (ex. float file data -> int array)
            # and that it happens automatically when assigning to the existing out array
            data_out[:] = data
        else:
            data_out = data[:].astype(var_info.dtype)
            mask_out = np.zeros_like(data_out, dtype=np.bool)

        try:
            factors = self[var_info.scaling_factors] if var_info.scaling_factors is not None else None
        except KeyError:
            LOG.debug("No scaling factors found for %s", item)
            factors = None

        if is_floating:
            # If the data is a float then we mask everything <= -999.0
            fill_max = float(var_info.kwargs.get("fill_max_float", -999.0))
            mask_out[:] = data_out <= fill_max
        else:
            # If the data is an integer then we mask everything >= fill_min_int
            fill_min = int(var_info.kwargs.get("fill_min_int", 65528))
            mask_out[:] = data_out >= fill_min

        if factors is not None:
            data_out, scaling_mask = self.scale_swath_data(data_out, mask_out, factors)

        return data_out, mask_out


class MultiFileReader(object):
    def __init__(self, file_type, file_readers, file_keys=None, **kwargs):
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

    def get_swath_data(self, item, extra_mask=None, filename=None):
        if self.file_keys is None:
            raise RuntimeError("Can not get swath data when no file key information was found")

        var_info = self.file_keys[item]
        granule_shapes = [x[item + "/shape"] for x in self.file_readers]
        num_rows = sum([x[0] for x in granule_shapes])
        num_cols = granule_shapes[0][1]

        if filename:
            raise NotImplementedError("Saving data arrays to disk is not supported yet")
            # data = np.memmap(filename, dtype=var_info.dtype, mode='w', shape=(num_rows, num_cols))
        else:
            data = np.empty((num_rows, num_cols), dtype=var_info.dtype)
            if extra_mask is not None:
                mask = extra_mask.copy()
            else:
                mask = np.zeros_like(data, dtype=np.bool)

        idx = 0
        for granule_shape, file_reader in zip(granule_shapes, self.file_readers):
            # Get the data from each individual file reader (assumes it gets the data with the right data type)
            file_reader.get_swath_data(item, filename=filename,
                                       data_out=data[idx: idx + granule_shape[0]],
                                       mask_out=mask[idx: idx + granule_shape[0]])
            idx += granule_shape[0]

        # FIXME: This may get ugly when using memmaps, maybe move projectable creation here instead
        return np.ma.array(data, mask=mask, copy=False)


class ViirsSDRReader(Reader):
    def __init__(self, *args, **kwargs):
        self.file_types = {}
        self.file_readers = {}
        self.file_keys = {}
        self.nav_sets = {}
        kwargs.setdefault("default_config_filename", "readers/viirs_sdr.cfg")

        # Load the configuration file and other defaults
        Reader.__init__(self, *args, **kwargs)

        # Determine what we know about the files provided and create file readers to read them
        file_types = self.identify_file_types(self.filenames)
        num_files = 0
        for file_type_name, file_type_files in file_types.items():
            file_type_files = self._get_swathsegment(file_type_files)
            LOG.debug("File type %s has %d files after segment selection", file_type_name, len(file_type_files))

            if len(file_type_files) == 0:
                raise IOError("No VIIRS SDR files matching!: " +
                              "Start time = " + str(self.start_time) +
                              "  End time = " + str(self.end_time))
            elif num_files and len(file_type_files) != num_files:
                raise IOError("Varying numbers of files found", file_type_name)
            else:
                num_files = len(file_type_files)

            file_reader = MultiFileReader(file_type_name, file_types[file_type_name], file_keys=self.file_keys)
            self.file_readers[file_type_name] = file_reader

        # TODO: Double check that we have all of the navigation file types for the files we do have
        # TODO: Find geolocation files based on headers if we don't have them

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
                if self.start_time >= file_start and self.start_time <= file_end:
                    segment_readers.append(file_reader)
                    continue
            else:
                # search for multiple granules
                # check that granule start time is inside interval
                if file_start >= self.start_time and file_start <= self.end_time:
                    segment_readers.append(file_reader)
                    continue

                # check that granule end time is inside interval
                if file_end >= self.start_time and file_end <= self.end_time:
                    segment_readers.append(file_reader)
                    continue

        return sorted(segment_readers, key=lambda x: x.start_time)

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
        self.nav_sets[name] = section_options

    def identify_file_types(self, filenames, default_file_reader="mpop.readers.viirs_sdr.SDRReader"):
        """Identify the type of a file by its filename or by its contents.

        Uses previously loaded information from the configuration file.
        """
        file_types = {}
        # remaining_filenames = [os.path.basename(fn) for fn in filenames]
        remaining_filenames = filenames[:]
        for file_type_name, file_type_info in self.file_types.items():
            file_types[file_type_name] = []
            file_reader_class = self._runtime_import(file_type_info.pop("file_reader", default_file_reader))
            for file_pattern in file_type_info["file_patterns"]:
                tmp_remaining = []
                tmp_matching = []
                for fn in remaining_filenames:
                    # Add a wildcard to the front for path information
                    # FIXME: Is there a better way to generalize this besides removing the path every time
                    if fnmatch(fn, "*" + file_pattern):
                        tmp_matching.append(file_reader_class(file_type_name, fn, file_keys=self.file_keys, **file_type_info))
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

    def _load_navigation(self, nav_name, extra_mask=None):
        nav_info = self.nav_sets[nav_name]
        lon_key = nav_info["longitude_key"]
        lat_key = nav_info["latitude_key"]
        file_type = nav_info["file_type"]
        file_reader = self.file_readers[file_type]

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

    def load(self, channels_to_load, **kwargs):
        # Ignore `calibrate` for now
        if kwargs:
            LOG.warning("Unsupported options for viirs reader: %s", str(kwargs))

        channels_to_load = set(channels_to_load) & set(self.channel_names)
        if len(channels_to_load) == 0:
            LOG.debug("No channels to load from this reader")
            return

        LOG.debug("Channels to load: " + str(channels_to_load))

        # Sanity check and get the navigation sets being used
        areas = {}
        channels_loaded = {}
        for chn in channels_to_load:
            channel_info = self.channels[chn]
            file_type = channel_info["file_type"]
            file_key = channel_info["file_key"]
            file_reader = self.file_readers[file_type]
            nav_name = channel_info["navigation"]

            # Get the swath data (fully scaled and in the correct data type)
            data = file_reader.get_swath_data(file_key)

            # Load the navigation information first
            if nav_name not in areas:
                # FIXME: This ignores the possibility that data masks are different between bands
                areas[nav_name] = area = self._load_navigation(nav_name, extra_mask=data.mask)
            else:
                area = areas[nav_name]

            # Create a projectable from info from the file data and the config file
            # FIXME: Units are provided by the config, but should use the type from the file by default
            projectable = Projectable(data=data,
                                      start_time=file_reader.start_time,
                                      end_time=file_reader.end_time,
                                      **self.channels[chn])
            projectable.info["area"] = area

            # We assume the same geolocation should apply to all M-bands!
            # ...and the same to all I-bands:

            # if 'institution' not in glob_info:
            ##    glob_info['institution'] = band.global_info['N_Dataset_Source']
            # if 'mission_name' not in glob_info:
            ##    glob_info['mission_name'] = band.global_info['Mission_Name']
            channels_loaded[chn] = projectable

        return channels_loaded


