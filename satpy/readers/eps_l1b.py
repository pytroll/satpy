#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Reader for eps level 1b data. Uses xml files as a format description."""

import functools
import logging

import dask.array as da
import numpy as np
import xarray as xr
from dask.delayed import delayed
from pyresample.geometry import SwathDefinition

from satpy import CHUNK_SIZE
from satpy._compat import cached_property
from satpy._config import get_config_path
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.xmlformat import XMLFormat

logger = logging.getLogger(__name__)

C1 = 1.191062e-05  # mW/(m2*sr*cm-4)
C2 = 1.4387863  # K/cm-1


def radiance_to_bt(arr, wc_, a__, b__):
    """Convert to BT in K."""
    return a__ + b__ * (C2 * wc_ / (da.log(1 + (C1 * (wc_ ** 3) / arr))))


def radiance_to_refl(arr, solar_flux):
    """Convert to reflectances in %."""
    return arr * np.pi * 100.0 / solar_flux


record_class = ["Reserved", "mphr", "sphr",
                "ipr", "geadr", "giadr",
                "veadr", "viadr", "mdr"]


def read_records(filename):
    """Read *filename* without scaling it afterwards."""
    format_fn = get_config_path("eps_avhrrl1b_6.5.xml")
    form = XMLFormat(format_fn)

    grh_dtype = np.dtype([("record_class", "|i1"),
                          ("INSTRUMENT_GROUP", "|i1"),
                          ("RECORD_SUBCLASS", "|i1"),
                          ("RECORD_SUBCLASS_VERSION", "|i1"),
                          ("RECORD_SIZE", ">u4"),
                          ("RECORD_START_TIME", "S6"),
                          ("RECORD_STOP_TIME", "S6")])

    max_lines = np.floor((CHUNK_SIZE ** 2) / 2048)

    dtypes = []
    cnt = 0
    counts = []
    classes = []
    prev = None
    with open(filename, "rb") as fdes:
        while True:
            grh = np.fromfile(fdes, grh_dtype, 1)
            if grh.size == 0:
                break
            rec_class = record_class[int(grh["record_class"])]
            sub_class = grh["RECORD_SUBCLASS"][0]

            expected_size = int(grh["RECORD_SIZE"])
            bare_size = expected_size - grh_dtype.itemsize
            try:
                the_type = form.dtype((rec_class, sub_class))
                # the_descr = grh_dtype.descr + the_type.descr
            except KeyError:
                the_type = np.dtype([('unknown', 'V%d' % bare_size)])
            the_descr = grh_dtype.descr + the_type.descr
            the_type = np.dtype(the_descr)
            if the_type.itemsize < expected_size:
                padding = [('unknown%d' % cnt, 'V%d' % (expected_size - the_type.itemsize))]
                cnt += 1
                the_descr += padding
            new_dtype = np.dtype(the_descr)
            key = (rec_class, sub_class)
            if key == prev:
                counts[-1] += 1
            else:
                dtypes.append(new_dtype)
                counts.append(1)
                classes.append(key)
                prev = key
            fdes.seek(expected_size - grh_dtype.itemsize, 1)

        sections = {}
        offset = 0
        for dtype, count, rec_class in zip(dtypes, counts, classes):
            fdes.seek(offset)
            if rec_class == ('mdr', 2):
                record = da.from_array(np.memmap(fdes, mode='r', dtype=dtype, shape=count, offset=offset),
                                       chunks=(max_lines,))
            else:
                record = np.fromfile(fdes, dtype=dtype, count=count)
            offset += dtype.itemsize * count
            if rec_class in sections:
                logger.debug('Multiple records for ', str(rec_class))
                sections[rec_class] = np.hstack((sections[rec_class], record))
            else:
                sections[rec_class] = record

    return sections, form


def create_xarray(arr):
    """Create xarray with correct dimensions."""
    res = arr
    res = xr.DataArray(res, dims=['y', 'x'])
    return res


class EPSAVHRRFile(BaseFileHandler):
    """Eps level 1b reader for AVHRR data."""

    spacecrafts = {"M01": "Metop-B",
                   "M02": "Metop-A",
                   "M03": "Metop-C", }

    sensors = {"AVHR": "avhrr-3"}

    units = {"reflectance": "%",
             "brightness_temperature": "K"}

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize FileHandler."""
        super(EPSAVHRRFile, self).__init__(
            filename, filename_info, filetype_info)

        self.area = None
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        self.form = None
        self.scanlines = None
        self.pixels = None
        self.sections = None
        self.get_full_angles = functools.lru_cache(maxsize=1)(
            self._get_full_angles_uncached
        )
        self.get_full_lonlats = functools.lru_cache(maxsize=1)(
            self._get_full_lonlats_uncached
        )

    def _read_all(self):
        logger.debug("Reading %s", self.filename)
        self.sections, self.form = read_records(self.filename)
        self.scanlines = self['TOTAL_MDR']
        if self.scanlines != len(self.sections[('mdr', 2)]):
            logger.warning("Number of declared records doesn't match number of scanlines in the file.")
            self.scanlines = len(self.sections[('mdr', 2)])
        self.pixels = self["EARTH_VIEWS_PER_SCANLINE"]

    def __getitem__(self, key):
        """Get value for given key."""
        for altkey in self.form.scales:
            try:
                try:
                    return self.sections[altkey][key] * self.form.scales[altkey][key]
                except TypeError:
                    val = self.sections[altkey][key].item().decode().split("=")[1]
                    try:
                        return float(val) * self.form.scales[altkey][key].item()
                    except ValueError:
                        return val.strip()
            except (KeyError, ValueError):
                continue
        raise KeyError("No matching value for " + str(key))

    def keys(self):
        """List of reader's keys."""
        keys = []
        for val in self.form.scales.values():
            keys += val.dtype.fields.keys()
        return keys

    def _get_full_lonlats_uncached(self):
        """Get the interpolated longitudes and latitudes."""
        raw_lats = np.hstack((self["EARTH_LOCATION_FIRST"][:, [0]],
                              self["EARTH_LOCATIONS"][:, :, 0],
                              self["EARTH_LOCATION_LAST"][:, [0]]))

        raw_lons = np.hstack((self["EARTH_LOCATION_FIRST"][:, [1]],
                              self["EARTH_LOCATIONS"][:, :, 1],
                              self["EARTH_LOCATION_LAST"][:, [1]]))
        return self._interpolate(raw_lons, raw_lats)

    def _interpolate(self, lons_like, lats_like):
        nav_sample_rate = self["NAV_SAMPLE_RATE"]
        if nav_sample_rate == 20 and self.pixels == 2048:
            lons_like_1km, lats_like_1km = self._interpolate_20km_to_1km(lons_like, lats_like)
            lons_like_1km = da.from_delayed(lons_like_1km, dtype=lons_like.dtype,
                                            shape=(self.scanlines, self.pixels))
            lats_like_1km = da.from_delayed(lats_like_1km, dtype=lats_like.dtype,
                                            shape=(self.scanlines, self.pixels))
            return lons_like_1km, lats_like_1km

        raise NotImplementedError("Lon/lat and angle expansion not implemented for " +
                                  "sample rate = " + str(nav_sample_rate) +
                                  " and earth views = " +
                                  str(self.pixels))

    @delayed(nout=2, pure=True)
    def _interpolate_20km_to_1km(self, lons, lats):
        # Note: delayed will cast input dask-arrays to numpy arrays (needed by metop20kmto1km).
        from geotiepoints import metop20kmto1km
        return metop20kmto1km(lons, lats)

    def _get_full_angles(self, solar_zenith, sat_zenith, solar_azimuth, sat_azimuth):

        nav_sample_rate = self["NAV_SAMPLE_RATE"]
        if nav_sample_rate == 20 and self.pixels == 2048:
            # Note: interpolation assumes second array values between -90 and 90
            # Solar and satellite zenith is between 0 and 180.
            sun_azi, sun_zen = self._interpolate(solar_azimuth, solar_zenith - 90)
            sun_zen += 90
            sat_azi, sat_zen = self._interpolate(sat_azimuth, sat_zenith - 90)
            sat_zen += 90
            return sun_azi, sun_zen, sat_azi, sat_zen
        else:
            raise NotImplementedError("Angles expansion not implemented for " +
                                      "sample rate = " + str(nav_sample_rate) +
                                      " and earth views = " +
                                      str(self.pixels))

    def _get_full_angles_uncached(self):
        """Get the interpolated angles."""
        solar_zenith = np.hstack((self["ANGULAR_RELATIONS_FIRST"][:, [0]],
                                  self["ANGULAR_RELATIONS"][:, :, 0],
                                  self["ANGULAR_RELATIONS_LAST"][:, [0]]))

        sat_zenith = np.hstack((self["ANGULAR_RELATIONS_FIRST"][:, [1]],
                                self["ANGULAR_RELATIONS"][:, :, 1],
                                self["ANGULAR_RELATIONS_LAST"][:, [1]]))

        solar_azimuth = np.hstack((self["ANGULAR_RELATIONS_FIRST"][:, [2]],
                                   self["ANGULAR_RELATIONS"][:, :, 2],
                                   self["ANGULAR_RELATIONS_LAST"][:, [2]]))
        sat_azimuth = np.hstack((self["ANGULAR_RELATIONS_FIRST"][:, [3]],
                                 self["ANGULAR_RELATIONS"][:, :, 3],
                                 self["ANGULAR_RELATIONS_LAST"][:, [3]]))

        return self._get_full_angles(solar_zenith,
                                     sat_zenith,
                                     solar_azimuth,
                                     sat_azimuth)

    def get_bounding_box(self):
        """Get bounding box."""
        if self.sections is None:
            self._read_all()
        lats = np.hstack([self["EARTH_LOCATION_FIRST"][0, [0]],
                          self["EARTH_LOCATION_LAST"][0, [0]],
                          self["EARTH_LOCATION_LAST"][-1, [0]],
                          self["EARTH_LOCATION_FIRST"][-1, [0]]])
        lons = np.hstack([self["EARTH_LOCATION_FIRST"][0, [1]],
                          self["EARTH_LOCATION_LAST"][0, [1]],
                          self["EARTH_LOCATION_LAST"][-1, [1]],
                          self["EARTH_LOCATION_FIRST"][-1, [1]]])
        return lons.ravel(), lats.ravel()

    def get_dataset(self, key, info):
        """Get calibrated channel data."""
        if self.sections is None:
            self._read_all()

        if key['name'] in ['longitude', 'latitude']:
            lons, lats = self.get_full_lonlats()
            if key['name'] == 'longitude':
                dataset = create_xarray(lons)
            else:
                dataset = create_xarray(lats)

        elif key['name'] in ['solar_zenith_angle', 'solar_azimuth_angle',
                             'satellite_zenith_angle', 'satellite_azimuth_angle']:
            dataset = self._get_angle_dataarray(key)
        elif key['name'] in ["1", "2", "3a", "3A", "3b", "3B", "4", "5"]:
            dataset = self._get_calibrated_dataarray(key)
        else:
            logger.info("Can't load channel in eps_l1b: " + str(key['name']))
            return

        dataset.attrs['platform_name'] = self.platform_name
        dataset.attrs['sensor'] = self.sensor_name
        if "calibration" in key:
            dataset.attrs["units"] = self.units[key["calibration"]]
        dataset.attrs.update(info)
        dataset.attrs.update(key.to_dict())
        return dataset

    def _get_angle_dataarray(self, key):
        """Get an angle dataarray."""
        sun_azi, sun_zen, sat_azi, sat_zen = self.get_full_angles()
        if key['name'] == 'solar_zenith_angle':
            dataset = create_xarray(sun_zen)
        elif key['name'] == 'solar_azimuth_angle':
            dataset = create_xarray(sun_azi)
        if key['name'] == 'satellite_zenith_angle':
            dataset = create_xarray(sat_zen)
        elif key['name'] == 'satellite_azimuth_angle':
            dataset = create_xarray(sat_azi)
        return dataset

    @cached_property
    def three_a_mask(self):
        """Mask for 3A."""
        return (self["FRAME_INDICATOR"] & 2 ** 16) != 2 ** 16

    @cached_property
    def three_b_mask(self):
        """Mask for 3B."""
        return (self["FRAME_INDICATOR"] & 2 ** 16) != 0

    def _get_calibrated_dataarray(self, key):
        """Get a calibrated dataarray."""
        if key['calibration'] not in ['reflectance', 'brightness_temperature', 'radiance']:
            raise ValueError('calibration type ' + str(key['calibration']) +
                             ' is not supported!')

        mask = None

        channel_name = key['name'].upper()

        radiance_indices = {"1": 0, "2": 1, "3A": 2, "3B": 2, "4": 3, "5": 4}
        array = self["SCENE_RADIANCES"][:, radiance_indices[channel_name], :]

        if channel_name in ["1", "2", "3A"]:
            if key['calibration'] == 'reflectance':
                array = radiance_to_refl(array,
                                         self[f"CH{channel_name}_SOLAR_FILTERED_IRRADIANCE"])
            if channel_name == "3A":
                mask = self.three_a_mask[:, np.newaxis]

        if channel_name in ["3B", "4", "5"]:
            if key['calibration'] == 'brightness_temperature':
                array = radiance_to_bt(array,
                                       self[f"CH{channel_name}_CENTRAL_WAVENUMBER"],
                                       self[f"CH{channel_name}_CONSTANT1"],
                                       self[f"CH{channel_name}_CONSTANT2_SLOPE"])
            if channel_name == "3B":
                mask = self.three_b_mask[:, np.newaxis]

        dataset = create_xarray(array)
        if mask is not None:
            dataset = dataset.where(~mask)
        return dataset

    def get_lonlats(self):
        """Get lonlats."""
        if self.area is None:
            lons, lats = self.get_full_lonlats()
            self.area = SwathDefinition(lons, lats)
            self.area.name = '_'.join([self.platform_name, str(self.start_time),
                                       str(self.end_time)])
        return self.area

    @property
    def platform_name(self):
        """Get platform name."""
        return self.spacecrafts[self["SPACECRAFT_ID"]]

    @property
    def sensor_name(self):
        """Get sensor name."""
        return self.sensors[self["INSTRUMENT_ID"]]

    @property
    def start_time(self):
        """Get start time."""
        # return datetime.strptime(self["SENSING_START"], "%Y%m%d%H%M%SZ")
        return self._start_time

    @property
    def end_time(self):
        """Get end time."""
        # return datetime.strptime(self["SENSING_END"], "%Y%m%d%H%M%SZ")
        return self._end_time
