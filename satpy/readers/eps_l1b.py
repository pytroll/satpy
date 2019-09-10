#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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

# Copyright (c) 2012, 2013, 2014 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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

"""Reader for eps level 1b data. Uses xml files as a format description.
"""

import logging
import os

import numpy as np
import xarray as xr

import dask.array as da
from dask.delayed import delayed
from pyresample.geometry import SwathDefinition
from satpy.config import CONFIG_PATH
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.xmlformat import XMLFormat
from satpy import CHUNK_SIZE

LOG = logging.getLogger(__name__)

C1 = 1.191062e-05  # mW/(m2*sr*cm-4)
C2 = 1.4387863  # K/cm-1


def radiance_to_bt(arr, wc_, a__, b__):
    """Convert to BT.
    """
    return a__ + b__ * (C2 * wc_ / (da.log(1 + (C1 * (wc_ ** 3) / arr))))


def radiance_to_refl(arr, solar_flux):
    """Convert to reflectances.
    """
    return arr * np.pi * 100.0 / solar_flux


record_class = ["Reserved", "mphr", "sphr",
                "ipr", "geadr", "giadr",
                "veadr", "viadr", "mdr"]


def read_raw(filename):
    """Read *filename* without scaling it afterwards.
    """

    form = XMLFormat(os.path.join(CONFIG_PATH, "eps_avhrrl1b_6.5.xml"))

    grh_dtype = np.dtype([("record_class", "|i1"),
                          ("INSTRUMENT_GROUP", "|i1"),
                          ("RECORD_SUBCLASS", "|i1"),
                          ("RECORD_SUBCLASS_VERSION", "|i1"),
                          ("RECORD_SIZE", ">u4"),
                          ("RECORD_START_TIME", "S6"),
                          ("RECORD_STOP_TIME", "S6")])

    dtypes = []
    cnt = 0
    with open(filename, "rb") as fdes:
        while True:
            grh = np.fromfile(fdes, grh_dtype, 1)
            if not grh:
                break
            rec_class = record_class[int(grh["record_class"])]
            sub_class = grh["RECORD_SUBCLASS"][0]

            expected_size = int(grh["RECORD_SIZE"])
            bare_size = expected_size - grh_dtype.itemsize
            try:
                the_type = form.dtype((rec_class, sub_class))
                the_descr = grh_dtype.descr + the_type.descr
            except KeyError:
                the_type = np.dtype([('unknown', 'V%d' % bare_size)])
            the_descr = grh_dtype.descr + the_type.descr
            the_type = np.dtype(the_descr)
            if the_type.itemsize < expected_size:
                padding = [('unknown%d' % cnt, 'V%d' % (expected_size - the_type.itemsize))]
                cnt += 1
                the_descr += padding
            dtypes.append(np.dtype(the_descr))
            fdes.seek(expected_size - grh_dtype.itemsize, 1)

        file_dtype = np.dtype([(str(num), the_dtype) for num, the_dtype in enumerate(dtypes)])
        records = np.memmap(fdes, mode='r', dtype=file_dtype, shape=1)[0]

    return records, form


def create_xarray(arr):
    res = arr
    res = xr.DataArray(res, dims=['y', 'x'])
    return res


class EPSAVHRRFile(BaseFileHandler):
    """Eps level 1b reader for AVHRR data.
    """
    spacecrafts = {"M01": "Metop-B",
                   "M02": "Metop-A",
                   "M03": "Metop-C", }

    sensors = {"AVHR": "avhrr-3"}

    def __init__(self, filename, filename_info, filetype_info):
        super(EPSAVHRRFile, self).__init__(
            filename, filename_info, filetype_info)

        self.lons, self.lats = None, None
        self.sun_azi, self.sun_zen, self.sat_azi, self.sat_zen = None, None, None, None
        self.area = None
        self.three_a_mask, self.three_b_mask = None, None
        self._start_time = filename_info['start_time']
        self._end_time = filename_info['end_time']
        self.records = None
        self.form = None
        self.mdrs = None
        self.scanlines = None
        self.pixels = None
        self.sections = None

    def _read_all(self, filename):
        LOG.debug("Reading %s", filename)
        self.records, self.form = read_raw(filename)
        self.mdrs = [record
                     for record in self.records
                     if record_class[record['record_class']] == "mdr"]
        self.iprs = [record
                     for record in self.records
                     if record_class[record['record_class']] == "ipr"]
        self.scanlines = len(self.mdrs)
        self.sections = {("mdr", 2): np.hstack(self.mdrs)}
        self.sections[("ipr", 0)] = np.hstack(self.iprs)
        for record in self.records:
            rec_class = record_class[record['record_class']]
            sub_class = record["RECORD_SUBCLASS"]
            if rec_class in ["mdr", "ipr"]:
                continue
            if (rec_class, sub_class) in self.sections:
                raise ValueError("Too many " + str((rec_class, sub_class)))
            else:
                self.sections[(rec_class, sub_class)] = record
        self.pixels = self["EARTH_VIEWS_PER_SCANLINE"]

    def __getitem__(self, key):
        for altkey in self.form.scales.keys():
            try:
                try:
                    return (da.from_array(self.sections[altkey][key], chunks=CHUNK_SIZE)
                            * self.form.scales[altkey][key])
                except TypeError:
                    val = self.sections[altkey][key].decode().split("=")[1]
                    try:
                        return float(val) * self.form.scales[altkey][key]
                    except ValueError:
                        return val.strip()
            except (KeyError, ValueError):
                continue
        raise KeyError("No matching value for " + str(key))

    def keys(self):
        """List of reader's keys.
        """
        keys = []
        for val in self.form.scales.values():
            keys += val.dtype.fields.keys()
        return keys

    @delayed(nout=2, pure=True)
    def _get_full_lonlats(self):
        lats = np.hstack((self["EARTH_LOCATION_FIRST"][:, [0]],
                          self["EARTH_LOCATIONS"][:, :, 0],
                          self["EARTH_LOCATION_LAST"][:, [0]]))

        lons = np.hstack((self["EARTH_LOCATION_FIRST"][:, [1]],
                          self["EARTH_LOCATIONS"][:, :, 1],
                          self["EARTH_LOCATION_LAST"][:, [1]]))

        nav_sample_rate = self["NAV_SAMPLE_RATE"]
        earth_views_per_scanline = self["EARTH_VIEWS_PER_SCANLINE"]
        if nav_sample_rate == 20 and earth_views_per_scanline == 2048:
            from geotiepoints import metop20kmto1km
            return metop20kmto1km(lons, lats)
        else:
            raise NotImplementedError("Lon/lat expansion not implemented for " +
                                      "sample rate = " + str(nav_sample_rate) +
                                      " and earth views = " +
                                      str(earth_views_per_scanline))

    def get_full_lonlats(self):
        """Get the interpolated lons/lats.
        """
        if self.lons is not None and self.lats is not None:
            return self.lons, self.lats

        self.lons, self.lats = self._get_full_lonlats()
        self.lons = da.from_delayed(self.lons, dtype=self["EARTH_LOCATIONS"].dtype,
                                    shape=(self.scanlines, self.pixels))
        self.lats = da.from_delayed(self.lats, dtype=self["EARTH_LOCATIONS"].dtype,
                                    shape=(self.scanlines, self.pixels))
        return self.lons, self.lats

    @delayed(nout=4, pure=True)
    def _get_full_angles(self):
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

        nav_sample_rate = self["NAV_SAMPLE_RATE"]
        earth_views_per_scanline = self["EARTH_VIEWS_PER_SCANLINE"]
        if nav_sample_rate == 20 and earth_views_per_scanline == 2048:
            from geotiepoints import metop20kmto1km
            # Note: interpolation asumes lat values values between -90 and 90
            # Solar and satellite zenith is between 0 and 180.
            solar_zenith -= 90
            sun_azi, sun_zen = metop20kmto1km(
                solar_azimuth, solar_zenith)
            sun_zen += 90
            sat_zenith -= 90
            sat_azi, sat_zen = metop20kmto1km(
                sat_azimuth, sat_zenith)
            sat_zen += 90
            return sun_azi, sun_zen, sat_azi, sat_zen
        else:
            raise NotImplementedError("Angles expansion not implemented for " +
                                      "sample rate = " + str(nav_sample_rate) +
                                      " and earth views = " +
                                      str(earth_views_per_scanline))

    def get_full_angles(self):
        """Get the interpolated lons/lats.
        """
        if (self.sun_azi is not None and self.sun_zen is not None and
                self.sat_azi is not None and self.sat_zen is not None):
            return self.sun_azi, self.sun_zen, self.sat_azi, self.sat_zen

        self.sun_azi, self.sun_zen, self.sat_azi, self.sat_zen = self._get_full_angles()
        self.sun_azi = da.from_delayed(self.sun_azi, dtype=self["ANGULAR_RELATIONS"].dtype,
                                       shape=(self.scanlines, self.pixels))
        self.sun_zen = da.from_delayed(self.sun_zen, dtype=self["ANGULAR_RELATIONS"].dtype,
                                       shape=(self.scanlines, self.pixels))
        self.sat_azi = da.from_delayed(self.sat_azi, dtype=self["ANGULAR_RELATIONS"].dtype,
                                       shape=(self.scanlines, self.pixels))
        self.sat_zen = da.from_delayed(self.sat_zen, dtype=self["ANGULAR_RELATIONS"].dtype,
                                       shape=(self.scanlines, self.pixels))
        return self.sun_azi, self.sun_zen, self.sat_azi, self.sat_zen

    def get_bounding_box(self):
        if self.mdrs is None:
            self._read_all(self.filename)
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
        if self.mdrs is None:
            self._read_all(self.filename)

        if key.name in ['longitude', 'latitude']:
            lons, lats = self.get_full_lonlats()
            if key.name == 'longitude':
                dataset = create_xarray(lons)
            else:
                dataset = create_xarray(lats)

        elif key.name in ['solar_zenith_angle', 'solar_azimuth_angle',
                          'satellite_zenith_angle', 'satellite_azimuth_angle']:
            sun_azi, sun_zen, sat_azi, sat_zen = self.get_full_angles()
            if key.name == 'solar_zenith_angle':
                dataset = create_xarray(sun_zen)
            elif key.name == 'solar_azimuth_angle':
                dataset = create_xarray(sun_azi)
            if key.name == 'satellite_zenith_angle':
                dataset = create_xarray(sat_zen)
            elif key.name == 'satellite_azimuth_angle':
                dataset = create_xarray(sat_azi)
        else:
            mask = None
            if key.calibration == 'counts':
                raise ValueError('calibration=counts is not supported! ' +
                                 'This reader cannot return counts')
            elif key.calibration not in ['reflectance', 'brightness_temperature', 'radiance']:
                raise ValueError('calibration type ' + str(key.calibration) +
                                 ' is not supported!')

            if key.name in ['3A', '3a'] and self.three_a_mask is None:
                self.three_a_mask = ((self["FRAME_INDICATOR"] & 2 ** 16) != 2 ** 16)

            if key.name in ['3B', '3b'] and self.three_b_mask is None:
                self.three_b_mask = ((self["FRAME_INDICATOR"] & 2 ** 16) != 0)

            if key.name not in ["1", "2", "3a", "3A", "3b", "3B", "4", "5"]:
                LOG.info("Can't load channel in eps_l1b: " + str(key.name))
                return

            if key.name == "1":
                if key.calibration == 'reflectance':
                    array = radiance_to_refl(self["SCENE_RADIANCES"][:, 0, :],
                                             self["CH1_SOLAR_FILTERED_IRRADIANCE"])
                else:
                    array = self["SCENE_RADIANCES"][:, 0, :]

            if key.name == "2":
                if key.calibration == 'reflectance':
                    array = radiance_to_refl(self["SCENE_RADIANCES"][:, 1, :],
                                             self["CH2_SOLAR_FILTERED_IRRADIANCE"])
                else:
                    array = self["SCENE_RADIANCES"][:, 1, :]

            if key.name.lower() == "3a":
                if key.calibration == 'reflectance':
                    array = radiance_to_refl(self["SCENE_RADIANCES"][:, 2, :],
                                             self["CH3A_SOLAR_FILTERED_IRRADIANCE"])
                else:
                    array = self["SCENE_RADIANCES"][:, 2, :]

                mask = np.empty(array.shape, dtype=bool)
                mask[:, :] = self.three_a_mask[:, np.newaxis]

            if key.name.lower() == "3b":
                if key.calibration == 'brightness_temperature':
                    array = radiance_to_bt(self["SCENE_RADIANCES"][:, 2, :],
                                           self["CH3B_CENTRAL_WAVENUMBER"],
                                           self["CH3B_CONSTANT1"],
                                           self["CH3B_CONSTANT2_SLOPE"])
                else:
                    array = self["SCENE_RADIANCES"][:, 2, :]
                mask = np.empty(array.shape, dtype=bool)
                mask[:, :] = self.three_b_mask[:, np.newaxis]

            if key.name == "4":
                if key.calibration == 'brightness_temperature':
                    array = radiance_to_bt(self["SCENE_RADIANCES"][:, 3, :],
                                           self["CH4_CENTRAL_WAVENUMBER"],
                                           self["CH4_CONSTANT1"],
                                           self["CH4_CONSTANT2_SLOPE"])
                else:
                    array = self["SCENE_RADIANCES"][:, 3, :]

            if key.name == "5":
                if key.calibration == 'brightness_temperature':
                    array = radiance_to_bt(self["SCENE_RADIANCES"][:, 4, :],
                                           self["CH5_CENTRAL_WAVENUMBER"],
                                           self["CH5_CONSTANT1"],
                                           self["CH5_CONSTANT2_SLOPE"])
                else:
                    array = self["SCENE_RADIANCES"][:, 4, :]

            dataset = create_xarray(array)
            if mask is not None:
                dataset = dataset.where(~mask)

        dataset.attrs['platform_name'] = self.platform_name
        dataset.attrs['sensor'] = self.sensor_name
        dataset.attrs.update(info)
        dataset.attrs.update(key.to_dict())
        return dataset

    def get_lonlats(self):
        if self.area is None:
            if self.lons is None or self.lats is None:
                self.lons, self.lats = self.get_full_lonlats()
            self.area = SwathDefinition(self.lons, self.lats)
            self.area.name = '_'.join([self.platform_name, str(self.start_time),
                                       str(self.end_time)])
        return self.area

    @property
    def platform_name(self):
        return self.spacecrafts[self["SPACECRAFT_ID"]]

    @property
    def sensor_name(self):
        return self.sensors[self["INSTRUMENT_ID"]]

    @property
    def start_time(self):
        # return datetime.strptime(self["SENSING_START"], "%Y%m%d%H%M%SZ")
        return self._start_time

    @property
    def end_time(self):
        # return datetime.strptime(self["SENSING_END"], "%Y%m%d%H%M%SZ")
        return self._end_time


if __name__ == '__main__':
    def norm255(a__):
        """normalize array to uint8.
        """
        arr = a__ * 1.0
        arr = (arr - arr.min()) * 255.0 / (arr.max() - arr.min())
        return arr.astype(np.uint8)

    def show(a__):
        """show array.
        """
        from PIL import Image
        Image.fromarray(norm255(a__), "L").show()

    import sys
    res = read_raw(sys.argv[1])
