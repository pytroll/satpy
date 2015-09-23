#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import os
import numpy as np
from mpop import CONFIG_PATH
from mpop.satin.xmlformat import XMLFormat
import logging
from mpop.readers import ConfigBasedReader, GenericFileReader
from datetime import datetime

LOG = logging.getLogger(__name__)

try:
    from numexpr import evaluate
except ImportError:
    from numpy import log
    evaluate = eval

C1 = 1.191062e-05  # mW/(m2*sr*cm-4)
C2 = 1.4387863  # K/cm-1


def radiance_to_bt(arr, wc_, a__, b__):
    """Convert to BT.
    """
    return evaluate("a__ + b__ * (C2 * wc_ / "
                    "(log(1 + (C1 * (wc_ ** 3) / arr))))")


def radiance_to_refl(arr, solar_flux):
    """Convert to reflectances.
    """
    return arr * np.pi * 100.0 / solar_flux


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

    record_class = ["Reserved", "mphr", "sphr",
                    "ipr", "geadr", "giadr",
                    "veadr", "viadr", "mdr"]

    records = []

    with open(filename, "rb") as fdes:
        while True:
            grh = np.fromfile(fdes, grh_dtype, 1)
            if not grh:
                break
            rec_class = record_class[grh["record_class"]]
            sub_class = grh["RECORD_SUBCLASS"][0]
            offset = fdes.tell()
            try:
                record = np.memmap(fdes,
                                   form.dtype((rec_class,
                                               sub_class)),
                                   mode='r',
                                   offset=offset,
                                   shape=(1, ))
            except KeyError:
                fdes.seek(grh["RECORD_SIZE"] - 20, 1)
            else:
                fdes.seek(offset + grh["RECORD_SIZE"] - 20, 0)
                records.append((rec_class, record, sub_class))

    return records, form

class EPSL1BReader(ConfigBasedReader):
    def _interpolate_navigation(self, lon, lat):
        from geotiepoints import metop20kmto1km
        return metop20kmto1km(lon, lat)

class AVHRREPSL1BFileReader(GenericFileReader):

    spacecrafts = {"M01": "Metop-B",
                   "M02": "Metop-A",
                   "M03": "Metop-C",}

    sensors = {"AVHR": "avhrr/3"}

    def __init__(self, file_type, filename, file_keys, **kwargs):
        self.file_keys = file_keys

        self.records, self.form = read_raw(filename)
        self.mdrs = [record[1]
                     for record in self.records
                     if record[0] == "mdr"]
        self.scanlines = len(self.mdrs)
        self.sections = {("mdr", 2): np.concatenate(self.mdrs)}
        for record in self.records:
            if record[0] == "mdr":
                continue
            if (record[0], record[2]) in self.sections:
                raise ValueError("Too many " + str((record[0], record[2])))
            else:
                self.sections[(record[0], record[2])] = record[1]
        self.lons, self.lats = None, None

        self.cache = {}


    def get_swath_data(self, item, dataset_name=None, data_out=None, mask_out=None):
        """
        :param item: usually a channel name
        :return:
        """
        geo_indices = {"latitude": 0, "longitude": 1}
        if item == "radiance":
            return self._get_channel(dataset_name, 2, data_out, mask_out)
        elif item in ["reflectance", "bt"]:
            return self._get_channel(dataset_name, 1, data_out, mask_out)
        elif item in ["longitude", "latitude"]:
            variable_names = self.file_keys[item].variable_name.split(",")
            if data_out is not None:
                data_out[:] = np.hstack((self[variable_names[0]][:, [geo_indices[item]]],
                                         self[variable_names[1]][:, :, geo_indices[item]],
                                         self[variable_names[2]][:, [geo_indices[item]]]))
                return
            else:
                return np.hstack((self[variable_names[0]][:, [geo_indices[item]]],
                                  self[variable_names[1]][:, :, geo_indices[item]],
                                  self[variable_names[2]][:, [geo_indices[item]]]))

        raise NotImplementedError("not done yet")

    def get_shape(self, item):
        if item in ["radiance", "reflectance", "bt"]:
            return int(self["TOTAL_MDR"]), int(self["EARTH_VIEWS_PER_SCANLINE"])
        if item in ["longitude", "latitude"]:
            return int(self["TOTAL_MDR"]), int(max(self["NUM_NAVIGATION_POINTS"]) + 2)

    def get_units(self, item):
        pass

    def get_platform_name(self):
        return self.spacecrafts[self["SPACECRAFT_ID"]]

    def get_sensor_name(self):
        return self.sensors[self["INSTRUMENT_ID"]]


    def get_begin_orbit_number(self):
        return self["ORBIT_START"]

    def get_end_orbit_number(self):
        return self["ORBIT_END"]

    @property
    def start_time(self):
        return datetime.strptime(self["SENSING_START"], "%Y%m%d%H%M%SZ")

    @property
    def end_time(self):
        return datetime.strptime(self["SENSING_END"], "%Y%m%d%H%M%SZ")

    def __getitem__(self, key):
        if key in self.file_keys:
            key = self.file_keys[key].variable_name.format(**self.file_info)

        if key in self.cache:
            return self.cache[key]
        for altkey in self.form.scales.keys():
            try:
                try:
                    self.cache[key] = (self.sections[altkey][key] * self.form.scales[altkey][key])
                    return self.cache[key]
                except TypeError:
                    val = self.sections[altkey][key][0].split("=")[1].strip()
                    try:
                        self.cache[key] = int(val) * self.form.scales[altkey][key]
                        return self.cache[key]
                    except ValueError: # it's probably a string
                        self.cache[key] = val
                        return self.cache[key]
            except ValueError:
                continue
        raise KeyError("No matching value for " + str(key))

    def keys(self):
        """List of reader's keys.
        """
        keys = []
        for val in self.form.scales.values():
            keys += val.dtype.fields.keys()
        return keys

    def _get_channel(self, chan, calib_type, data_out=None, mask_out=None):
        """Get calibrated channel data.
        *calib_type* = 0: Counts
        *calib_type* = 1: Reflectances and brightness temperatures
        *calib_type* = 2: Radiances
        """

        if calib_type == 0:
            raise ValueError('calibrate=0 is not supported! ' +
                             'This reader cannot return counts')
        elif calib_type != 1 and calib_type != 2:
            raise ValueError('calibrate=' + str(calib_type) +
                             'is not supported!')

        if chan not in ["1", "2", "3a", "3A", "3b", "3B", "4", "5"]:
            LOG.info("Can't load channel in eps_l1b: " + str(chan))
            return

        radiance_shape = self["SCENE_RADIANCES"].shape
        if data_out is None:
            data_out = np.ma.empty((radiance_shape[0], radiance_shape[2]), dtype=self["SCENE_RADIANCES"].dtype)
        if data_out is None:
            mask_out = np.ma.empty((radiance_shape[0], radiance_shape[2]), dtype=bool)

        if chan == "1":
            if calib_type == 1:
                data_out[:] = radiance_to_refl(self["SCENE_RADIANCES"][:, 0, :],
                                      self["CH1_SOLAR_FILTERED_IRRADIANCE"])
            else:
                data_out[:] = self["SCENE_RADIANCES"][:, 0, :]
            mask_out[:] = False
        if chan == "2":
            if calib_type == 1:
                data_out[:] = radiance_to_refl(self["SCENE_RADIANCES"][:, 1, :],
                                      self["CH2_SOLAR_FILTERED_IRRADIANCE"])
            else:
                data_out[:] = self["SCENE_RADIANCES"][:, 1, :]
            mask_out[:] = False

        if chan.lower() == "3a":
            frames = (self["FRAME_INDICATOR"] & 2 ** 16) != 0
            if calib_type == 1:
                data_out[frames, :] = radiance_to_refl(self["SCENE_RADIANCES"][frames, 2, :],
                                              self["CH3A_SOLAR_FILTERED_IRRADIANCE"])
            else:
                data_out[frames, :] = np.ma.array(self["SCENE_RADIANCES"][frames, 2, :])
            mask_out[~frames, :] = True
            mask_out[frames, :] = False

        if chan.lower() == "3b":
            frames = (self["FRAME_INDICATOR"] & 2 ** 16) == 0
            if calib_type == 1:
                data_out[:] = radiance_to_bt(self["SCENE_RADIANCES"][:, 2, :],
                                    self["CH3B_CENTRAL_WAVENUMBER"],
                                    self["CH3B_CONSTANT1"],
                                    self["CH3B_CONSTANT2_SLOPE"])

            else:
                data_out[:] = self["SCENE_RADIANCES"][:, 2, :]
            mask_out[~frames, :] = True
            mask_out[frames, :] = False
        if chan == "4":
            if calib_type == 1:
                data_out[:] = radiance_to_bt(self["SCENE_RADIANCES"][:, 3, :],
                                    self["CH4_CENTRAL_WAVENUMBER"],
                                    self["CH4_CONSTANT1"],
                                    self["CH4_CONSTANT2_SLOPE"])
            else:
                data_out[:] = self["SCENE_RADIANCES"][:, 3, :]
            mask_out[:] = False

        if chan == "5":
            if calib_type == 1:
                data_out[:] = radiance_to_bt(self["SCENE_RADIANCES"][:, 4, :],
                                    self["CH5_CENTRAL_WAVENUMBER"],
                                    self["CH5_CONSTANT1"],
                                    self["CH5_CONSTANT2_SLOPE"])
            else:
                data_out[:] = self["SCENE_RADIANCES"][:, 4, :]
            mask_out[:] = False

        return data_out


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
