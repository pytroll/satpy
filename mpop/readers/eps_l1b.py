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


class AVHRREPSL1BFileReader(GenericFileReader):

    spacecrafts = {"M01": "Metop-B",
                   "M02": "Metop-A",
                   "M03": "Metop-C",}

    sensors = {"AVHR": "avhrr/3"}

    def create_file_handle(self, filename, **kwargs):
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
        # no single file handle for these files, so just return None and override any parent class methods
        return filename, None

    def get_swath_data(self, item, dataset_id=None, data_out=None, mask_out=None):
        """
        :param item: usually a channel name
        :return:
        """
        var_info = self.file_keys[item]
        if "geo_index" in var_info.kwargs:
            variable_names = var_info.variable_name.split(",")
            geo_index = int(var_info.kwargs["geo_index"])
            if data_out is not None:
                data_out[:] = np.hstack((self[variable_names[0]][:, [geo_index]],
                                         self[variable_names[1]][:, :, geo_index],
                                         self[variable_names[2]][:, [geo_index]]))
                return
            else:
                return np.hstack((self[variable_names[0]][:, [geo_index]],
                                  self[variable_names[1]][:, :, geo_index],
                                  self[variable_names[2]][:, [geo_index]]))
        elif item.startswith("radiance") or item.startswith("reflectance") or item.startswith("bt"):
            return self._get_channel(item, dataset_id, data_out, mask_out)
        else:
            raise ValueError("Unknown file key: %s" % (item,))

    def _get_channel(self, item, chan, data_out=None, mask_out=None):
        """Get calibrated channel data.
        """
        var_info = self.file_keys[item]
        var_name = var_info.variable_name

        data_arr = self[var_name]
        radiance_shape = data_arr.shape
        if data_out is None:
            data_out = np.ma.empty((radiance_shape[0], radiance_shape[2]), dtype=data_arr.dtype)
        if data_out is None:
            mask_out = np.ma.empty((radiance_shape[0], radiance_shape[2]), dtype=bool)

        if "frame_indicator" in var_info.kwargs:
            frame_indicator = int(var_info.kwargs["frame_indicator"])
            frames = (self[self.file_keys["frame_indicator"].variable_name] & 2 ** 16) == frame_indicator
            data_out[frames, :] = data_arr[frames, int(var_info.kwargs.get("band_index", 0)), :]
            mask_out[~frames, :] = True
            mask_out[frames, :] = False
        else:
            data_out[:] = data_arr[:, int(var_info.kwargs.get("band_index", 0)), :]
            mask_out[:] = False

        # FIXME: Temporary, calibrate data here
        calib_type = var_info.kwargs.get("calib_type", None)
        if calib_type == "bt":
            wl = self[self.file_keys[var_info.kwargs["cw_key"]].variable_name]
            chan_const = self[self.file_keys[var_info.kwargs["channel_constant_key"]].variable_name]
            slope_const = self[self.file_keys[var_info.kwargs["slope_constant_key"]].variable_name]
            data_out[:] = radiance_to_bt(data_out,
                                         wl,
                                         chan_const,
                                         slope_const)
        elif calib_type == "reflectance":
            sfi = self[self.file_keys[var_info.kwargs["solar_irradiance"]].variable_name]
            data_out[:] = radiance_to_refl(data_out, sfi)

        # Simple unit conversion
        file_units = self.get_file_units(item)
        output_units = getattr(var_info, "units", file_units)
        if file_units == "1" and output_units == "%":
            data_out[:] *= 100.0

        return data_out

    def get_shape(self, item):
        if item in ["longitude", "latitude"]:
            return int(self["TOTAL_MDR"]), int(max(self["NUM_NAVIGATION_POINTS"]) + 2)
        else:
            return int(self["TOTAL_MDR"]), int(self[self.file_keys["views_per_scanline"].variable_name])

    @property
    def platform_name(self):
        return self.spacecrafts[self["SPACECRAFT_ID"]]

    @property
    def sensor_name(self):
        return self.sensors[self["INSTRUMENT_ID"]]

    def get_file_units(self, item):
        return getattr(self.file_keys[item], "file_units", None)

    @property
    def geofilename(self):
        return self.filename

    @property
    def ring_lonlats(self):
        raise NotImplementedError

    @property
    def begin_orbit_number(self):
        return self["ORBIT_START"]

    @property
    def end_orbit_number(self):
        return self["ORBIT_END"]

    def _get_start_time(self):
        return datetime.strptime(self[self.file_keys["start_time"].variable_name], "%Y%m%d%H%M%SZ")

    def _get_end_time(self):
        return datetime.strptime(self[self.file_keys["end_time"].variable_name], "%Y%m%d%H%M%SZ")

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
                    except ValueError:  # it's probably a string
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


class EPSL1BReader(ConfigBasedReader):
    def __init__(self, default_file_reader=AVHRREPSL1BFileReader, **kwargs):
        super(EPSL1BReader, self).__init__(default_file_reader=default_file_reader, **kwargs)

    def _interpolate_navigation(self, lon, lat):
        from geotiepoints import metop20kmto1km
        return metop20kmto1km(lon, lat)


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
