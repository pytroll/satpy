#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Pytroll Community

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

"""Reader for eps amsu-a level 1b data. Uses xml files as a format description.
"""

import logging
import os

import numpy as np
import xarray as xr

import dask.array as da
from satpy.config import CONFIG_PATH
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.xmlformat import XMLFormat
from satpy import CHUNK_SIZE

LOG = logging.getLogger(__name__)


record_class = ["Reserved", "mphr", "sphr",
                "ipr", "geadr", "giadr",
                "veadr", "viadr", "mdr"]


def read_raw(filename):
    """Read *filename* without scaling it afterwards.
    """

    form = XMLFormat(os.path.join(CONFIG_PATH, "eps_amsual1b_6.4.xml"))

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


class EPSAMSUAFile(BaseFileHandler):
    """Eps level 1b reader for AVHRR data.
    """
    spacecrafts = {"M01": "Metop-B",
                   "M02": "Metop-A",
                   "M03": "Metop-C", }

    sensors = {"AMSA": "amsu-a"}

    def __init__(self, filename, filename_info, filetype_info):
        super(EPSAMSUAFile, self).__init__(
            filename, filename_info, filetype_info)

        self.lons, self.lats = None, None
        self.sun_azi, self.sun_zen, self.sat_azi, self.sat_zen = None, None, None, None
        self.area = None
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
        self.pixels = 30

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

    def get_bounding_box(self):
        if self.mdrs is None:
            self._read_all(self.filename)
        lats = np.hstack([self["EARTH_LOCATION"][0, 0, 0],
                          self["EARTH_LOCATION"][-1, 0, 0],
                          self["EARTH_LOCATION"][-1, -1, 0],
                          self["EARTH_LOCATION"][0, -1, 0]])
        lons = np.hstack([self["EARTH_LOCATION"][0, 0, 1],
                          self["EARTH_LOCATION"][-1, 0, 1],
                          self["EARTH_LOCATION"][-1, -1, 1],
                          self["EARTH_LOCATION"][0, -1, 1]])
        return lons.ravel(), lats.ravel()

    def get_dataset(self, key, info):
        """Get calibrated channel data."""
        if self.mdrs is None:
            self._read_all(self.filename)

        if key.name in ['longitude', 'latitude']:
            lats = self['EARTH_LOCATION'][:, :, 0]
            lons = self['EARTH_LOCATION'][:, :, 1]

            if key.name == 'longitude':
                dataset = create_xarray(lons)
            else:
                dataset = create_xarray(lats)
        elif key.name == 'terrain_elevation':
            dataset = create_xarray(self['TERRAIN_ELEVATION'])
        elif key.name in ['solar_zenith_angle', 'solar_azimuth_angle',
                          'satellite_zenith_angle', 'satellite_azimuth_angle']:
            if key.name == 'solar_zenith_angle':
                dataset = create_xarray(self['ANGULAR_RELATION'][:, :, 0])
            elif key.name == 'solar_azimuth_angle':
                dataset = create_xarray(self['ANGULAR_RELATION'][:, :, 2])
            if key.name == 'satellite_zenith_angle':
                dataset = create_xarray(self['ANGULAR_RELATION'][:, :, 1])
            elif key.name == 'satellite_azimuth_angle':
                dataset = create_xarray(self['ANGULAR_RELATION'][:, :, 3])
        else:
            if key.calibration == 'counts':
                raise ValueError('calibration=counts is not supported! ' +
                                 'This reader cannot return counts')
            elif key.calibration not in ['reflectance', 'brightness_temperature', 'radiance']:
                raise ValueError('calibration type ' + str(key.calibration) +
                                 ' is not supported!')

            if int(key.name) not in range(1, 16):
                LOG.info("Can't load channel in eps_l1b: " + str(key.name))
                return

            array = self["SCENE_RADIANCE"][:, :, int(key.name)]

            dataset = create_xarray(array)

        dataset.attrs['platform_name'] = self.platform_name
        dataset.attrs['sensor'] = self.sensor_name
        dataset.attrs.update(info)
        dataset.attrs.update(key.to_dict())
        return dataset

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
