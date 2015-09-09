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

import glob
import os
from ConfigParser import ConfigParser

import numpy as np
from mpop import CONFIG_PATH
from mpop.satin.xmlformat import XMLFormat
import logging
from mpop.projectable import Projectable

LOG = logging.getLogger(__name__)

try:
    from pyresample import geometry
except ImportError:
    pass

try:
    import numexpr as ne
except ImportError:
    pass

C1 = 1.191062e-05  # mW/(m2*sr*cm-4)
C2 = 1.4387863  # K/cm-1



def to_bt(arr, wc_, a__, b__):
    """Convert to BT.
    """
    try:
        return ne.evaluate("a__ + b__ * (C2 * wc_ / "
                           "(log(1 + (C1 * (wc_ ** 3) / arr))))")
    except NameError:
        return a__ + b__ * (C2 * wc_ / np.log(1 + (C1 * (wc_ ** 3) / arr)))


def to_refl(arr, solar_flux):
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
            try:
                rec_class = record_class[grh["record_class"]]
                sub_class = grh["RECORD_SUBCLASS"][0]
                record = np.fromfile(fdes,
                                     form.dtype((rec_class,
                                                 sub_class)),
                                     1)
                records.append((rec_class, record, sub_class))
            except KeyError:
                fdes.seek(grh["RECORD_SIZE"] - 20, 1)

    return records, form


def get_filename(satscene, level):
    """Get the filename.
    """
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-" + level,
                                    raw=True):
        options[option] = value
    values = {"INSTRUMENT": satscene.instrument_name[:4].upper(),
              "FNAME": satscene.satname[0].upper() + satscene.number
              }
    filename = os.path.join(
        options["dir"],
        (satscene.time_slot.strftime(options["filename"]) % values))
    LOG.debug("Looking for file %s" % satscene.time_slot.strftime(filename))
    file_list = glob.glob(satscene.time_slot.strftime(filename))

    if len(file_list) > 1:
        raise IOError("More than one l1b file matching!")
    elif len(file_list) == 0:
        raise IOError("No l1b file matching!")
    return file_list[0]


from mpop.readers import ConfigBasedReader
from trollsift.parser import parse

class EPSL1BReader(ConfigBasedReader):
    def __init__(self, **kwargs):
        super(EPSL1BReader, self).__init__(**kwargs)


    # def load(self, datasets_to_load, calibration_level=None, **kwargs):
    #     if kwargs:
    #         LOG.warning("Unsupported options for eps avhrr reader: %s", str(kwargs))
    #
    #     print "in load", datasets_to_load, calibration_level
    #     res = {}
    #     for dataset in datasets_to_load:
    #         data = self.file_reader.get_swath_data("reflectance")
    #         res[dataset] = Projectable(data=data,
    #                                    start_time=file_reader.start_time,
    #                                    end_time=file_reader.end_time,
    #                                    **kwargs)
    #     return res


    def _load_navigation(self, nav_name, dep_file_type, extra_mask=None):
        """Load the `nav_name` navigation.

        For VIIRS, if we haven't loaded the geolocation file read the `dep_file_type` header
        to figure out where it is.
        """
        nav_info = self.navigations[nav_name]
        lon_key = nav_info["longitude_key"]
        lat_key = nav_info["latitude_key"]
        file_type = nav_info["file_type"]

        file_reader = self.file_readers[file_type]

        #gross_lon_data = file_reader.get_swath_data(lon_key, extra_mask=extra_mask)
        #gross_lat_data = file_reader.get_swath_data(lat_key, extra_mask=extra_mask)
        gross_lon_data = file_reader.get_swath_data(lon_key)
        gross_lat_data = file_reader.get_swath_data(lat_key)
        print gross_lat_data
        from geotiepoints import metop20kmto1km
        lon_data, lat_data = metop20kmto1km(gross_lon_data, gross_lat_data)


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

class AVHRREPSL1BFileReader(object):

    spacecrafts = {"M01": "Metop-B",
                   "M02": "Metop-A",
                   "M03": "Metop-C",}

    sensors = {"AVHR": "avhrr/3"}

    #def __init__(self, file_type, filename, file_key, **kwargs):
    def __init__(self, file_type, filename, file_keys, **kwargs):
        print "we get", kwargs
        info = parse(kwargs["file_patterns"][0], filename)
        self.start_time = info["start_time"]
        self.end_time = info["end_time"]
        self._reader = EpsAvhrrL1bReader(filename)
        self.file_keys = file_keys
        print file_keys
        print self._reader["TOTAL_MDR"], self._reader["EARTH_VIEWS_PER_SCANLINE"]
        pass

    def get_swath_data(self, item, dataset_name=None, data_out=None, mask_out=None):
        """
        :param item: usually a channel name
        :return:
        """
        print "in get_swath_data", item
        ch_indices = {"1": 0, "2": 1, "3a": 2, "3b": 2, "4": 3, "5": 4}
        geo_indices = {"latitude": 0, "longitude": 1}
        if item == "radiance":
            if data_out is not None:
                data_out[:] = self._reader[self.file_keys[item].variable_name][:, ch_indices[dataset_name], :]
                return data_out
            else:
                return self._reader[self.file_keys[item].variable_name][:, ch_indices[dataset_name], :]
        if item in ["longitude", "latitude"]:
            variable_names = self.file_keys[item].variable_name.split(",")
            if data_out is not None:
                data_out[:] = np.hstack((self._reader[variable_names[0]][:, [geo_indices[item]]],
                                         self._reader[variable_names[1]][:, :, geo_indices[item]],
                                         self._reader[variable_names[2]][:, [geo_indices[item]]]))
                return
            else:
                return np.hstack((self._reader[variable_names[0]][:, [geo_indices[item]]],
                                  self._reader[variable_names[1]][:, :, geo_indices[item]],
                                  self._reader[variable_names[2]][:, [geo_indices[item]]]))






        raise NotImplementedError("not done yet")

    def __getitem__(self, item):
        print "in getitem", item
        raise NotImplementedError("not done yet")

    def get_shape(self, item):
        if item in ["radiance", "reflectance", "bt"]:
            return int(self._reader["TOTAL_MDR"]), int(self._reader["EARTH_VIEWS_PER_SCANLINE"])
        if item in ["longitude", "latitude"]:
            return int(self._reader["TOTAL_MDR"]), int(max(self._reader["NUM_NAVIGATION_POINTS"]) + 2)
            #return int(self._reader["TOTAL_MDR"]), int(self._reader["EARTH_VIEWS_PER_SCANLINE"])

    def get_units(self, item):
        pass

    def get_platform_name(self):
        return self.spacecrafts[self._reader["SPACECRAFT_ID"]]

    def get_sensor_name(self):
        return self.sensors[self._reader["INSTRUMENT_ID"]]


    def get_begin_orbit_number(self):
        return self._reader["ORBIT_START"]

    def get_end_orbit_number(self):
        return self._reader["ORBIT_END"]

class EpsAvhrrL1bReader(object):

    """Eps level 1b reader for AVHRR data.
    """

    def __init__(self, filename):
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

    def __getitem__(self, key):
        for altkey in self.form.scales.keys():
            try:
                try:
                    return (self.sections[altkey][key]
                            * self.form.scales[altkey][key])
                except TypeError:
                    val = self.sections[altkey][key][0].split("=")[1].strip()
                    try:
                        return int(val) * self.form.scales[altkey][key]
                    except ValueError: # it's probably a string
                        return val
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

    def get_full_lonlats(self):
        """Get the interpolated lons/lats.
        """
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
            self.lons, self.lats = metop20kmto1km(lons, lats)
        else:
            raise NotImplementedError("Lon/lat expansion not implemented for " +
                                      "sample rate = " + str(nav_sample_rate) +
                                      " and earth views = " +
                                      str(earth_views_per_scanline))
        return self.lons, self.lats

    def get_lonlat(self, row, col):
        """Get lons/lats for given indices. WARNING: if the lon/lats were not
        expanded, this will refer to the tiepoint data.
        """
        if self.lons is None or self.lats is None:
            self.lats = np.hstack((self["EARTH_LOCATION_FIRST"][:, [0]],
                                   self["EARTH_LOCATIONS"][:, :, 0],
                                   self["EARTH_LOCATION_LAST"][:, [0]]))

            self.lons = np.hstack((self["EARTH_LOCATION_FIRST"][:, [1]],
                                   self["EARTH_LOCATIONS"][:, :, 1],
                                   self["EARTH_LOCATION_LAST"][:, [1]]))
        return self.lons[row, col], self.lats[row, col]

    def get_channels(self, channels, calib_type):
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

        if ("3a" in channels or
                "3A" in channels or
                "3b" in channels or
                "3B" in channels):
            three_a = ((self["FRAME_INDICATOR"] & 2 ** 16) == 2 ** 16)
            three_b = ((self["FRAME_INDICATOR"] & 2 ** 16) == 0)

        chans = {}
        for chan in channels:
            if chan not in ["1", "2", "3a", "3A", "3b", "3B", "4", "5"]:
                LOG.info("Can't load channel in eps_l1b: " + str(chan))
                continue

            if chan == "1":
                if calib_type == 1:
                    chans[chan] = np.ma.array(
                        to_refl(self["SCENE_RADIANCES"][:, 0, :],
                                self["CH1_SOLAR_FILTERED_IRRADIANCE"]))
                else:
                    chans[chan] = np.ma.array(
                        self["SCENE_RADIANCES"][:, 0, :])
            if chan == "2":
                if calib_type == 1:
                    chans[chan] = np.ma.array(
                        to_refl(self["SCENE_RADIANCES"][:, 1, :],
                                self["CH2_SOLAR_FILTERED_IRRADIANCE"]))
                else:
                    chans[chan] = np.ma.array(
                        self["SCENE_RADIANCES"][:, 1, :])

            if chan.lower() == "3a":
                if calib_type == 1:
                    chans[chan] = np.ma.array(
                        to_refl(self["SCENE_RADIANCES"][:, 2, :],
                                self["CH2_SOLAR_FILTERED_IRRADIANCE"]))
                else:
                    chans[chan] = np.ma.array(self["SCENE_RADIANCES"][:, 2, :])

                chans[chan][three_b, :] = np.nan
                chans[chan] = np.ma.masked_invalid(chans[chan])
            if chan.lower() == "3b":
                if calib_type == 1:
                    chans[chan] = np.ma.array(
                        to_bt(self["SCENE_RADIANCES"][:, 2, :],
                              self["CH3B_CENTRAL_WAVENUMBER"],
                              self["CH3B_CONSTANT1"],
                              self["CH3B_CONSTANT2_SLOPE"]))
                else:
                    chans[chan] = self["SCENE_RADIANCES"][:, 2, :]
                chans[chan][three_a, :] = np.nan
                chans[chan] = np.ma.masked_invalid(chans[chan])
            if chan == "4":
                if calib_type == 1:
                    chans[chan] = np.ma.array(
                        to_bt(self["SCENE_RADIANCES"][:, 3, :],
                              self["CH4_CENTRAL_WAVENUMBER"],
                              self["CH4_CONSTANT1"],
                              self["CH4_CONSTANT2_SLOPE"]))
                else:
                    chans[chan] = np.ma.array(
                        self["SCENE_RADIANCES"][:, 3, :])

            if chan == "5":
                if calib_type == 1:
                    chans[chan] = np.ma.array(
                        to_bt(self["SCENE_RADIANCES"][:, 4, :],
                              self["CH5_CENTRAL_WAVENUMBER"],
                              self["CH5_CONSTANT1"],
                              self["CH5_CONSTANT2_SLOPE"]))
                else:
                    chans[chan] = np.ma.array(self["SCENE_RADIANCES"][:, 4, :])

        return chans


def get_lonlat(scene, row, col):
    """Get the longitutes and latitudes for the give *rows* and *cols*.
    """
    try:
        filename = get_filename(scene, "granules")
    except IOError:
        #from mpop.satin.eps1a import get_lonlat_avhrr
        # return get_lonlat_avhrr(scene, row, col)
        from pyorbital.orbital import Orbital
        import pyproj
        from datetime import timedelta
        start_time = scene.time_slot
        end_time = scene.time_slot + timedelta(minutes=3)

        orbital = Orbital("METOP-A")
        track_start = orbital.get_lonlatalt(start_time)
        track_end = orbital.get_lonlatalt(end_time)

        geod = pyproj.Geod(ellps='WGS84')
        az_fwd, az_back, dist = geod.inv(track_start[0], track_start[1],
                                         track_end[0], track_end[1])

        del dist

        M02_WIDTH = 2821885.8962408099

        pos = ((col - 1024) * M02_WIDTH) / 2048.0
        if row > 520:
            lonlatdist = geod.fwd(track_end[0], track_end[1],
                                  az_back - 86.253533216206648,  -pos)
        else:
            lonlatdist = geod.fwd(track_start[0], track_start[1],
                                  az_fwd - 86.253533216206648,  pos)

        return lonlatdist[0], lonlatdist[1]

    try:
        if scene.lons is None or scene.lats is None:
            records, form = read_raw(filename)
            mdrs = [record[1]
                    for record in records
                    if record[0] == "mdr"]
            sphrs = [record for record in records
                     if record[0] == "sphr"]
            sphr = sphrs[0][1]
            scene.lons, scene.lats = _get_lonlats(mdrs, sphr, form)
        return scene.lons[row, col], scene.lats[row, col]
    except AttributeError:
        records, form = read_raw(filename)
        mdrs = [record[1]
                for record in records
                if record[0] == "mdr"]
        sphrs = [record for record in records
                 if record[0] == "sphr"]
        sphr = sphrs[0][1]
        scene.lons, scene.lats = _get_lonlats(mdrs, sphr, form)
        return scene.lons[row, col], scene.lats[row, col]


def _get_lonlats(mdrs, sphr, form):
    """Get sparse arrays of lon/lats.
    """

    scanlines = len(mdrs)
    mdrs = np.concatenate(mdrs)

    lats = np.hstack((mdrs["EARTH_LOCATION_FIRST"][:, [0]]
                      * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"][:, 0],
                      mdrs["EARTH_LOCATIONS"][:, :, 0]
                      * form.scales[("mdr", 2)]["EARTH_LOCATIONS"][:, :, 0],
                      mdrs["EARTH_LOCATION_LAST"][:, [0]]
                      * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"][:, 0]))

    lons = np.hstack((mdrs["EARTH_LOCATION_FIRST"][:, [1]]
                      * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"][:, 1],
                      mdrs["EARTH_LOCATIONS"][:, :, 1]
                      * form.scales[("mdr", 2)]["EARTH_LOCATIONS"][:, :, 1],
                      mdrs["EARTH_LOCATION_LAST"][:, [1]]
                      * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"][:, 1]))

    nav_sample_rate = int(sphr["NAV_SAMPLE_RATE"][0].split("=")[1])
    earth_views_per_scanline = \
        int(sphr["EARTH_VIEWS_PER_SCANLINE"][0].split("=")[1])

    geo_samples = np.round(earth_views_per_scanline / nav_sample_rate) + 3
    samples = np.zeros(geo_samples, dtype=np.intp)
    samples[1:-1] = np.arange(geo_samples - 2) * 20 + 5 - 1
    samples[-1] = earth_views_per_scanline - 1

    mask = np.ones((scanlines, earth_views_per_scanline))
    mask[:, samples] = 0
    geolats = np.ma.empty((scanlines, earth_views_per_scanline),
                          dtype=lats.dtype)
    geolats.mask = mask
    geolats[:, samples] = lats
    geolons = np.ma.empty((scanlines, earth_views_per_scanline),
                          dtype=lons.dtype)
    geolons.mask = mask
    geolons[:, samples] = lons

    return geolons, geolats


def get_corners(filename):
    """Get the corner lon/lats of the file.
    """
    records, form = read_raw(filename)

    mdrs = [record[1]
            for record in records
            if record[0] == "mdr"]

    ul_ = (mdrs[0]["EARTH_LOCATION_FIRST"]
           * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"])
    ur_ = (mdrs[0]["EARTH_LOCATION_LAST"]
           * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"])
    ll_ = (mdrs[-1]["EARTH_LOCATION_FIRST"]
           * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"])
    lr_ = (mdrs[-1]["EARTH_LOCATION_LAST"]
           * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"])

    return ul_, ur_, ll_, lr_


def load(scene, *args, **kwargs):
    """Loads the *channels* into the satellite *scene*.
    A possible *calibrate* keyword argument is passed to the AAPP reader
    Should be 0 for off, 1 for default, and 2 for radiances only.
    However, as the AAPP-lvl1b file contains radiances this reader cannot
    return counts, so calibrate=0 is not allowed/supported. The radiance to
    counts conversion is not possible.
    """

    del args
    calibrate = kwargs.get("calibrate", True)
    if calibrate == 0:
        raise ValueError('calibrate=0 is not supported! ' +
                         'This reader cannot return counts')

    if kwargs.get("filename") is not None:
        filename = kwargs["filename"]
    else:
        filename = (kwargs.get("filename", None) or
                    get_filename(scene, "level2"))

    if isinstance(filename, (list, tuple, set)):
        filenames = filename
    else:
        filenames = [filename]
    LOG.debug("Using file(s) %s", str(filename))
    readers = [EpsAvhrrL1bReader(filename) for filename in filenames]

    arrs = {}
    llons = []
    llats = []
    loaded_channels = set()

    for reader in readers:

        for chname, arr in reader.get_channels(scene.channels_to_load,
                                               calibrate).items():
            arrs.setdefault(chname, []).append(arr)
            loaded_channels.add(chname)

        if scene.orbit is None:
            scene.orbit = int(reader["ORBIT_START"][0])
            scene.info["orbit_number"] = scene.orbit
        lons, lats = reader.get_full_lonlats()
        llons.append(lons)
        llats.append(lats)

    for chname in loaded_channels:
        scene[chname] = np.vstack(arrs[chname])
        if chname in ["1", "2", "3A"]:
            scene[chname].info["units"] = "%"
        elif chname in ["4", "5", "3B"]:
            scene[chname].info["units"] = "K"

    lons = np.vstack(llons)
    lats = np.vstack(llats)

    try:
        scene.area = geometry.SwathDefinition(lons, lats)
    except NameError:
        scene.lons, scene.lats = lons, lats


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

if __name__ == '__main__':
    import sys
    res = read_raw(sys.argv[1])
