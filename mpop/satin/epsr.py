#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012 SMHI

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

"""
"""

from mpop.satin.xmlformat import XMLFormat
from datetime import datetime
import numpy as np

C1 = 1.191062e-05 # mW/(m2*sr*cm-4)
C2 = 1.4387863 # K/cm-1

def to_bt(arr, wc_, a__, b__):
    """Convert to BT.
    """
    val = np.log(1 + (C1 * (wc_ ** 3) / arr))
    t_star = C2 * wc_ / val
    return a__ + b__ * t_star

def to_refl(arr, solar_flux):
    """Convert to reflectances.
    """
    return arr * np.pi * 100.0 / solar_flux

CHANNELS = {"1": 0,
            "2": 1,
            "3A": 2,
            "3B": 3,
            "4": 4,
            "5": 5,}

def read(filename, channel_names):
    pass

def read_raw(filename):
    """Read *filename* without scaling it afterwards.
    """

    from mpop import CONFIG_PATH
    form = XMLFormat(os.path.join(CONFIG_PATH, "eps_avhrrl1b_6.5.xml"))

    grh_dtype = np.dtype([("RECORD_CLASS", "|i1"),
                          ("INSTRUMENT_GROUP", "|i1"),
                          ("RECORD_SUBCLASS", "|i1"),
                          ("RECORD_SUBCLASS_VERSION", "|i1"),
                          ("RECORD_SIZE", ">u4"),
                          ("RECORD_START_TIME", "S6"),
                          ("RECORD_STOP_TIME", "S6")])

    RECORD_CLASS = ["Reserved", "mphr", "sphr",
                    "ipr", "geadr", "giadr",
                    "veadr", "viadr", "mdr"]


    records = []

    with open(filename, "rb") as fdes:
        while True:
            grh = np.fromfile(fdes, grh_dtype, 1)
            if not grh:
                break
            try:
                rec_class = RECORD_CLASS[grh["RECORD_CLASS"]]
                sub_class = grh["RECORD_SUBCLASS"][0]
                record = np.fromfile(fdes,
                                     form.dtype((rec_class,
                                                 sub_class)),
                                     1)
                records.append((rec_class, record, sub_class))
            except KeyError:
                fdes.seek(grh["RECORD_SIZE"] - 20, 1)

    return records, form

from ConfigParser import ConfigParser
import os
from mpop import CONFIG_PATH
from mpop.satin.logger import LOG
import glob

def get_filename(satscene):
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-granules",
                                    raw = True):
        options[option] = value
    values = {"INSTRUMENT": satscene.instrument_name[:4].upper(),
              "FNAME": satscene.satname[0].upper() + satscene.number
              }
    filename = os.path.join(
        options["dir"],
        (satscene.time_slot.strftime(options["filename"])%values))
    LOG.debug("Looking for file %s"%satscene.time_slot.strftime(filename))
    file_list = glob.glob(satscene.time_slot.strftime(filename))

    if len(file_list) > 1:
        raise IOError("More than one l1b file matching!")
    elif len(file_list) == 0:
        raise IOError("No l1b file matching!")
    return file_list[0]

def get_lonlat(scene, row, col):
    filename = get_filename(scene)
    try:
        if scene.lons is None or scene.lats is None:
            scene.lons, scene.lats = _get_lonlats(filename)
    except AttributeError:
        scene.lons, scene.lats = _get_lonlats(filename)
    return scene.lons[row, col], scene.lats[row, col]

def _get_lonlats(filename):
    records, form = read_raw(filename)
    
    mdrs = [record[1]
            for record in records
            if record[0] == "mdr"]

    scanlines = len(mdrs)

    mdrs = np.concatenate(mdrs)
    lats = np.hstack((mdrs["EARTH_LOCATION_FIRST"][:, [0]] * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"][:, 0],
                      mdrs["EARTH_LOCATIONS"][:, :, 0] * form.scales[("mdr", 2)]["EARTH_LOCATIONS"][:, :, 0],
                      mdrs["EARTH_LOCATION_LAST"][:, [0]] * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"][:, 0]))
    
    lons = np.hstack((mdrs["EARTH_LOCATION_FIRST"][:, [1]] * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"][:, 1],
                      mdrs["EARTH_LOCATIONS"][:, :, 1] * form.scales[("mdr", 2)]["EARTH_LOCATIONS"][:, :, 0],
                      mdrs["EARTH_LOCATION_LAST"][:, [1]] * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"][:, 1]))

    sphr = records[1][1]
    nav_sample_rate = int(sphr["NAV_SAMPLE_RATE"][0].split("=")[1])
    earth_views_per_scanline = int(sphr["EARTH_VIEWS_PER_SCANLINE"][0].split("=")[1])

    geo_samples = np.round(earth_views_per_scanline / nav_sample_rate) + 3
    samples = np.zeros(geo_samples, dtype=np.intp)
    samples[1:-1] = np.arange(geo_samples - 2) * 20 + 5 - 1
    samples[-1] = earth_views_per_scanline - 1

    mask = np.ones((scanlines, earth_views_per_scanline))
    mask[:, samples] = 0
    geolats = np.ma.empty((scanlines, earth_views_per_scanline), dtype=lats.dtype)
    geolats.mask = mask
    geolats[:, samples] = lats
    geolons = np.ma.empty((scanlines, earth_views_per_scanline), dtype=lons.dtype)
    geolons.mask = mask
    geolons[:, samples] = lons 

    return geolons, geolats

def get_corners(filename):

    records, form = read_raw(filename)

    mdrs = [record[1]
            for record in records
            if record[0] == "mdr"]

    #mdrs = np.concatenate(mdrs)

    ul = mdrs[0]["EARTH_LOCATION_FIRST"] * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"]
    ur = mdrs[0]["EARTH_LOCATION_LAST"] * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"]
    ll = mdrs[-1]["EARTH_LOCATION_FIRST"] * form.scales[("mdr", 2)]["EARTH_LOCATION_FIRST"]
    lr = mdrs[-1]["EARTH_LOCATION_LAST"] * form.scales[("mdr", 2)]["EARTH_LOCATION_LAST"]

    return ul, ur, ll, lr

def norm255(a):
    arr = a * 1.0
    arr = (arr - arr.min()) * 255.0 / (arr.max() - arr.min())
    return arr.astype(np.uint8)

def show(a):
    import Image
    Image.fromarray(norm255(a), "L").show()

if __name__ == '__main__':
    tic = datetime.now()
    gtic = tic

    form = XMLFormat("eps_avhrrl1b_6.5.xml")


    toc = datetime.now()

    print "reading format took", toc - tic


    #filename = "AVHR_xxx_1B_M02_20120321100103Z_20120321100403Z_N_O_20120321105619Z"
    # night
    #filename = "AVHR_xxx_1B_M02_20120321100403Z_20120321100703Z_N_O_20120321105847Z"
    # day
    filename = "AVHR_xxx_1B_M02_20120322101903Z_20120322102203Z_N_O_20120322110900Z"
    filename = "/home/a001673/usr/src/kai-1.9.new/truc.eps1b"
    tic = datetime.now()

    print get_corners(filename)

    toc = datetime.now()

    print "corners took", toc - tic
    
    tic = datetime.now()

    records, form = read_raw(filename)
    
    toc = datetime.now()

    print "reading took", toc - tic

    tic = datetime.now()

    srecords = []
    for record in records:
        srecords.append((record[0],
                         form.apply_scales(record[1]),
                         record[2]))

    toc = datetime.now()

    print "scaling took", toc - tic

    # take mdr

    tic = datetime.now()

    mdrs = [record[1]
            for record in srecords
            if record[0] == "mdr"]

    scanlines = len(mdrs)

    mdrs = np.concatenate(mdrs)

    giadr = [record[1]
              for record in srecords
              if record[0] == "giadr" and record[2] == 1][0]

    toc = datetime.now()

    print "selecting mdrs", toc - tic


    # filter 3a and calibrate

    tic = datetime.now()


    three_a = ((mdrs["FRAME_INDICATOR"] & 2**16) == 2**16)
    three_b = ((mdrs["FRAME_INDICATOR"] & 2**16) == 0)


    channels = np.empty((mdrs["SCENE_RADIANCES"].shape[0],
                         6,
                         mdrs["SCENE_RADIANCES"].shape[2]))
    channels[:, 0, :] = to_refl(mdrs["SCENE_RADIANCES"][:, 0, :],
                                giadr["CH1_SOLAR_FILTERED_IRRADIANCE"])
    channels[:, 1, :] = to_refl(mdrs["SCENE_RADIANCES"][:, 1, :],
                                giadr["CH2_SOLAR_FILTERED_IRRADIANCE"])
    channels[three_a, 2, :] = to_refl(mdrs["SCENE_RADIANCES"][three_a, 2, :],
                                      giadr["CH3A_SOLAR_FILTERED_IRRADIANCE"])
    channels[three_b, 2, :] = np.nan
    channels[three_b, 3, :] = to_bt(mdrs["SCENE_RADIANCES"][three_b, 2, :],
                                    giadr["CH3B_CENTRAL_WAVENUMBER"],
                                    giadr["CH3B_CONSTANT1"],
                                    giadr["CH3B_CONSTANT2_SLOPE"])
    channels[three_a, 3, :] = np.nan
    channels[:, 4, :] = to_bt(mdrs["SCENE_RADIANCES"][:, 3, :],
                              giadr["CH4_CENTRAL_WAVENUMBER"],
                              giadr["CH4_CONSTANT1"],
                              giadr["CH4_CONSTANT2_SLOPE"])
    channels[:, 5, :] = to_bt(mdrs["SCENE_RADIANCES"][:, 4, :],
                              giadr["CH5_CENTRAL_WAVENUMBER"],
                              giadr["CH5_CONSTANT1"],
                              giadr["CH5_CONSTANT2_SLOPE"])
    channels = np.ma.masked_invalid(channels)
    
    
    toc = datetime.now()

    print "calibration took", toc - tic

    tic = datetime.now()

    lats = np.hstack((mdrs["EARTH_LOCATION_FIRST"][:, [0]], mdrs["EARTH_LOCATIONS"][:, :, 0], mdrs["EARTH_LOCATION_LAST"][:, [0]]))
    
    lons = np.hstack((mdrs["EARTH_LOCATION_FIRST"][:, [1]], mdrs["EARTH_LOCATIONS"][:, :, 1], mdrs["EARTH_LOCATION_LAST"][:, [1]]))

    sphr = srecords[1][1]
    nav_sample_rate = int(sphr["NAV_SAMPLE_RATE"][0].split("=")[1])
    earth_views_per_scanline = int(sphr["EARTH_VIEWS_PER_SCANLINE"][0].split("=")[1])

    geo_samples = np.round(earth_views_per_scanline / nav_sample_rate) + 3
    samples = np.zeros(geo_samples, dtype=np.intp)
    samples[1:-1] = np.arange(geo_samples - 2) * 20 + 5 - 1
    samples[-1] = earth_views_per_scanline - 1

    mask = np.ones((scanlines, earth_views_per_scanline))
    mask[:, samples] = 0
    geolats = np.ma.empty((scanlines, earth_views_per_scanline), dtype=lats.dtype)
    geolats.mask = mask
    geolats[:, samples] = lats
    geolons = np.ma.empty((scanlines, earth_views_per_scanline), dtype=lons.dtype)
    geolons.mask = mask
    geolons[:, samples] = lons 

    # get index of 44
    # np.nonzero(samples == 44)
    
    toc = datetime.now()

    print "geolocation took", toc - tic
    print "grand total", datetime.now() - gtic
    show(channels[:, 4, :])
