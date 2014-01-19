#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014 Martin Raspaud

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

"""Compact viirs format.
"""

import h5py
import numpy as np
from pyresample.geometry import SwathDefinition
from datetime import timedelta
import glob
from ConfigParser import ConfigParser
import os
import logging

from mpop import CONFIG_PATH
from mpop.imageo.geo_image import GeoImage

logger = logging.getLogger(__name__)

c = 299792458 # m.s-1
h = 6.6260755e-34 # m2kg.s-1
k = 1.380658e-23 # m2kg.s-2.K-1

def load(satscene, *args, **kwargs):

    time_start, time_end = kwargs.get("time_interval",
                                      (satscene.time_slot, None))

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name+"-level2",
                                    raw = True):
        options[option] = value


    template = os.path.join(options["dir"], options["filename"])

    second = timedelta(seconds=1)
    files_to_load = []

    if time_end is not None:
        time = time_start - second * 85
        files_to_load = []
        while time <= time_end:
            fname = time.strftime(template)
            flist = glob.glob(fname)
            try:
                files_to_load.append(flist[0])
                time += second*80
            except IndexError:
                pass
            time += second

    else:
        files_to_load = glob.glob(time_start.strftime(template))
    
    chan_dict = {"M01": "M1",
                 "M02": "M2",
                 "M03": "M3",
                 "M04": "M4",
                 "M05": "M5",
                 "M06": "M6",
                 "M07": "M7",
                 "M08": "M8",
                 "M09": "M9",
                 "M10": "M10",
                 "M11": "M11",
                 "M12": "M12",
                 "M13": "M13",
                 "M14": "M14",
                 "M15": "M15",
                 "M16": "M16"}

    chans = [chan_dict[chn] for chn in satscene.channels_to_load]

    datas = []
    lonlats = []

    for fname in files_to_load:
        h5f = h5py.File(fname, "r")
        datas.append(read(h5f, chans))
        lonlats.append(navigate(h5f))
        h5f.close()

    lons = np.ma.vstack([lonlat[0] for lonlat in lonlats])
    lats = np.ma.vstack([lonlat[1] for lonlat in lonlats])

    for nb, chn in enumerate(satscene.channels_to_load):
        data = np.ma.vstack([dat[nb] for dat in datas])
        satscene[chn] = data
        
    area_def = SwathDefinition(np.ma.masked_where(data.mask, lons),
                               np.ma.masked_where(data.mask, lats))

    for chn in satscene.channels_to_load:
        satscene[chn].area = area_def
    
def read(h5f, channels, calibrate=1):
    
    chan_dict = dict([(key.split("-")[1], key)
                      for key in h5f["All_Data"].keys()
                      if key.startswith("VIIRS")])
        
    res = []
    
    for channel in channels:
        rads = h5f["All_Data"][chan_dict[channel]]["Radiance"]
        arr = np.ma.masked_greater(rads.value, 65526)
        arr = np.ma.where(arr <= rads.attrs['Threshold'],
                          arr * rads.attrs['RadianceScaleLow'] + rads.attrs['RadianceOffsetLow'],
                          arr * rads.attrs['RadianceScaleHigh'] + rads.attrs['RadianceOffsetHigh'],)
        if calibrate == 0:
            raise NotImplementedError("Can't get counts from this data")
        if calibrate == 1:
            # do calibrate
            try:
                a_vis = rads.attrs['EquivalentWidth']
                b_vis = rads.attrs['IntegratedSolarIrradiance']
                dse = rads.attrs['EarthSunDistanceNormalised']
                arr *= 100 * np.pi * a_vis / b_vis * (dse ** 2)
            except KeyError:
                a_ir = rads.attrs['BandCorrectionCoefficientA']
                b_ir = rads.attrs['BandCorrectionCoefficientB']
                lambda_c = rads.attrs['CentralWaveLength']
                arr *= 1e6
                arr = (h*c) / (k * lambda_c * np.log(1 +
                                                     (2 * h * c**2) /
                                                     ((lambda_c**5) * arr)))
                arr *= a_ir
                arr += b_ir
        elif calibrate != 2:
            raise ValueError("Calibrate parameter should be 1 or 2")
        res.append(arr)
    return res

def navigate(h5f):
    geostuff = h5f["All_Data"]["VIIRS-MOD-GEO_All"]
    c_align = geostuff["AlignmentCoefficient"].value[np.newaxis, np.newaxis,
                                                     :, np.newaxis]
    c_exp = geostuff["ExpansionCoefficient"].value[np.newaxis, np.newaxis,
                                                   :, np.newaxis]
    lon = geostuff["Longitude"].value
    lat = geostuff["Latitude"].value
    s_track, s_scan = ((np.mgrid[0:768, 0:3200] % 16) + 0.5) / 16
    s_track = s_track.reshape(48, 16, 200, 16)
    s_scan = s_scan.reshape(48, 16, 200, 16)
    
    a_scan = s_scan + s_scan*(1-s_scan)*c_exp + s_track*(1 - s_track) * c_align
    a_track = s_track

    lon_a = lon[::2, np.newaxis, :-1, np.newaxis]
    lon_b = lon[::2, np.newaxis, 1:, np.newaxis]
    lon_c = lon[1::2, np.newaxis, 1:, np.newaxis]
    lon_d = lon[1::2, np.newaxis, :-1, np.newaxis]
    lat_a = lat[::2, np.newaxis, :-1, np.newaxis]
    lat_b = lat[::2, np.newaxis, 1:, np.newaxis]
    lat_c = lat[1::2, np.newaxis, 1:, np.newaxis]
    lat_d = lat[1::2, np.newaxis, :-1, np.newaxis]

    flon = ((1 - a_track) * ((1 - a_scan) * lon_a + a_scan * lon_b) +
            a_track * ((1 - a_scan) * lon_d + a_scan * lon_c))
    flat = ((1 - a_track) * ((1 - a_scan) * lat_a + a_scan * lat_b) +
            a_track * ((1 - a_scan) * lat_d + a_scan * lat_c))
    return flon.reshape(768, 3200), flat.reshape(768, 3200)

if __name__ == '__main__':
    #filename = "/local_disk/data/satellite/polar/compact_viirs/SVMC_npp_d20140114_t1245125_e1246367_b11480_c20140114125427496143_eum_ops.h5"
    filename = "/local_disk/data/satellite/polar/compact_viirs/mymy.h5"
    h5f = h5py.File(filename, 'r')
    # ch1, ch2, ch3, ch4 = read(h5f, ["M5", "M4", "M2", "M12"])
    # img = GeoImage((ch1, ch2, ch3),
    #                None,
    #                None,
    #                fill_value=None,
    #                mode="RGB")

    # img.enhance(stretch="linear")
    # img.enhance(gamma=2.0)
    # img.show()

    lons, lats = navigate(h5f)
