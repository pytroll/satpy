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

"""Interface to HRPT level 0 format. Needs pyorbital.

Since the loading and calibration goes quite fast, all channels are calibrated
at the same time, so don't hesitate to load all the channels anyway.

Contrarily to AAPP, no smoothing, sigma or gross filtering is taking place.

TODO:
 - Faster navigation (pyorbital).

"""
from ConfigParser import ConfigParser
import os
import logging
import glob
import numpy as np
import numexpr as ne
from mpop.plugin_base import Reader
from mpop import CONFIG_PATH
from pyresample.geometry import SwathDefinition

logger = logging.getLogger(__name__)

# Constants
c1 = 1.1910427e-5 #mW/(m2-sr-cm-4)
c2 = 1.4387752 #cm-K 


calib = {"noaa 15": # copy from noaa 16
         {
             # VIS
             "intersections": np.array([497.5, 500.3, 498.7]),
             "slopes_l": np.array([0.0523, 0.0513, 0.0262]),
             "slopes_h": np.array([0.1528, 0.1510, 0.1920]),
             "intercepts_l": np.array([-2.016, -1.943, -1.01]),
             "intercepts_h": np.array([-51.91, -51.77, -84.2]),
             # IR
             "d0": np.array([276.355, 276.142, 275.996, 276.132, 0]),
             "d1": np.array([0.0562, 0.05605, 0.05486, 0.0594, 0]),
             "d2": np.array([-1.590e-5, -1.707e-5, -1.223e-5, -1.344e-5, 0]),
             "d3": np.array([2.486e-8, 2.595e-8, 1.862e-8, 2.112e-8, 0]),
             "d4": np.array([-1.199e-11, -1.224e-11,
                             -0.853e-11, -1.001e-11, 0]),
             "prt_weights": np.array((.25, .25, .25, .25)),
             "vc": np.array((2700.1148, 917.2289, 838.1255)),
             "A": np.array((1.592459, 0.332380, 0.674623)),
             "B": np.array((0.998147, 0.998522, 0.998363)),
             "N_S": np.array([0, -2.467, -2.009]),
             "b0": np.array([0, 2.96, 2.25]),
             "b1": np.array([0, -0.05411, -0.03665]),
             "b2": np.array([0, 0.00024532, 0.00014854]),
             },
         "noaa 16":
         {
             # VIS
             "intersections": np.array([497.5, 500.3, 498.7]),
             "slopes_l": np.array([0.0523, 0.0513, 0.0262]),
             "slopes_h": np.array([0.1528, 0.1510, 0.1920]),
             "intercepts_l": np.array([-2.016, -1.943, -1.01]),
             "intercepts_h": np.array([-51.91, -51.77, -84.2]),
             # IR
             "d0": np.array([276.355, 276.142, 275.996, 276.132, 0]),
             "d1": np.array([0.0562, 0.05605, 0.05486, 0.0594, 0]),
             "d2": np.array([-1.590e-5, -1.707e-5, -1.223e-5, -1.344e-5, 0]),
             "d3": np.array([2.486e-8, 2.595e-8, 1.862e-8, 2.112e-8, 0]),
             "d4": np.array([-1.199e-11, -1.224e-11,
                             -0.853e-11, -1.001e-11, 0]),
             "prt_weights": np.array((.25, .25, .25, .25)),
             "vc": np.array((2700.1148, 917.2289, 838.1255)),
             "A": np.array((1.592459, 0.332380, 0.674623)),
             "B": np.array((0.998147, 0.998522, 0.998363)),
             "N_S": np.array([0, -2.467, -2.009]),
             "b0": np.array([0, 2.96, 2.25]),
             "b1": np.array([0, -0.05411, -0.03665]),
             "b2": np.array([0, 0.00024532, 0.00014854]),
             },
         "noaa 18": # FIXME: copy of noaa 19
         {
             # VIS
             "intersections": np.array([496.43, 500.37, 496.11]),
             "slopes_l": np.array([0.055091, 0.054892, 0.027174]),
             "slopes_h": np.array([0.16253, 0.16325, 0.18798]),
             "intercepts_l": np.array([-2.1415, -2.1288, -1.0881]),
             "intercepts_h": np.array([-55.863, -56.445, -81.491]),
             # IR
             "d0": np.array([276.601, 276.683, 276.565, 276.615, 0]),
             "d1": np.array([0.05090, 0.05101, 0.05117, 0.05103, 0]),
             "d2": np.array([1.657e-6, 1.482e-6, 1.313e-6, 1.484e-6, 0]),
             "d3": np.array([0, 0, 0, 0, 0]),
             "d4": np.array([0, 0, 0, 0, 0]),
             "prt_weights": np.array((1, 1, 1, 1)),
             "vc": np.array((2659.7952, 928.1460, 833.2532)),
             "A": np.array((1.698704, 0.436645, 0.253179)),
             "B": np.array((0.996960, 0.998607, 0.999057)),
             "N_S": np.array([0, -5.49, -3.39]),
             "b0": np.array([0, 5.70, 3.58]),
             "b1": np.array([0, -0.11187, -0.05991]),
             "b2": np.array([0, 0.00054668, 0.00024985]),
             },
         "noaa 19":
         {
             # VIS
             "intersections": np.array([496.43, 500.37, 496.11]),
             "slopes_l": np.array([0.055091, 0.054892, 0.027174]),
             "slopes_h": np.array([0.16253, 0.16325, 0.18798]),
             "intercepts_l": np.array([-2.1415, -2.1288, -1.0881]),
             "intercepts_h": np.array([-55.863, -56.445, -81.491]),
             # IR
             "d0": np.array([276.601, 276.683, 276.565, 276.615, 0]),
             "d1": np.array([0.05090, 0.05101, 0.05117, 0.05103, 0]),
             "d2": np.array([1.657e-6, 1.482e-6, 1.313e-6, 1.484e-6, 0]),
             "d3": np.array([0, 0, 0, 0, 0]),
             "d4": np.array([0, 0, 0, 0, 0]),
             "prt_weights": np.array((1, 1, 1, 1)),
             "vc": np.array((2659.7952, 928.1460, 833.2532)),
             "A": np.array((1.698704, 0.436645, 0.253179)),
             "B": np.array((0.996960, 0.998607, 0.999057)),
             "N_S": np.array([0, -5.49, -3.39]),
             "b0": np.array([0, 5.70, 3.58]),
             "b1": np.array([0, -0.11187, -0.05991]),
             "b2": np.array([0, 0.00054668, 0.00024985]),
             },
         "metop-a":
         {
             # VIS
             "intersections": np.array([501, 500, 502]),
             "slopes_l": np.array([0.0537, 0.0545, 0.0264]),
             "slopes_h": np.array([0.1587, 0.1619, 0.1837]),
             "intercepts_l": np.array([-2.1719, -2.167, -1.0868]),
             "intercepts_h": np.array([-54.7824, -55.913, -80.0116]),
             # IR
             "d0": np.array([276.6194, 276.6511, 276.6597, 276.3685, 0]),
             "d1": np.array([0.050919, 0.050892, 0.050845, 0.050992, 0]),
             "d2": np.array([1.470892e-6, 1.489e-6, 1.520646e-6, 1.48239e-6, 0]),
             "d3": np.array([0, 0, 0, 0, 0]),
             "d4": np.array([0, 0, 0, 0, 0]),
             "prt_weights": np.array((1, 1, 1, 1)) / 4.0,
             "vc": np.array((2687, 927.2, 837.7)),
             "A": np.array((2.06699, 0.55126, 0.34716)),
             "B": np.array((0.996577, 0.998509, 0.998947)),
             "N_S": np.array([0, -4.98, -3.40]),
             "b0": np.array([0, 5.44, 3.84]),
             "b1": np.array([0, 0.89848 - 1, 0.93751 - 1]),
             "b2": np.array([0, 0.00046964, 0.00025239]),
             },
         "metop-b":
         {
             # VIS
             "intersections": np.array([501, 503, 501]),
             "slopes_l": np.array([0.053572113, 0.051817433, 0.023518528]),
             "slopes_h": np.array([0.15871941, 0.15264062, 0.16376181]),
             "intercepts_l": np.array([-2.1099778, -2.0923391, -0.9879577]),
             "intercepts_h": np.array([-54.751018, -52.806460, -71.229881]),
             # IR
             "d0": np.array([276.5853, 276.5335, 276.5721, 276.5750, 0]),
             "d1": np.array([0.050933, 0.051033, 0.051097, 0.05102, 0]),
             "d2": np.array([1.54333e-6, 1.49751e-6, 1.42928e-6, 1.50841e-6, 0]),
             "d3": np.array([0, 0, 0, 0, 0]),
             "d4": np.array([0, 0, 0, 0, 0]),
             "prt_weights": np.array((1, 1, 1, 1)) / 4.0,
             "vc": np.array((2687, 927.2, 837.7)),
             "A": np.array((2.06699, 0.55126, 0.34716)),
             "B": np.array((0.996577, 0.998509, 0.998947)),
             "N_S": np.array([0, -4.75, -4.39]),
             "b0": np.array([0, 4.85, 4.36]),
             "b1": np.array([0, 0.903229 - 1, 0.923365 - 1]),
             "b2": np.array([0, 0.00048091, 0.00033524]),
             }

         }

SATELLITES = {7: "noaa 15",
              3: "noaa 16",
              5: "noaa 18",
              13: "noaa 18",
              15: "noaa 19"}

def bfield(array, bit):
    """return the bit array.
    """
    return (array & 2**(9 - bit + 1)).astype(np.bool)  

class HRPTReader(Reader):
    """HRPT minor frame reader.
    """
    pformat = "hrpt_hmf"
    
    def load(self, satscene):
        """Read data from file and load it into *satscene*.
        """
        conf = ConfigParser()
        conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
        options = {}
        for option, value in conf.items(satscene.instrument_name + "-level2",
                                        raw = True):
            options[option] = value
        CASES[satscene.instrument_name](self, satscene, options)

    def load_avhrr(self, satscene, options):
        """Read avhrr data from file and load it into *satscene*.
        """

        if "filename" not in options:
            raise IOError("No filename given, cannot load.")
        filename = os.path.join(
            options["dir"],
            (satscene.time_slot.strftime(options["filename"])))

        file_list = glob.glob(satscene.time_slot.strftime(filename))

        if len(file_list) > 1:
            raise IOError("More than one hrpt file matching!")
        elif len(file_list) == 0:
            raise IOError("No hrpt file matching!: " +
                          satscene.time_slot.strftime(filename))


        filename = file_list[0]
        array = read_file(filename)

        sat = (array["id"]["id"] & (2 ** 6 - 1)) >> 3
        sat = SATELLITES[sat[len(sat) / 2]]

        lon, lat, alt = navigate(array["timecode"], sat)
        area = SwathDefinition(lon.reshape(2048, -1), lat.reshape(2048, -1))
        satscene.area = area

        vis = vis_cal(array["image_data"][:, :, :3], sat)
        ir_ = ir_cal(array["image_data"][:, :, 2:], array["telemetry"]["PRT"],
                     array["back_scan"], array["space_data"], sat)

        channels = np.empty(array["image_data"].shape, dtype=np.float64)
        channels[:, :, :2] = vis[:, :, :2]
        channels[:, :, 3:] = ir_[:, :, 1:]
        ch3a = bfield(array["id"]["id"], 10)
        ch3b = np.logical_not(ch3a)
        channels[ch3a, :, 2] = vis[ch3a, :, 2]
        channels[ch3b, :, 2] = ir_[ch3b, :, 0]

        for chan in satscene.channels_to_load:
            if chan == "1":
                satscene["1"] = np.ma.array(vis[:, :, 0])
            if chan == "2":
                satscene["2"] = np.ma.array(vis[:, :, 1])
            if chan == "3A":
                satscene["3A"] = np.ma.array(vis[:, :, 2],
                                             mask=np.tile(ch3a, (1, 2048)))
            if chan == "3B":
                satscene["3B"] = np.ma.array(ir_[:, :, 0],
                                             mask=np.tile(ch3b, (1, 2048)))
            if chan == "4":
                satscene["4"] = np.ma.array(ir_[:, :, 1])
            if chan == "5":
                satscene["5"] = np.ma.array(ir_[:, :, 2])

## Reading
## http://www.ncdc.noaa.gov/oa/pod-guide/ncdc/docs/klm/html/c4/sec4-1.htm#t413-1

def read_file(filename):
    """Read the file using numpy
    """
    dtype = np.dtype([('frame_sync', '>u2', (6, )),
                      ('id', [('id', '>u2'),
                              ('spare', '>u2')]),
                      ('timecode', '>u2', (4, )),
                      ('telemetry', [("ramp_calibration", '>u2', (5, )),
                                     ("PRT", '>u2', (3, )),
                                     ("ch3_patch_temp", '>u2'),
                                     ("spare", '>u2'),]),
                      ('back_scan', '>u2', (10, 3)),
                      ('space_data', '>u2', (10, 5)),
                      ('sync', '>u2'),
                      ('TIP_data', '>u2', (520, )),
                      ('spare', '>u2', (127, )),
                      ('image_data', '>u2', (2048, 5)),
                      ('aux_sync', '>u2', (100, ))])

    arr = np.memmap(filename, dtype=dtype)
    #arr = arr.newbyteorder()
    return arr

## navigation

from pyorbital.orbital import Orbital
from datetime import datetime, timedelta, time
from pyorbital.geoloc import ScanGeometry, compute_pixels, get_lonlatalt

def timecode(tc_array):
    word = tc_array[0]
    day = word >> 1
    word = tc_array[1]
    msecs = ((127) & word) * 1024
    word = tc_array[2]
    msecs += word & 1023
    msecs *= 1024
    word = tc_array[3]
    msecs += word & 1023
    return datetime(2014, 1, 1) + timedelta(days=int(day) - 1,
                                            milliseconds=int(msecs))
def navigate(timecodes, satellite):
    orb = Orbital(satellite)


    first_time = timecode(timecodes[0])
    first_time = datetime(first_time.year, first_time.month, first_time.day)

    hrpttimes = [timecode(x) - first_time for x in timecodes]
    hrpttimes = np.array([x.seconds + x.microseconds / 1000000.0
                          for x in hrpttimes])
    
    scan_points = np.arange(2048)
    if satellite == "noaa 16":
        scan_angle = 55.25
    else:
        scan_angle = 55.37
    scans_nb = len(hrpttimes)
    avhrr_inst = np.vstack(((scan_points / 1023.5 - 1)
                            * np.deg2rad(-scan_angle),
                            np.zeros((len(scan_points),)))).T
    avhrr_inst = np.tile(avhrr_inst, [scans_nb, 1])

    offset = hrpttimes

    times = (np.tile(scan_points * 0.000025, [scans_nb, 1])
             + np.expand_dims(offset, 1))

    sgeom = ScanGeometry(avhrr_inst, times.ravel())

    s_times = sgeom.times(first_time)

    rpy = (0, 0, 0)
    pixels_pos = compute_pixels((orb.tle._line1, orb.tle._line2), sgeom, s_times, rpy)
    pos_time = get_lonlatalt(pixels_pos, s_times)
    return pos_time
    
    

## VIS calibration

def vis_cal(vis_data, sat):
    """Calibrates the visual data using dual gain.
    """
    logger.debug("Visual calibration")
    vis = np.empty(vis_data.shape, dtype=np.float64)
    for i in range(3):
        ch = vis_data[:, :, i]
        intersect = calib[sat]["intersections"][i]
        slope_l = calib[sat]["slopes_l"][i]
        slope_h = calib[sat]["slopes_h"][i]
        intercept_l = calib[sat]["intercepts_l"][i]
        intercept_h = calib[sat]["intercepts_h"][i]

        vis[:, :, i] = ne.evaluate("where(ch < intersect, ch * slope_l + intercept_l, ch * slope_h + intercept_h)")
    return vis

## IR calibration
def ir_cal(ir_data, telemetry, back_scan, space_data, sat):
    alen = ir_data.shape[0]
    logger.debug("IR calibration")
    logger.debug(" Preparing telemetry...")
    factor = np.ceil(alen / 5.0) + 1

    displacement = (telemetry[0:5, :] == np.array([0, 0, 0])).sum(1).argmax() + 1
    offset = 4 - (displacement - 1)

    globals().update(calib[sat])
    
    bd0 = np.tile(d0.reshape(-1, 1), (factor, 3))[offset:offset + alen]
    bd1 = np.tile(d1.reshape(-1, 1), (factor, 3))[offset:offset + alen]
    bd2 = np.tile(d2.reshape(-1, 1), (factor, 3))[offset:offset + alen]
    bd3 = np.tile(d3.reshape(-1, 1), (factor, 3))[offset:offset + alen]
    bd4 = np.tile(d4.reshape(-1, 1), (factor, 3))[offset:offset + alen]

    PRT = telemetry

    T_PRT = bd0 + PRT * (bd1 + PRT * (bd2 + PRT * (bd3 + PRT * bd4)))

    sublen = np.floor((T_PRT.shape[0] - displacement) / 5.0) * 5
    TMP_PRT = T_PRT[displacement:displacement + sublen]

    logger.debug(" Computing blackbody temperatures...")
    
    MEAN = ((TMP_PRT[::5] +
             TMP_PRT[1::5] +
             TMP_PRT[2::5] +
             TMP_PRT[3::5]) / 4).repeat(5, 0)

    if displacement == 0:
        T_BB_beg = None
    elif displacement == 1:
        T_BB_beg = MEAN[0]
    else:
        T_BB_beg = np.tile(T_PRT[:displacement].sum(0) / (displacement - 1), (displacement, 1))
    if sublen + displacement >=T_PRT.shape[0]:
        T_BB_end = None
    else:
        T_BB_end = np.tile(T_PRT[sublen+displacement:].mean(0), (T_PRT.shape[0] - sublen - displacement, 1))

    if T_BB_beg is not None:
        to_stack = [T_BB_beg, MEAN]
    else:
        to_stack = [MEAN]

    if T_BB_end is not None:
        to_stack.append(T_BB_end)

    T_BB = np.vstack(to_stack)

    if sat in ["noaa 15", "noaa 16"]:
        # three readings for klm
        T_BB = T_BB.mean(0)

    T_BB_star = A + B * T_BB

    N_BB = (c1 * vc ** 3) / (np.exp((c2 * vc)/(T_BB_star)) - 1)

    C_S = space_data[:,:, 2:].mean(1)
    C_BB = back_scan.mean(1)

    C_E = ir_data

    # aapp style
    #G = (N_BB - N_S) / (C_BB - C_S)
    #k1 = G**2 * b2
    #k2 = (b1 + 1) *G - 2 * k1 * C_S + 2*b2 * G * N_S
    #k3 = b0 + (b1 + 1) * N_S - (b1 + 1) *G * C_S + b2 * (N_S - G * N_S) ** 2
    #N_E = k1[:, np.newaxis, :] * C_E * C_E + k2[:, np.newaxis, :] * C_E + k3[:, np.newaxis, :]


    logger.debug(" Computing linear part of radiances...")

    C_Sr = C_S[:, np.newaxis, :]
    Cr = ((N_BB - N_S) / (C_S - C_BB))[:, np.newaxis, :]
    N_lin = ne.evaluate("(N_S + (Cr * (C_Sr - C_E)))")

    logger.debug(" Computing radiance correction...")
    # the +1 (N_lin) here is for Ne = Nlin + Ncor 
    N_E = ne.evaluate("((b2 * N_lin + b1 + 1) * N_lin + b0)")

    logger.debug(" Computing channels brightness temperatures...")
    T_E_star = ne.evaluate("(c2 * vc / (log(1 + c1 * vc**3 / N_E)))")
    T_E = ne.evaluate("(T_E_star - A) / B")

    return T_E


CASES = {
    "avhrr": HRPTReader.load_avhrr
    }


if __name__ == '__main__':
    import sys
    array = read_file(sys.argv[1])
    sat = (array["id"]["id"] & (2 ** 6 - 1)) >> 3
    sat = int(np.round(np.mean(sat)))
    sat = SATELLITES[sat]

    vis = vis_cal(array["image_data"][:, :, :3], sat)
    ir_ = ir_cal(array["image_data"][:, :, 2:], array["telemetry"]["PRT"],
                 array["back_scan"], array["space_data"], sat)

    channels = np.empty(array["image_data"].shape, dtype=np.float64)
    channels[:, :, :2] = vis[:, :, :2]
    channels[:, :, 3:] = ir_[:, :, 1:]
    ch3a = bfield(array["id"]["id"], 10)
    ch3b = np.logical_not(ch3a)
    channels[ch3a, :, 2] = vis[ch3a, :, 2]
    channels[ch3b, :, 2] = ir_[ch3b, :, 0]

    lon, lat, alt = navigate(array["timecode"], sat)
    area = SwathDefinition(lon.reshape(2048, -1), lat.reshape(2048, -1))

