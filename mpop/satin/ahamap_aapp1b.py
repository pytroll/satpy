#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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
"""Interface to AAPP level 1b format. Uses the AHAMAP reader.
"""
import warnings
warnings.warn(__name__ + " is deprecated, please use aapp1b instead.",
              DeprecationWarning)

import glob
import os.path
from ConfigParser import ConfigParser

import math
import numpy as np
import logging

from mpop import CONFIG_PATH

LOG = logging.getLogger(__name__)

# Using ahamap: FIXME!

EPSILON = 0.001


def load(satscene, *args, **kwargs):
    """Read data from file and load it into *satscene*.
    """
    del args, kwargs
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2",
                                    raw=True):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options)


def load_avhrr(satscene, options):
    """Read avhrr data from file and load it into *satscene*.
    """

    if "filename" not in options:
        raise IOError("No filename given, cannot load.")

    chns = satscene.channels_to_load & set(["1", "2", "3A", "3B", "4", "5"])
    if len(chns) == 0:
        return

    values = {"orbit": satscene.orbit,
              "satname": satscene.satname,
              "number": satscene.number,
              "instrument": satscene.instrument_name,
              "satellite": satscene.fullname
              }

    filename = os.path.join(satscene.time_slot.strftime(options["dir"]) % values,
                            satscene.time_slot.strftime(options["filename"])
                            % values)

    file_list = glob.glob(filename)

    if len(file_list) > 1:
        raise IOError("More than one l1b file matching!")
    elif len(file_list) == 0:
        raise IOError("No l1b file matching!: " +
                      filename)

    filename = file_list[0]

    LOG.debug("Loading from " + filename)

    import avhrr  # AHAMAP module

    avh = avhrr.avhrr(filename)
    avh.get_unprojected()
    instrument_data = avh.build_raw()

    available_channels = set([])
    data_channels = {}

    for chn in instrument_data.data:
        channel_name = chn.info.info["channel_id"][3:].upper()
        available_channels |= set([channel_name])
        data_channels[channel_name] = chn.data

    for chn in satscene.channels_to_load:
        if chn in available_channels:
            if chn in ["1", "2", "3A"]:
                gain = instrument_data.info["vis_gain"]
                intercept = instrument_data.info["vis_intercept"]
                units = "%"
            else:
                gain = instrument_data.info["ir_gain"]
                intercept = instrument_data.info["ir_intercept"]
                units = "K"

            chn_array = np.ma.array(data_channels[chn])
            missing_data = instrument_data.info["missing_data"]
            chn_array = np.ma.masked_inside(chn_array,
                                            missing_data - EPSILON,
                                            missing_data + EPSILON)
            no_data = instrument_data.info["nodata"]
            chn_array = np.ma.masked_inside(chn_array,
                                            no_data - EPSILON,
                                            no_data + EPSILON)

            satscene[chn] = chn_array
            satscene[chn].data = np.ma.masked_less(satscene[chn].data *
                                                   gain +
                                                   intercept,
                                                   0)

            satscene[chn].info['units'] = units
        else:
            LOG.warning("Channel " + str(chn) + " not available, not loaded.")

    # Compulsory global attribudes
    satscene.info["title"] = (satscene.satname.capitalize() + satscene.number +
                              " satellite, " +
                              satscene.instrument_name.capitalize() +
                              " instrument.")
    satscene.info["institution"] = "Original data disseminated by EumetCast."
    satscene.add_to_history("HRIT/LRIT data read by mipp/mpop.")
    satscene.info["references"] = "No reference."
    satscene.info["comments"] = "No comment."

    lons = instrument_data.londata / math.pi * 180
    lats = instrument_data.latdata / math.pi * 180

    try:
        from pyresample import geometry
        satscene.area = geometry.SwathDefinition(lons=lons, lats=lats)
    except ImportError:
        satscene.area = None
        satscene.lat = lats
        satscene.lon = lons


CASES = {
    "avhrr": load_avhrr
}
