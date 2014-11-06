#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>
#   Lars Orum Rasmussen <ras@dmi.dk>

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

"""Interface to SAR level-1 data. Using the MIPP reader.
"""
import ConfigParser
import os

from mipp import xsar
from mipp import ReaderError, CalibrationError

from mpop import CONFIG_PATH
import logging

LOG = logging.getLogger(__name__)

try:
    # Work around for on demand import of pyresample. pyresample depends
    # on scipy.spatial which memory leaks on multiple imports
    is_pyresample_loaded = False
    from pyresample import geometry
    from mpop.projector import get_area_def
    is_pyresample_loaded = True
except ImportError:
    LOG.warning("pyresample missing. Can only work in satellite projection")


def load(satscene, calibrate=True):
    """Read data from file and load it into *satscene*. The *calibrate*
    argument is passed to mipp (should be 0 for off, 1 for default, and 2 for
    radiances only).
    """
    conf = ConfigParser.ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2"):
        options[option] = value

    for section in conf.sections():
        if(section.startswith(satscene.instrument_name) and
           not (section == "satellite") and
           not section[:-1].endswith("-level") and
           not section.endswith("-granules")):
            options[section] = conf.items(section)

    CASES.get(satscene.instrument_name, load_generic)(satscene,
                                                      options,
                                                      calibrate)


def load_generic(satscene, options, calibrate=True):
    """Read sar data from file and load it into *satscene*.
    """
    os.environ["PPP_CONFIG_DIR"] = CONFIG_PATH

    LOG.debug("Channels to load from %s: %s" % (satscene.instrument_name,
                                                satscene.channels_to_load))

    # Compulsory global attribudes
    satscene.info["title"] = (satscene.satname.capitalize() + satscene.number +
                              " satellite, " +
                              satscene.instrument_name.capitalize() +
                              " instrument.")
    satscene.info["institution"] = "No institution."
    satscene.add_to_history("SAR data read by mipp/mpop.")
    satscene.info["references"] = "No reference."
    satscene.info["comments"] = "No comment."

    for chn in satscene.channels_to_load:
        try:
            metadata, data = xsar.sat.load(satscene.fullname,
                                           satscene.time_slot,
                                           chn,
                                           mask=True,
                                           calibrate=calibrate)
        except CalibrationError:
            LOG.warning(
                "Loading non calibrated data since calibration failed.")
            metadata, data = xsar.sat.load(satscene.fullname,
                                           satscene.time_slot,
                                           chn,
                                           mask=True,
                                           calibrate=False)
        except ReaderError:
            # if channel can't be found, go on with next channel
            continue

        satscene[chn] = data
        satscene[chn].info['units'] = metadata.calibration_unit

        if is_pyresample_loaded:
            # Build an area on the fly from the mipp metadata
            proj_params = getattr(metadata, "proj4_params").split()
            proj_dict = {}
            for param in proj_params:
                key, val = [i.strip() for i in param.split("=")]
                proj_dict[key] = val

            # Build area_def on-the-fly
            satscene[chn].area = geometry.AreaDefinition(
                satscene.satname + satscene.instrument_name +
                str(metadata.area_extent) +
                str(data.shape),
                "On-the-fly area",
                proj_dict["proj"],
                proj_dict,
                data.shape[1],
                data.shape[0],
                metadata.area_extent)
        else:
            LOG.info("Could not build area, pyresample missing...")

CASES = {}
