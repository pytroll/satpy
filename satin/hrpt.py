#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

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

"""Interface to HRPT level 0 format. Uses AAPP and the aapp1b reader.
"""

import os.path
import glob
import satin.aapp1b
import tempfile
import subprocess
from ConfigParser import ConfigParser
import logging
import shutil

WORKING_DIR = "/tmp"
SATPOS_DIR = "/data/24/saf/pps/opt/AAPP/data/satpos"

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep)[:-1])

CONFIG_PATH = (os.environ.get('PPP_CONFIG_DIR', '') or
               os.path.join(BASE_PATH, 'etc'))

LOG = logging.getLogger("hrpt loader")

def load(satscene):
    """Read data from file and load it into *satscene*.
    """
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level1",
                                    raw = True):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options)

def load_avhrr(satscene, options):
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

    (handle, tempname) = tempfile.mkstemp(prefix="hrpt_decommuted",
                                          dir=WORKING_DIR)

    os.close(handle)
    del handle
    decommutation(filename, tempname, satscene, options)
    calibration_navigation(tempname, satscene, options)

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    new_dir = conf.get(satscene.instrument_name + "-level2", "dir")
    new_name = conf.get(satscene.instrument_name + "-level2", "filename")
    pathname = os.path.join(new_dir, satscene.time_slot.strftime(new_name))
    LOG.debug("Saving to "+pathname)
    shutil.move(tempname, pathname)
    
    satin.aapp1b.load(satscene)

def calibration_navigation(filename, satscene, options):
    """Perform calibration on *filename*
    """
    import pysdh2orbnum
    formated_date = satscene.time_slot.strftime("%d/%m/%y %H:%M:%S.000")
    satpos_file = os.path.join(SATPOS_DIR,
                               "satpos_" + options["shortname"] +
                               satscene.time_slot.strftime("_%Y%m%d") +
                               ".txt")
    LOG.debug(formated_date)
    LOG.debug(satpos_file)
    orbit_number = str(pysdh2orbnum.sdh2orbnum(options["shortname"],
                                               formated_date,
                                               satpos_file))
    avhrcl = ("cd /tmp;" +
              "$AAPP_PREFIX/AAPP/bin/avhrcl -c -l -s " +
              options["shortname"] + " -d " +
              satscene.time_slot.strftime("%Y%m%d") + " -h " +
              satscene.time_slot.strftime("%H%M") + " -n " +
              orbit_number + " " +
              filename)

    LOG.debug("Running " + avhrcl)
    proc = subprocess.Popen(avhrcl, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if out:
        LOG.debug(out)
    if err:
        LOG.error(err)    

    
    
def decommutation(filename_from, filename_to, satscene, options):
    """Perform decommutation on *filename_from* and save the result in
    *filename_to*.
    """
    import pysdh2orbnum
    
    

    (handle, tempname) = tempfile.mkstemp(prefix="decommutation",
                                          suffix=".par",
                                          dir=WORKING_DIR)
    os.close(handle)
    handle = open(tempname, "w")
    handle.write("1,0,0,0,0,0,0,0,0,0,0,1\n")
    handle.write("10,11,15,15,16,0,0,13,0,0,0,14\n")
    handle.write(str(satscene.time_slot.year) + "\n")
    handle.write("0\n")
    
    satpos_file = os.path.join(SATPOS_DIR,
                               "satpos_" + options["shortname"] +
                               satscene.time_slot.strftime("_%Y%m%d") +
                               ".txt")
    formated_date = satscene.time_slot.strftime("%d/%m/%y %H:%M:%S.000")
    orbit_start = str(pysdh2orbnum.sdh2orbnum(options["shortname"],
                                              formated_date,
                                              satpos_file))

    satpos_file = os.path.join(SATPOS_DIR,
                               "satpos_" + options["shortname"] +
                               satscene.end_time.strftime("_%Y%m%d") +
                               ".txt")
    formated_date = satscene.end_time.strftime("%d/%m/%y %H:%M:%S.000")
    orbit_end = str(pysdh2orbnum.sdh2orbnum(options["shortname"],
                                            formated_date,
                                            satpos_file))

    handle.write(orbit_start + "," + orbit_end + "\n")
    handle.close()
    
    decom = "$AAPP_PREFIX/AAPP/bin/decommutation"
    cmd = " ".join(["cd " + WORKING_DIR + ";",
                    decom, "ATOVS", tempname, filename_from])
    LOG.debug("Running " + cmd)
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if out:
        LOG.debug(out)
    if err:
        LOG.error(err)

    shutil.move(os.path.join(WORKING_DIR, "hrpt.l1b"), filename_to)
    LOG.debug("Decommutation done")

def get_lat_lon(satscene, resolution):
    """Read lat and lon.
    """
    del resolution
    
    return LL_CASES[satscene.instrument_name](satscene, None)

def get_lat_lon_avhrr(satscene, options):
    """Read lat and lon.
    """
    del options
    
    return satscene.lat, satscene.lon


LL_CASES = {
    "avhrr": get_lat_lon_avhrr
    }

CASES = {
    "avhrr": load_avhrr
    }
