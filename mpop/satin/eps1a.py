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

"""Interface to EPS level 1a format. Uses AAPP and the aapp1b reader.
"""
import glob
import logging
import os.path
import shutil
import subprocess
import tempfile
from ConfigParser import ConfigParser

import mpop.satin.aapp1b


WORKING_DIR = "/tmp"
SATPOS_DIR = "/data/24/saf/pps/opt/AAPP/data/satpos"

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep)[:-1])

CONFIG_PATH = (os.environ.get('PPP_CONFIG_DIR', '') or
               os.path.join(BASE_PATH, 'etc'))

LOG = logging.getLogger("eps1a loader")

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
        raise IOError("More than one l1a file matching!")
    elif len(file_list) == 0:
        raise IOError("No l1a file matching!: "+
                      satscene.time_slot.strftime(filename))

    
    filename = file_list[0]

    (handle, tempname) = tempfile.mkstemp(prefix="eps1a_decommuted",
                                          dir=WORKING_DIR)

    os.close(handle)
    del handle
    decommutation(filename, tempname)
    calibration_navigation(tempname, satscene, options)

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    new_dir = conf.get(satscene.instrument_name + "-level2", "dir")
    new_name = conf.get(satscene.instrument_name + "-level2", "filename")
    pathname = os.path.join(new_dir, satscene.time_slot.strftime(new_name))
    LOG.debug("Saving to "+pathname)
    shutil.move(tempname, pathname)
    
    mpop.satin.aapp1b.load(satscene)

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

#     prl1bavh = ("cd /tmp; " +
#                 ". $AAPP_PREFIX/ATOVS_CONF; " +
#                 "$AAPP_PREFIX/AAPP/bin/prl1bavh -s 1 -e 1 " + filename)
#     LOG.debug("Running " + prl1bavh)
#     proc = subprocess.Popen(prl1bavh, shell=True,
#                             stdout=subprocess.PIPE,
#                             stderr=subprocess.PIPE)
#     (out, err) = proc.communicate()
#     if out:
#         LOG.debug(out)
#     if err:
#         LOG.error(err)    
    
    
def decommutation(filename_from, filename_to):
    """Perform decommutation on *filename_from* and save the result in
    *filename_to*.
    """
    decom = "$AAPP_PREFIX/metop-tools/bin/decom-avhrr-metop"
    flags = "-ignore_degraded_inst_mdr -ignore_degraded_proc_mdr"
    cmd = (decom+" "+
           flags+" "+
           filename_from+" "+
           filename_to)
    LOG.debug("Running " + cmd)
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if out:
        LOG.debug(out)
    if err:
        LOG.error(err)
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
