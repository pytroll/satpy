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
import glob
import logging
import os.path
import shutil
import subprocess
import tempfile
from ConfigParser import ConfigParser

from mpop.utils import ensure_dir
import mpop.satin.aapp1b
from mpop.satellites import get_satellite_class

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
    
    mpop.satin.aapp1b.load(satscene)

def convert_to_1b(in_filename, out_filename,
                  time_slot_start, time_slot_end,
                  shortname, orbit):
    """Convert hrpt file to level 1b.
    """
    (handle, tempname) = tempfile.mkstemp(prefix="hrpt_decommuted",
                                          dir=WORKING_DIR)

    os.close(handle)
    del handle
    LOG.debug("Decommuting...")
    decommutation(in_filename, tempname,
                  time_slot_start, time_slot_end,
                  shortname)
    LOG.debug("Calibrating, navigating...")
    calibration_navigation(tempname, time_slot_start, shortname)

    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, "regional" + shortname + ".cfg"))
    new_dir = conf.get("avhrr-level2", "dir", raw=True)
    new_name = conf.get("avhrr-level2", "filename", raw=True)
    options = {"satellite": shortname, "orbit": orbit}
    pathname = time_slot_start.strftime(os.path.join(new_dir, new_name))%options
    LOG.debug("Saving to "+pathname)
    ensure_dir(pathname)
    shutil.move(tempname, pathname)

def calibration_navigation(filename, time_slot, shortname):
    """Perform calibration on *filename*
    """
    import pysdh2orbnum
    LOG.info("Calibrating "+filename)
    formated_date = time_slot.strftime("%d/%m/%y %H:%M:%S.000")
    satpos_file = os.path.join(SATPOS_DIR,
                               "satpos_" + shortname +
                               time_slot.strftime("_%Y%m%d") +
                               ".txt")
    LOG.debug(formated_date)
    LOG.debug(satpos_file)
    orbit_number = str(pysdh2orbnum.sdh2orbnum(shortname,
                                               formated_date,
                                               satpos_file))
    avhrcl = ("cd /tmp;" +
              "$AAPP_PREFIX/AAPP/bin/avhrcl -c -l -s " +
              shortname + " -d " +
              time_slot.strftime("%Y%m%d") + " -h " +
              time_slot.strftime("%H%M") + " -n " +
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

    anacl1 = ("cd /tmp;" +
              "$ANA_PATH/bin/ana_lmk_loc -D " + filename)

    LOG.debug("Running " + anacl1)
    proc = subprocess.Popen(anacl1, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if out:
        LOG.debug(out)
    if err:
        LOG.error(err)    
    
    anacl2 = ("cd /tmp;" +
              "$ANA_PATH/bin/ana_estatt -s " + shortname +
              " -d " + time_slot.strftime("%Y%m%d") +
              " -h " + time_slot.strftime("%H%M") + " -n " +
              orbit_number)

    LOG.debug("Running " + anacl2)
    proc = subprocess.Popen(anacl2, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if out:
        LOG.debug(out)
    if err:
        LOG.error(err)
        
    avhrcl2 = ("cd /tmp;" +
              "$AAPP_PREFIX/AAPP/bin/avhrcl -l -s " +
              shortname + " -d " +
              time_slot.strftime("%Y%m%d") + " -h " +
              time_slot.strftime("%H%M") + " -n " +
              orbit_number + " " +
              filename)
    LOG.debug("Running " + avhrcl2)
    proc = subprocess.Popen(avhrcl2, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if out:
        LOG.debug(out)
    if err:
        LOG.error(err)
        
def decommutation(filename_from, filename_to,
                  time_slot_start, time_slot_end,
                  shortname):
    """Perform decommutation on *filename_from* and save the result in
    *filename_to*.
    """
    import pysdh2orbnum
    LOG.info("Decommuting "+filename_from)
    
    

    (handle, tempname) = tempfile.mkstemp(prefix="decommutation",
                                          suffix=".par",
                                          dir=WORKING_DIR)
    os.close(handle)
    handle = open(tempname, "w")
    handle.write("1,0,0,0,0,0,0,0,0,0,0,1\n")
    handle.write("10,11,15,15,16,0,0,13,0,0,0,14\n")
    handle.write(str(time_slot_start.year) + "\n")
    handle.write("0\n")
    
    satpos_file = os.path.join(SATPOS_DIR,
                               "satpos_" + shortname +
                               time_slot_start.strftime("_%Y%m%d") +
                               ".txt")
    formated_date = time_slot_start.strftime("%d/%m/%y %H:%M:%S.000")
    orbit_start = str(pysdh2orbnum.sdh2orbnum(shortname,
                                              formated_date,
                                              satpos_file))

    satpos_file = os.path.join(SATPOS_DIR,
                               "satpos_" + shortname +
                               time_slot_end.strftime("_%Y%m%d") +
                               ".txt")
    formated_date = time_slot_end.strftime("%d/%m/%y %H:%M:%S.000")
    orbit_end = str(pysdh2orbnum.sdh2orbnum(shortname,
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

def get_orbit(time_slot, shortname):
    import pysdh2orbnum
    formated_date = time_slot.strftime("%d/%m/%y %H:%M:%S.000")
    satpos_file = os.path.join(SATPOS_DIR,
                               "satpos_" + shortname +
                               time_slot.strftime("_%Y%m%d") +
                               ".txt")
    return str(pysdh2orbnum.sdh2orbnum(shortname,
                                       formated_date,
                                       satpos_file))


def concatenate(granules):
    """Concatenate hrpt files.
    """
    filenames = [os.path.join(granule.directory, granule.file_name)
                 for granule in granules]
    
    arg_string = " ".join(filenames)
    
    if filenames[0].endswith(".bz2"):
        cat_cmd = "bzcat"
    else:
        cat_cmd = "cat"


    conffile = os.path.join(CONFIG_PATH, granules[0].fullname + ".cfg")
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, conffile))
    
    directory = conf.get('avhrr-level1','dir')
    filename = conf.get('avhrr-level1','filename')
    filename = granules[0].time_slot.strftime(filename)
    
    output_name = os.path.join(directory, filename)
    cmd = cat_cmd + " " + arg_string + " > " + output_name
    LOG.debug(cmd)
    proc = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    (out, err) = proc.communicate()
    if out:
        LOG.debug(out)
    if err:
        LOG.error(err)

    LOG.debug("Done concatenating level0 files.")

    new_dir = conf.get(granules[0].instrument_name + "-level2", "dir")
    new_name = conf.get(granules[0].instrument_name + "-level2", "filename")
    pathname = os.path.join(new_dir, granules[0].time_slot.strftime(new_name))
    shortname = conf.get('avhrr-level1','shortname')

    convert_to_1b(output_name, pathname,
                  granules[0].time_slot,
                  granules[-1].time_slot + granules[-1].granularity,
                  shortname,
                  get_orbit(granules[0].time_slot,
                            shortname))
    os.remove(output_name)
    klass = get_satellite_class(granules[0].satname,
                                granules[0].number,
                                granules[0].variant)
    scene = klass(time_slot=granules[0].time_slot,
                  orbit=get_orbit(granules[0].time_slot, shortname))

    return scene

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

def get_lonlat(satscene, row, col):
    """Read lat and lon.
    """

    return LONLAT_CASES[satscene.instrument_name](satscene, row, col)

def get_lonlat_avhrr(satscene, row, col):
    """Read longitude and latitude for a given pixel.
    """
    # Needs the SATID AAPP env variable to be set to find satid.txt...

    import pyaapp
    import math
    import datetime
    t_start = satscene.time_slot
    epoch = datetime.datetime(1950, 1, 1)
    t50_start = (t_start - epoch)
    jday_start = t50_start.seconds / (3600.0 *24) + t50_start.days
    jday_end = jday_start
    if(satscene.satname == "metop"):
        satname = "M02"
    else:
        satname = satscene.satname + satscene.number

    if satscene.time_slot.hour == 0 and satscene.time_slot.minute == 0:
        satpos_time = satscene.time_slot - datetime.timedelta(minutes=1)
    else:
        satpos_time = satscene.time_slot
        
    satpos_file = ("/data/24/saf/pps/import/ANC_data/source/satpos_"+
                   satname+"_"+
                   satpos_time.strftime("%Y%m%d")+".txt")

    pyaapp.read_satpos_file(jday_start, jday_end,
                            satscene.satname+" "+str(int(satscene.number)),
                            satpos_file)
    att = pyaapp.prepare_attitude(int(satscene.number), 0, 0, 0)
    lonlat = pyaapp.linepixel2lonlat(int(satscene.number), row, col, att,
                                     jday_start, jday_end)[1:3]
    return (lonlat[0] * 180.0 / math.pi, lonlat[1] * 180.0 / math.pi)

LONLAT_CASES = {
    "avhrr": get_lonlat_avhrr
    }


LL_CASES = {
    "avhrr": get_lat_lon_avhrr
    }

CASES = {
    "avhrr": load_avhrr
    }
