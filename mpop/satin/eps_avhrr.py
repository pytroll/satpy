#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2012, 2014.

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
"""Interface to EPS Avhrr/3 level 1b format.
http://oiswww.eumetsat.org/WEBOPS/eps-pg/AVHRR/AVHRR-PG-11L1bFormat.htm
http://www.eumetsat.int/idcplg?IdcService=GET_FILE&\
dDocName=PDF_TEN_96167-EPS-GPFS&RevisionSelectionMethod=LatestReleased
http://www.eumetsat.int/idcplg?IdcService=GET_FILE&\
dDocName=PDF_TEN_97231-EPS-AVHRR&RevisionSelectionMethod=LatestReleased
"""
import struct
import datetime
import numpy as np
import math
from scipy import interpolate
import os.path
import glob
from ConfigParser import ConfigParser
from mpop import CONFIG_PATH
import logging

LOG = logging.getLogger(__name__)

RECORD_CLASS = ["Reserved", "MPHR", "SPHR",
                "IPR", "GEADR", "GIADR",
                "VEADR", "VIADR", "MDR"]

INSTRUMENT_GROUP = ["GENERIC", "AMSU-A", "ASCAT", "ATOVS", "AVHRR/3", "GOME",
                    "GRAS", "HIRS/4", "IASA", "MHS", "SEM", "ADCS", "SBUV",
                    "DUMMY", "ARCHIVE", "IASI_L2"]

MPHR = [100, 100, 100, 100, 100, 37, 36, 36, 35, 36, 48, 48, 48, 48, 37, 38, 38,
        38, 38, 48, 48, 34, 34, 36, 48, 48, 38, 38, 44, 51, 44, 44, 44, 44, 44,
        44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44,
        35, 48, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 41, 41, 41,
        34]

SPHR = [38, 36]

GIADR_SUB = [None, "GIADR-RADIANCE", "GIADR-ANALOG"]

MAX_SCAN_LINES = 2000


def get_bit(bitstring, bit):
    """Get a given *bit* from *bitstring*.
    """
    return bitstring & (2 ** bit)


def read_u_bytes(fdes, size):
    """Read unsigned bytes, and scale it by 10 ** *sf_*.
    """
    cases = {
        1: ">B",
        2: ">H",
        4: ">I",
        8: ">Q"
    }
    return struct.unpack(cases[size], fdes.read(size))[0]


def read_bytes(fdes, size, sf_=0):
    """Read signed bytes, and scale it by 10 ** *sf_*.
    """
    cases = {
        1: ">b",
        2: ">h",
        4: ">i",
        8: ">q"
    }
    if sf_ != 0:
        return struct.unpack(cases[size], fdes.read(size))[0] * 10.0 ** sf_
    else:
        return struct.unpack(cases[size], fdes.read(size))[0]


def read_short_cds(fdes):
    """Read a short cds date.
    """

    difference = datetime.timedelta(days=read_u_bytes(fdes, 2),
                                    milliseconds=read_u_bytes(fdes, 4))
    epoch = datetime.datetime(2000, 1, 1)
    return epoch + difference


def read_long_cds(fdes):
    """Read a long cds date.
    """

    difference = datetime.timedelta(days=read_u_bytes(fdes, 2),
                                    milliseconds=read_u_bytes(fdes, 4),
                                    microseconds=read_u_bytes(fdes, 2))

    epoch = datetime.datetime(2000, 1, 1)
    return epoch + difference


def read_ascii_field(fdes, size):
    """Read an ascii field.
    """
    field_name = fdes.read(30).strip()
    fdes.read(2)
    field_value = fdes.read(size - 33).strip()
    fdes.read(1)
    return (field_name, field_value)


def read_bitstring(fdes, size):
    """Read a bit string.
    """
    cases = {
        1: ">B",
        2: ">H",
        4: ">I",
        8: ">Q"
    }
    return struct.unpack(cases[size], fdes.read(size))[0]


def print_bitstring(s__):
    """Print a bitstring.
    """

    res = ""
    ts_ = s__

    i = 0
    for i in range(16):
        res = str(ts_ & 1) + res
        ts_ = ts_ >> 1

    del i
    print res


def read_grh(fdes):
    """Read GRH.
    """

    grh = {}
    record_class = fdes.read(1)
    if record_class == "":
        return "EOF"
    grh["RECORD_CLASS"] = RECORD_CLASS[ord(record_class)]
    grh["INSTRUMENT_GROUP"] = INSTRUMENT_GROUP[ord(fdes.read(1))]
    grh["RECORD_SUBCLASS"] = ord(fdes.read(1))
    grh["RECORD_SUBCLASS_VERSION"] = ord(fdes.read(1))
    grh["RECORD_SIZE"] = read_u_bytes(fdes, 4)
    grh["RECORD_START_TIME"] = read_short_cds(fdes)
    grh["RECORD_STOP_TIME"] = read_short_cds(fdes)
    return grh


def read_mphr(fdes, grh, metadata):
    """Read MPHR.
    """

    del grh

    for i in MPHR:
        field_name, field_value = read_ascii_field(fdes, i)
        try:
            metadata[field_name] = eval(field_value)
        except:
            metadata[field_name] = str(field_value)

    return metadata


def read_sphr(fdes, grh, metadata):
    """Read SPHR.
    """

    del grh

    if(metadata["INSTRUMENT_ID"] != "AVHR"):
        raise NotImplementedError("Only Avhrr for now...")

    fdes.read(49)
    for i in SPHR:
        field_name, field_value = read_ascii_field(fdes, i)
        try:
            metadata[field_name] = eval(field_value)
        except:
            metadata[field_name] = str(field_value)


def read_ipr(fdes, grh, metadata):
    """Read IPR.
    """

    del grh, metadata

    ipr = {}
    ipr["TARGET_RECORD_CLASS"] = read_u_bytes(fdes, 1)
    ipr["TARGET_INSTRUMENT_GROUP"] = read_u_bytes(fdes, 1)
    ipr["TARGET_RECORD_SUBCLASS"] = read_u_bytes(fdes, 1)
    ipr["TARGET_RECORD_OFFSET"] = read_u_bytes(fdes, 4)
    return ipr


def read_geadr(fdes, grh, metadata):
    """Read GEADR.
    """

    del metadata

    geadr = {}
    field_name, field_val = read_ascii_field(fdes, grh["RECORD_SIZE"] - 20)
    geadr[field_name] = field_val
    return geadr


def read_giadr(fdes, grh, metadata):
    """Read GIADR.
    """
    if(metadata["INSTRUMENT_ID"] != "AVHR"):
        raise NotImplementedError("Only Avhrr for now...")
    if(metadata["PROCESSING_LEVEL"] != "1B"):
        raise NotImplementedError("Only level 1B for now...")

    if grh["RECORD_SUBCLASS"] == 1:
        return read_giadr_radiance(fdes, grh, metadata)
    elif grh["RECORD_SUBCLASS"] == 2:
        return read_giadr_analog(fdes, grh, metadata)
    elif grh["RECORD_SUBCLASS"] == 99:
        fdes.read(grh["RECORD_SIZE"] - 20)
        return
    else:
        raise ValueError("Undefined subclass " +
                         str(grh["RECORD_SUBCLASS"]) +
                         ", version " +
                         str(grh["RECORD_SUBCLASS_VERSION"]) +
                         "...")


def read_giadr_radiance(fdes, grh, metadata):
    """Read GIADR.
    """

    del grh
    del metadata

    giadr = {}
    giadr["RAMP_CALIBRATION_COEFFICIENT"] = read_bitstring(fdes, 2)
    giadr["YEAR_RECENT_CALIBRATION"] = read_u_bytes(fdes, 2)
    giadr["DAY_RECENT_CALIBRATION"] = read_u_bytes(fdes, 2)
    giadr["PRIMARY_CALIBRATION_ALGORITHM_ID"] = read_u_bytes(fdes, 2)
    giadr["PRIMARY_CALIBRATION_ALGORITHM_OPTION"] = read_bitstring(fdes, 2)
    giadr["SECONDARY_CALIBRATION_ALGORITHM_ID"] = read_u_bytes(fdes, 2)
    giadr["SECONDARY_CALIBRATION_ALGORITHM_OPTION"] = read_bitstring(fdes, 2)
    giadr["IR_TEMPERATURE1_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE1_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE1_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE1_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE1_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE1_COEFFICIENT6"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE2_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE2_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE2_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE2_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE2_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE2_COEFFICIENT6"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE3_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE3_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE3_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE3_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE3_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE3_COEFFICIENT6"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE4_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE4_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE4_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE4_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE4_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["IR_TEMPERATURE4_COEFFICIENT6"] = read_bytes(fdes, 2)
    giadr["CH1_SOLAR_FILTERED_IRRADIANCE"] = read_bytes(fdes, 2, -1)
    giadr["CH1_EQUIVALENT FILTER_WIDTH"] = read_bytes(fdes, 2, -3)
    giadr["CH2_SOLAR_FILTERED_IRRADIANCE"] = read_bytes(fdes, 2, -1)
    giadr["CH2_EQUIVALENT FILTER_WIDTH"] = read_bytes(fdes, 2, -3)
    giadr["CH3A_SOLAR_FILTERED_IRRADIANCE"] = read_bytes(fdes, 2, -1)
    giadr["CH3A_EQUIVALENT FILTER_WIDTH"] = read_bytes(fdes, 2, -3)
    giadr["CH3B_CENTRAL_WAVENUMBER"] = read_bytes(fdes, 4, -2)
    giadr["CH3B_CONSTANT1"] = read_bytes(fdes, 4, -5)
    giadr["CH3B_CONSTANT2_SLOPE"] = read_bytes(fdes, 4, -6)
    giadr["CH4_CENTRAL_WAVENUMBER"] = read_bytes(fdes, 4, -3)
    giadr["CH4_CONSTANT1"] = read_bytes(fdes, 4, -5)
    giadr["CH4_CONSTANT2_SLOPE"] = read_bytes(fdes, 4, -6)
    giadr["CH5_CENTRAL_WAVENUMBER"] = read_bytes(fdes, 4, -3)
    giadr["CH5_CONSTANT1"] = read_bytes(fdes, 4, -5)
    giadr["CH5_CONSTANT2_SLOPE"] = read_bytes(fdes, 4, -6)

    return giadr


def read_giadr_analog(fdes, grh, metadata):
    """Read GIADR.
    """
    giadr = {}

    giadr["PATCH_TEMPERATURE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_EXTENDED_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_EXTENDED_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_EXTENDED_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_EXTENDED_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["PATCH_TEMPERATURE_EXTENDED_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["PATCH_POWER_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["PATCH_POWER_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["PATCH_POWER_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["PATCH_POWER_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["PATCH_POWER_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["RADIATOR_TEMPERATURE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["RADIATOR_TEMPERATURE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["RADIATOR_TEMPERATURE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["RADIATOR_TEMPERATURE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["RADIATOR_TEMPERATURE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE1_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE1_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE1_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE1_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE1_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE2_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE2_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE2_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE2_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE2_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE3_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE3_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE3_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE3_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE3_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE4_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE4_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE4_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE4_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["BLACKBODY_TEMPERATURE4_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_CURRENT_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_CURRENT_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_CURRENT_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_CURRENT_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_CURRENT_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["MOTOR_CURRENT_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["MOTOR_CURRENT_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["MOTOR_CURRENT_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["MOTOR_CURRENT_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["MOTOR_CURRENT_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["EARTH_SHIELD_POSITION_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["EARTH_SHIELD_POSITION_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["EARTH_SHIELD_POSITION_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["EARTH_SHIELD_POSITION_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["EARTH_SHIELD_POSITION_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_TEMPERATURE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_TEMPERATURE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_TEMPERATURE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_TEMPERATURE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["ELECTRONIC_TEMPERATURE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["COOLER_HOUSING_TEMPERATURE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["COOLER_HOUSING_TEMPERATURE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["COOLER_HOUSING_TEMPERATURE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["COOLER_HOUSING_TEMPERATURE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["COOLER_HOUSING_TEMPERATURE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["BASEPLATE_TEMPERATURE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["BASEPLATE_TEMPERATURE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["BASEPLATE_TEMPERATURE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["BASEPLATE_TEMPERATURE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["BASEPLATE_TEMPERATURE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["MOTOR_HOUSING_TEMPERATURE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["MOTOR_HOUSING_TEMPERATURE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["MOTOR_HOUSING_TEMPERATURE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["MOTOR_HOUSING_TEMPERATURE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["MOTOR_HOUSING_TEMPERATURE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["AD_CONVERTER_TEMPERATURE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["AD_CONVERTER_TEMPERATURE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["AD_CONVERTER_TEMPERATURE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["AD_CONVERTER_TEMPERATURE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["AD_CONVERTER_TEMPERATURE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["DETECTOR4_BIAS_VOLTAGE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["DETECTOR4_BIAS_VOLTAGE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["DETECTOR4_BIAS_VOLTAGE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["DETECTOR4_BIAS_VOLTAGE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["DETECTOR4_BIAS_VOLTAGE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["DETECTOR5_BIAS_VOLTAGE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["DETECTOR5_BIAS_VOLTAGE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["DETECTOR5_BIAS_VOLTAGE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["DETECTOR5_BIAS_VOLTAGE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["DETECTOR5_BIAS_VOLTAGE_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["CH3B_BLACKBODY_VIEW_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["CH3B_BLACKBODY_VIEW_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["CH3B_BLACKBODY_VIEW_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["CH3B_BLACKBODY_VIEW_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["CH3B_BLACKBODY_VIEW_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["CH4_BLACKBODY_VIEW_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["CH4_BLACKBODY_VIEW_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["CH4_BLACKBODY_VIEW_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["CH4_BLACKBODY_VIEW_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["CH4_BLACKBODY_VIEW_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["CH5_BLACKBODY_VIEW_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["CH5_BLACKBODY_VIEW_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["CH5_BLACKBODY_VIEW_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["CH5_BLACKBODY_VIEW_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["CH5_BLACKBODY_VIEW_COEFFICIENT5"] = read_bytes(fdes, 2)
    giadr["REFERENCE_VOLTAGE_COEFFICIENT1"] = read_bytes(fdes, 2)
    giadr["REFERENCE_VOLTAGE_COEFFICIENT2"] = read_bytes(fdes, 2)
    giadr["REFERENCE_VOLTAGE_COEFFICIENT3"] = read_bytes(fdes, 2)
    giadr["REFERENCE_VOLTAGE_COEFFICIENT4"] = read_bytes(fdes, 2)
    giadr["REFERENCE_VOLTAGE_COEFFICIENT5"] = read_bytes(fdes, 2)

    return giadr


def read_veadr(fdes, grh, metadata):
    """Read VEADR.
    """
    veadr = {}
    field_name, field_val = read_ascii_field(fdes, grh["RECORD_SIZE"] - 20)
    veadr[field_name] = field_val
    return veadr


def read_mdr(fdes, grh, metadata):
    """Read MDR.
    """
    if grh["RECORD_SUBCLASS"] != 2:
        raise ValueError("Only l1b supported for now")

    return read_mdr_1b(fdes, grh, metadata)


def read_mdr_1b(fdes, grh, metadata):
    """Read MDR section, 1B type.
    """
    mdr = {}

    mdr["DEGRADED_INST_MDR"] = read_u_bytes(fdes, 1)
    mdr["DEGRADED_PROC_MDR"] = read_u_bytes(fdes, 1)
    mdr["EARTH_VIEWS_PER_SCANLINE"] = read_bytes(fdes, 2)
    scanlength = mdr["EARTH_VIEWS_PER_SCANLINE"]
    array = (np.fromfile(file=fdes, dtype=">i2", count=scanlength * 5) *
             10 ** -2)
    array = array.reshape(5, scanlength)
    array[2, :] *= 10 ** -2
    mdr["SCENE_RADIANCES"] = array

    # Channels 1, 2, 3a in units of W/(m^2.sr).
    # Channels 3b, 4, 5 in units of mW/(m^2.sr.cm^-1).
    # Channels 1, 2, 4 & 5 with scale factor = 2.
    # Channels 3a or 3b with scale factor = 4.

    mdr["TIME_ATTITUDE"] = read_u_bytes(fdes, 4)
    mdr["EULER_ANGLE"] = (read_bytes(fdes, 2), read_bytes(fdes, 2),
                          read_bytes(fdes, 2))
    mdr["NAVIGATION_STATUS"] = read_bitstring(fdes, 4)
    mdr["SPACECRAFT_ALTITUDE"] = read_u_bytes(fdes, 4)
    mdr["ANGULAR_RELATIONS_FIRST"] = (read_bytes(fdes, 2), read_bytes(fdes, 2),
                                      read_bytes(fdes, 2), read_bytes(fdes, 2))
    mdr["ANGULAR_RELATIONS_LAST"] = (read_bytes(fdes, 2), read_bytes(fdes, 2),
                                     read_bytes(fdes, 2), read_bytes(fdes, 2))
    mdr["EARTH_LOCATION_FIRST"] = np.array((read_bytes(fdes, 4, -4),
                                            read_bytes(fdes, 4, -4)))
    mdr["EARTH_LOCATION_LAST"] = np.array((read_bytes(fdes, 4, -4),
                                           read_bytes(fdes, 4, -4)))

    mdr["NUM_NAVIGATION_POINTS"] = read_bytes(fdes, 2)
    mdr["ANGULAR_RELATIONS"] = np.fromfile(file=fdes, dtype=">i2",
                                           count=412) * 10 ** -2

    mdr["EARTH_LOCATIONS"] = np.fromfile(file=fdes, dtype=">i4",
                                         count=206) * 10 ** -4

    mdr["QUALITY_INDICATOR"] = read_bitstring(fdes, 4)
    mdr["SCAN_LINE_QUALITY"] = read_bitstring(fdes, 4)
    mdr["CALIBRATION_QUALITY"] = (read_bitstring(fdes, 2),
                                  read_bitstring(fdes, 2),
                                  read_bitstring(fdes, 2))
    mdr["COUNT_ERROR_FRAME"] = read_u_bytes(fdes, 2)
    mdr["CH123A_CURVE_SLOPE1"] = (read_bytes(fdes, 4),
                                  read_bytes(fdes, 4),
                                  read_bytes(fdes, 4))
    mdr["CH123A_CURVE_INTERCEPT1"] = (read_bytes(fdes, 4),
                                      read_bytes(fdes, 4),
                                      read_bytes(fdes, 4))
    mdr["CH123A_CURVE_SLOPE2"] = (read_bytes(fdes, 4),
                                  read_bytes(fdes, 4),
                                  read_bytes(fdes, 4))
    mdr["CH123A_CURVE_INTERCEPT2"] = (read_bytes(fdes, 4),
                                      read_bytes(fdes, 4),
                                      read_bytes(fdes, 4))
    mdr["CH123A_CURVE_INTERCEPTION"] = (read_bytes(fdes, 4),
                                        read_bytes(fdes, 4),
                                        read_bytes(fdes, 4))
    mdr["CH123A_TEST_CURVE_SLOPE1"] = (read_bytes(fdes, 4),
                                       read_bytes(fdes, 4),
                                       read_bytes(fdes, 4))
    mdr["CH123A_TEST_CURVE_INTERCEPT1"] = (read_bytes(fdes, 4),
                                           read_bytes(fdes, 4),
                                           read_bytes(fdes, 4))
    mdr["CH123A_TEST_CURVE_SLOPE2"] = (read_bytes(fdes, 4),
                                       read_bytes(fdes, 4),
                                       read_bytes(fdes, 4))
    mdr["CH123A_TEST_CURVE_INTERCEPT2"] = (read_bytes(fdes, 4),
                                           read_bytes(fdes, 4),
                                           read_bytes(fdes, 4))
    mdr["CH123A_TEST_CURVE_INTERCEPTION"] = (read_bytes(fdes, 4),
                                             read_bytes(fdes, 4),
                                             read_bytes(fdes, 4))
    mdr["CH123A_PRELAUNCH_CURVE_SLOPE"] = (read_bytes(fdes, 4),
                                           read_bytes(fdes, 4),
                                           read_bytes(fdes, 4))
    mdr["CH123A_PRELAUNCH_CURVE_INTERCEPT1"] = (read_bytes(fdes, 4),
                                                read_bytes(fdes, 4),
                                                read_bytes(fdes, 4))
    mdr["CH123A_PRELAUNCH_CURVE_SLOPE2"] = (read_bytes(fdes, 4),
                                            read_bytes(fdes, 4),
                                            read_bytes(fdes, 4))
    mdr["CH123A_PRELAUNCH_CURVE_INTERCEPT2"] = (read_bytes(fdes, 4),
                                                read_bytes(fdes, 4),
                                                read_bytes(fdes, 4))
    mdr["CH123A_PRELAUNCH_CURVE_INTERCEPTION"] = (read_bytes(fdes, 4),
                                                  read_bytes(fdes, 4),
                                                  read_bytes(fdes, 4))
    mdr["CH3B45_SECOND_TERM"] = (read_bytes(fdes, 4),
                                 read_bytes(fdes, 4),
                                 read_bytes(fdes, 4))
    mdr["CH3B45_FIRST_TERM"] = (read_bytes(fdes, 4),
                                read_bytes(fdes, 4),
                                read_bytes(fdes, 4))
    mdr["CH3B45_ZEROTH_TERM"] = (read_bytes(fdes, 4),
                                 read_bytes(fdes, 4),
                                 read_bytes(fdes, 4))
    mdr["CH3B45_TEST_SECOND_TERM"] = (read_bytes(fdes, 4),
                                      read_bytes(fdes, 4),
                                      read_bytes(fdes, 4))
    mdr["CH3B45_TEST_FIRST_TERM"] = (read_bytes(fdes, 4),
                                     read_bytes(fdes, 4),
                                     read_bytes(fdes, 4))
    mdr["CH3B45_TEST_ZEROTH_TERM"] = (read_bytes(fdes, 4),
                                      read_bytes(fdes, 4),
                                      read_bytes(fdes, 4))
    mdr["CLOUD_INFORMATION"] = np.fromfile(file=fdes, dtype="<i2",
                                           count=scanlength)

    mdr["FRAME_SYNCHRONISATION"] = (read_u_bytes(fdes, 2),
                                    read_u_bytes(fdes, 2),
                                    read_u_bytes(fdes, 2),
                                    read_u_bytes(fdes, 2),
                                    read_u_bytes(fdes, 2),
                                    read_u_bytes(fdes, 2))
    mdr["FRAME_INDICATOR"] = (read_bitstring(fdes, 2), read_bitstring(fdes, 2))

    mdr["TIME_CODE"] = (read_bitstring(fdes, 2), read_bitstring(fdes, 2),
                        read_bitstring(fdes, 2), read_bitstring(fdes, 2))
    mdr["RAMP_CALIB"] = (read_bitstring(fdes, 2), read_bitstring(fdes, 2),
                         read_bitstring(fdes, 2), read_bitstring(fdes, 2),
                         read_bitstring(fdes, 2))
    mdr["INTERNAL_TARGET_TEMPERATURE_COUNT"] = (read_bitstring(fdes, 2),
                                                read_bitstring(fdes, 2),
                                                read_bitstring(fdes, 2))

    # Digital B telemetry
    mdr["INSTRUMENT_INVALID_WORD_FLAG"] = read_bitstring(fdes, 2)
    mdr["DIGITAL_B_DATA"] = read_bitstring(fdes, 2)

    # Analog housekeeping data
    mdr["INSTRUMENT_INVALID_ANALOG_WORD_FLAG"] = read_bitstring(fdes, 4)
    mdr["PATCH_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["PATCH_EXTENDED_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["PATCH_POWER"] = read_u_bytes(fdes, 2)
    mdr["RADIATOR_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["BLACKBODY_TEMPERATURE1"] = read_u_bytes(fdes, 2)
    mdr["BLACKBODY_TEMPERATURE2"] = read_u_bytes(fdes, 2)
    mdr["BLACKBODY_TEMPERATURE3"] = read_u_bytes(fdes, 2)
    mdr["BLACKBODY_TEMPERATURE4"] = read_u_bytes(fdes, 2)
    mdr["ELECTRONIC_CURRENT"] = read_u_bytes(fdes, 2)
    mdr["MOTOR_CURRENT"] = read_u_bytes(fdes, 2)
    mdr["EARTH_SHIELD_POSITION"] = read_u_bytes(fdes, 2)
    mdr["ELECTRONIC_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["COOLER_HOUSING_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["BASEPLATE_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["MOTOR_HOUSING_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["AD_CONVERTER_TEMPERATURE"] = read_u_bytes(fdes, 2)
    mdr["DETECTOR4_VOLTAGE"] = read_u_bytes(fdes, 2)
    mdr["DETECTOR5_VOLTAGE"] = read_u_bytes(fdes, 2)
    mdr["CH3_BLACKBODY_VIEW"] = read_u_bytes(fdes, 2)
    mdr["CH4_BLACKBODY_VIEW"] = read_u_bytes(fdes, 2)
    mdr["CH5_BLACKBODY_VIEW"] = read_u_bytes(fdes, 2)
    mdr["REFERENCE_VOLTAGE"] = read_u_bytes(fdes, 2)

    return mdr

C1 = 1.191062e-05  # mW/(m2*sr*cm-4)
C2 = 1.4387863  # K/cm-1


def to_bt(arr, wc_, a__, b__):
    """Convert to BT.
    """
    val = np.log(1 + (C1 * (wc_ ** 3) / arr))
    t_star = C2 * wc_ / val
    return a__ + b__ * t_star


def to_refl(arr, solar_flux):
    """Convert to reflectances.
    """
    return arr * math.pi * 100.0 / solar_flux


def calibrate(channels, info_giadr):
    """convert the radiances to reflectances and bts.
    """
    channels[0, :, :] = to_refl(channels[0, :, :],
                                info_giadr["CH1_SOLAR_FILTERED_IRRADIANCE"])
    channels[1, :, :] = to_refl(channels[1, :, :],
                                info_giadr["CH2_SOLAR_FILTERED_IRRADIANCE"])
    channels[2, :, :] = to_refl(channels[2, :, :],
                                info_giadr["CH3A_SOLAR_FILTERED_IRRADIANCE"])
    channels[3, :, :] = to_bt(channels[3, :, :],
                              info_giadr["CH3B_CENTRAL_WAVENUMBER"],
                              info_giadr["CH3B_CONSTANT1"],
                              info_giadr["CH3B_CONSTANT2_SLOPE"])
    channels[4, :, :] = to_bt(channels[4, :, :],
                              info_giadr["CH4_CENTRAL_WAVENUMBER"],
                              info_giadr["CH4_CONSTANT1"],
                              info_giadr["CH4_CONSTANT2_SLOPE"])
    channels[5, :, :] = to_bt(channels[5, :, :],
                              info_giadr["CH5_CENTRAL_WAVENUMBER"],
                              info_giadr["CH5_CONSTANT1"],
                              info_giadr["CH5_CONSTANT2_SLOPE"])


def read(fdes):
    """Read the entire file.
    """

    metadata = {}

    cnt = 0

    channels = None
    lons = None
    lats = None
    samples = None

    geo_samples = 0
    scanlength = 0

    g3a = False
    g3b = False

    while True:

        grh = read_grh(fdes)
        if grh == "EOF":
            break

        record_class = grh["RECORD_CLASS"]
        if record_class not in CASES:
            raise NotImplementedError

        res = CASES[grh["RECORD_CLASS"]](fdes, grh, metadata)

        if record_class == "SPHR":
            scanlength = metadata["EARTH_VIEWS_PER_SCANLINE"]
            llats = np.zeros((MAX_SCAN_LINES, scanlength))
            llons = np.zeros((MAX_SCAN_LINES, scanlength))

            channels = np.ma.ones((6, MAX_SCAN_LINES, scanlength),
                                  dtype=np.float) * np.infty
            geo_samples = np.round(
                scanlength / metadata["NAV_SAMPLE_RATE"]) + 3
            samples = np.zeros(geo_samples, dtype=np.intp)
            samples[1:-1] = np.arange(geo_samples - 2) * 20 + 5 - 1
            samples[-1] = scanlength - 1
            lats = np.zeros((MAX_SCAN_LINES, geo_samples))
            lons = np.zeros((MAX_SCAN_LINES, geo_samples))

        if record_class == "GIADR" and grh["RECORD_SUBCLASS"] == 1:
            info_giadr = res

        if record_class == "MDR":
            three_a = get_bit(res["FRAME_INDICATOR"][0], 0)
            if three_a:
                channels[0:3, cnt, :] = res["SCENE_RADIANCES"][0:3]
                channels[4:, cnt, :] = res["SCENE_RADIANCES"][3:]
                g3a = True
            else:
                channels[0:2, cnt, :] = res["SCENE_RADIANCES"][0:2]
                channels[3:, cnt, :] = res["SCENE_RADIANCES"][2:]
                g3b = True

            lats[cnt, 1:-1] = res["EARTH_LOCATIONS"]\
                [np.arange(geo_samples - 2) * 2]
            lats[cnt, 0] = res["EARTH_LOCATION_FIRST"][0]
            lats[cnt, -1] = res["EARTH_LOCATION_LAST"][0]
            lons[cnt, 1:-1] = res["EARTH_LOCATIONS"]\
                [np.arange(geo_samples - 2) * 2 + 1]
            lons[cnt, 0] = res["EARTH_LOCATION_FIRST"][1]
            lons[cnt, -1] = res["EARTH_LOCATION_LAST"][1]
            # unwraping datum shift line
            lons[cnt, :] = np.rad2deg(np.unwrap(np.deg2rad(lons[cnt, :])))

            xnew = np.arange(scanlength)
            tck = interpolate.splrep(samples, lats[cnt, :], s=0)
            llats[cnt, :] = interpolate.splev(xnew, tck, der=0)
            tck = interpolate.splrep(samples, lons[cnt, :], s=0)
            llons[cnt, :] = interpolate.splev(xnew, tck, der=0)

            cnt += 1

    channels = channels[:, :cnt, :]
    llats = llats[:cnt, :]
    llons = llons[:cnt, :]
    llons[llons > 180] -= 360
    llons[llons < -180] += 360

    calibrate(channels, info_giadr)
    return channels, llats, llons, g3a, g3b, metadata["ORBIT_START"]


CASES = {"MPHR": read_mphr,
         "SPHR": read_sphr,
         "IPR": read_ipr,
         "GEADR": read_geadr,
         "GIADR": read_giadr,
         "VEADR": read_veadr,
         "MDR": read_mdr}


EPSILON = 0.001


def load(satscene):
    """Read data from file and load it into *satscene*.
    """
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level2",
                                    raw=True):
        options[option] = value
    LOAD_CASES[satscene.instrument_name](satscene, options)


def load_avhrr(satscene, options):
    """Read avhrr data from file and load it into *satscene*.
    """

    if "filename" not in options:
        raise IOError("No filename given, cannot load.")
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

    try:
        fdes = open(file_list[0])
        channels, lats, lons, g3a, g3b, orbit = read(fdes)

    finally:
        fdes.close()

    channels = np.ma.masked_invalid(channels)

    satscene["1"] = channels[0, :, :]
    satscene["2"] = channels[1, :, :]
    satscene["4"] = channels[4, :, :]
    satscene["5"] = channels[5, :, :]
    if g3a:
        satscene["3A"] = channels[2, :, :]
    if g3b:
        satscene["3B"] = channels[3, :, :]

    print "Inside eps_avhrr.load_avhrr: orbit = ", orbit
    #satscene.orbit = str(int(orbit) + 1)
    satscene.orbit = str(int(orbit))

    try:
        from pyresample import geometry
        satscene.area = geometry.SwathDefinition(lons=lons, lats=lats)
    except ImportError:
        satscene.area = None
        satscene.lat = lats
        satscene.lon = lons


def get_lonlat(satscene, row, col):
    try:
        if (satscene.area is None and
                (satscene.lat is None or satscene.lon is None)):
            load(satscene)
    except AttributeError:
        load(satscene)
    try:
        return satscene.area.lons[row, col], satscene.area.lats[row, col]
    except AttributeError:
        return satscene.lon[row, col], satscene.lat[row, col]


def get_lat_lon(satscene, resolution):
    """Read lat and lon.
    """
    del resolution

    return LAT_LON_CASES[satscene.instrument_name](satscene, None)


def get_lat_lon_avhrr(satscene, options):
    """Read lat and lon.
    """
    del options

    return satscene.lat, satscene.lon


LAT_LON_CASES = {
    "avhrr": get_lat_lon_avhrr
}

LOAD_CASES = {
    "avhrr": load_avhrr
}


if __name__ == "__main__":
    pass
