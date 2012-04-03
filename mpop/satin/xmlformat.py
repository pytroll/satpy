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

import numpy as np
from xml.etree.ElementTree import ElementTree


def process_delimiter(elt):
    pass

def process_field(elt):
    if elt.get("scaling-factor") is not None:
        try:
            return 10 / float(elt.get("scaling-factor").replace("^", "e"))
        except ValueError:
            return 10 / np.array(elt.get("scaling-factor").replace("^", "e").split(","), dtype=np.float)
    else:
        return 1.0
def get_len(elt):
    length = elt.get("length")
    if length.startswith("$"):
        length = VARIABLES[length[1:]]
    return int(length)

def process_array(elt):
    length = get_len(elt)
    ones = np.ones((length, ))
    chld = elt.getchildren()[0]
    scale = CASES[chld.tag](chld)
    try:
        if scale.shape == (length, get_len(chld)):
            return scale
    except AttributeError:
        pass
    try:
        return (ones * scale)
    except ValueError:
        return (np.repeat(scale, length).reshape(len(scale), length))
    


CASES = {"delimiter": process_delimiter,
         "field": process_field,
         "array": process_array,
         }

VARIABLES = {}

def fill_scales(xml_file, product, scales):
    tree = ElementTree()
    tree.parse(xml_file)


    params = tree.find("parameters")

    global VARIABLES
    for param in params.getchildren():
        VARIABLES[param.get("name")] = param.get("value")


    scales = {}


    prod = tree.find("product/" + product)
    res = []
    for i in prod:
        CASES[i.tag](i, scales)

if __name__ == '__main__':
    print get_scales("eps_avhrrl1b_6.5.xml", "mdr")
"""

from xml.etree.ElementTree import ElementTree

import numpy as np

VARIABLES = {}

TYPEC = {"boolean": ">i1",
         "integer2": ">i2",
         "integer4": ">i4",
         "uinteger2": ">u2",
         "uinteger4": ">u4",
         }


def process_delimiter(elt, ascii=False):
    del elt, ascii

def process_field(elt, ascii=False):

    # TODO: if there is a variable defined in this field and it is different
    # from the default, change the value and restart.
    
    scale = np.uint8(1)
    if elt.get("type") == "bitfield" and not ascii:
        current_type = ">u" + str(int(elt.get("length")) / 8)
        scale = np.dtype(current_type).type(1)
    elif(elt.get("length") is not None):
        if ascii:
            add = 33
        else:
            add = 0
        current_type = "S" + str(int(elt.get("length")) + add)
    else:
        current_type = TYPEC[elt.get("type")]
        try:
            scale = 10 / float(elt.get("scaling-factor", "10").replace("^", "e"))
        except ValueError:
            scale = 10 / np.array(elt.get("scaling-factor").replace("^", "e").split(","), dtype=np.float)
        
    return ((elt.get("name"), current_type, scale))


def process_array(elt, ascii=False):
    del ascii
    chld = elt.getchildren()
    if len(chld) > 1:
        print "stop"
        raise ValueError()
    chld = chld[0]
    try:
        name, current_type, scale = CASES[chld.tag](chld)
        size = None
    except ValueError:
        name, current_type, size, scale = CASES[chld.tag](chld)
    del name
    myname = elt.get("name") or elt.get("label")
    if elt.get("length").startswith("$"):
        length = int(VARIABLES[elt.get("length")[1:]])
    else:
        length = int(elt.get("length"))
    if size is not None:
        return (myname, current_type, (length, ) + size, scale)
    else:
        return (myname, current_type, (length, ), scale)

CASES = {"delimiter": process_delimiter,
         "field": process_field,
         "array": process_array,
         }

def to_dtype(val):
    """Parse *val* to return a dtype.
    """
    return np.dtype([i[:-1] for i in val])

def to_scaled_dtype(val):
    """Parse *val* to return a dtype.
    """
    res = []
    for i in val:
        if i[1].startswith("S"):
            res.append((i[0], i[1]) + i[2:-1])
        else:
            try:
                res.append((i[0], i[-1].dtype) + i[2:-1])
            except AttributeError:
                res.append((i[0], type(i[-1])) + i[2:-1])

    return np.dtype(res)
 

def to_scales(val):
    """Parse *val* to return an array of scale factors.
    """
    res = []
    for i in val:
        if len(i) == 3:
            res.append((i[0], type(i[2])))
        else:
            res.append((i[0], type(i[3]), i[2]))

    dtype = np.dtype(res)

    scales = np.zeros((1, ), dtype=dtype)

    for i in val:
        try:
            scales[i[0]] = i[-1]
        except ValueError:
            scales[i[0]] = np.repeat(np.array(i[-1]), i[2][1]).reshape(i[2])

    return scales


def parse_format(xml_file):

    global VARIABLES
    tree = ElementTree()
    tree.parse(xml_file)

    products = tree.find("product")


    params = tree.find("parameters")

    for param in params.getchildren():
        VARIABLES[param.get("name")] = param.get("value")


    types_scales = {}

    for prod in products:
        ascii = (prod.tag in ["mphr", "sphr"])
        res = []
        for i in prod:
            lres = CASES[i.tag](i, ascii)
            if lres is not None:
                res.append(lres)
        types_scales[(prod.tag, int(prod.get("subclass")))] = res


    types = {}
    stypes = {}
    scales = {}

    for key, val in types_scales.items():
        types[key] = to_dtype(val)
        stypes[key] = to_scaled_dtype(val)
        scales[key] = to_scales(val)

    return types, stypes, scales

def apply_scales(array, scales, dtype):
    new_array = np.empty(array.shape, dtype)
    #print dtype
    for i in array.dtype.names:
        try:
            new_array[i] = array[i] * scales[i]
        except TypeError:
            if np.all(scales[i] == 1):
                new_array[i] = array[i]
            else:
                raise
    return new_array

class XMLFormat(object):
    
    def __init__(self, filename):
        self.types, self.stypes, self.scales = parse_format(filename)

        self.translator = {}

        for key, val in self.types.items():
            self.translator[val] = (self.scales[key], self.stypes[key])

    def dtype(self, key):
        return self.types[key]

    def apply_scales(self, array):
        return apply_scales(array, *self.translator[array.dtype])

if __name__ == '__main__':
    TYPES, STYPES, SCALES = parse_format("eps_avhrrl1b_6.5.xml")

    mdr = np.ones((1, ), dtype=TYPES[("mphr", 2)])

    sca = SCALES[("mphr", 2)]

    res = apply_scales(mdr, sca, STYPES[("mphr", 2)])
    print res

#    print apply_scales(mdr.view(STYPES[("mdr", 2)], sca)

    # unit tests
    types = {('mphr', 0): [('PRODUCT_NAME', 'S100'), ('PARENT_PRODUCT_NAME_1', 'S100'), ('PARENT_PRODUCT_NAME_2', 'S100'), ('PARENT_PRODUCT_NAME_3', 'S100'), ('PARENT_PRODUCT_NAME_4', 'S100'), ('INSTRUMENT_ID', 'S37'), ('INSTRUMENT_MODEL', 'S36'), ('PRODUCT_TYPE', 'S36'), ('PROCESSING_LEVEL', 'S35'), ('SPACECRAFT_ID', 'S36'), ('SENSING_START', 'S48'), ('SENSING_END', 'S48'), ('SENSING_START_THEORETICAL', 'S48'), ('SENSING_END_THEORETICAL', 'S48'), ('PROCESSING_CENTRE', 'S37'), ('PROCESSOR_MAJOR_VERSION', 'S38'), ('PROCESSOR_MINOR_VERSION', 'S38'), ('FORMAT_MAJOR_VERSION', 'S38'), ('FORMAT_MINOR_VERSION', 'S38'), ('PROCESSING_TIME_START', 'S48'), ('PROCESSING_TIME_END', 'S48'), ('PROCESSING_MODE', 'S34'), ('DISPOSITION_MODE', 'S34'), ('RECEIVING_GROUND_STATION', 'S36'), ('RECEIVE_TIME_START', 'S48'), ('RECEIVE_TIME_END', 'S48'), ('ORBIT_START', 'S38'), ('ORBIT_END', 'S38'), ('ACTUAL_PRODUCT_SIZE', 'S44'), ('STATE_VECTOR_TIME', 'S51'), ('SEMI_MAJOR_AXIS', 'S44'), ('ECCENTRICITY', 'S44'), ('INCLINATION', 'S44'), ('PERIGEE_ARGUMENT', 'S44'), ('RIGHT_ASCENSION', 'S44'), ('MEAN_ANOMALY', 'S44'), ('X_POSITION', 'S44'), ('Y_POSITION', 'S44'), ('Z_POSITION', 'S44'), ('X_VELOCITY', 'S44'), ('Y_VELOCITY', 'S44'), ('Z_VELOCITY', 'S44'), ('EARTH_SUN_DISTANCE_RATIO', 'S44'), ('LOCATION_TOLERANCE_RADIAL', 'S44'), ('LOCATION_TOLERANCE_CROSSTRACK', 'S44'), ('LOCATION_TOLERANCE_ALONGTRACK', 'S44'), ('YAW_ERROR', 'S44'), ('ROLL_ERROR', 'S44'), ('PITCH_ERROR', 'S44'), ('SUBSAT_LATITUDE_START', 'S44'), ('SUBSAT_LONGITUDE_START', 'S44'), ('SUBSAT_LATITUDE_END', 'S44'), ('SUBSAT_LONGITUDE_END', 'S44'), ('LEAP_SECOND', 'S35'), ('LEAP_SECOND_UTC', 'S48'), ('TOTAL_RECORDS', 'S39'), ('TOTAL_MPHR', 'S39'), ('TOTAL_SPHR', 'S39'), ('TOTAL_IPR', 'S39'), ('TOTAL_GEADR', 'S39'), ('TOTAL_GIADR', 'S39'), ('TOTAL_VEADR', 'S39'), ('TOTAL_VIADR', 'S39'), ('TOTAL_MDR', 'S39'), ('COUNT_DEGRADED_INST_MDR', 'S39'), ('COUNT_DEGRADED_PROC_MDR', 'S39'), ('COUNT_DEGRADED_INST_MDR_BLOCKS', 'S39'), ('COUNT_DEGRADED_PROC_MDR_BLOCKS', 'S39'), ('DURATION_OF_PRODUCT', 'S41'), ('MILLISECONDS_OF_DATA_PRESENT', 'S41'), ('MILLISECONDS_OF_DATA_MISSING', 'S41'), ('SUBSETTED_PRODUCT', 'S34')], ('giadr', 2): [('PATCH_TEMPERATURE_COEFFICIENT1', '>i2'), ('PATCH_TEMPERATURE_COEFFICIENT2', '>i2'), ('PATCH_TEMPERATURE_COEFFICIENT3', '>i2'), ('PATCH_TEMPERATURE_COEFFICIENT4', '>i2'), ('PATCH_TEMPERATURE_COEFFICIENT5', '>i2'), ('PATCH_TEMPERATURE_EXTENDED_COEFFICIENT1', '>i2'), ('PATCH_TEMPERATURE_EXTENDED_COEFFICIENT2', '>i2'), ('PATCH_TEMPERATURE_EXTENDED_COEFFICIENT3', '>i2'), ('PATCH_TEMPERATURE_EXTENDED_COEFFICIENT4', '>i2'), ('PATCH_TEMPERATURE_EXTENDED_COEFFICIENT5', '>i2'), ('PATCH_POWER_COEFFICIENT1', '>i2'), ('PATCH_POWER_COEFFICIENT2', '>i2'), ('PATCH_POWER_COEFFICIENT3', '>i2'), ('PATCH_POWER_COEFFICIENT4', '>i2'), ('PATCH_POWER_COEFFICIENT5', '>i2'), ('RADIATOR_TEMPERATURE_COEFFICIENT1', '>i2'), ('RADIATOR_TEMPERATURE_COEFFICIENT2', '>i2'), ('RADIATOR_TEMPERATURE_COEFFICIENT3', '>i2'), ('RADIATOR_TEMPERATURE_COEFFICIENT4', '>i2'), ('RADIATOR_TEMPERATURE_COEFFICIENT5', '>i2'), ('BLACKBODY_TEMPERATURE1_COEFFICIENT1', '>i2'), ('BLACKBODY_TEMPERATURE1_COEFFICIENT2', '>i2'), ('BLACKBODY_TEMPERATURE1_COEFFICIENT3', '>i2'), ('BLACKBODY_TEMPERATURE1_COEFFICIENT4', '>i2'), ('BLACKBODY_TEMPERATURE1_COEFFICIENT5', '>i2'), ('BLACKBODY_TEMPERATURE2_COEFFICIENT1', '>i2'), ('BLACKBODY_TEMPERATURE2_COEFFICIENT2', '>i2'), ('BLACKBODY_TEMPERATURE2_COEFFICIENT3', '>i2'), ('BLACKBODY_TEMPERATURE2_COEFFICIENT4', '>i2'), ('BLACKBODY_TEMPERATURE2_COEFFICIENT5', '>i2'), ('BLACKBODY_TEMPERATURE3_COEFFICIENT1', '>i2'), ('BLACKBODY_TEMPERATURE3_COEFFICIENT2', '>i2'), ('BLACKBODY_TEMPERATURE3_COEFFICIENT3', '>i2'), ('BLACKBODY_TEMPERATURE3_COEFFICIENT4', '>i2'), ('BLACKBODY_TEMPERATURE3_COEFFICIENT5', '>i2'), ('BLACKBODY_TEMPERATURE4_COEFFICIENT1', '>i2'), ('BLACKBODY_TEMPERATURE4_COEFFICIENT2', '>i2'), ('BLACKBODY_TEMPERATURE4_COEFFICIENT3', '>i2'), ('BLACKBODY_TEMPERATURE4_COEFFICIENT4', '>i2'), ('BLACKBODY_TEMPERATURE4_COEFFICIENT5', '>i2'), ('ELECTRONIC_CURRENT_COEFFICIENT1', '>i2'), ('ELECTRONIC_CURRENT_COEFFICIENT2', '>i2'), ('ELECTRONIC_CURRENT_COEFFICIENT3', '>i2'), ('ELECTRONIC_CURRENT_COEFFICIENT4', '>i2'), ('ELECTRONIC_CURRENT_COEFFICIENT5', '>i2'), ('MOTOR_CURRENT_COEFFICIENT1', '>i2'), ('MOTOR_CURRENT_COEFFICIENT2', '>i2'), ('MOTOR_CURRENT_COEFFICIENT3', '>i2'), ('MOTOR_CURRENT_COEFFICIENT4', '>i2'), ('MOTOR_CURRENT_COEFFICIENT5', '>i2'), ('EARTH_SHIELD_POSITION_COEFFICIENT1', '>i2'), ('EARTH_SHIELD_POSITION_COEFFICIENT2', '>i2'), ('EARTH_SHIELD_POSITION_COEFFICIENT3', '>i2'), ('EARTH_SHIELD_POSITION_COEFFICIENT4', '>i2'), ('EARTH_SHIELD_POSITION_COEFFICIENT5', '>i2'), ('ELECTRONIC_TEMPERATURE_COEFFICIENT1', '>i2'), ('ELECTRONIC_TEMPERATURE_COEFFICIENT2', '>i2'), ('ELECTRONIC_TEMPERATURE_COEFFICIENT3', '>i2'), ('ELECTRONIC_TEMPERATURE_COEFFICIENT4', '>i2'), ('ELECTRONIC_TEMPERATURE_COEFFICIENT5', '>i2'), ('COOLER_HOUSING_TEMPERATURE_COEFFICIENT1', '>i2'), ('COOLER_HOUSING_TEMPERATURE_COEFFICIENT2', '>i2'), ('COOLER_HOUSING_TEMPERATURE_COEFFICIENT3', '>i2'), ('COOLER_HOUSING_TEMPERATURE_COEFFICIENT4', '>i2'), ('COOLER_HOUSING_TEMPERATURE_COEFFICIENT5', '>i2'), ('BASEPLATE_TEMPERATURE_COEFFICIENT1', '>i2'), ('BASEPLATE_TEMPERATURE_COEFFICIENT2', '>i2'), ('BASEPLATE_TEMPERATURE_COEFFICIENT3', '>i2'), ('BASEPLATE_TEMPERATURE_COEFFICIENT4', '>i2'), ('BASEPLATE_TEMPERATURE_COEFFICIENT5', '>i2'), ('MOTOR_HOUSING_TEMPERATURE_COEFFICIENT1', '>i2'), ('MOTOR_HOUSING_TEMPERATURE_COEFFICIENT2', '>i2'), ('MOTOR_HOUSING_TEMPERATURE_COEFFICIENT3', '>i2'), ('MOTOR_HOUSING_TEMPERATURE_COEFFICIENT4', '>i2'), ('MOTOR_HOUSING_TEMPERATURE_COEFFICIENT5', '>i2'), ('AD_CONVERTER_TEMPERATURE_COEFFICIENT1', '>i2'), ('AD_CONVERTER_TEMPERATURE_COEFFICIENT2', '>i2'), ('AD_CONVERTER_TEMPERATURE_COEFFICIENT3', '>i2'), ('AD_CONVERTER_TEMPERATURE_COEFFICIENT4', '>i2'), ('AD_CONVERTER_TEMPERATURE_COEFFICIENT5', '>i2'), ('DETECTOR4_BIAS_VOLTAGE_COEFFICIENT1', '>i2'), ('DETECTOR4_BIAS_VOLTAGE_COEFFICIENT2', '>i2'), ('DETECTOR4_BIAS_VOLTAGE_COEFFICIENT3', '>i2'), ('DETECTOR4_BIAS_VOLTAGE_COEFFICIENT4', '>i2'), ('DETECTOR4_BIAS_VOLTAGE_COEFFICIENT5', '>i2'), ('DETECTOR5_BIAS_VOLTAGE_COEFFICIENT1', '>i2'), ('DETECTOR5_BIAS_VOLTAGE_COEFFICIENT2', '>i2'), ('DETECTOR5_BIAS_VOLTAGE_COEFFICIENT3', '>i2'), ('DETECTOR5_BIAS_VOLTAGE_COEFFICIENT4', '>i2'), ('DETECTOR5_BIAS_VOLTAGE_COEFFICIENT5', '>i2'), ('CH3B_BLACKBODY_VIEW_COEFFICIENT1', '>i2'), ('CH3B_BLACKBODY_VIEW_COEFFICIENT2', '>i2'), ('CH3B_BLACKBODY_VIEW_COEFFICIENT3', '>i2'), ('CH3B_BLACKBODY_VIEW_COEFFICIENT4', '>i2'), ('CH3B_BLACKBODY_VIEW_COEFFICIENT5', '>i2'), ('CH4_BLACKBODY_VIEW_COEFFICIENT1', '>i2'), ('CH4_BLACKBODY_VIEW_COEFFICIENT2', '>i2'), ('CH4_BLACKBODY_VIEW_COEFFICIENT3', '>i2'), ('CH4_BLACKBODY_VIEW_COEFFICIENT4', '>i2'), ('CH4_BLACKBODY_VIEW_COEFFICIENT5', '>i2'), ('CH5_BLACKBODY_VIEW_COEFFICIENT1', '>i2'), ('CH5_BLACKBODY_VIEW_COEFFICIENT2', '>i2'), ('CH5_BLACKBODY_VIEW_COEFFICIENT3', '>i2'), ('CH5_BLACKBODY_VIEW_COEFFICIENT4', '>i2'), ('CH5_BLACKBODY_VIEW_COEFFICIENT5', '>i2'), ('REFERENCE_VOLTAGE_COEFFICIENT1', '>i2'), ('REFERENCE_VOLTAGE_COEFFICIENT2', '>i2'), ('REFERENCE_VOLTAGE_COEFFICIENT3', '>i2'), ('REFERENCE_VOLTAGE_COEFFICIENT4', '>i2'), ('REFERENCE_VOLTAGE_COEFFICIENT5', '>i2')], ('sphr', 0): [('SRC_DATA_QUAL', 'S49'), ('EARTH_VIEWS_PER_SCANLINE', 'S38'), ('NAV_SAMPLE_RATE', 'S36')], ('giadr', 1): [('RAMP_CALIBRATION_COEFFICIENT', '>u2'), ('YEAR_RECENT_CALIBRATION', '>u2'), ('DAY_RECENT_CALIBRATION', '>u2'), ('PRIMARY_CALIBRATION_ALGORITHM_ID', '>u2'), ('PRIMARY_CALIBRATION_ALGORITHM_OPTION', '>u2'), ('SECONDARY_CALIBRATION_ALGORITHM_ID', '>u2'), ('SECONDARY_CALIBRATION_ALGORITHM_OPTION', '>u2'), ('IR_TEMPERATURE1_COEFFICIENT1', '>i2'), ('IR_TEMPERATURE1_COEFFICIENT2', '>i2'), ('IR_TEMPERATURE1_COEFFICIENT3', '>i2'), ('IR_TEMPERATURE1_COEFFICIENT4', '>i2'), ('IR_TEMPERATURE1_COEFFICIENT5', '>i2'), ('IR_TEMPERATURE1_COEFFICIENT6', '>i2'), ('IR_TEMPERATURE2_COEFFICIENT1', '>i2'), ('IR_TEMPERATURE2_COEFFICIENT2', '>i2'), ('IR_TEMPERATURE2_COEFFICIENT3', '>i2'), ('IR_TEMPERATURE2_COEFFICIENT4', '>i2'), ('IR_TEMPERATURE2_COEFFICIENT5', '>i2'), ('IR_TEMPERATURE2_COEFFICIENT6', '>i2'), ('IR_TEMPERATURE3_COEFFICIENT1', '>i2'), ('IR_TEMPERATURE3_COEFFICIENT2', '>i2'), ('IR_TEMPERATURE3_COEFFICIENT3', '>i2'), ('IR_TEMPERATURE3_COEFFICIENT4', '>i2'), ('IR_TEMPERATURE3_COEFFICIENT5', '>i2'), ('IR_TEMPERATURE3_COEFFICIENT6', '>i2'), ('IR_TEMPERATURE4_COEFFICIENT1', '>i2'), ('IR_TEMPERATURE4_COEFFICIENT2', '>i2'), ('IR_TEMPERATURE4_COEFFICIENT3', '>i2'), ('IR_TEMPERATURE4_COEFFICIENT4', '>i2'), ('IR_TEMPERATURE4_COEFFICIENT5', '>i2'), ('IR_TEMPERATURE4_COEFFICIENT6', '>i2'), ('CH1_SOLAR_FILTERED_IRRADIANCE', '>i2'), ('CH1_EQUIVALENT_FILTER_WIDTH', '>i2'), ('CH2_SOLAR_FILTERED_IRRADIANCE', '>i2'), ('CH2_EQUIVALENT_FILTER_WIDTH', '>i2'), ('CH3A_SOLAR_FILTERED_IRRADIANCE', '>i2'), ('CH3A_EQUIVALENT_FILTER_WIDTH', '>i2'), ('CH3B_CENTRAL_WAVENUMBER', '>i4'), ('CH3B_CONSTANT1', '>i4'), ('CH3B_CONSTANT2_SLOPE', '>i4'), ('CH4_CENTRAL_WAVENUMBER', '>i4'), ('CH4_CONSTANT1', '>i4'), ('CH4_CONSTANT2_SLOPE', '>i4'), ('CH5_CENTRAL_WAVENUMBER', '>i4'), ('CH5_CONSTANT1', '>i4'), ('CH5_CONSTANT2_SLOPE', '>i4')], ('mdr', 2): [('DEGRADED_INST_MDR', '>i1'), ('DEGRADED_PROC_MDR', '>i1'), ('EARTH_VIEWS_PER_SCANLINE', '>i2'), ('SCENE_RADIANCES', '>i2', (5, 2048)), ('TIME_ATTITUDE', '>u4'), ('EULER_ANGLE', '>i2', (3,)), ('NAVIGATION_STATUS', '>u4'), ('SPACECRAFT_ALTITUDE', '>u4'), ('ANGULAR_RELATIONS_FIRST', '>i2', (4,)), ('ANGULAR_RELATIONS_LAST', '>i2', (4,)), ('EARTH_LOCATION_FIRST', '>i4', (2,)), ('EARTH_LOCATION_LAST', '>i4', (2,)), ('NUM_NAVIGATION_POINTS', '>i2'), ('ANGULAR_RELATIONS', '>i2', (103, 4)), ('EARTH_LOCATIONS', '>i4', (103, 2)), ('QUALITY_INDICATOR', '>u4'), ('SCAN_LINE_QUALITY', '>u4'), ('CALIBRATION_QUALITY', '>u2', (3,)), ('COUNT_ERROR_FRAME', '>u2'), ('CH123A_CURVE_SLOPE1', '>i4', (3,)), ('CH123A_CURVE_INTERCEPT1', '>i4', (3,)), ('CH123A_CURVE_SLOPE2', '>i4', (3,)), ('CH123A_CURVE_INTERCEPT2', '>i4', (3,)), ('CH123A_CURVE_INTERCEPTION', '>i4', (3,)), ('CH123A_TEST_CURVE_SLOPE1', '>i4', (3,)), ('CH123A_TEST_CURVE_INTERCEPT1', '>i4', (3,)), ('CH123A_TEST_CURVE_SLOPE2', '>i4', (3,)), ('CH123A_TEST_CURVE_INTERCEPT2', '>i4', (3,)), ('CH123A_TEST_CURVE_INTERCEPTION', '>i4', (3,)), ('CH123A_PRELAUNCH_CURVE_SLOPE1', '>i4', (3,)), ('CH123A_PRELAUNCH_CURVE_INTERCEPT1', '>i4', (3,)), ('CH123A_PRELAUNCH_CURVE_SLOPE2', '>i4', (3,)), ('CH123A_PRELAUNCH_CURVE_INTERCEPT2', '>i4', (3,)), ('CH123A_PRELAUNCH_CURVE_INTERCEPTION', '>i4', (3,)), ('CH3B45_SECOND_TERM', '>i4', (3,)), ('CH3B45_FIRST_TERM', '>i4', (3,)), ('CH3B45_ZEROTH_TERM', '>i4', (3,)), ('CH3B45_TEST_SECOND_TERM', '>i4', (3,)), ('CH3B45_TEST_FIRST_TERM', '>i4', (3,)), ('CH3B45_TEST_ZEROTH_TERM', '>i4', (3,)), ('CLOUD_INFORMATION', '>u2', (2048,)), ('FRAME_SYNCHRONISATION', '>u2', (6,)), ('FRAME_INDICATOR', '>u4'), ('TIME_CODE', '>u8'), ('RAMP_CALIB', '>u2', (5,)), ('INTERNAL_TARGET_TEMPERATURE_COUNT', '>u2', (3,)), ('INSTRUMENT_INVALID_WORD_FLAG', '>u2'), ('DIGITAL_B_DATA', '>u2'), ('INSTRUMENT_INVALID_ANALOG_WORD_FLAG', '>u4'), ('PATCH_TEMPERATURE', '>u2'), ('PATCH_EXTENDED_TEMPERATURE', '>u2'), ('PATCH_POWER', '>u2'), ('RADIATOR_TEMPERATURE', '>u2'), ('BLACKBODY_TEMPERATURE1', '>u2'), ('BLACKBODY_TEMPERATURE2', '>u2'), ('BLACKBODY_TEMPERATURE3', '>u2'), ('BLACKBODY_TEMPERATURE4', '>u2'), ('ELECTRONIC_CURRENT', '>u2'), ('MOTOR_CURRENT', '>u2'), ('EARTH_SHIELD_POSITION', '>u2'), ('ELECTRONIC_TEMPERATURE', '>u2'), ('COOLER_HOUSING_TEMPERATURE', '>u2'), ('BASEPLATE_TEMPERATURE', '>u2'), ('MOTOR_HOUSING_TEMPERATURE', '>u2'), ('AD_CONVERTER_TEMPERATURE', '>u2'), ('DETECTOR4_VOLTAGE', '>u2'), ('DETECTOR5_VOLTAGE', '>u2'), ('CH3_BLACKBODY_VIEW', '>u2'), ('CH4_BLACKBODY_VIEW', '>u2'), ('CH5_BLACKBODY_VIEW', '>u2'), ('REFERENCE_VOLTAGE', '>u2')]}

    #print newtypes == types

