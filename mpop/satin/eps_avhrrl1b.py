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

"""New reader for eps avhrr level 1b
"""
from datetime import datetime
from xml.etree.ElementTree import ElementTree

import numpy as np

from test_scale import fill_scales

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
    if elt.get("type") == "bitfield" and not ascii:
        current_type = ">u" + str(int(elt.get("length")) / 8)
    elif(elt.get("length") is not None):
        if ascii:
            add = 33
        else:
            add = 0
        current_type = "S" + str(int(elt.get("length")) + add)
    else:
        current_type = TYPEC[elt.get("type")]
    return ((elt.get("name"), current_type))


def process_array(elt, ascii=False):
    del ascii
    chld = elt.getchildren()
    if len(chld) > 1:
        print "stop"
        raise ValueError()
    chld = chld[0]
    try:
        name, current_type = CASES[chld.tag](chld)
        size = None
    except ValueError:
        name, current_type, size = CASES[chld.tag](chld)
    del name
    myname = elt.get("name") or elt.get("label")
    if elt.get("length").startswith("$"):
        length = int(VARIABLES[elt.get("length")[1:]])
    else:
        length = int(elt.get("length"))
    if size is not None:
        return (myname, current_type, (length, ) + size)
    else:
        return (myname, current_type, (length, ))

CASES = {"delimiter": process_delimiter,
         "field": process_field,
         "array": process_array,
         }


def parse_format(xml_file):

    global VARIABLES
    tree = ElementTree()
    tree.parse(xml_file)

    products = tree.find("product")


    params = tree.find("parameters")

    for param in params.getchildren():
        VARIABLES[param.get("name")] = param.get("value")


    types = {}

    for prod in products:
        ascii = (prod.tag in ["mphr", "sphr"])
        res = []
        for i in prod:
            lres = CASES[i.tag](i, ascii)
            if lres is not None:
                res.append(lres)
        types[(prod.tag, int(prod.get("subclass")))] = np.dtype(res)

    return types

def norm255(a):
    arr = a * 1.0
    arr = (arr - arr.min()) * 255.0 / (arr.max() - arr.min())
    return arr.astype(np.uint8)

def show(a):
    import Image
    print norm255(a).dtype
    Image.fromarray(norm255(a), "L").show()

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
    return arr * math.pi * 100.0 / solar_flux

if __name__ == '__main__':
    tic = datetime.now()



    types = parse_format("eps_avhrrl1b_6.5.xml")

    #filename = "AVHR_xxx_1B_M02_20120321100103Z_20120321100403Z_N_O_20120321105619Z"
    filename = "AVHR_xxx_1B_M02_20120321100403Z_20120321100703Z_N_O_20120321105847Z"
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
                record = np.fromfile(fdes,
                                     types[(rec_class,
                                            grh["RECORD_SUBCLASS"][0])],
                                     1)
                records.append((rec_class, record, grh["RECORD_SUBCLASS"][0]))
            except KeyError:
                fdes.seek(grh["RECORD_SIZE"] - 20, 1)
                
    mdrs = [record[1]
            for record in records
            if record[0] == "mdr"]

    scales = np.empty((1, ), dtype=mdrs[0].dtype)
    print scales
    fill_scales(scales, "eps_avhrrl1b_6.5.xml")
    mdrs = np.concatenate(mdrs)


    giadr = [record[1]
              for record in records
              if record[0] == "giadr" and record[2] == 1][0]

    print mdrs["SCENE_RADIANCES"].shape

    three_a = mdrs["FRAME_INDICATOR"] & 2**15
    print three_a.shape

    # filter 3a and calibrate

    channels = np.empty((mdrs["SCENE_RADIANCES"].shape[0], 6, mdrs["SCENE_RADIANCES"].shape[2]))
    channels[:, 0, :] = mdrs["SCENE_RADIANCES"][:, 0, :]
    channels[:, 1, :] = mdrs["SCENE_RADIANCES"][:, 1, :]
    print (giadr["CH4_CENTRAL_WAVENUMBER"],
                              giadr["CH4_CONSTANT1"],
                              giadr["CH4_CONSTANT2_SLOPE"])
    channels[:, 4, :] = to_bt(mdrs["SCENE_RADIANCES"][:, 3, :] / 100.0,
                              giadr["CH4_CENTRAL_WAVENUMBER"] * 1e-3,
                              giadr["CH4_CONSTANT1"] * 1e-5,
                              giadr["CH4_CONSTANT2_SLOPE"] * 1e-6)
    channels[:, 5, :] = mdrs["SCENE_RADIANCES"][:, 4, :]
    channels[three_a, 2, :] = mdrs["SCENE_RADIANCES"][three_a, 2, :]
    channels[three_a, 3, :] = mdrs["SCENE_RADIANCES"][1 - three_a, 2, :]

    
    
    toc = datetime.now()

    print toc - tic

    show(channels[:, 4, :])
