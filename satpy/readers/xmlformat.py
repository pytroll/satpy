#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2012 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Reads a format from an xml file to create dtypes and scaling factor arrays."""

from __future__ import annotations

from xml.etree.ElementTree import ElementTree

import numpy as np

VARIABLES: dict[str, str] = {}

TYPEC = {"boolean": ">i1",
         "integer2": ">i2",
         "integer4": ">i4",
         "uinteger2": ">u2",
         "uinteger4": ">u4", }


def process_delimiter(elt, ascii=False):
    """Process a 'delimiter' tag."""
    del elt, ascii


def process_field(elt, ascii=False):
    """Process a 'field' tag."""
    # NOTE: if there is a variable defined in this field and it is different
    # from the default, we could change the value and restart.

    scale = np.uint8(1)
    if elt.get("type") == "bitfield" and not ascii:
        current_type = ">u" + str(int(elt.get("length")) // 8)
        scale = np.dtype(current_type).type(1)
    elif (elt.get("length") is not None):
        if ascii:
            add = 33
        else:
            add = 0
        current_type = "S" + str(int(elt.get("length")) + add)
    else:
        current_type = TYPEC[elt.get("type")]
        try:
            scale = (10 /
                     float(elt.get("scaling-factor", "10").replace("^", "e")))
        except ValueError:
            scale = (10 / np.array(
                elt.get("scaling-factor").replace("^", "e").split(","),
                dtype=np.float64))

    return ((elt.get("name"), current_type, scale))


def process_array(elt, ascii=False):
    """Process an 'array' tag."""
    del ascii
    chld = list(elt)
    if len(chld) > 1:
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
         "array": process_array, }


def to_dtype(val):
    """Parse *val* to return a dtype."""
    return np.dtype([i[:-1] for i in val])


def to_scaled_dtype(val):
    """Parse *val* to return a dtype."""
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
    """Parse *val* to return an array of scale factors."""
    res = []
    for i in val:
        if len(i) == 3:
            res.append((i[0], type(i[2])))
        else:
            try:
                res.append((i[0], i[3].dtype, i[2]))
            except AttributeError:
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
    """Parse the xml file to create types, scaling factor types, and scales."""
    tree = ElementTree()
    tree.parse(xml_file)
    for param in tree.find("parameters"):
        VARIABLES[param.get("name")] = param.get("value")

    types_scales = {}

    for prod in tree.find("product"):
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


def _apply_scales(array, scales, dtype):
    """Apply scales to the array."""
    new_array = np.empty(array.shape, dtype)
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
    """XMLFormat object."""

    def __init__(self, filename):
        """Init the format reader."""
        self.types, self.stypes, self.scales = parse_format(filename)

        self.translator = {}

        for key, val in self.types.items():
            self.translator[val] = (self.scales[key], self.stypes[key])

    def dtype(self, key):
        """Get the dtype for the format object."""
        return self.types[key]

    def apply_scales(self, array):
        """Apply scales to *array*."""
        return _apply_scales(array, *self.translator[array.dtype])


if __name__ == '__main__':
    pass
