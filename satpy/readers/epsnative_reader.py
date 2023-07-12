# Copyright 2017-2022, European Organisation for the Exploitation of Meteorological Satellites (EUMETSAT)
# Copyright (c) 2023 Satpy developers

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

# This module is based on source code obtained from the
# epct_plugin_gis package developed by B-Open Solutions srl for EUMETSAT under
# contract EUM/C0/17/4600001943/0PN and released under Apache License
# Version 2.0, January 2004, http://www.apache.org/licenses/.  The original
# source including revision history and details on authorship can be found at
# https://gitlab.eumetsat.int/open-source/data-tailor-plugins/epct_plugin_gis

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FIXME DOC.

The ``epsnative_reader`` module provides the support to read from EPS native products records info
and metadata.
"""

import collections
import datetime
import os

import numpy as np
import pandas as pd
import yaml

from satpy._config import get_config_path

dtype_to_gdal_type = {
    ">i1": "Byte",
    ">i2": "Int16",
    ">i4": "Int32",
    ">u1": "Byte",
    ">u2": "UInt16",
    ">u4": "UInt32",
    ">f4": "Float32",
}
eps_classes = ["reserved", "mphr", "sphr", "ipr", "geadr", "giadr", "veadr", "viadr", "mdr"]
eps_type_to_dtype = {
    "boolean": ">i1",
    "ubyte": ">u1",
    "integer": ">i",
    "integer1": ">i1",
    "integer2": ">i2",
    "integer4": ">i4",
    "uinteger": ">u1",
    "uinteger1": ">u1",
    "uinteger2": ">u2",
    "uinteger4": ">u4",
    "vuinteger2": "byte,>u2",
    "vinteger2": "byte,>i2",
    "vinteger4": "byte,>i4",
    "short_cds_time": ">u2,>u4",
    "enumerated": ">i1",
    "bitst": ">ux",
}
grh_type = [
    ("RECORD_CLASS", "|i1"),
    ("INSTRUMENT_GROUP", "|i1"),
    ("RECORD_SUBCLASS", "|i1"),
    ("RECORD_SUBCLASS_VERSION", "|i1"),
    ("RECORD_SIZE", ">u4"),
    ("RECORD_START_TIME", eps_type_to_dtype["short_cds_time"]),
    ("RECORD_STOP_TIME", eps_type_to_dtype["short_cds_time"]),
]
ipr_type = [
    ("TARGET_RECORD_CLASS", "|i1"),
    ("TARGET_INSTRUMENT_GROUP", "|i1"),
    ("TARGET_RECORD_SUBCLASS", "|i1"),
    ("TARGET_RECORD_OFFSET", ">u4"),
]
NODATA_VALUE = -9999
UINTEGER1_BIT = 8
FORMAT_TO_MDR_CLASS = {
    "ASCATL1SZF": {13: ("mdr", 3, 5), 12: ("mdr", 3, 4), 11: ("mdr", 3, 4), 10: ("mdr", 3, 4)},
    "ASCATL1SZO": {13: ("mdr", 2, 4), 12: ("mdr", 2, 3), 11: ("mdr", 2, 3), 10: ("mdr", 2, 3)},
    "ASCATL1SZR": {13: ("mdr", 1, 4), 12: ("mdr", 1, 3), 11: ("mdr", 1, 3), 10: ("mdr", 1, 3)},
    "ASCATL2SMO": {12: ("mdr", 5, 1), 11: ("mdr", 5, 1), 10: ("mdr", 5, 0)},
    "ASCATL2SMR": {12: ("mdr", 4, 1), 11: ("mdr", 4, 1), 10: ("mdr", 4, 0)},
}


def get_class_tuple(class_string):
    """FIXME DOC.

    :param class_string:
    :return:
    """
    class_name = class_string.split("_")
    try:
        class_name[1:] = map(int, class_name[1:])
    except ValueError:
        class_name = (class_name[0], "_".join(class_name[1:]))
    return tuple(class_name)


def assemble_descriptor(product):
    """FIXME DOC.

    :param product:
    :return:
    """
    csv_list = [
            get_config_path("readers/eps_native_format/IASISND02/mdr_1_4.csv"),
            get_config_path("readers/eps_native_format/IASISND02/giadr_1_4.csv"),
            get_config_path("readers/eps_native_format/mphr_0_2.csv")]
    descriptor = {}
    for csv in csv_list:
        class_name = get_class_tuple(os.path.basename(csv).split(".")[0])
        ds = pd.read_csv(csv, sep=",")
        ds.columns = ds.columns.str.replace(" ", "_")
        descriptor[tuple(class_name)] = ds
    return descriptor


def scds_to_datetime(days, milliseconds):
    """FIXME DOC.

    :param int days:
    :param int milliseconds:
    :return datetime.datetime:
    """
    epoch = datetime.datetime(2000, 1, 1)
    return epoch + datetime.timedelta(days=int(days), milliseconds=int(milliseconds))


def grh_reader(eps_fileobj):
    """FIXME DOC.

    :param eps_fileobj:
    :return:
    """
    dtp = np.dtype(grh_type)
    # test for ndarray instead of memmap, because an empty slice from a memmap
    # becomes an ndarray, and this happens at EOF
    if isinstance(eps_fileobj, np.ndarray):
        grh_array = eps_fileobj[:dtp.itemsize].view(dtp)
    else:
        grh_array = np.fromfile(eps_fileobj, dtp, 1)
    # When the EOF is reached
    if grh_array.size == 0:
        return ()
    rec_class = eps_classes[grh_array["RECORD_CLASS"][0]]
    rec_subclass = grh_array["RECORD_SUBCLASS"][0]
    rec_subclass_version = grh_array["RECORD_SUBCLASS_VERSION"][0]
    rec_size = grh_array["RECORD_SIZE"][0]
    rec_start_time = scds_to_datetime(*grh_array["RECORD_START_TIME"][0])
    rec_stop_time = scds_to_datetime(*grh_array["RECORD_STOP_TIME"][0])
    return rec_class, rec_subclass, rec_subclass_version, rec_size, rec_start_time, rec_stop_time


def mphr_reader(input_product):
    """FIXME DOC.

    :param input_product:
    :return:
    """
    mphr_descriptor = pd.read_csv(
            get_config_path("readers/eps_native_format/mphr_0_2.csv"),
            sep=",")
    mphr_content = collections.OrderedDict()
    with open(input_product, "rb") as eps_fileobj:
        eps_fileobj.seek(20)
        for _, row in mphr_descriptor.iterrows():
            # empty row or general header
            if np.isnan(row["TYPE_SIZE"]) or row["FIELD"] == "RECORD_HEADER":
                continue
            else:
                dtype = f"S30,S2,S{int(row['TYPE_SIZE'])},S1"
                row_content = np.fromfile(eps_fileobj, dtype, 1)
                key = row_content[0][0].decode().strip()
                value = row_content[0][2].decode().strip(" x")
                if value.lstrip("-").isdigit():
                    value = int(value)
                    if np.isfinite(row["SF"]):
                        value /= 10 ** row["SF"]
                mphr_content[key] = value
    return mphr_content


def first_class_occurrence(input_product, class_name):
    """FIXME DOC.

    :param input_product:
    :param class_name:
    :return:

    """
    header_size = 20  # bytes
    offset = None
    with open(input_product, "rb") as eps_fileobj:
        while True:
            grh = grh_reader(eps_fileobj)
            if len(grh) == 0:
                break
            elif grh[0] != class_name:
                class_size = grh[3]
                eps_fileobj.seek(class_size - header_size, 1)
                continue
            else:
                offset = eps_fileobj.tell() - header_size
                break
    return grh, offset


def ipr_reader(eps_fileobj):
    """FIXME DOC.

    :param eps_fileobj:
    :return:
    """
    ipr_array = np.fromfile(eps_fileobj, np.dtype(ipr_type), 1)
    data = list(ipr_array[0])
    data[0] = eps_classes[data[0]]
    return data


def read_ipr_sequence(input_product):
    """FIXME DOC.

    :param input_product:
    :return:
    """
    dummy_id = 13
    _, ipr_offset = first_class_occurrence(input_product, "ipr")
    ipr_sequence = []
    with open(input_product, "rb") as eps_fileobj:
        eps_fileobj.seek(ipr_offset)
        while True:
            grh = grh_reader(eps_fileobj)
            if grh[0] != "ipr":
                break
            ipr_content = ipr_reader(eps_fileobj)
            info = {
                "class": (ipr_content[0], ipr_content[2]),
                "instrument": ipr_content[1],
                "offset": ipr_content[-1],
                "is_dummy": ipr_content[1] == dummy_id,
            }
            ipr_sequence.append(info)
    return ipr_sequence


def reckon_dtype(eps_type):
    """FIXME DOC.

    Convert EPS type into a suitable string which represents a ``numpy dtype`` object, according
    to whether the tag belongs to an ASCII record or not. For ASCII records see: "EPS Generic
    Product Format Specification", par. 4.3.1.

    :param str eps_type:
    :return string: string which represents a ``numpy dtype`` object
    """
    # remove "-" character, e.g. "u-integer2"
    eps_type = eps_type.replace("-", "")
    if eps_type.startswith("bitst"):
        eps_type, length = eps_type.replace("(", " ").split()
        length = int(length.split(")")[0])
        dtype = eps_type_to_dtype.get(eps_type)
        # if the length in bytes is an odd number the numpy dtype is a sequence of uinteger1
        if ((length // UINTEGER1_BIT) % 2 != 0) or (length // UINTEGER1_BIT) > UINTEGER1_BIT:
            dtype = ",".join([">u1"] * (length // UINTEGER1_BIT))
        else:
            dtype = dtype.replace("x", str(length // UINTEGER1_BIT))
    else:
        dtype = eps_type_to_dtype.get(eps_type)
    return dtype


def bands_to_records_reader(product):
    """FIXME DOC.

    :param product:
    :return:
    """
    band_to_record_path = get_config_path("readers/eps_native_format/IASISND02/band_to_record.yaml")
    with open(band_to_record_path, "rb") as fp:
        contents = fp.read()
    return yaml.safe_load(contents)
