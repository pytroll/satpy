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

#
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

"""FIXME DOC."""

import datetime
import os

import pytest

from satpy.readers import epsnative_reader

TEST_DATA_PATH = os.environ.get("EPCT_TEST_DATA_DIR", "")
TEST_DATA = os.path.join(
    TEST_DATA_PATH,
    "EPS",
    "AVHRRL1",
    "AVHR_xxx_1B_M02_20181212113403Z_20181212131303Z_N_O_20181212130905Z.nat",
)
PDUS_DATA = [
    os.path.join(
        TEST_DATA_PATH,
        "EPS",
        "AVHRRL1",
        "AVHR_xxx_1B_M01_20180120003103Z_20180120003403Z_N_O_20180120004248Z",
    ),
    os.path.join(
        TEST_DATA_PATH,
        "EPS",
        "AVHRRL1",
        "AVHR_xxx_1B_M01_20180120003403Z_20180120003703Z_N_O_20180120013050Z",
    ),
    os.path.join(
        TEST_DATA_PATH,
        "EPS",
        "AVHRRL1",
        "AVHR_xxx_1B_M01_20180120003703Z_20180120004003Z_N_O_20180120013202Z",
    ),
]

sample_file_str = pytest.mark.parametrize(
        "iasisndl2_file",
        ["string"],
        indirect=["iasisndl2_file"])


def test_get_class_tuple():
    """FIXME DOC."""
    class_string = "class_1_2_3"
    assert epsnative_reader.get_class_tuple(class_string) == ("class", 1, 2, 3)

    class_string = "class_dummy_dummy2"
    assert epsnative_reader.get_class_tuple(class_string) == ("class", "dummy_dummy2")


def test_assemble_descriptor():
    """FIXME DOC."""
    descriptor = epsnative_reader.assemble_descriptor("IASISNDL2")
    exp_keys = {('mphr', 0, 2), ('giadr', 1, 4), ('mdr', 1, 4)}

    assert set(descriptor.keys()) == exp_keys


@sample_file_str
def test_grh_reader(iasisndl2_file):
    """FIXME DOC."""
    grh = epsnative_reader.grh_reader(iasisndl2_file)
    assert len(grh) == 6
    assert grh[0] == "mphr"
    assert grh[1] == 0
    assert grh[2] == 2
    assert grh[3] == 3307
    assert grh[4] == datetime.datetime(2019, 6, 5, 0, 23, 52, 653000)
    assert grh[5] == datetime.datetime(2019, 6, 5, 2, 8, 56, 225000)


@sample_file_str
def test_mphr_reader(iasisndl2_file):
    """FIXME DOC."""
    mphr_content = epsnative_reader.mphr_reader(iasisndl2_file)

    assert len(mphr_content) == 72
    assert mphr_content["INSTRUMENT_ID"] == "IASI"
    assert mphr_content["ORBIT_START"] == 34826
    assert mphr_content["ORBIT_END"] == 34827


@sample_file_str
def test_first_class_occurrence(iasisndl2_file):
    """FIXME DOC."""
    class_name = "mphr"
    grh, offset = epsnative_reader.first_class_occurrence(iasisndl2_file, class_name)
    assert offset == 0

    class_name = "giadr"
    grh, offset = epsnative_reader.first_class_occurrence(iasisndl2_file, class_name)
    assert offset == 3361
    assert grh[0] == class_name
    assert grh[1] == 1
    assert grh[2] == 4

    class_name = "mdr"
    grh, offset = epsnative_reader.first_class_occurrence(iasisndl2_file, class_name)
    assert offset == 4818
    assert grh[0] == class_name
    assert grh[1] == 1
    assert grh[2] == 4


@sample_file_str
def test_read_ipr_sequence(iasisndl2_file):
    """FIXME DOC."""
    ipr_sequence = epsnative_reader.read_ipr_sequence(iasisndl2_file)

    assert len(ipr_sequence) == 2
    assert ipr_sequence[-1]["class"] == ("mdr", 1)
    assert ipr_sequence[-1]["offset"] == 4818


def test_reckon_dtype():
    """FIXME DOC."""
    assert epsnative_reader.reckon_dtype("boolean") == ">i1"
    assert epsnative_reader.reckon_dtype("u-integer1") == ">u1"
    assert epsnative_reader.reckon_dtype("vinteger2") == "byte,>i2"
    assert epsnative_reader.reckon_dtype("bitst(16)") == ">u2"
    assert epsnative_reader.reckon_dtype("bitst(24)") == ">u1,>u1,>u1"


def test_bands_to_records_reader():
    """FIXME DOC."""
    records = epsnative_reader.bands_to_records_reader("IASISNDL2")

    assert len(records) == 74
    assert records["atmospheric_temperature"]["record_name"] == "ATMOSPHERIC_TEMPERATURE"
    assert records["solar_azimuth"]["metadata"]["units"] == "degrees"
    assert records["surface_emissivity"]["metadata"]["scale_factor"] == 4
