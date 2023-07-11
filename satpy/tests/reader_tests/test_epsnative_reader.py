# Copyright 2017-2022, European Organisation for the Exploitation of Meteorological Satellites (EUMETSAT)
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

# -*- coding: utf-8 -*-
import os

from epct_plugin_gis import epsnative_reader


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

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data")
SAMPLE_DATA = os.path.join(SAMPLE_PATH, "AVHRRL1.nat")


def test_get_class_tuple():
    class_string = "class_1_2_3"
    assert epsnative_reader.get_class_tuple(class_string) == ("class", 1, 2, 3)

    class_string = "class_dummy_dummy2"
    assert epsnative_reader.get_class_tuple(class_string) == ("class", "dummy_dummy2")


def test_csv_list_by_product():
    csv_list = epsnative_reader.csv_list_by_product("AVHRRL1")
    exp_csv = ["giadr_1_3.csv", "giadr_2_2.csv", "mdr_2_5.csv", "mphr_0_2.csv"]

    assert len(csv_list) == 4
    for csv_path in csv_list:
        assert os.path.isfile(csv_path)
        assert os.path.basename(csv_path) in exp_csv


def test_assemble_descriptor():
    descriptor = epsnative_reader.assemble_descriptor("AVHRRL1")
    exp_keys = {("giadr", 1, 3), ("giadr", 2, 2), ("mdr", 2, 5), ("mphr", 0, 2)}

    assert set(descriptor.keys()) == exp_keys


def test_grh_reader():
    grh = epsnative_reader.grh_reader(SAMPLE_DATA)

    assert len(grh) == 6
    assert grh[0] == "mphr"
    assert grh[1] == 0
    assert grh[2] == 2
    assert grh[3] == 3307
    assert grh[4].isoformat() == "2018-01-22T13:28:03.130000"
    assert grh[5].isoformat() == "2018-01-22T13:28:04.797000"


def test_find_mphr_csv():
    mphr_csv = epsnative_reader.find_mphr_csv()

    assert os.path.isfile(mphr_csv)


def test_mphr_reader():
    mphr_content = epsnative_reader.mphr_reader(SAMPLE_DATA)

    assert len(mphr_content) == 72
    assert mphr_content["INSTRUMENT_ID"] == "AVHR"
    assert mphr_content["ORBIT_START"] == 58431
    assert mphr_content["ORBIT_END"] == 58432


def test_first_class_occurrence():
    class_name = "mphr"
    grh, offset = epsnative_reader.first_class_occurrence(SAMPLE_DATA, class_name)
    assert offset == 0

    class_name = "giadr"
    grh, offset = epsnative_reader.first_class_occurrence(SAMPLE_DATA, class_name)
    assert offset == 4587
    assert grh[0] == class_name
    assert grh[1] == 1
    assert grh[2] == 3

    class_name = "mdr"
    grh, offset = epsnative_reader.first_class_occurrence(SAMPLE_DATA, class_name)
    assert offset == 5077
    assert grh[0] == class_name
    assert grh[1] == 2
    assert grh[2] == 4


def test_read_ipr_sequence():
    ipr_sequence = epsnative_reader.read_ipr_sequence(SAMPLE_DATA)

    assert len(ipr_sequence) == 11
    assert ipr_sequence[-1]["class"] == ("mdr", 2)
    assert ipr_sequence[-1]["offset"] == 5077


def test_read_grh_of_target_class():
    target_offset = 0
    with open(SAMPLE_DATA, "rb") as eps_fileobj:
        target_grh = epsnative_reader.read_grh_of_target_class(eps_fileobj, target_offset)
    assert target_grh[0] == "mphr"
    assert target_grh[1] == 0
    assert target_grh[2] == 2
    assert target_grh[3] == 3307

    target_offset = 5077
    with open(SAMPLE_DATA, "rb") as eps_fileobj:
        target_grh = epsnative_reader.read_grh_of_target_class(eps_fileobj, target_offset)
    assert target_grh[0] == "mdr"
    assert target_grh[1] == 2
    assert target_grh[2] == 4
    assert target_grh[3] == 26660


def test_add_info_about_class():
    mphr = epsnative_reader.mphr_reader(SAMPLE_DATA)
    current_class = {"class": "", "offset": 5077}
    with open(SAMPLE_DATA, "rb") as eps_fileobj:
        current_class = epsnative_reader.add_info_about_class(eps_fileobj, current_class, {}, mphr)

    assert current_class["class_id"] == ("mdr", 2, 4)
    assert current_class["class_size"] == 26660
    assert current_class["nr_records"] == 10.0


def test_reckon_dtype():
    assert epsnative_reader.reckon_dtype("boolean") == ">i1"
    assert epsnative_reader.reckon_dtype("u-integer1") == ">u1"
    assert epsnative_reader.reckon_dtype("vinteger2") == "byte,>i2"
    assert epsnative_reader.reckon_dtype("bitst(16)") == ">u2"
    assert epsnative_reader.reckon_dtype("bitst(24)") == ">u1,>u1,>u1"


def test_bands_to_records_reader():
    records = epsnative_reader.bands_to_records_reader("AVHRRL1")

    assert len(records) == 7
    assert records["channel_5"]["record_name"] == "SCENE_RADIANCES"
    assert records["channel_5"]["band_position"] == 4
    assert records["channel_5"]["convention"] == "BIL"


def test_add_record_info_to_band():
    record_info = epsnative_reader.add_record_info_to_band("AVHRRL1")

    assert len(record_info) == 7
    assert record_info[0]["band_id"] == "channel_1"
    assert record_info[0]["shape"] == [2048, 5]
    assert record_info[-1]["band_id"] == "longitude"
    assert record_info[-1]["shape"] == [2, 103]


def test_create_toc():
    toc = epsnative_reader.create_toc(SAMPLE_DATA)

    assert len(toc) == 11
    assert toc[0]["class_id"] == ("geadr", 1, 1)
    assert toc[-1]["class_id"] == ("mdr", 2, 4)
    assert toc[-1]["y_offset"] == 0
