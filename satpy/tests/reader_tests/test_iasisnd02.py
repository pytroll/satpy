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

"""Test reading IASI L2 SND."""

import datetime

import dask
import numpy as np

from ..utils import CustomScheduler


def test_read_giadr(sample_file):
    """Test reading GIADR."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_giadr
    descriptor = assemble_descriptor("IASISND02")
    class_data = read_giadr(sample_file, descriptor)

    assert len(class_data) == 19
    assert class_data["PRESSURE_LEVELS_OZONE"]["units"] == "Pa"
    assert class_data["PRESSURE_LEVELS_OZONE"]["values"].size == 101
    assert class_data["PRESSURE_LEVELS_OZONE"]["values"].min() == 50
    assert class_data["PRESSURE_LEVELS_OZONE"]["values"].max() == 11000000


def test_datetime_to_second_since_2000():
    """Test counting seconds since 2000."""
    from satpy.readers.iasi_l2_eps import datetime_to_second_since_2000
    date = datetime.datetime(2000, 1, 1, 0, 1)
    sec = datetime_to_second_since_2000(date)

    assert sec == 60


def test_read_all_rows(sample_file):
    """Test reading all rows from data."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_all_rows

    sensing_start = datetime.datetime.strptime("20190605002352Z", "%Y%m%d%H%M%SZ")
    sensing_stop = datetime.datetime.strptime("20190605020856Z", "%Y%m%d%H%M%SZ")
    descriptor = assemble_descriptor("IASISND02")

    mdr_class_offset = 271719118  # last row
    with open(sample_file, "rb") as epsfile_obj:
        data_before_errors_section, algorithms_data, errors_data = read_all_rows(
            epsfile_obj, descriptor, mdr_class_offset, sensing_start, sensing_stop
        )

    assert len(data_before_errors_section) == 1
    assert data_before_errors_section[0]["INTEGRATED_CO"].min() == 7205

    assert len(algorithms_data) == 1
    assert algorithms_data[0]["params"] == {"NERR": 69, "CO_NBR": 79, "HNO3_NBR": 0, "O3_NBR": 0}

    assert len(errors_data) == 1
    assert errors_data[0]["SURFACE_Z"].min() == 30


def test_read_product_data(sample_file):
    """Test reading product data."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_product_data
    descriptor = assemble_descriptor("IASISND02")
    mdr_class_offset = 260002007  # not quite the last row

    with open(sample_file, "rb") as epsfile_obj:
        (sdbs, sad, sed) = read_product_data(
                epsfile_obj,
                descriptor,
                mdr_class_offset,
                datetime.datetime(2019, 6, 5, 0, 23, 52),
                datetime.datetime(2019, 6, 5, 2, 8, 56))
    assert sdbs["SURFACE_TEMPERATURE"].shape == (37, 120)
    assert sdbs["ATMOSPHERIC_TEMPERATURE"].shape == (37, 101, 120)
    np.testing.assert_array_equal(
            sdbs["SURFACE_TEMPERATURE"][0, 58:64],
            np.array([27252, 27223, 27188, 27136, 27227, 65535], dtype=np.uint16))
    np.testing.assert_array_equal(
            sdbs["SURFACE_TEMPERATURE"][30, 80:86],
            np.array([27366, 65535, 27368, 27369, 27350, 27406], dtype=np.uint16))
    assert isinstance(sdbs["ATMOSPHERIC_TEMPERATURE"], dask.array.Array)


def test_load(sample_file, tmp_path):
    """Test read at a scene and write again."""
    from satpy import Scene
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        sc = Scene(filenames=[sample_file], reader=["iasi_l2_eps2"])
        sc.load(["surface_temperature"])
    assert sc["surface_temperature"].dims == ("y", "x")
    np.testing.assert_allclose(
            sc["surface_temperature"][0, 104:110],
            np.array([262.55, 261.83, 265.85, 265.81, 256.15, 255.23]))
    np.testing.assert_allclose(
            sc["surface_temperature"][1, 104:110],
            np.array([261.9, 266.88, 268.47, 266.76, 256.63, 261.32]))
    np.testing.assert_allclose(
            sc["surface_temperature"][2, 108:112],
            np.array([262.33, 265.42, 268.32, 266.21]))
    np.testing.assert_allclose(
            sc["surface_temperature"][10, 83:89],
            np.array([271.11, 280.07, 278.65, 278.84, 278.37, 281.54]))

    assert isinstance(sc["surface_temperature"], dask.array.Array)
