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
import pandas as pd
import pyresample
import pytest

from ..utils import CustomScheduler

sample_file_str = pytest.mark.parametrize(
        "iasisndl2_file",
        ["string"],
        indirect=["iasisndl2_file"])

sample_file_file = pytest.mark.parametrize(
        "iasisndl2_file",
        ["file"],
        indirect=["iasisndl2_file"])

sample_file_mmap = pytest.mark.parametrize(
        "iasisndl2_file",
        ["mmap"],
        indirect=["iasisndl2_file"])


@sample_file_str
def test_read_giadr(iasisndl2_file):
    """Test reading GIADR."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_giadr
    descriptor = assemble_descriptor("IASISND02")
    class_data = read_giadr(iasisndl2_file, descriptor)

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


@sample_file_mmap
def test_read_all_rows(iasisndl2_file):
    """Test reading all rows from data."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_all_rows

    sensing_start = datetime.datetime.strptime("20190605002352Z", "%Y%m%d%H%M%SZ")
    sensing_stop = datetime.datetime.strptime("20190605020856Z", "%Y%m%d%H%M%SZ")
    descriptor = assemble_descriptor("IASISND02")

    mdr_class_offset = 271719118  # last row
    data_before_errors_section, algorithms_data, errors_data = read_all_rows(
        iasisndl2_file, descriptor, mdr_class_offset, sensing_start, sensing_stop
    )

    assert len(data_before_errors_section) == 1
    assert data_before_errors_section[0]["INTEGRATED_CO"].min() == 7205

    assert len(algorithms_data) == 1
    assert algorithms_data[0]["params"] == {"NERR": 69, "CO_NBR": 79, "HNO3_NBR": 0, "O3_NBR": 0}

    assert len(errors_data) == 1
    assert errors_data[0]["SURFACE_Z"].min() == 30

    np.testing.assert_array_equal(
            data_before_errors_section[0]["ATMOSPHERIC_TEMPERATURE"][50, 50:55],
            np.array([22859, 22844, 22848, 22816, 22812], dtype=np.uint16))


@sample_file_mmap
def test_read_nerr_values(iasisndl2_file):
    """Test reading the number of uncertainty values."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_nerr_values

    descriptor = assemble_descriptor("IASISND02")[("mdr", 1, 4)]
    nerr = read_nerr_values(iasisndl2_file, descriptor, 271719118)
    np.testing.assert_array_equal(nerr, [69])


@sample_file_mmap
def test_read_values(iasisndl2_file):
    """Test reading values from a mmap."""
    from satpy.readers.iasi_l2_eps import read_values

    row = pd.Series(
            np.array(
                ['NERR', 'Number of error data records for current scan line',
                 0.0, np.nan, 1, '1', 1, 'u-byte', 1, 1.0, 207747.0],
                dtype=object),
            index=pd.Index(
                ['FIELD', 'DESCRIPTION', 'SF', 'UNITS', 'DIM1', 'DIM2', 'DIM3',
                 'TYPE', 'TYPE_SIZE', 'FIELD_SIZE', 'OFFSET'],
                dtype=object),
            name=52)
    try:
        iasisndl2_file.seek(271926865)
    except AttributeError:
        iasisndl2_file = iasisndl2_file[271926865:]

    vals = read_values(iasisndl2_file, row, False)
    np.testing.assert_array_equal(vals, [69])


@sample_file_mmap
def test_read_records_before_error_section(iasisndl2_file):
    """Test reading records before the error section."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_records_before_error_section

    descriptor = assemble_descriptor("IASISND02")[("mdr", 1, 4)]
    data = read_records_before_error_section(
        iasisndl2_file,
        descriptor[1:54],
        271719118)
    np.testing.assert_array_equal(
            data["SURFACE_TEMPERATURE"][48:55],
            np.array([29209, 29276, 29386, 29220, 29266, 29568, 29302],
                     dtype=np.uint16))
    np.testing.assert_array_equal(
            data["ATMOSPHERIC_TEMPERATURE"][20:25, 50],
            np.array([23917, 23741, 23583, 23436, 23290],
                     dtype=np.uint16))


@sample_file_file
def test_read_product_data(iasisndl2_file):
    """Test reading product data."""
    from satpy.readers.epsnative_reader import assemble_descriptor
    from satpy.readers.iasi_l2_eps import read_product_data
    descriptor = assemble_descriptor("IASISND02")
    mdr_class_offset = 260002007  # not quite the last row

    (sdbs, sad, sed) = read_product_data(
            iasisndl2_file,
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


@sample_file_str
def test_load(iasisndl2_file, tmp_path):
    """Test read at a scene and write again."""
    from satpy import Scene
    with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
        sc = Scene(filenames=[iasisndl2_file], reader=["iasi_l2_eps"])
        sc.load(["surface_temperature", "atmospheric_temperature",
                 "atmospheric_water_vapour", "pressure_levels_temp"])
    assert sc["surface_temperature"].dims == ("y", "x")
    np.testing.assert_allclose(
            sc["surface_temperature"][0, 100:104],
            np.array([270.39, 270.21, 269.65, 269.66]))
    np.testing.assert_allclose(
            sc["surface_temperature"][1, 106:110],
            np.array([269.91, 270.07, 270.02, 270.41]))
    np.testing.assert_allclose(
            sc["surface_temperature"][30, 60:66],
            np.array([282.27, 283.18, 285.67, 282.98, 282.81, 282.9]))
    np.testing.assert_array_equal(
            sc["surface_temperature"][30, :5],
            [np.nan]*5)
    np.testing.assert_array_equal(
            sc["atmospheric_water_vapour"][0, 0, :5],
            [np.nan]*5)

    assert isinstance(sc["surface_temperature"].data, dask.array.Array)
    assert isinstance(sc["surface_temperature"].attrs["area"],
                      pyresample.SwathDefinition)
    assert sc["surface_temperature"].attrs["standard_name"] == "surface_temperature"
    assert sc["surface_temperature"].attrs["units"] == "K"
    assert "area" not in sc["pressure_levels_temp"].attrs
