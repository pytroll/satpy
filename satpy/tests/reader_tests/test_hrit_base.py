#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""The HRIT base reader tests package."""

import bz2
import datetime as dt
import gzip
import os
from unittest import mock

import numpy as np
import pytest

from satpy.readers import FSFile
from satpy.readers.hrit_base import HRITFileHandler
from satpy.tests.utils import RANDOM_GEN

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


# From a compressed msg hrit file.
# uncompressed data field length 17223680
# compressed data field length 1578312
mda = {"file_type": 0, "total_header_length": 6198, "data_field_length": 17223680, "number_of_bits_per_pixel": 10,
       "number_of_columns": 3712, "number_of_lines": 464, "compression_flag_for_data": 0,
       "projection_name": b"GEOS(+000.0)                    ",
       "cfac": -13642337, "lfac": -13642337, "coff": 1856, "loff": 1856,
       "annotation_header": b"H-000-MSG4__-MSG4________-VIS006___-000001___-202208180730-C_",
       "cds_p_field": 64, "timestamp": (23605, 27911151), "GP_SC_ID": 324,
       "spectral_channel_id": 1,
       "segment_sequence_number": 1, "planned_start_segment_number": 1, "planned_end_segment_number": 8,
       "data_field_representation": 3,
       "image_segment_line_quality": np.array([(1, (0, 0), 1, 1, 0)] * 464,
                                              dtype=[("line_number_in_grid", ">i4"),
                                                     ("line_mean_acquisition", [("days", ">u2"),
                                                                                ("milliseconds", ">u4")]),
                                                     ("line_validity", "u1"),
                                                     ("line_radiometric_quality", "u1"),
                                                     ("line_geometric_quality", "u1")]),
       "projection_parameters": {"a": 6378169.0, "b": 6356583.8, "h": 35785831.0, "SSP_longitude": 0.0},
       "orbital_parameters": {}}

mda_compressed = mda.copy()
mda_compressed["data_field_length"] = 1578312
mda_compressed["compression_flag_for_data"] = 1


def new_get_hd(instance, hdr_info):
    """Generate some metadata."""
    if os.fspath(instance.filename).endswith(".C_"):
        instance.mda = mda_compressed.copy()
    else:
        instance.mda = mda.copy()


def new_get_hd_compressed(instance, hdr_info):
    """Generate some metadata."""
    instance.mda = mda.copy()
    instance.mda["compression_flag_for_data"] = 1
    instance.mda["data_field_length"] = 1578312


@pytest.fixture
def stub_hrit_file(tmp_path):
    """Create a stub hrit file."""
    filename = tmp_path / "some_hrit_file"
    create_stub_hrit(filename)
    return filename


def create_stub_hrit(filename, open_fun=open, meta=mda):
    """Create a stub hrit file."""
    stub_hrit_data = create_stub_hrit_data(meta)

    with open_fun(filename, mode="wb") as fd:
        fd.write(stub_hrit_data)
    return filename

def create_stub_hrit_data(meta):
    """Create the data for the stub hrit."""
    nbits = meta["number_of_bits_per_pixel"]
    lines = meta["number_of_lines"]
    cols = meta["number_of_columns"]
    total_bits = lines * cols * nbits
    arr = RANDOM_GEN.integers(0, 256,
                              size=int(total_bits / 8),
                              dtype=np.uint8)
    header_data = b" " * meta["total_header_length"]
    bytes_data = arr.tobytes()
    stub_hrit_data = header_data + bytes_data
    return stub_hrit_data


@pytest.fixture
def stub_bzipped_hrit_file(tmp_path):
    """Create a stub bzipped hrit file."""
    filename = tmp_path / "some_hrit_file.bz2"
    create_stub_hrit(filename, open_fun=bz2.open)
    return filename


@pytest.fixture
def stub_gzipped_hrit_file(tmp_path):
    """Create a stub gzipped hrit file."""
    filename = tmp_path / "some_hrit_file.gz"
    create_stub_hrit(filename, open_fun=gzip.open)
    return filename


@pytest.fixture
def stub_compressed_hrit_file(tmp_path):
    """Create a stub compressed hrit file."""
    filename = tmp_path / "some_hrit_file.C_"
    create_stub_hrit(filename, meta=mda_compressed)
    return filename


class TestHRITFileHandler:
    """Test the HRITFileHandler."""

    def setup_method(self, method):
        """Set up the hrit file handler for testing."""
        del method

        with mock.patch.object(HRITFileHandler, "_get_hd", new=new_get_hd):
            self.reader = HRITFileHandler("filename",
                                          {"platform_shortname": "MSG3",
                                           "start_time": dt.datetime(2016, 3, 3, 0, 0)},
                                          {"filetype": "info"},
                                          [mock.MagicMock(), mock.MagicMock(),
                                           mock.MagicMock()])

            self.reader.mda["cfac"] = 5
            self.reader.mda["lfac"] = 5
            self.reader.mda["coff"] = 10
            self.reader.mda["loff"] = 10
            self.reader.mda["projection_parameters"]["SSP_longitude"] = 44

    def test_get_xy_from_linecol(self):
        """Test get_xy_from_linecol."""
        x__, y__ = self.reader.get_xy_from_linecol(0, 0, (10, 10), (5, 5))
        assert -131072 == x__
        assert -131072 == y__
        x__, y__ = self.reader.get_xy_from_linecol(10, 10, (10, 10), (5, 5))
        assert x__ == 0
        assert y__ == 0
        x__, y__ = self.reader.get_xy_from_linecol(20, 20, (10, 10), (5, 5))
        assert 131072 == x__
        assert 131072 == y__

    def test_get_area_extent(self):
        """Test getting the area extent."""
        res = self.reader.get_area_extent((20, 20), (10, 10), (5, 5), 33)
        exp = (-71717.44995740513, -71717.44995740513,
               79266.655216079365, 79266.655216079365)
        assert res == exp

    def test_get_area_def(self):
        """Test getting an area definition."""
        from pyproj import CRS

        area = self.reader.get_area_def("VIS06")

        expected_crs = CRS(dict(proj="geos", a=6378169.0, b=6356583.8, h=35785831.0, lon_0=44.0, units="m"))
        assert area.crs == expected_crs

        assert area.area_extent == (-77771774058.38356, -77771774058.38356,
                                    30310525626438.438, 3720765401003.719)

    def test_read_band_filepath(self, stub_hrit_file):
        """Test reading a single band from a filepath."""
        self.reader.filename = stub_hrit_file

        res = self.reader.read_band("VIS006", None)
        assert res.compute().shape == (464, 3712)

    def test_read_band_FSFile(self, stub_hrit_file):
        """Test reading a single band from an FSFile."""
        import fsspec
        filename = stub_hrit_file

        fs_file = fsspec.open(filename)
        self.reader.filename = FSFile(fs_file)

        res = self.reader.read_band("VIS006", None)
        assert res.compute().shape == (464, 3712)

    def test_read_band_bzipped2_filepath(self, stub_bzipped_hrit_file):
        """Test reading a single band from a bzipped file."""
        self.reader.filename = stub_bzipped_hrit_file

        res = self.reader.read_band("VIS006", None)
        assert res.compute().shape == (464, 3712)

    def test_read_band_gzip_stream(self, stub_gzipped_hrit_file):
        """Test reading a single band from a gzip stream."""
        import fsspec
        filename = stub_gzipped_hrit_file

        fs_file = fsspec.open(filename, compression="gzip")
        self.reader.filename = FSFile(fs_file)

        res = self.reader.read_band("VIS006", None)
        assert res.compute().shape == (464, 3712)

    def test_start_end_time(self):
        """Test reading and converting start/end time."""
        assert self.reader.start_time == dt.datetime(2016, 3, 3, 0, 0)
        assert self.reader.start_time == self.reader.observation_start_time
        assert self.reader.end_time == dt.datetime(2016, 3, 3, 0, 0) + dt.timedelta(minutes=15)
        assert self.reader.end_time == self.reader.observation_end_time


def fake_decompress(filename):
    """Fake decompression."""
    del filename
    return create_stub_hrit_data(mda)


class TestHRITFileHandlerCompressed:
    """Test the HRITFileHandler with compressed segments."""

    def test_read_band_filepath(self, stub_compressed_hrit_file):
        """Test reading a single band from a filepath."""
        filename = stub_compressed_hrit_file

        with mock.patch("satpy.readers.hrit_base.decompress", side_effect=fake_decompress) as mock_decompress:
            with mock.patch.object(HRITFileHandler, "_get_hd", side_effect=new_get_hd, autospec=True) as get_hd:
                self.reader = HRITFileHandler(filename,
                                              {"platform_shortname": "MSG3",
                                               "start_time": dt.datetime(2016, 3, 3, 0, 0)},
                                              {"filetype": "info"},
                                              [mock.MagicMock(), mock.MagicMock(),
                                               mock.MagicMock()])

                res = self.reader.read_band("VIS006", None)
                assert get_hd.call_count == 1
                assert mock_decompress.call_count == 0
                assert res.compute().shape == (464, 3712)
                assert mock_decompress.call_count == 1
