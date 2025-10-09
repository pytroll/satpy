#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""The hrit ahi reader tests package."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from unittest import mock

import dask.array as da
import numpy as np
import pytest
from xarray import DataArray

from satpy.readers.hrit_jma import (
    AREA_NAMES,
    FULL_DISK,
    HIMAWARI8,
    NORTH_HEMIS,
    PLATFORMS,
    SOUTH_HEMIS,
    UNKNOWN_AREA,
    UNKNOWN_PLATFORM,
    HRITJMAFileHandler,
)
from satpy.tests.utils import make_dataid


def create_fake_ahi_hrit(hrit_path: Path, metadata_overrides: dict | None = None) -> None:
    """Create a fake AHI HRIT file on disk."""
    num_rows = 11000  # 1km
    num_cols = 11000  # 1km
    coff_loffs = f"LINE:=1\rCOFF:={num_cols / 2}\rLOFF:={num_rows / 2}"
    for lnum in range(1000, num_rows + 1, 1000):
        coff_loffs += f"LINE:={lnum}\rCOFF:={num_cols / 2}\rLOFF:={num_rows / 2}"
        if lnum != num_rows:
            coff_loffs += f"LINE:={lnum + 1}\rCOFF:={num_cols / 2}\rLOFF:={num_rows / 2}"
    coff_loffs_bytes = coff_loffs.encode()
    acq_times_bytes = _get_acq_time(num_rows)


    with hrit_path.open(mode="wb") as fp:
        header_data = (
            # header 0
            np.void((0, 16), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            # 4km ->np.void((0, 1841, 121000000),
            np.void((0, 2219, 1936000000),
                    dtype=[("file_type", "u1"), ("total_header_length", ">u4"), ("data_field_length", ">u8")]),

            # header 1
            np.void((1, 9), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            # np.void((16, 2750, 2750, 0),
            np.void((16, num_cols, num_rows, 0),
                    dtype=[("number_of_bits_per_pixel", "u1"), ("number_of_columns", ">u2"), ("number_of_lines", ">u2"),
                           ("compression_flag_for_data", "u1")]),

            # header 2
            np.void((2, 51), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.void((b"GEOS(140.70)                    ", 40932549, 40932549, 5500, 5500),
                    dtype=[("projection_name", "S32"),
                           ("cfac", ">i4"), ("lfac", ">i4"),
                           ("coff", ">i4"), ("loff", ">i4")]),

            # header 3
            np.void((3, 85), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.array(b"$HALFTONE:=16\r_NAME:=VISIBLE\r_UNIT:=ALBEDO(%)\r0:=-0.10\r1023:=100.00\r65535:=100.00\r",
                     dtype="|S82"),

            # header 4
            np.void((4, 27), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.array(b"IMG_DK01VIS_201809100300", dtype="|S24"),

            # header 5
            np.void((5, 10), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.void((64, (22167, 11162999)),
                    dtype=[("cds_p_field", "u1"), ("timestamp", [("Days", ">u2"), ("Milliseconds", ">u4")])]),

            # header 128
            np.void((128, 7), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.void((0, 1, 1),
                    dtype=[("image_segm_seq_no", "u1"), ("total_no_image_segm", "u1"), ("line_no_image_segm", ">u2")]),

            # header 130
            np.void((130, len(coff_loffs_bytes) + 3), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.bytes_(coff_loffs_bytes),

            # header 131
            np.void((131, len(acq_times_bytes) + 3), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.bytes_(acq_times_bytes),

            # header 132
            np.void((132, 12), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
            np.array(b"NO_ERROR\r", dtype="|S9"),
        )
        for header_arr in header_data:
            if metadata_overrides and header_arr.dtype.fields is not None:
                for key in header_arr.dtype.fields:
                    if key in metadata_overrides:
                        header_arr[key] = metadata_overrides[key]
            header_arr.tofile(fp)


@mock.patch("satpy.readers.hrit_jma.HRITFileHandler.__init__")
def _get_reader(mocked_init, mda, filename_info=None, filetype_info=None, reader_kwargs=None):
    if not filename_info:
        filename_info = {}
    if not filetype_info:
        filetype_info = {}
    if not reader_kwargs:
        reader_kwargs = {}
    HRITJMAFileHandler.filename = "filename"
    HRITJMAFileHandler.mda = mda
    HRITJMAFileHandler._start_time = filename_info.get("start_time")
    return HRITJMAFileHandler("filename", filename_info, filetype_info, **reader_kwargs)


def _get_acq_time(nlines):
    """Get sample header entry for scanline acquisition times.

    Lines: 1, 21, 41, 61, ..., nlines
    Times: 1970-01-01 00:00 + (1, 21, 41, 61, ..., nlines) seconds

    So the interpolated times are expected to be 1970-01-01 +
    (1, 2, 3, 4, ..., nlines) seconds. Note that there will be some
    floating point inaccuracies, because timestamps are stored
    with only 6 decimals precision.
    """
    # AHI:
    # 11000 == 1km
    # 2750 == 4km
    mjd_1970 = 40587.0
    lines_sparse = np.array(list(range(1, nlines, 20)) + [nlines])
    times_sparse = mjd_1970 + lines_sparse / 24 / 3600
    acq_time_s = ["LINE:={}\rTIME:={:.6f}\r".format(line, time)
                  for line, time in zip(lines_sparse, times_sparse)]
    acq_time_b = "".join(acq_time_s).encode()
    return acq_time_b


def _get_mda(loff=5500.0, coff=5500.0, nlines=11000, ncols=11000,
             segno=0, numseg=1, vis=True, platform="Himawari-8"):
    """Create metadata dict like HRITFileHandler would do it."""
    if vis:
        idf = b"$HALFTONE:=16\r_NAME:=VISIBLE\r_UNIT:=ALBEDO(%)\r" \
              b"0:=-0.10\r1023:=100.00\r65535:=100.00\r"
    else:
        idf = b"$HALFTONE:=16\r_NAME:=INFRARED\r_UNIT:=KELVIN\r" \
              b"0:=329.98\r1023:=130.02\r65535:=130.02\r"
    proj_h8 = b"GEOS(140.70)                    "
    proj_mtsat2 = b"GEOS(145.00)                    "
    proj_name = proj_h8 if platform == "Himawari-8" else proj_mtsat2
    return {"image_segm_seq_no": np.uint8(segno),
            "total_no_image_segm": np.uint8(numseg),
            "projection_name": np.bytes_(proj_name),
            "projection_parameters": {
                "a": 6378169.00,
                "b": 6356583.80,
                "h": 35785831.00,
            },
            "cfac": np.int32(10233128),
            "lfac": np.int32(10233128),
            "coff": np.int32(coff),
            "loff": np.int32(loff),
            "number_of_columns": np.uint16(ncols),
            "number_of_lines": np.uint16(nlines),
            "image_data_function": np.bytes_(idf),
            "image_observation_time": np.bytes_(_get_acq_time(nlines)),
            }


def test_init(tmp_path):
    """Test creating the file handler."""
    mda = _get_mda()
    mda_expected = mda.copy()
    mda_expected.update(
        {"planned_end_segment_number": np.uint8(1),
         "planned_start_segment_number": np.uint8(1),
         "segment_sequence_number": np.uint8(0),
         "unit": "ALBEDO(%)"})
    mda_expected["projection_parameters"]["SSP_longitude"] = 140.7

    hrit_path = tmp_path / "IMG_DK01B04_202509261940_001"
    create_fake_ahi_hrit(hrit_path, metadata_overrides=mda)

    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})

    # Test addition of extra metadata
    # expected dict doesn"t have every possible key so we only check what is expected
    for mda_key, exp_val in mda_expected.items():
        assert reader.mda[mda_key] == exp_val

    # Check projection name
    assert reader.projection_name == "GEOS(140.70)"

    # Check calibration table
    cal_expected = np.array([[0, -0.1],
                             [1023,  100],
                             [65535,  100]])
    assert np.all(reader.calibration_table == cal_expected)

    # Check if scanline timestamps are there (dedicated test below)
    assert isinstance(reader.acq_time, np.ndarray)
    assert reader.acq_time.dtype == np.dtype("datetime64[ns]")

    # Check platform
    assert reader.platform == HIMAWARI8


@pytest.mark.parametrize(
    ("segno", "is_segmented"),
    [
        (0, False),
        (1, True),
        (8, True),
    ]
)
def test_segmented_checks(segno, is_segmented):
    """Test segments are identified."""
    mda = _get_mda(segno=segno)
    reader = _get_reader(mda=mda)
    assert reader.is_segmented == is_segmented


@pytest.mark.parametrize(
    ("filename_info", "area_id"),
    [
        ({"area": 1}, 1),
        ({"area": 1234}, UNKNOWN_AREA),
        ({}, UNKNOWN_AREA)
    ]
)
def test_check_areas(filename_info, area_id):
    """Test area names coming from the filename."""
    mda = _get_mda()
    reader = _get_reader(mda=mda, filename_info=filename_info)
    assert reader.area_id == area_id


@pytest.mark.parametrize(("proj_name", "platform"), list(PLATFORMS.items()) + [("invalid", UNKNOWN_PLATFORM)])
def test_get_platform(proj_name, platform, caplog):
    """Test platform identification."""
    with mock.patch("satpy.readers.hrit_jma.HRITJMAFileHandler.__init__") as mocked_init:
        mocked_init.return_value = None
        reader = HRITJMAFileHandler()

    reader.projection_name = proj_name
    assert reader._get_platform() == platform

    if proj_name == "invalid":
        assert "Unable to determine platform" in caplog.text


@pytest.mark.parametrize(
    ("mda_info", "area_name", "extent"),
    [
        # Non-segmented, full disk
        (
            {
                "loff": 1375.0, "coff": 1375.0,
                "nlines": 2750, "ncols": 2750,
                "segno": 0, "numseg": 1,
            },
            FULL_DISK,
            (-5498000.088960204, -5498000.088960204, 5502000.089024927, 5502000.089024927),
        ),
        # Non-segmented, northern hemisphere
        (
            {
                "loff": 1325.0, "coff": 1375.0,
                "nlines": 1375, "ncols": 2750,
                "segno": 0, "numseg": 1,
            },
            NORTH_HEMIS,
            (-5498000.088960204, -198000.00320373234, 5502000.089024927, 5302000.085788833),
        ),
        # Non-segmented, southern hemisphere
        (
            {
                "loff": 50, "coff": 1375.0,
                "nlines": 1375, "ncols": 2750,
                "segno": 0, "numseg": 1,
            },
            SOUTH_HEMIS,
            (-5498000.088960204, -5298000.085724112, 5502000.089024927, 202000.0032684542),
        ),
        # Segmented, segment #1
        (
            {
                "loff": 1375.0, "coff": 1375.0,
                "nlines": 275, "ncols": 2750,
                "segno": 1, "numseg": 10,
            },
            FULL_DISK,
            (-5498000.088960204, 4402000.071226413, 5502000.089024927, 5502000.089024927),
        ),
        # Segmented, segment #7
        (
            {
                "loff": 1375.0, "coff": 1375.0,
                "nlines": 275, "ncols": 2750,
                "segno": 7, "numseg": 10,
            },
            FULL_DISK,
            (-5498000.088960204, -2198000.035564665, 5502000.089024927, -1098000.0177661523),
        ),
    ]
)
def test_get_area_def(mda_info, area_name, extent):
    """Test getting an AreaDefinition."""
    mda = _get_mda(**mda_info)
    reader = _get_reader(mda=mda, filename_info={"area": area_name})
    area = reader.get_area_def("some_id")
    assert area.area_extent == extent
    assert area.description == AREA_NAMES[area_name]["long"]


@pytest.mark.parametrize("calibration", ["counts", "reflectance", "brightness_temperature"])
def test_calibrate(calibration):
    """Test calibration."""
    counts = np.linspace(0, 1200, 25).reshape(5, 5)
    counts[-1, -1] = 65535
    counts = DataArray(da.from_array(counts, chunks=5))
    refl = np.array(
        [[-0.1,            4.79247312,   9.68494624,  14.57741935,  19.46989247],
         [24.36236559,  29.25483871,  34.14731183,  39.03978495,  43.93225806],
         [48.82473118,  53.7172043,   58.60967742,  63.50215054,  68.39462366],
         [73.28709677,  78.17956989,  83.07204301,  87.96451613,  92.85698925],
         [97.74946237,  100.,         100.,         100.,         np.nan]]
    )
    bt = np.array(
        [[329.98,            320.20678397, 310.43356794, 300.66035191, 290.88713587],
         [281.11391984, 271.34070381, 261.56748778, 251.79427175, 242.02105572],
         [232.24783969, 222.47462366, 212.70140762, 202.92819159, 193.15497556],
         [183.38175953, 173.6085435,  163.83532747, 154.06211144, 144.28889541],
         [134.51567937, 130.02,       130.02,       130.02,       np.nan]]
    )

    # Choose an area near the subsatellite point to avoid masking of space pixels
    mda = _get_mda(nlines=5, ncols=5, loff=1375.0, coff=1375.0, segno=0, vis=calibration == "reflectance")
    reader = _get_reader(mda=mda)

    res = reader.calibrate(data=counts, calibration=calibration)
    exp = {
        "counts": counts.values,
        "reflectance": refl,
        "brightness_temperature": bt,
    }
    np.testing.assert_allclose(exp[calibration], res.values)


def test_mask_space():
    """Test masking of space pixels."""
    mda = _get_mda(loff=1375.0, coff=1375.0, nlines=275, ncols=1375, segno=1, numseg=10)
    reader = _get_reader(mda=mda)
    data = DataArray(da.ones((275, 1375), chunks=1024))
    masked = reader._mask_space(data)

    # First line of the segment should be space, in the middle of the
    # last line there should be some valid pixels
    np.testing.assert_allclose(masked.values[0, :], np.nan)
    np.testing.assert_array_equal(masked.values[-1, 588:788], 1)


@mock.patch("satpy.readers.hrit_jma.HRITFileHandler.get_dataset")
def test_get_dataset(base_get_dataset):
    """Test getting a dataset."""
    mda = _get_mda(loff=1375.0, coff=1375.0, nlines=275, ncols=1375, segno=1, numseg=10)
    reader = _get_reader(mda=mda)
    key = make_dataid(name="VIS", calibration="reflectance")

    base_get_dataset.return_value = DataArray(da.ones((275, 1375),
                                                      chunks=1024),
                                              dims=("y", "x"))

    # Check attributes
    res = reader.get_dataset(key, {"units": "%", "sensor": "ahi"})
    assert res.attrs["units"] == "%"
    assert res.attrs["sensor"] == "ahi"
    assert res.attrs["platform_name"] == HIMAWARI8
    assert res.attrs["orbital_parameters"] == {"projection_longitude": 140.7,
                                               "projection_latitude": 0.0,
                                               "projection_altitude": 35785831.0}

    # Check if acquisition time is a coordinate
    assert "acq_time" in res.coords

    # Check called methods
    with mock.patch.object(reader, "_mask_space") as mask_space:
        with mock.patch.object(reader, "calibrate") as calibrate:
            reader.get_dataset(key, {"units": "%", "sensor": "ahi"})
            mask_space.assert_called()
            calibrate.assert_called()


def test_sensor_mismatch(caplog):
    """Test that a file for a different sensor is detected."""
    reader = _get_reader(mda=_get_mda())
    key = make_dataid(name="VIS", calibration="reflectance")
    with mock.patch("satpy.readers.hrit_jma.HRITFileHandler.get_dataset") as base_get_dataset:
        base_get_dataset.return_value = DataArray(
            da.ones((11000, 11000), chunks=1024),
            dims=("y", "x"))
        reader.get_dataset(key, {"units": "%", "sensor": "jami"})
    assert "Sensor-Platform mismatch" in caplog.text


@pytest.mark.parametrize("platform", ["Himawari-8", "MTSAT-2"])
def test_get_acq_time(platform):
    """Test computation of scanline acquisition times."""
    dt_line = np.arange(1, 11000+1).astype("timedelta64[s]")
    acq_time_exp = np.datetime64("1970-01-01", "ns") + dt_line
    # Results are not exactly identical because timestamps are stored in
    # the header with only 6 decimals precision (max diff here: 45 msec).
    mda = _get_mda(platform=platform)
    reader = _get_reader(mda=mda)
    np.testing.assert_allclose(reader.acq_time.astype(np.int64),
                               acq_time_exp.astype(np.int64),
                               atol=45000000)


@pytest.mark.parametrize("platform", ["Himawari-8", "MTSAT-2"])
@pytest.mark.parametrize("use_acq_time", [None, False, True])
def test_start_time_from_aqc_time(platform, use_acq_time):
    """Test that by the datetime from the metadata returned when `use_acquisition_time_as_start_time=True`."""
    start_time = dt.datetime(2022, 1, 20, 12, 10)
    mda = _get_mda(platform=platform)
    reader_kwargs = {"use_acquisition_time_as_start_time": True} if use_acq_time else {}
    reader = _get_reader(
        mda=mda,
        filename_info={"start_time": start_time},
        reader_kwargs=reader_kwargs,
    )
    if use_acq_time:
        assert reader.start_time == dt.datetime(1970, 1, 1, 0, 0, 1, 36799)
        assert reader.end_time == dt.datetime(1970, 1, 1, 3, 3, 20, 16000)
    else:
        assert reader.start_time == start_time


@pytest.mark.parametrize(
    ("mjd", "dt_str"),
    [
        (0, "1858-11-17"),
        (40587.5, "1970-01-01 12:00"),
    ]
)
def test_mjd2datetime64(mjd, dt_str):
    """Test conversion from modified julian day to datetime64."""
    from satpy.readers.hrit_jma import mjd2datetime64
    assert mjd2datetime64(np.array([mjd])) == np.datetime64(dt_str, "ns")
