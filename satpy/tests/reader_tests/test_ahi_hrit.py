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


def create_fake_ahi_hrit(
        hrit_path: Path,
        num_rows: int = 11000,  # default 1km
        num_cols: int = 11000,  # default 1km
        is_vis: bool = True,
        annotation: str | None = None,
        metadata_overrides: dict | None = None,
) -> None:
    """Create a fake AHI HRIT file on disk."""
    header_data = _get_fake_header_data(num_rows, num_cols, is_vis, annotation or hrit_path.name)
    _update_header_with_metadata(header_data, metadata_overrides)
    data_arr = np.ones((num_rows, num_cols), dtype=np.uint16).ravel()

    with hrit_path.open(mode="wb") as fp:
        for header_arr in header_data:
            header_arr.tofile(fp)
        data_arr.tofile(fp)


def _get_fake_header_data(
        num_rows: int,
        num_cols: int,
        is_vis: bool,
        annotation: str,
) -> tuple[np.ndarray, ...]:
    coff_loffs_bytes = _get_line_offsets(num_rows, num_cols)
    acq_times_bytes = _get_acq_time(num_rows)
    if is_vis:
        idf = "$HALFTONE:=16\r_NAME:=VISIBLE\r_UNIT:=ALBEDO(%)\r0:=-0.10\r1023:=100.00\r65535:=100.00\r"
    else:
        idf = "$HALFTONE:=16\r_NAME:=INFRARED\r_UNIT:=KELVIN\r0:=329.98\r1023:=130.02\r65535:=130.02\r"

    header_data = (
        # header 0
        np.void((0, 16), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
        # total_header_length is updated below
        np.void((0, 0, 0),
                dtype=[("file_type", "u1"), ("total_header_length", ">u4"), ("data_field_length", ">u8")]),

        # header 1
        np.void((1, 9), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
        np.void((16, num_cols, num_rows, 0),
                dtype=[("number_of_bits_per_pixel", "u1"), ("number_of_columns", ">u2"), ("number_of_lines", ">u2"),
                       ("compression_flag_for_data", "u1")]),

        # header 2
        np.void((2, 51), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
        np.void((b"GEOS(140.70)                    ", 10233128, 10233128, 5500, 5500),
                dtype=[("projection_name", "S32"),
                       ("cfac", ">i4"), ("lfac", ">i4"),
                       ("coff", ">i4"), ("loff", ">i4")]),

        # header 3
        np.void((3, len(idf) + 3), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
        np.bytes_(idf),

        # header 4
        np.void((4, len(annotation) + 3), dtype=[("hdr_id", "u1"), ("record_length", ">u2")]),
        np.bytes_(annotation),

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
    total_header_length = sum([x.size * x.dtype.itemsize for x in header_data])
    header_data[1]["total_header_length"] = total_header_length
    return header_data


def _update_header_with_metadata(header_data: tuple[np.ndarray, ...], metadata_overrides: dict | None) -> None:
    if metadata_overrides is None:
        return
    for header_arr in header_data:
        keys_to_update = set(metadata_overrides.keys()) & set(header_arr.dtype.fields or [])
        for key in keys_to_update:
            new_val = metadata_overrides[key]
            if np.issubdtype(header_arr[key].dtype, np.bytes_):
                # null-pad the provided bytes
                new_val = np.array(new_val, dtype=header_arr[key].dtype)
            header_arr[key] = new_val


def _get_line_offsets(num_rows: int, num_cols: int) -> bytes:
    coff_loffs = f"LINE:=1\rCOFF:={num_cols / 2}\rLOFF:={num_rows / 2}"
    for lnum in range(1000, num_rows + 1, 1000):
        coff_loffs += f"LINE:={lnum}\rCOFF:={num_cols / 2}\rLOFF:={num_rows / 2}"
        if lnum != num_rows:
            coff_loffs += f"LINE:={lnum + 1}\rCOFF:={num_cols / 2}\rLOFF:={num_rows / 2}"
    return coff_loffs.encode()


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


@pytest.mark.parametrize(
    ("channel", "is_vis", "shape"),
    [
        ("VIS", True, (11000, 11000)),
        ("IR4", False, (2750, 2750)),
    ],
)
def test_header_parsing(tmp_path, channel, is_vis, shape):
    """Test creating the file handler."""
    hrit_path = tmp_path / f"IMG_DK01{channel}_202509261940_001"
    create_fake_ahi_hrit(
        hrit_path,
        shape[0],
        shape[1],
        is_vis=is_vis,
    )
    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})

    assert reader.mda["image_segm_seq_no"] == np.uint8(0)
    assert reader.mda["total_no_image_segm"] == np.uint8(1)
    assert reader.mda["projection_parameters"] == {
        "a": 6378169.00,
        "b": 6356583.80,
        "h": 35785831.00,
        "SSP_longitude": 140.7,
    }
    assert reader.mda["cfac"] == np.int32(10233128)
    assert reader.mda["lfac"] == np.int32(10233128)
    assert reader.mda["coff"] == np.int32(5500.0)
    assert reader.mda["loff"] == np.int32(5500.0)
    assert reader.mda["number_of_lines"] == np.uint16(shape[0])
    assert reader.mda["number_of_columns"] == np.uint16(shape[1])
    assert reader.mda["planned_end_segment_number"] == np.uint8(1)
    assert reader.mda["planned_start_segment_number"] == np.uint8(1)
    assert reader.mda["segment_sequence_number"] == np.uint8(0)
    assert reader.mda["unit"] == ("ALBEDO(%)" if is_vis else "KELVIN")
    assert reader.projection_name == "GEOS(140.70)"

    # Check calibration table
    if is_vis:
        cal_expected = np.array([[0, -0.1],
                                 [1023,  100],
                                 [65535,  100]])
    else:
        cal_expected = np.array([[0, 329.98],
                                 [1023, 130.02],
                                 [65535, 130.02]])
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
def test_segmented_checks(tmp_path, segno, is_segmented):
    """Test segments are identified."""
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(
        hrit_path,
        1100,
        11000,
        metadata_overrides={"image_segm_seq_no": segno},
    )
    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})
    assert reader.is_segmented == is_segmented


@pytest.mark.parametrize(
    ("extra_filename_info", "area_id"),
    [
        ({"area": FULL_DISK}, FULL_DISK),
        ({"area": 1234}, UNKNOWN_AREA),
        ({}, UNKNOWN_AREA)
    ]
)
def test_check_areas(tmp_path, extra_filename_info, area_id):
    """Test area names coming from the filename."""
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(hrit_path)
    filename_info = {"start_time": dt.datetime.now()}
    filename_info.update(extra_filename_info)
    reader = HRITJMAFileHandler(hrit_path, filename_info, {})
    assert reader.area_id == area_id


@pytest.mark.parametrize(("proj_name", "platform"), list(PLATFORMS.items()) + [("MERCATOR(0.0)", UNKNOWN_PLATFORM)])
def test_get_platform(tmp_path, proj_name, platform, caplog):
    """Test platform identification."""
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(hrit_path, metadata_overrides={"projection_name": proj_name})
    filename_info = {"start_time": dt.datetime.now()}
    reader = HRITJMAFileHandler(hrit_path, filename_info, {})

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
def test_get_area_def(tmp_path, mda_info, area_name, extent):
    """Test getting an AreaDefinition."""
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(
        hrit_path,
        mda_info["nlines"],
        mda_info["ncols"],
        metadata_overrides={
            "loff": mda_info["loff"],
            "coff": mda_info["coff"],
            "image_segm_seq_no": mda_info["segno"],
            "total_no_image_segm": mda_info["numseg"],
        },
    )
    reader = HRITJMAFileHandler(
        hrit_path,
        {"start_time": dt.datetime.now(), "area": area_name},
        {})

    area = reader.get_area_def("some_id")
    assert area.area_extent == extent
    assert area.description == AREA_NAMES[area_name]["long"]


@pytest.mark.parametrize("calibration", ["counts", "reflectance", "brightness_temperature"])
def test_calibrate(tmp_path, calibration):
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

    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(
        hrit_path,
        5,
        5,
        is_vis=calibration == "reflectance",
        # Choose an area near the subsatellite point to avoid masking of space pixels
        metadata_overrides={
            "loff": 1375.0,
            "coff": 1375.0,
        },
    )
    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})

    res = reader.calibrate(data=counts, calibration=calibration)
    exp = {
        "counts": counts.values,
        "reflectance": refl,
        "brightness_temperature": bt,
    }
    np.testing.assert_allclose(exp[calibration], res.values)


def test_mask_space(tmp_path):
    """Test masking of space pixels."""
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(
        hrit_path,
        275,
        1375,
        # Choose an area near the subsatellite point to avoid masking of space pixels
        metadata_overrides={
            "loff": 1375.0,
            "coff": 1375.0,
            "image_segm_seq_no": 1,
            "total_no_image_segm": 10,
        },
    )
    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})

    data = DataArray(da.ones((275, 1375), chunks=1024))
    masked = reader._mask_space(data)

    # First line of the segment should be space, in the middle of the
    # last line there should be some valid pixels
    np.testing.assert_allclose(masked.values[0, :], np.nan)
    np.testing.assert_array_equal(masked.values[-1, 588:788], 1)


def test_get_dataset(tmp_path):
    """Test getting a dataset."""
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(
        hrit_path,
        275,
        1375,
        # Choose an area near the subsatellite point to avoid masking of space pixels
        metadata_overrides={
            "loff": 1375.0,
            "coff": 1375.0,
            "image_segm_seq_no": 1,
            "total_no_image_segm": 10,
        },
    )
    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})

    key = make_dataid(name="VIS", calibration="reflectance")
    with mock.patch.object(reader, "_mask_space", wraps=reader._mask_space) as mask_space, \
            mock.patch.object(reader, "calibrate", wraps=reader.calibrate) as calibrate:
        res = reader.get_dataset(key, {"units": "%", "sensor": "ahi"})
        mask_space.assert_called()
        calibrate.assert_called()

    # Check attributes
    assert res.attrs["units"] == "%"
    assert res.attrs["sensor"] == "ahi"
    assert res.attrs["platform_name"] == HIMAWARI8
    assert res.attrs["orbital_parameters"] == {"projection_longitude": 140.7,
                                               "projection_latitude": 0.0,
                                               "projection_altitude": 35785831.0}

    # Check if acquisition time is a coordinate
    assert "acq_time" in res.coords


def test_sensor_mismatch(tmp_path, caplog):
    """Test that a file for a different sensor is detected."""
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(hrit_path)
    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})

    key = make_dataid(name="VIS", calibration="reflectance")
    reader.get_dataset(key, {"units": "%", "sensor": "jami"})
    assert "Sensor-Platform mismatch" in caplog.text


@pytest.mark.parametrize("projection_name", ["GEOS(140.70)", "GEOS(145.00)"])
def test_get_acq_time(tmp_path, projection_name):
    """Test computation of scanline acquisition times."""
    dt_line = np.arange(1, 11000+1).astype("timedelta64[s]")
    acq_time_exp = np.datetime64("1970-01-01", "ns") + dt_line
    # Results are not exactly identical because timestamps are stored in
    # the header with only 6 decimals precision (max diff here: 45 msec).

    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(hrit_path, metadata_overrides={"projection_name": projection_name})
    reader = HRITJMAFileHandler(hrit_path, {"start_time": dt.datetime.now()}, {})
    np.testing.assert_allclose(reader.acq_time.astype(np.int64),
                               acq_time_exp.astype(np.int64),
                               atol=45000000)


@pytest.mark.parametrize("projection_name", ["GEOS(140.70)", "GEOS(145.00)"])
@pytest.mark.parametrize("use_acq_time", [None, False, True])
def test_start_time_from_aqc_time(tmp_path, projection_name, use_acq_time):
    """Test that by the datetime from the metadata returned when `use_acquisition_time_as_start_time=True`."""
    start_time = dt.datetime(2022, 1, 20, 12, 10)
    hrit_path = tmp_path / "IMG_DK01VIS_202509261940_001"
    create_fake_ahi_hrit(hrit_path, metadata_overrides={"projection_name": projection_name})
    reader_kwargs = {"use_acquisition_time_as_start_time": True} if use_acq_time else {}
    reader = HRITJMAFileHandler(hrit_path, {"start_time": start_time}, {}, **reader_kwargs)

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
