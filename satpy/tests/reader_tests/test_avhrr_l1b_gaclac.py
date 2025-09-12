#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2019 Satpy developers
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

"""Pygac interface."""
import abc
import datetime as dt

import numpy as np
import pygac.gac_klm
import pygac.gac_pod
import pygac.klm_reader
import pygac.pod_reader
import pytest
import xarray as xr
from pygac.gac_reader import GACReader

from satpy import Scene
from satpy.readers.avhrr_l1b_gaclac import GACLACFile

GAC_PATTERN = '{creation_site:3s}.{transfer_mode:4s}.{platform_id:2s}.D{start_time:%y%j.S%H%M}.E{end_time:%H%M}.B{orbit_number:05d}{end_orbit_last_digits:02d}.{station:2s}'  # noqa
EOSIP_PATTERN = '{platform_id:3s}_RPRO_AVH_L1B_1P_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_{orbit_number:06d}/image.l1b'  # noqa

GAC_POD_FILENAMES = ["NSS.GHRR.NA.D79184.S1150.E1337.B0008384.WI",
                     "NSS.GHRR.NA.D79184.S2350.E0137.B0008384.WI",
                     "NSS.GHRR.NA.D80021.S0927.E1121.B0295354.WI",
                     "NSS.GHRR.NA.D80021.S1120.E1301.B0295455.WI",
                     "NSS.GHRR.NA.D80021.S1256.E1450.B0295556.GC",
                     "NSS.GHRR.NE.D83208.S1219.E1404.B0171819.WI",
                     "NSS.GHRR.NG.D88002.S0614.E0807.B0670506.WI",
                     "NSS.GHRR.TN.D79183.S1258.E1444.B0369697.GC",
                     "NSS.GHRR.TN.D80003.S1147.E1332.B0630506.GC",
                     "NSS.GHRR.TN.D80003.S1328.E1513.B0630507.GC",
                     "NSS.GHRR.TN.D80003.S1509.E1654.B0630608.GC"]

GAC_KLM_FILENAMES = ["NSS.GHRR.NK.D01235.S0252.E0446.B1703233.GC",
                     "NSS.GHRR.NL.D01288.S2315.E0104.B0549495.GC",
                     "NSS.GHRR.NM.D04111.S2305.E0050.B0947778.GC",
                     "NSS.GHRR.NN.D13011.S0559.E0741.B3939192.WI",
                     "NSS.GHRR.NP.D15361.S0121.E0315.B3547172.SV",
                     "NSS.GHRR.M1.D15362.S0031.E0129.B1699697.SV",
                     "NSS.GHRR.M2.D10178.S2359.E0142.B1914142.SV"]

LAC_POD_FILENAMES = ["BRN.HRPT.ND.D95152.S1730.E1715.B2102323.UB",
                     "BRN.HRPT.ND.D95152.S1910.E1857.B2102424.UB",
                     "BRN.HRPT.NF.D85152.S1345.E1330.B0241414.UB",
                     "BRN.HRPT.NJ.D95152.S1233.E1217.B0216060.UB"]

LAC_KLM_FILENAMES = ["BRN.HRPT.M1.D14152.S0958.E1012.B0883232.UB",
                     "BRN.HRPT.M1.D14152.S1943.E1958.B0883838.UB",
                     "BRN.HRPT.M2.D12153.S0912.E0922.B2914747.UB",
                     "BRN.HRPT.NN.D12153.S0138.E0152.B3622828.UB",
                     "BRN.HRPT.NN.D12153.S0139.E0153.B3622828.UB",
                     "BRN.HRPT.NN.D12153.S1309.E1324.B3623535.UB",
                     "BRN.HRPT.NP.D12153.S0003.E0016.B1707272.UB",
                     "BRN.HRPT.NP.D12153.S1134.E1148.B1707979.UB",
                     "BRN.HRPT.NP.D16184.S1256.E1311.B3813131.UB",
                     "BRN.HRPT.NP.D16184.S1438.E1451.B3813232.UB",
                     "BRN.HRPT.NP.D16184.S1439.E1451.B3813232.UB",
                     "BRN.HRPT.NP.D16185.S1245.E1259.B3814545.UB",
                     "BRN.HRPT.NP.D16185.S1427.E1440.B3814646.UB",
                     "NSS.FRAC.M2.D12153.S1729.E1910.B2915354.SV",
                     "NSS.LHRR.NP.D16306.S1803.E1814.B3985555.WI"]

LAC_EOSIP_FILENAMES = ["N06_RPRO_AVH_L1B_1P_20061206T010808_20061206T012223_007961/image.l1b"]


class TestGACLACFile:
    """Test the GACLAC file handler."""

    def _get_fh(self, filename="NSS.GHRR.NG.D88002.S0614.E0807.B0670506.WI",
                **kwargs):
        """Create a file handler."""
        from trollsift import parse
        filename_info = parse(GAC_PATTERN, filename)
        return GACLACFile(filename, filename_info, {}, **kwargs)

    def _get_eosip_fh(self, filename, **kwargs):
        """Create a file handler."""
        from trollsift import parse
        filename_info = parse(EOSIP_PATTERN, filename)
        return GACLACFile(filename, filename_info, {}, **kwargs)

    def test_init(self):
        """Test GACLACFile initialization."""
        from pygac.gac_klm import GACKLMReader
        from pygac.gac_pod import GACPODReader
        from pygac.lac_klm import LACKLMReader
        from pygac.lac_pod import LACPODReader

        kwargs = {"start_line": 1,
                  "end_line": 2,
                  "strip_invalid_coords": True,
                  "interpolate_coords": True,
                  "adjust_clock_drift": True,
                  "tle_dir": "tle_dir",
                  "tle_name": "tle_name",
                  "tle_thresh": 123,
                  "calibration": "calibration"}
        for filenames, reader_cls in zip([GAC_POD_FILENAMES, GAC_KLM_FILENAMES, LAC_POD_FILENAMES, LAC_KLM_FILENAMES],
                                         [GACPODReader, GACKLMReader, LACPODReader, LACKLMReader]):
            for filename in filenames:
                fh = self._get_fh(filename, **kwargs)
                assert fh.start_time < fh.end_time
                assert fh.reader_class is reader_cls


    def test_init_eosip(self):
        """Test GACLACFile initialization."""
        from pygac.lac_pod import LACPODReader

        kwargs = {"start_line": 1,
                  "end_line": 2,
                  "strip_invalid_coords": True,
                  "interpolate_coords": True,
                  "adjust_clock_drift": True,
                  "tle_dir": "tle_dir",
                  "tle_name": "tle_name",
                  "tle_thresh": 123,
                  "calibration": "calibration"}
        for filenames, reader_cls in zip([LAC_EOSIP_FILENAMES],
                                         [LACPODReader]):
            for filename in filenames:
                fh = self._get_eosip_fh(filename, **kwargs)
                assert fh.start_time < fh.end_time
                assert fh.reader_class is reader_cls
                assert fh.reader_kwargs["header_date"] > dt.date(1994, 11, 15)


@pytest.fixture(scope="class")
def tmp_dir(tmp_path_factory):
    """Create temporary directory."""
    return tmp_path_factory.mktemp("stubs")


@pytest.fixture(scope="class")
def tle_dir(tmp_dir):
    """Create TLE directory."""
    tle_dir = tmp_dir / "tle"
    tle_dir.mkdir()
    return tle_dir


class GacTestBase(metaclass=abc.ABCMeta):
    """Base class for GAC reader tests."""

    @abc.abstractproperty
    def filename(self):
        """GAC file name."""

    @abc.abstractproperty
    def num_lines_exp(self):
        """Expected number of scanlines."""

    @abc.abstractproperty
    def num_lines_exp_slice(self):
        """Expected number of scanlines after slicing."""

    @abc.abstractproperty
    def start_time_exp(self):
        """Expected start time."""

    @abc.abstractproperty
    def end_time_exp(self):
        """Expected end time."""

    @abc.abstractproperty
    def start_time_exp_slice(self):
        """Expected start time after slicing."""

    @abc.abstractproperty
    def end_time_exp_slice(self):
        """Expected end time after slicing."""

    @abc.abstractproperty
    def platform_name_exp(self):
        """Expected platform name."""

    @pytest.fixture(scope="class")
    def reader_kwargs(self, tle_dir):
        """Reader keyword arguments."""
        return {"tle_dir": str(tle_dir), "tle_name": "TLE_%(satname)s.txt", "strip_invalid_coords": False}

    @pytest.fixture(autouse=True, scope="class")
    def tle_file(self, tle_dir, tle):
        """Write TLE file."""
        filename = tle_dir / f"TLE_{self.platform_name_exp}.txt"
        with filename.open("w") as fh:
            fh.write(tle)

    @pytest.fixture(scope="class")
    def stub_file(self, tmp_dir, stub):
        """Write stub file."""
        filename = tmp_dir / self.filename
        with filename.open("wb") as fh:
            for array in stub:
                array.tofile(fh)
        return filename

    def test_get_channel_calibrated(self, stub_file, reader_kwargs):
        """Test getting calibrated channel."""
        scene = Scene(filenames=[stub_file], reader="avhrr_l1b_gaclac",
                      reader_kwargs=reader_kwargs)
        scene.load(["1"])
        assert scene["1"].shape == (self.num_lines_exp, 409)
        assert scene["1"].dims == ("y", "x")
        assert scene["1"].start_time == self.start_time_exp
        assert scene["1"].end_time == self.end_time_exp
        assert scene["1"].units == "%"
        assert "tle" in scene["1"].attrs["orbital_parameters"]
        assert scene["1"].attrs["platform_name"] == self.platform_name_exp

    def test_get_channel_counts(self, stub_file, reader_kwargs):
        """Test getting raw channel counts."""
        scene = Scene(filenames=[stub_file], reader="avhrr_l1b_gaclac",
                      reader_kwargs=reader_kwargs)
        scene.load(["1"], calibration="counts")
        assert scene["1"].units == "count"

    def test_slice(self, stub_file, reader_kwargs):
        """Test slicing a range of scanlines from the overpass."""
        reader_kwargs = reader_kwargs | {"start_line": 10, "end_line": 20}
        scene = Scene(filenames=[stub_file], reader="avhrr_l1b_gaclac",
                      reader_kwargs=reader_kwargs)
        scene.load(["1"])
        assert scene["1"].shape == (self.num_lines_exp_slice, 409)
        assert scene["1"].start_time == self.start_time_exp_slice
        assert scene["1"].end_time == self.end_time_exp_slice

    def test_get_latlon(self, stub_file, reader_kwargs):
        """Test getting lat/lon coordinates."""
        scene = Scene(filenames=[stub_file], reader="avhrr_l1b_gaclac",
                      reader_kwargs=reader_kwargs)
        scene.load(["latitude"])
        assert scene["latitude"].shape == (self.num_lines_exp, 409)
        assert scene["latitude"].dims == ("y", "x")
        assert scene["latitude"].units == "degrees_north"

    def test_get_angles(self, stub_file, reader_kwargs):
        """Test getting sun/sensor angles."""
        scene = Scene(filenames=[stub_file], reader="avhrr_l1b_gaclac",
                      reader_kwargs=reader_kwargs)
        scene.load(["sensor_zenith_angle"])
        assert scene["sensor_zenith_angle"].shape == (self.num_lines_exp, 409)
        assert scene["sensor_zenith_angle"].dims == ("y", "x")
        assert scene["sensor_zenith_angle"].units == "degrees"

    def test_get_qual_flags(self, stub_file, reader_kwargs):
        """Test getting quality flags."""
        scene = Scene(filenames=[stub_file], reader="avhrr_l1b_gaclac",
                      reader_kwargs=reader_kwargs)
        scene.load(["qual_flags"])
        assert scene["qual_flags"].shape == (self.num_lines_exp, 7)
        assert scene["qual_flags"].dims == ("y", "num_flags")



class TestReadingGacKlmFile(GacTestBase):
    """Test reading GAC KLM file."""
    filename = "NSS.GHRR.NK.D09362.S2359.E0001.B6044445.GC"
    num_lines_exp = 55
    num_lines_exp_slice = 11
    # Start time doesn't make sense, probably side effect of using the stub
    start_time_exp = dt.datetime(2009, 12, 29, 9, 5, 57, 500000)
    end_time_exp = dt.datetime(2009, 12, 29, 0, 0, 16, 500000)
    start_time_exp_slice = dt.datetime(2009, 12, 28, 23, 59, 54, 500000)
    end_time_exp_slice = dt.datetime(2009, 12, 28, 23, 59, 59, 500000)
    platform_name_exp = "noaa15"

    @pytest.fixture(scope="class")
    def tle(self):
        """NOAA-15 two line elements."""
        return """1 25338U 98030A   09361.80631861 -.00000112  00000-0 -29536-4 0  2104
2 25338  98.6031 346.9543 0010160 209.9249 150.1326 14.24798410604342
1 25338U 98030A   09362.36812158 -.00000012  00000-0  13243-4 0  2102
2 25338  98.6030 347.5048 0010160 208.4397 151.6285 14.24799183604423
1 25338U 98030A   09363.49172435  .00000077  00000-0  51611-4 0  2116
2 25338  98.6031 348.6054 0010232 205.1043 154.9644 14.24799790604581"""

    @pytest.fixture(scope="class")
    def stub(self):
        """Create GAC KLM stub."""
        num_lines = 55
        times = xr.DataArray(
            [
                dt.datetime(2009, 12, 28, 23, 59, 50) + dt.timedelta(milliseconds=line * 500)
                for line in range(num_lines)
            ],
        )
        msecs_since_00 = (1000 * (
                    times.dt.hour * 3600 + times.dt.minute * 60 + times.dt.second) + times.dt.microsecond / 1000)
        telemetry = np.zeros(1, dtype=pygac.klm_reader.analog_telemetry_v2)
        scans = np.ones(num_lines, dtype=pygac.gac_klm.scanline)
        scans["scan_line_number"] = np.arange(num_lines)
        scans["scan_line_year"] = times.dt.year.values
        scans["scan_line_day_of_year"] = times.dt.dayofyear.values
        scans["scan_line_utc_time_of_day"] = msecs_since_00.values
        scans["telemetry"]["PRT"] = np.repeat(np.arange(num_lines), 3).reshape((num_lines, 3))

        hdr = np.zeros(1, dtype=pygac.klm_reader.header)
        hdr["data_type_code"] = 2  # GAC
        hdr["data_set_name"] = self.filename
        hdr["noaa_spacecraft_identification_code"] = 4  # NOAA-15
        hdr["noaa_level_1b_format_version_number"] = 2
        hdr["count_of_data_records"] = num_lines
        hdr["start_of_data_set_year"] = times.dt.year.values[0]
        hdr["start_of_data_set_day_of_year"] = times.dt.dayofyear.values[0]
        hdr["start_of_data_set_utc_time_of_day"] = msecs_since_00.values[0]

        spare = np.zeros(pygac.gac_klm.GACKLMReader().offset - (hdr.itemsize + telemetry.itemsize), dtype="u1")

        return [hdr, telemetry, spare, scans]

    def test_get_latlon_no_interpolation(self, stub_file, reader_kwargs):
        """Test getting lat/lon coordinates without interpolation.

        Only works for KLM, because the stub file is too small.
        """
        reader_kwargs = reader_kwargs | {"interpolate_coords": False}
        scene = Scene(filenames=[stub_file], reader="avhrr_l1b_gaclac",
                      reader_kwargs=reader_kwargs)
        scene.load(["latitude"])
        assert scene["latitude"].shape == (self.num_lines_exp, 51)
        assert scene["latitude"].dims == ("y", "x_every_eighth")


def _get_times(num_lines):
    times = xr.DataArray(
        [
            dt.datetime(2009, 12, 28, 23, 59, 50) + dt.timedelta(milliseconds=line/GACReader.scan_freq)
            for line in range(num_lines)
        ],
    )
    msecs_since_00 = (
        1000 * (times.dt.hour * 3600 + times.dt.minute * 60 + times.dt.second) +
        times.dt.microsecond / 1000
    )
    return times, msecs_since_00

class TestReadingGacPodFile(GacTestBase):
    """Test reading GAC POD file."""
    filename = "NSS.GHRR.NJ.D09362.S2359.E0001.B6044445.GC"
    num_lines_exp = 54
    num_lines_exp_slice = 11
    start_time_exp = dt.datetime(2009, 12, 28, 23, 59, 49, 800000)
    end_time_exp = dt.datetime(2009, 12, 29, 0, 0, 16, 300000)
    start_time_exp_slice = dt.datetime(2009, 12, 28, 23, 59, 54, 800000)
    end_time_exp_slice = dt.datetime(2009, 12, 28, 23, 59, 59, 800000)
    platform_name_exp = "noaa14"

    @pytest.fixture(scope="class")
    def tle(self):
        """NOAA-14 two line elements."""
        return """1 23455U 94089A   09363.50221683 -.00000013  00000-0  16673-4 0  8853
2 23455  98.8939  80.3339 0009410   2.6550 357.4665 14.13775739773522
1 23455U 94089A   09363.50221683 -.00000013  00000-0  16673-4 0  8864
2 23455  98.8939  80.3339 0009410   2.6550 357.4665 14.13775739773522
1 23455U 94089A   09363.78530466 -.00000027  00000-0  94367-5 0  8879
2 23455  98.8938  80.6155 0009463   1.7605 358.3582 14.13775588773563"""

    @pytest.fixture(scope="class")
    def stub(self):
        """Create GAC POD stub."""
        num_lines = 55
        times, msecs_since_00 = _get_times(num_lines)
        times_enc = encode_timestamps_pod(
            times.dt.year.values,
            times.dt.dayofyear.values,
            msecs_since_00.values.astype(int)
        )

        scans = np.ones(num_lines, dtype=pygac.gac_pod.scanline)
        scans["scan_line_number"] = np.arange(num_lines)
        scans["time_code"] = times_enc
        scans["telemetry"] = 100 * np.arange(35 * num_lines).reshape((num_lines, 35))

        hdr0 = np.zeros(1, dtype=pygac.pod_reader.header0)
        hdr0["start_time"] = times_enc[0]
        hdr0["number_of_scans"] = num_lines
        hdr0["noaa_spacecraft_identification_code"] = 3  # NOAA-14

        hdr3 = np.zeros(1, dtype=pygac.pod_reader.header3)

        spare = np.zeros(pygac.gac_pod.GACPODReader().offset - (hdr0.itemsize + hdr3.itemsize), dtype="u1")
        return [hdr0, hdr3, spare, scans]


def encode_timestamps_pod(year, jday, msec):
    """Encode timestamps like in POD files.

    Reverse engineered from Pygac code by an LLM.
    """
    yoff = year - 2000  # assuming we're in 2009 for this test case
    enc0 = (yoff << 9) | (jday & 0x1FF)
    enc2 = msec & 0xFFFF
    enc1 = (msec >> 16) & 0x7FF
    return np.array([enc0, enc1, enc2]).transpose()


def test_encode_timestamps_pod():
    """Test timestamp encoding/decoding round trip."""
    year = np.array([2009, 2009])
    jday = np.array([362, 362])
    msec = np.array([0, 123456])
    encoded = encode_timestamps_pod(year, jday, msec)
    year_dec, jday_dec, msec_dec = pygac.pod_reader.PODReader.decode_timestamps(encoded)
    np.testing.assert_equal(year_dec, year)
    np.testing.assert_equal(jday_dec, jday)
    np.testing.assert_equal(msec_dec, msec)
