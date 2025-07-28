#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""The ami_l1b reader tests package."""
import contextlib
from typing import Iterator
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pytest import approx, raises  # noqa: PT013

from satpy.readers.ami_l1b import AMIL1bNetCDF
from satpy.tests.utils import make_dataid

FAKE_VIS_DATA = (((np.arange(10.).reshape((2, 5)) + 1.) * 50.0 + 1.0) / 0.5).astype(np.uint16)
FAKE_IR_DATA = ((np.arange(10).reshape((2, 5))) + 7000).astype(np.uint16)


class FakeDataset(object):
    """Mimic xarray Dataset object."""

    def __init__(self, info, attrs):
        """Initialize test data."""
        for var_name, var_data in list(info.items()):
            if isinstance(var_data, np.ndarray):
                info[var_name] = xr.DataArray(var_data)
        self.info = info
        self.attrs = attrs

    def __getitem__(self, key):
        """Mimic getitem method."""
        return self.info[key]

    def __contains__(self, key):
        """Mimic contains method."""
        return key in self.info

    def rename(self, *args, **kwargs):
        """Mimic rename method."""
        return self

    def close(self):
        """Act like close method."""
        return


def _get_fake_counts(rad_data: np.ndarray, attrs: dict) -> xr.DataArray:
    counts = xr.DataArray(
        da.from_array(rad_data, chunks="auto"),
        dims=("y", "x"),
        attrs=attrs,
    )
    return counts


@contextlib.contextmanager
def _fake_reader(counts_data: xr.DataArray) -> Iterator[AMIL1bNetCDF]:
    sc_position = xr.DataArray(0., attrs={
        "sc_position_center_pixel": [-26113466.1974016, 33100139.1630508, 3943.75470244799],
    })
    fake_ds = FakeDataset(
        {
            "image_pixel_values": counts_data,
            "sc_position": sc_position,
            "gsics_coeff_intercept": [0.1859369],
            "gsics_coeff_slope": [0.9967594],
        },
        {
            "satellite_name": "GK-2A",
            "observation_start_time": 623084431.957882,
            "observation_end_time": 623084975.606133,
            "projection_type": "GEOS",
            "sub_longitude": 2.23751210105673,
            "cfac": 81701355.6133574,
            "lfac": -81701355.6133574,
            "coff": 11000.5,
            "loff": 11000.5,
            "nominal_satellite_height": 42164000.,
            "earth_equatorial_radius": 6378137.,
            "earth_polar_radius": 6356752.3,
            "number_of_columns": 22000,
            "number_of_lines": 22000,
            "observation_mode": "FD",
            "channel_spatial_resolution": "0.5",
            "Radiance_to_Albedo_c": 1,
            "DN_to_Radiance_Gain": -0.0144806550815701,
            "DN_to_Radiance_Offset": 118.050903320312,
            "Teff_to_Tbb_c0": -0.141418528203155,
            "Teff_to_Tbb_c1": 1.00052232906885,
            "Teff_to_Tbb_c2": -0.00000036287276076109,
            "light_speed": 2.9979245800E+08,
            "Boltzmann_constant_k": 1.3806488000E-23,
            "Plank_constant_h": 6.6260695700E-34,
        }
    )
    with mock.patch("satpy.readers.ami_l1b.xr") as xr_:
        xr_.open_dataset.return_value = fake_ds
        yield AMIL1bNetCDF("filename",
                            {"platform_shortname": "gk2a"},
                            {"file_type": "ir087"})


@pytest.fixture
def fake_vis_reader():
    """Create fake reader for loading visible data."""
    attrs = _fake_vis_attrs()
    counts_data_arr = _get_fake_counts(FAKE_VIS_DATA, attrs)
    with _fake_reader(counts_data_arr) as reader:
        yield reader


def _fake_vis_attrs():
    return {
        "channel_name": "VI006",
        "detector_side": 2,
        "number_of_total_pixels": 484000000,
        "number_of_error_pixels": 113892451,
        "max_pixel_value": 32768,
        "min_pixel_value": 6,
        "average_pixel_value": 8228.98770845248,
        "stddev_pixel_value": 13621.130386551,
        "number_of_total_bits_per_pixel": 16,
        "number_of_data_quality_flag_bits_per_pixel": 2,
        "number_of_valid_bits_per_pixel": np.array([12]).astype(np.uint8),
        "data_quality_flag_meaning":
            "0:good_pixel, 1:conditionally_usable_pixel, 2:out_of_scan_area_pixel, 3:error_pixel",
        "ground_sample_distance_ew": 1.4e-05,
        "ground_sample_distance_ns": 1.4e-05,
    }


@pytest.fixture
def fake_ir_reader():
    """Create fake reader for loading IR data."""
    attrs = _fake_ir_attrs()
    counts_data_arr = _get_fake_counts(FAKE_IR_DATA, attrs)
    with _fake_reader(counts_data_arr) as reader:
        yield reader


def _fake_ir_attrs():
    return {
        "channel_name": "IR087",
        "detector_side": 2,
        "number_of_total_pixels": 484000000,
        "number_of_error_pixels": 113892451,
        "max_pixel_value": 32768,
        "min_pixel_value": 6,
        "average_pixel_value": 8228.98770845248,
        "stddev_pixel_value": 13621.130386551,
        "number_of_total_bits_per_pixel": 16,
        "number_of_data_quality_flag_bits_per_pixel": 2,
        "number_of_valid_bits_per_pixel": np.array([13]).astype(np.uint8),
        "data_quality_flag_meaning":
            "0:good_pixel, 1:conditionally_usable_pixel, 2:out_of_scan_area_pixel, 3:error_pixel",
        "ground_sample_distance_ew": 1.4e-05,
        "ground_sample_distance_ns": 1.4e-05,
    }


@pytest.fixture
def fake_ir_reader2():
    """Create fake reader for testing radiance clipping."""
    counts_arr = FAKE_IR_DATA.copy()
    counts_arr[0, 0] = 16364
    attrs = _fake_ir_attrs()
    counts_data_arr = _get_fake_counts(counts_arr, attrs)
    with _fake_reader(counts_data_arr) as reader:
        yield reader


class TestAMIL1bNetCDF:
    """Test the AMI L1b reader."""

    def _check_orbital_parameters(self, orb_params):
        """Check that orbital parameters match expected values."""
        exp_params = {
            "projection_altitude": 35785863.0,
            "projection_latitude": 0.0,
            "projection_longitude": 128.2,
            "satellite_actual_altitude": 35782654.56070405,
            "satellite_actual_latitude": 0.005364927,
            "satellite_actual_longitude": 128.2707,
        }
        for key, val in exp_params.items():
            assert val == approx(orb_params[key], abs=1e-3)

    def test_filename_grouping(self):
        """Test that filenames are grouped properly."""
        from satpy.readers.core.grouping import group_files
        filenames = [
            "gk2a_ami_le1b_ir087_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_ir096_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_ir105_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_ir112_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_ir123_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_ir133_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_nr013_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_nr016_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_sw038_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_vi004_fd010ge_201909300300.nc",
            "gk2a_ami_le1b_vi005_fd010ge_201909300300.nc",
            "gk2a_ami_le1b_vi006_fd005ge_201909300300.nc",
            "gk2a_ami_le1b_vi008_fd010ge_201909300300.nc",
            "gk2a_ami_le1b_wv063_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_wv069_fd020ge_201909300300.nc",
            "gk2a_ami_le1b_wv073_fd020ge_201909300300.nc"]
        groups = group_files(filenames, reader="ami_l1b")
        assert len(groups) == 1
        assert len(groups[0]["ami_l1b"]) == 16

    def test_basic_attributes(self, fake_vis_reader):
        """Test getting basic file attributes."""
        import datetime as dt
        assert fake_vis_reader.start_time == dt.datetime(2019, 9, 30, 3, 0, 31, 957882)
        assert fake_vis_reader.end_time == dt.datetime(2019, 9, 30, 3, 9, 35, 606133)

    def test_get_dataset(self, fake_vis_reader):
        """Test getting radiance data."""
        from satpy.tests.utils import make_dataid
        key = make_dataid(name="VI006", calibration="radiance")
        res = fake_vis_reader.get_dataset(key, {
            "file_key": "image_pixel_values",
            "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
            "units": "W m-2 um-1 sr-1",
        })
        exp = {"calibration": "radiance",
               "modifiers": (),
               "platform_name": "GEO-KOMPSAT-2A",
               "sensor": "ami",
               "units": "W m-2 um-1 sr-1"}
        for key, val in exp.items():
            assert val == res.attrs[key]
        self._check_orbital_parameters(res.attrs["orbital_parameters"])

    def test_bad_calibration(self):
        """Test that asking for a bad calibration fails."""
        from satpy.tests.utils import make_dataid
        with raises(ValueError, match="_bad_ invalid value for .*"):
            _ = make_dataid(name="VI006", calibration="_bad_")

    @mock.patch("satpy.readers.core.abi.geometry.AreaDefinition")
    def test_get_area_def(self, adef, fake_vis_reader):
        """Test the area generation."""
        fake_vis_reader.get_area_def(None)

        assert adef.call_count == 1
        call_args = tuple(adef.call_args)[0]
        exp = {"a": 6378137.0, "b": 6356752.3, "h": 35785863.0,
               "lon_0": 128.2, "proj": "geos", "units": "m"}
        for key, val in exp.items():
            assert key in call_args[3]
            assert val == approx(call_args[3][key])
        assert call_args[4] == fake_vis_reader.nc.attrs["number_of_columns"]
        assert call_args[5] == fake_vis_reader.nc.attrs["number_of_lines"]
        np.testing.assert_allclose(call_args[6],
                                   [-5511022.902, -5511022.902, 5511022.902, 5511022.902])

    def test_get_dataset_vis(self, fake_vis_reader):
        """Test get visible calibrated data."""
        from satpy.tests.utils import make_dataid
        key = make_dataid(name="VI006", calibration="reflectance")
        res = fake_vis_reader.get_dataset(key, {
            "file_key": "image_pixel_values",
            "standard_name": "toa_bidirectional_reflectance",
            "units": "%",
        })
        exp = {"calibration": "reflectance",
               "modifiers": (),
               "platform_name": "GEO-KOMPSAT-2A",
               "sensor": "ami",
               "units": "%"}
        for key, val in exp.items():
            assert val == res.attrs[key]
        self._check_orbital_parameters(res.attrs["orbital_parameters"])

    def test_get_dataset_counts(self, fake_vis_reader):
        """Test get counts data."""
        from satpy.tests.utils import make_dataid
        key = make_dataid(name="VI006", calibration="counts")
        res = fake_vis_reader.get_dataset(key, {
            "file_key": "image_pixel_values",
            "standard_name": "counts",
            "units": "1",
        })
        exp = {"calibration": "counts",
               "modifiers": (),
               "platform_name": "GEO-KOMPSAT-2A",
               "sensor": "ami",
               "units": "1"}
        for key, val in exp.items():
            assert val == res.attrs[key]
        self._check_orbital_parameters(res.attrs["orbital_parameters"])


class TestAMIL1bNetCDFIRCal:
    """Test IR specific things about the AMI reader."""
    ds_id = make_dataid(name="IR087", wavelength=[8.415, 8.59, 8.765],
                        calibration="brightness_temperature")
    ds_info = {
        "file_key": "image_pixel_values",
        "wavelength": [8.415, 8.59, 8.765],
        "standard_name": "toa_brightness_temperature",
        "units": "K",
    }

    def test_default_calibrate(self, fake_ir_reader):
        """Test default (pyspectral) IR calibration."""
        from satpy.readers.ami_l1b import rad2temp
        with mock.patch("satpy.readers.ami_l1b.rad2temp", wraps=rad2temp) as r2t_mock:
            res = fake_ir_reader.get_dataset(self.ds_id, self.ds_info)
            r2t_mock.assert_called_once()
        expected = np.array([[238.34385135, 238.31443527, 238.28500087, 238.25554813, 238.22607701],
                             [238.1965875, 238.16707956, 238.13755317, 238.10800829, 238.07844489]])
        np.testing.assert_allclose(res.data.compute(), expected, equal_nan=True)
        # make sure the attributes from the file are in the data array
        assert res.attrs["standard_name"] == "toa_brightness_temperature"

    def test_infile_calibrate(self, fake_ir_reader):
        """Test IR calibration using in-file coefficients."""
        from satpy.readers.ami_l1b import rad2temp
        fake_ir_reader.calib_mode = "FILE"
        with mock.patch("satpy.readers.ami_l1b.rad2temp", wraps=rad2temp) as r2t_mock:
            res = fake_ir_reader.get_dataset(self.ds_id, self.ds_info)
            r2t_mock.assert_not_called()
        expected = np.array([[238.34385135, 238.31443527, 238.28500087, 238.25554813, 238.22607701],
                             [238.1965875, 238.16707956, 238.13755317, 238.10800829, 238.07844489]])
        # file coefficients are pretty close, give some wiggle room
        np.testing.assert_allclose(res.data.compute(), expected, equal_nan=True, atol=0.04)
        # make sure the attributes from the file are in the data array
        assert res.attrs["standard_name"] == "toa_brightness_temperature"

    def test_gsics_radiance_corr(self, fake_ir_reader):
        """Test IR radiance adjustment using in-file GSICS coefs."""
        from satpy.readers.ami_l1b import rad2temp
        fake_ir_reader.calib_mode = "GSICS"
        expected = np.array([[238.036797, 238.007106, 237.977396, 237.947668, 237.91792],
                             [237.888154, 237.85837, 237.828566, 237.798743, 237.768902]])
        with mock.patch("satpy.readers.ami_l1b.rad2temp", wraps=rad2temp) as r2t_mock:
            res = fake_ir_reader.get_dataset(self.ds_id, self.ds_info)
            r2t_mock.assert_not_called()
        # file coefficients are pretty close, give some wiggle room
        np.testing.assert_allclose(res.data.compute(), expected, equal_nan=True, atol=0.01)
        # make sure the attributes from the file are in the data array
        assert res.attrs["standard_name"] == "toa_brightness_temperature"

    def test_user_radiance_corr(self, fake_ir_reader):
        """Test IR radiance adjustment using user-supplied coefs."""
        from satpy.readers.ami_l1b import rad2temp
        fake_ir_reader.calib_mode = "FILE"
        fake_ir_reader.user_calibration = {"IR087": {"slope": 0.99669,
                                                  "offset": 0.16907}}
        expected = np.array([[238.073713, 238.044043, 238.014354, 237.984647, 237.954921],
                             [237.925176, 237.895413, 237.865631, 237.835829, 237.806009]])
        with mock.patch("satpy.readers.ami_l1b.rad2temp", wraps=rad2temp) as r2t_mock:
            res = fake_ir_reader.get_dataset(self.ds_id, self.ds_info)
            r2t_mock.assert_not_called()
        # file coefficients are pretty close, give some wiggle room
        np.testing.assert_allclose(res.data.compute(), expected, equal_nan=True, atol=0.01)
        # make sure the attributes from the file are in the data array
        assert res.attrs["standard_name"] == "toa_brightness_temperature"

    @pytest.mark.parametrize("clip", [False, True])
    def test_clipneg(self, fake_ir_reader2, clip):
        """Test that negative radiances are clipped."""
        ds_id = make_dataid(name="IR087", wavelength=[8.415, 8.59, 8.765],
                            calibration="radiance")
        fake_ir_reader2.clip_negative_radiances = clip
        res = np.array(fake_ir_reader2.get_dataset(ds_id, self.ds_info))
        if clip:
            np.testing.assert_allclose(res[0, 0], 0.004603, atol=0.0001)
        else:
            assert res[0, 0] < 0
