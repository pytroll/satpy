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

"""Module for testing the satpy.readers.nc_slstr module."""

import datetime as dt
import unittest
import unittest.mock as mock

import numpy as np
import pytest
import xarray as xr

from satpy.dataset.dataid import DataID, ModifierTuple, WavelengthRange
from satpy.readers.slstr_l1b import NCSLSTR1B, NCSLSTRAngles, NCSLSTRFlag, NCSLSTRGeo

local_id_keys_config = {"name": {
    "required": True,
},
    "wavelength": {
        "type": WavelengthRange,
    },
    "resolution": None,
    "calibration": {
        "enum": [
            "reflectance",
            "brightness_temperature",
            "radiance",
            "counts"
        ]
    },
    "stripe": {
        "enum": [
            "a",
            "b",
            "c",
            "i",
            "f",
        ]
    },
    "view": {
        "enum": [
            "nadir",
            "oblique",
        ]
    },
    "modifiers": {
        "required": True,
        "default": ModifierTuple(),
        "type": ModifierTuple,
    },
}


class TestSLSTRL1B(unittest.TestCase):
    """Common setup for SLSTR_L1B tests."""

    @mock.patch("satpy.readers.slstr_l1b.xr")
    def setUp(self, xr_):
        """Create a fake dataset using the given radiance data."""
        self.base_data = np.array(([1., 2., 3., 4., 5., 6., 7., 8., 9.],
                                   [7., 8., 9., 10., 11., 12., 13., 14., 15.],
                                   [16., 17., 18., 19., 20., 21., 22., 23., 24.],
                                   [25., 26., 27., 28., 29., 30., 31., 32., 33.]))
        self.ang_data = np.array(([345., 355., 1., 5., 10.],
                                  [346., 355., 0.5, 4.2, 15.],
                                  [342., 356., 1., 5.3, 12.],
                                  [344.3, 356.1, 0.0001, 4.9, 9.2],))

        self.tx_data = np.array(([101000, 100500, 100000, 99500, 99000],
                                 [101000, 100500, 100000, 99500, 99000],
                                 [101000, 100500, 100000, 99500, 99000],
                                 [101000, 100500, 100000, 99500, 99000]))
        self.ty_data = np.array(([199500, 199500, 199500, 199500, 199500],
                                 [200000, 200000, 200000, 200000, 200000],
                                 [200500, 200500, 200500, 200500, 200500],
                                 [201000, 201000, 201000, 201000, 201000],))

        self.ix_data = np.array(([100800, 100600, 100400, 100200, 100000, 99800, 99600, 99400, 99200],
                                 [100800, 100600, 100400, 100200, 100000, 99800, 99600, 99400, 99200],
                                 [100800, 100600, 100400, 100200, 100000, 99800, 99600, 99400, 99200],
                                 [100800, 100600, 100400, 100200, 100000, 99800, 99600, 99400, 99200]))
        self.iy_data = np.array(([199800, 199800, 199800, 199800, 199800, 199800, 199800, 199800, 199800],
                                 [200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000, 200000],
                                 [200600, 200600, 200600, 200600, 200600, 200600, 200600, 200600, 200600],
                                 [200800, 200800, 200800, 200800, 200800, 200800, 200800, 200800, 200800]))

        self.det_data = np.array(([0, 1, 1, 0, 1, 0, 1, 0, 1],
                                  [1, 0, 0, 0, 0, 1, 0, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 1],
                                  [0, 1, 1, 0, 1, 1, 1, 1, 0]))
        self.start_time = "2020-05-10T12:01:15.585Z"
        self.end_time = "2020-05-10T12:06:18.012Z"

        self.rad = xr.DataArray(
            self.base_data,
            dims=("columns", "rows"),
            attrs={"scale_factor": 1.0, "add_offset": 0.0,
                   "_FillValue": -32768, "units": "mW.m-2.sr-1.nm-1",})
        det = xr.DataArray(
            self.base_data,
            dims=("columns", "rows"),
            attrs={"scale_factor": 1.0, "add_offset": 0.0,"_FillValue": 255,})
        x_in = xr.DataArray(self.ix_data,dims=("columns", "rows"))
        y_in = xr.DataArray(self.iy_data,dims=("columns", "rows"))

        saa = xr.DataArray(self.ang_data,dims=("columns_t", "rows_t"))
        x_tx = xr.DataArray(self.tx_data,dims=("columns_t", "rows_t"))
        y_tx = xr.DataArray(self.ty_data,dims=("columns_t", "rows_t"))

        self.fake_dataset = xr.Dataset(
            data_vars={
                "S5_radiance_an": self.rad,
                "S9_BT_ao": self.rad,
                "foo_radiance_an": self.rad,
                "S5_solar_irradiances": self.rad,
                "geometry_tn": self.rad,
                "latitude_an": self.rad,
                "x_in": x_in,
                "y_in": y_in,
                "x_an": x_in,
                "y_an": y_in,
                "flags_an": self.rad,
                "detector_an": det,
                "x_tx": x_tx,
                "y_tx": y_tx,
                "solar_azimuth_tn": saa,
            },
            attrs={
                "start_time": self.start_time,
                "stop_time": self.end_time,
            },
        )


def make_dataid(**items):
    """Make a data id."""
    return DataID(local_id_keys_config, **items)


class TestSLSTRReader(TestSLSTRL1B):
    """Test various nc_slstr file handlers."""

    class FakeSpl:
        """Fake return function for SPL interpolation."""

        @staticmethod
        def ev(foo_x, foo_y):
            """Fake function to return interpolated data."""
            return np.zeros((3, 2))

    @mock.patch("satpy.readers.slstr_l1b.xr")
    @mock.patch("scipy.interpolate.RectBivariateSpline")
    def test_instantiate(self, bvs_, xr_):
        """Test initialization of file handlers."""
        bvs_.return_value = self.FakeSpl
        xr_.open_dataset.return_value = self.fake_dataset

        good_start = dt.datetime.strptime(self.start_time,
                                          "%Y-%m-%dT%H:%M:%S.%fZ")
        good_end = dt.datetime.strptime(self.end_time,
                                        "%Y-%m-%dT%H:%M:%S.%fZ")

        ds_id = make_dataid(name="foo", calibration="radiance",
                            stripe="a", view="nadir")
        ds_id_500 = make_dataid(name="foo", calibration="radiance",
                                stripe="a", view="nadir", resolution=500)
        filename_info = {"mission_id": "S3A", "dataset_name": "foo",
                         "start_time": 0, "end_time": 0,
                         "stripe": "a", "view": "n"}
        test = NCSLSTR1B("somedir/S1_radiance_an.nc", filename_info, "c")
        assert test.view == "nadir"
        assert test.stripe == "a"
        with pytest.warns(UserWarning, match=r"No radiance adjustment supplied for channel"):
            test.get_dataset(ds_id, dict(filename_info, **{"file_key": "foo"}))
        assert test.start_time == good_start
        assert test.end_time == good_end
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        filename_info = {"mission_id": "S3A", "dataset_name": "foo",
                         "start_time": 0, "end_time": 0,
                         "stripe": "c", "view": "o"}
        test = NCSLSTR1B("somedir/S1_radiance_co.nc", filename_info, "c")
        assert test.view == "oblique"
        assert test.stripe == "c"
        test.get_dataset(ds_id, dict(filename_info, **{"file_key": "foo"}))
        assert test.start_time == good_start
        assert test.end_time == good_end
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        filename_info = {"mission_id": "S3A", "dataset_name": "foo",
                         "start_time": 0, "end_time": 0,
                         "stripe": "a", "view": "n"}
        test = NCSLSTRGeo("somedir/geometry_an.nc", filename_info, "c")
        test.get_dataset(ds_id, dict(filename_info, **{"file_key": "latitude_{stripe:1s}{view:1s}"}))
        assert test.start_time == good_start
        assert test.end_time == good_end
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        test = NCSLSTRFlag("somedir/S1_radiance_an.nc", filename_info, "c")
        test.get_dataset(ds_id, dict(filename_info, **{"file_key": "flags_{stripe:1s}{view:1s}"}))
        assert test.view == "nadir"
        assert test.stripe == "a"
        assert test.start_time == good_start
        assert test.end_time == good_end
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()

        test = NCSLSTRAngles("somedir/S1_radiance_an.nc", filename_info, "c")
        test.get_dataset(ds_id, dict(filename_info, **{"file_key": "geometry_t{view:1s}"}))
        assert test.start_time == good_start
        assert test.end_time == good_end
        xr_.open_dataset.assert_called()
        xr_.open_dataset.reset_mock()
        test.get_dataset(ds_id_500, dict(filename_info, **{"file_key": "geometry_t{view:1s}"}))


class TestSLSTRCalibration(TestSLSTRL1B):
    """Test the implementation of the calibration factors."""

    @mock.patch("satpy.readers.slstr_l1b.xr")
    def test_radiance_calibration(self, xr_):
        """Test radiance calibration steps."""
        from satpy.readers.slstr_l1b import CHANCALIB_FACTORS
        xr_.open_dataset.return_value = self.fake_dataset

        ds_id = make_dataid(name="foo", calibration="radiance",
                            stripe="a", view="nadir")
        filename_info = {"mission_id": "S3A", "dataset_name": "foo",
                         "start_time": 0, "end_time": 0,
                         "stripe": "a", "view": "n"}

        test = NCSLSTR1B("somedir/S1_radiance_co.nc", filename_info, "c")
        # Check warning is raised if we don't have calibration
        with pytest.warns(UserWarning, match=r"No radiance adjustment supplied for channel"):
            test.get_dataset(ds_id, dict(filename_info, **{"file_key": "foo"}))

        # Check user calibration is used correctly
        test = NCSLSTR1B("somedir/S1_radiance_co.nc", filename_info, "c",
                         user_calibration={"foo_nadir": 0.4})
        data = test.get_dataset(ds_id, dict(filename_info, **{"file_key": "foo"}))
        np.testing.assert_allclose(data.values, self.base_data * 0.4)

        # Check internal calibration is used correctly
        ds_id = make_dataid(name="S5", calibration="radiance", stripe="a", view="nadir")
        filename_info["dataset_name"] = "S5"
        test = NCSLSTR1B("somedir/S1_radiance_an.nc", filename_info, "c")
        data = test.get_dataset(ds_id, dict(filename_info, **{"file_key": "S5"}))
        np.testing.assert_allclose(data.values,
                                   self.base_data * CHANCALIB_FACTORS["S5_nadir"])

    @mock.patch("satpy.readers.slstr_l1b.xr")
    @mock.patch("satpy.readers.slstr_l1b.da")
    def test_reflectance_calibration(self, da_, xr_):
        """Test reflectance calibration."""
        xr_.open_dataset.return_value = self.fake_dataset
        da_.map_blocks.return_value = self.rad / 100.
        filename_info = {"mission_id": "S3A", "dataset_name": "S5",
                         "start_time": 0, "end_time": 0,
                         "stripe": "a", "view": "n"}
        ds_id = make_dataid(name="S5", calibration="reflectance", stripe="a", view="nadir")
        test = NCSLSTR1B("somedir/S1_radiance_an.nc", filename_info, "c")
        data = test.get_dataset(ds_id, dict(filename_info, **{"file_key": "S5"}))
        assert data.units == "%"
        np.testing.assert_allclose(data.values, self.rad * np.pi)

    def test_cal_rad(self):
        """Test the radiance to reflectance converter."""
        rad = np.array([10., 20., 30., 40., 50., 60., 70.])
        didx = np.array([1, 2., 1., 3., 2., 2., 0.])
        solflux = np.array([100., 200., 300., 400.])

        good_rad = np.array([1. / 20., 1. / 15., 3. / 20., 1. / 10., 1. / 6., 2. / 10., 7. / 10.])

        out_rad = NCSLSTR1B._cal_rad(rad, didx, solflux)
        np.testing.assert_allclose(out_rad, good_rad)


class TestSLSTRAngles(TestSLSTRL1B):
    """Test the implementation of the angle reconstruction."""

    @mock.patch("satpy.readers.slstr_l1b.xr.open_dataset")
    def test_radiance_calibration(self, xr_):
        """Test radiance calibration steps."""
        xr_.return_value = self.fake_dataset

        res = np.array([[350.17659522, 353.31335956, 356.1441152, 358.55896366,
                         0.4400064, 1.79903533, 3.18424353, 5.28496166, 8.80226547],
                        [350.02717525, 353.48225908, 356.37905054, 358.72059252,
                         0.5, 1.81840943, 3.23819125, 5.44552178, 9.13958444],
                        [349.30986294, 354.38594648, 357.53203205, 359.49732861,
                         1.00001215, 2.61842911, 4.45389453, 6.49348929, 8.72979111],
                        [349.68116993, 354.62824808, 357.55405421, 359.31582013,
                         0.74004914, 2.46431559, 4.44154365, 6.45655991, 8.30055946]])

        ds_id = make_dataid(name="solar_azimuth_angle", view="nadir")
        filename_info = {"mission_id": "S3A", "dataset_name": "solar_azimuth_angle",
                         "start_time": 0, "end_time": 0,
                         "stripe": "t", "view": "n"}

        test = NCSLSTRAngles("somedir/geometry_tn.nc", filename_info, "c")
        data = test.get_dataset(ds_id, dict(filename_info, **{"file_key": "solar_azimuth_tn"}))
        assert data.attrs["units"] == "degrees"
        assert data.shape == res.shape
        np.testing.assert_allclose(data.values, res)
