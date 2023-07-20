#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Module for testing the satpy.readers.sar-rs02 module."""

import os
import shutil
from datetime import datetime
from tempfile import mkdtemp

import numpy as np
import rasterio
from trollsift import parse

from satpy.readers.sar_rs02 import SARRS02CalibrationFileHandler, SARRS02MeasurementFileHandler
from satpy.tests.utils import make_dsq

imagery_pattern = ('{mission_id:3s}_{start_time:%Y%m%d_%H%M%S}_{filler1:4s}_{scan_mode:s}_{polarization_set:s}_{process'
                   'ing_level:s}_{filler2:6s}_{filler3:4s}_{filler4:8s}/imagery_{polarization:2s}.tif')


def create_fake_rs02_dataset():
    """Create a fake Radarsat 2 dataset."""
    dirpath = mkdtemp()
    rs2_name = "RS2_20220114_045216_0076_SCWA_HHHV_SGF_948791_1370_46579936"
    rs2_dir = os.path.join(dirpath, rs2_name)
    os.makedirs(rs2_dir)

    arr = np.full((30, 20), 2, dtype=np.uint16)

    for filename in ["imagery_HH.tif", "imagery_HV.tif"]:
        with rasterio.open(
                os.path.join(rs2_dir, filename),
                'w',
                driver='GTiff',
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype=arr.dtype,
                crs='+proj=latlong',
        ) as dst:
            dst.write(arr, 1)

    lut_prefix = ('<lut copyright="RADARSAT-2 Data and Products (c) MDA Geospatial Services Inc., 2022 - All Rights Res'
                  'erved.">'
                  '<offset>0.000000e+00</offset>'
                  '<gains>')
    lut_suffix = '</gains></lut>'
    sigma = ("3.178132e+09 3.177661e+09 3.177190e+09 3.176724e+09 3.176248e+09 3.175782e+09 3.175311e+09 3.174839e+09 "
             "3.174369e+09 3.173898e+09 3.173427e+09 3.172957e+09 3.172487e+09 3.172016e+09 3.171551e+09 3.171075e+09 "
             "3.170605e+09 3.170140e+09 3.169665e+09 3.169195e+09")

    beta = ("2.415050e+09 2.414618e+09 2.414185e+09 2.413757e+09 2.413320e+09 2.412892e+09 2.412460e+09 2.412027e+09 "
            "2.411595e+09 2.411163e+09 2.410731e+09 2.410299e+09 2.409867e+09 2.409435e+09 2.409008e+09 2.408572e+09 "
            "2.408141e+09 2.407713e+09 2.407278e+09 2.406846e+09")

    gamma = ("2.065928e+09 2.065709e+09 2.065489e+09 2.065273e+09 2.065051e+09 2.064835e+09 2.064615e+09 2.064396e+09 "
             "2.064177e+09 2.063958e+09 2.063739e+09 2.063520e+09 2.063301e+09 2.063081e+09 2.062866e+09 2.062644e+09 "
             "2.062425e+09 2.062209e+09 2.061987e+09 2.061768e+09")

    for lutname, arr in zip(["lutBeta.xml", "lutSigma.xml", "lutGamma.xml"], [beta, sigma, gamma]):
        with open(os.path.join(rs2_dir, lutname), mode="w") as dst:
            dst.write(lut_prefix + arr + lut_suffix)

    return dirpath, rs2_name


class TestSARRS02FileHandlers:
    """Test the Radarsat 2 file handlers."""

    def setup(self):
        """Set up test case."""
        self.base_dir, rs2_dir = create_fake_rs02_dataset()
        self.rs2_dir = os.path.join(self.base_dir, rs2_dir)
        basename = os.path.join(rs2_dir, "imagery_HV.tif")
        self.filename = os.path.join(self.base_dir, basename)
        self.filename_info = parse(imagery_pattern, basename)

    def teardown(self):
        """Tear down the test case."""
        shutil.rmtree(self.base_dir)

    def test_image_shape(self):
        """Test that image has correct shape."""
        fh = SARRS02MeasurementFileHandler(self.filename, self.filename_info, None, None, None, None)
        data = fh.get_dataset(make_dsq(name="hv", polarization="hv"))
        assert data.shape == (30, 20)

    def test_lut_reading(self):
        """Test that calibration luts are read correctly."""
        filename_calibration = os.path.join(self.rs2_dir, "lutGamma.xml")
        filename_info = self.filename_info.copy()
        cal_fh = SARRS02CalibrationFileHandler(filename_calibration, filename_info, None)
        cal_data = cal_fh.get_dataset()
        assert cal_data.shape == (20, )
        assert cal_data[0] == 2.065928e+09

    def test_calibration_to_gamma(self):
        """Test the calibration of the data to gamma."""
        calibration = "gamma"
        self.check_calibrated_data(calibration)

    def check_calibrated_data(self, calibration):
        """Check the calibrated data is correct."""
        cal_data, fh = self.get_fh_with_calibration(calibration)
        data = fh.get_dataset(make_dsq(name="measurement", polarization="hv")).astype(float)
        calibrated = fh.get_dataset(make_dsq(name="hv", polarization="hv", calibration=calibration))
        np.testing.assert_allclose(calibrated, data * data / cal_data)

    def get_fh_with_calibration(self, calibration):
        """Get a filehandler with calibration filehandlers provided."""
        calibrations = {"beta_nought": ("lutBeta.xml", "cal_beta_fh"),
                        "sigma_nought": ("lutSigma.xml", "cal_sigma_fh"),
                        "gamma": ("lutGamma.xml", "cal_gamma_fh")}
        kwargs = dict(cal_beta_fh=None, cal_sigma_fh=None, cal_gamma_fh=None)
        lut_filename, keyword = calibrations[calibration]
        filename_calibration = os.path.join(self.rs2_dir, lut_filename)
        cal_fh = SARRS02CalibrationFileHandler(filename_calibration, self.filename_info, None)
        cal_data = cal_fh.get_dataset()
        kwargs[keyword] = cal_fh
        fh = SARRS02MeasurementFileHandler(self.filename, self.filename_info, None, **kwargs)
        return cal_data, fh

    def test_calibration_to_sigma(self):
        """Test the calibration of the data to sigma nought."""
        calibration = "sigma_nought"
        self.check_calibrated_data(calibration)

    def test_calibration_to_beta(self):
        """Test the calibration of the data to beta nought."""
        calibration = "beta_nought"
        self.check_calibrated_data(calibration)

    def test_wrong_file_makes_get_dataset_return_none(self):
        """Test that providing a file with wrong polarization makes get_dataset return None."""
        fh = SARRS02MeasurementFileHandler(self.filename, self.filename_info, None, None, None, None)
        data = fh.get_dataset(make_dsq(name="hh", polarization="hh"))
        assert data is None

    def test_units_are_provided(self):
        """Test that the resulting DataArray has units."""
        fh = SARRS02MeasurementFileHandler(self.filename, self.filename_info, None, None, None, None)
        data = fh.get_dataset(make_dsq(name="hv", polarization="hv"))
        assert data.attrs["units"] == "1"

    def test_units_are_db_when_requested(self):
        """Test that the resulting DataArray has units in dB when needed."""
        fh = SARRS02MeasurementFileHandler(self.filename, self.filename_info, None, None, None, None)
        data = fh.get_dataset(make_dsq(name="hv", polarization="hv", quantity="dB"))
        assert data.attrs["units"] == "dB"

    def test_start_time(self):
        """Test that the start time of the file handler is correct."""
        fh = SARRS02MeasurementFileHandler(self.filename, self.filename_info, None, None, None, None)
        assert fh.start_time == datetime(2022, 1, 14, 4, 52, 16)

    def test_calibration_start_time(self):
        """Test that the start time of the calibration file handler is correct."""
        filename_calibration = os.path.join(self.rs2_dir, "lutGamma.xml")
        fh = SARRS02CalibrationFileHandler(filename_calibration, self.filename_info, None)
        assert fh.start_time == datetime(2022, 1, 14, 4, 52, 16)
