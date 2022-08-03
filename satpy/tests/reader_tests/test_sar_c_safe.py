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
"""Module for testing the satpy.readers.sar-c_safe module."""

import unittest
import unittest.mock as mock
from enum import Enum
from io import BytesIO

import dask.array as da
import numpy as np
import xarray as xr

from satpy.dataset import DataQuery
from satpy.readers.sar_c_safe import SAFEXMLAnnotation, SAFEXMLCalibration, SAFEXMLNoise


class TestSAFEGRD(unittest.TestCase):
    """Test the SAFE GRD file handler."""

    @mock.patch('rasterio.open')
    def setUp(self, mocked_rio_open):
        """Set up the test case."""
        from satpy.readers.sar_c_safe import SAFEGRD
        filename_info = {'mission_id': 'S1A', 'dataset_name': 'foo', 'start_time': 0, 'end_time': 0,
                         'polarization': 'vv'}
        filetype_info = 'bla'
        self.noisefh = mock.MagicMock()
        self.noisefh.get_noise_correction.return_value = xr.DataArray(np.zeros((2, 2)), dims=['y', 'x'])
        self.calfh = mock.MagicMock()
        self.calfh.get_calibration_constant.return_value = 1
        self.calfh.get_calibration.return_value = xr.DataArray(np.ones((2, 2)), dims=['y', 'x'])
        self.annotationfh = mock.MagicMock()

        self.test_fh = SAFEGRD('S1A_IW_GRDH_1SDV_20190201T024655_20190201T024720_025730_02DC2A_AE07.SAFE/measurement/'
                               's1a-iw-grd-vv-20190201t024655-20190201t024720-025730-02dc2a-001.tiff',
                               filename_info, filetype_info, self.calfh, self.noisefh, self.annotationfh)
        self.mocked_rio_open = mocked_rio_open

    def test_instantiate(self):
        """Test initialization of file handlers."""
        assert self.test_fh._polarization == 'vv'
        assert self.test_fh.calibration == self.calfh
        assert self.test_fh.noise == self.noisefh
        self.mocked_rio_open.assert_called()

    @mock.patch('rioxarray.open_rasterio')
    def test_read_calibrated_natural(self, mocked_rioxarray_open):
        """Test the calibration routines."""
        calibration = mock.MagicMock()
        calibration.name = "sigma_nought"
        mocked_rioxarray_open.return_value = xr.DataArray(da.from_array(np.array([[0, 1], [2, 3]])), dims=['y', 'x'])
        xarr = self.test_fh.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                  calibration=calibration, quantity='natural'), info=dict())
        np.testing.assert_allclose(xarr, [[np.nan, 2], [5, 10]])

    @mock.patch('rioxarray.open_rasterio')
    def test_read_calibrated_dB(self, mocked_rioxarray_open):
        """Test the calibration routines."""
        calibration = mock.MagicMock()
        calibration.name = "sigma_nought"
        mocked_rioxarray_open.return_value = xr.DataArray(da.from_array(np.array([[0, 1], [2, 3]])), dims=['y', 'x'])
        xarr = self.test_fh.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                  calibration=calibration, quantity='dB'), info=dict())
        np.testing.assert_allclose(xarr, [[np.nan, 3.0103], [6.9897, 10]])

    def test_read_lon_lats(self):
        """Test reading lons and lats."""

        class FakeGCP:

            def __init__(self, *args):
                self.row, self.col, self.x, self.y, self.z = args

        gcps = [FakeGCP(0, 0, 0, 0, 0),
                FakeGCP(0, 3, 1, 0, 0),
                FakeGCP(3, 0, 0, 1, 0),
                FakeGCP(3, 3, 1, 1, 0),
                FakeGCP(0, 7, 2, 0, 0),
                FakeGCP(3, 7, 2, 1, 0),
                FakeGCP(7, 7, 2, 2, 0),
                FakeGCP(7, 3, 1, 2, 0),
                FakeGCP(7, 0, 0, 2, 0),
                FakeGCP(0, 15, 3, 0, 0),
                FakeGCP(3, 15, 3, 1, 0),
                FakeGCP(7, 15, 3, 2, 0),
                FakeGCP(15, 15, 3, 3, 0),
                FakeGCP(15, 7, 2, 3, 0),
                FakeGCP(15, 3, 1, 3, 0),
                FakeGCP(15, 0, 0, 3, 0),
                ]

        crs = dict(init='epsg:4326')

        self.mocked_rio_open.return_value.gcps = [gcps, crs]
        self.mocked_rio_open.return_value.shape = [16, 16]

        query = DataQuery(name="longitude", polarization="vv")
        xarr = self.test_fh.get_dataset(query, info=dict())
        expected = np.array([[3.79492915e-16, 5.91666667e-01, 9.09722222e-01,
                              1.00000000e+00, 9.08333333e-01, 6.80555556e-01,
                              3.62500000e-01, 8.32667268e-17, -3.61111111e-01,
                              -6.75000000e-01, -8.95833333e-01, -9.77777778e-01,
                              -8.75000000e-01, -5.41666667e-01, 6.80555556e-02,
                              1.00000000e+00],
                             [1.19166667e+00, 1.32437500e+00, 1.36941964e+00,
                              1.34166667e+00, 1.25598214e+00, 1.12723214e+00,
                              9.70282738e-01, 8.00000000e-01, 6.31250000e-01,
                              4.78898810e-01, 3.57812500e-01, 2.82857143e-01,
                              2.68898810e-01, 3.30803571e-01, 4.83437500e-01,
                              7.41666667e-01],
                             [1.82638889e+00, 1.77596726e+00, 1.72667765e+00,
                              1.67757937e+00, 1.62773172e+00, 1.57619402e+00,
                              1.52202558e+00, 1.46428571e+00, 1.40203373e+00,
                              1.33432894e+00, 1.26023065e+00, 1.17879819e+00,
                              1.08909084e+00, 9.90167942e-01, 8.81088790e-01,
                              7.60912698e-01],
                             [2.00000000e+00, 1.99166667e+00, 1.99305556e+00,
                              2.00000000e+00, 2.00833333e+00, 2.01388889e+00,
                              2.01250000e+00, 2.00000000e+00, 1.97222222e+00,
                              1.92500000e+00, 1.85416667e+00, 1.75555556e+00,
                              1.62500000e+00, 1.45833333e+00, 1.25138889e+00,
                              1.00000000e+00],
                             [1.80833333e+00, 2.01669643e+00, 2.18011267e+00,
                              2.30119048e+00, 2.38253827e+00, 2.42676446e+00,
                              2.43647747e+00, 2.41428571e+00, 2.36279762e+00,
                              2.28462160e+00, 2.18236607e+00, 2.05863946e+00,
                              1.91605017e+00, 1.75720663e+00, 1.58471726e+00,
                              1.40119048e+00],
                             [1.34722222e+00, 1.89627976e+00, 2.29940830e+00,
                              2.57341270e+00, 2.73509779e+00, 2.80126842e+00,
                              2.78872945e+00, 2.71428571e+00, 2.59474206e+00,
                              2.44690334e+00, 2.28757440e+00, 2.13356009e+00,
                              2.00166525e+00, 1.90869473e+00, 1.87145337e+00,
                              1.90674603e+00],
                             [7.12500000e-01, 1.67563988e+00, 2.36250177e+00,
                              2.80892857e+00, 3.05076318e+00, 3.12384850e+00,
                              3.06402742e+00, 2.90714286e+00, 2.68903770e+00,
                              2.44555485e+00, 2.21253720e+00, 2.02582766e+00,
                              1.92126913e+00, 1.93470451e+00, 2.10197669e+00,
                              2.45892857e+00],
                             [5.55111512e-16, 1.40000000e+00, 2.38095238e+00,
                              3.00000000e+00, 3.31428571e+00, 3.38095238e+00,
                              3.25714286e+00, 3.00000000e+00, 2.66666667e+00,
                              2.31428571e+00, 2.00000000e+00, 1.78095238e+00,
                              1.71428571e+00, 1.85714286e+00, 2.26666667e+00,
                              3.00000000e+00],
                             [-6.94444444e-01, 1.11458333e+00, 2.36631944e+00,
                              3.13888889e+00, 3.51041667e+00, 3.55902778e+00,
                              3.36284722e+00, 3.00000000e+00, 2.54861111e+00,
                              2.08680556e+00, 1.69270833e+00, 1.44444444e+00,
                              1.42013889e+00, 1.69791667e+00, 2.35590278e+00,
                              3.47222222e+00],
                             [-1.27500000e+00, 8.64613095e-01, 2.33016227e+00,
                              3.21785714e+00, 3.62390731e+00, 3.64452239e+00,
                              3.37591199e+00, 2.91428571e+00, 2.35585317e+00,
                              1.79682398e+00, 1.33340774e+00, 1.06181406e+00,
                              1.07825255e+00, 1.47893282e+00, 2.36006448e+00,
                              3.81785714e+00],
                             [-1.64583333e+00, 6.95312500e-01, 2.28404018e+00,
                              3.22916667e+00, 3.63950893e+00, 3.62388393e+00,
                              3.29110863e+00, 2.75000000e+00, 2.10937500e+00,
                              1.47805060e+00, 9.64843750e-01, 6.78571429e-01,
                              7.28050595e-01, 1.22209821e+00, 2.26953125e+00,
                              3.97916667e+00],
                             [-1.71111111e+00, 6.51904762e-01, 2.23951247e+00,
                              3.16507937e+00, 3.54197279e+00, 3.48356009e+00,
                              3.10320862e+00, 2.51428571e+00, 1.83015873e+00,
                              1.16419501e+00, 6.29761905e-01, 3.40226757e-01,
                              4.08956916e-01, 9.49319728e-01, 2.07468254e+00,
                              3.89841270e+00],
                             [-1.37500000e+00, 7.79613095e-01, 2.20813846e+00,
                              3.01785714e+00, 3.31605017e+00, 3.20999858e+00,
                              2.80698342e+00, 2.21428571e+00, 1.53918651e+00,
                              8.88966837e-01, 3.70907738e-01, 9.22902494e-02,
                              1.60395408e-01, 6.82504252e-01, 1.76589782e+00,
                              3.51785714e+00],
                             [-5.41666667e-01, 1.12366071e+00, 2.20147747e+00,
                              2.77976190e+00, 2.94649235e+00, 2.78964711e+00,
                              2.39720451e+00, 1.85714286e+00, 1.25744048e+00,
                              6.86075680e-01, 2.31026786e-01, -1.97278912e-02,
                              2.17899660e-02, 4.43558673e-01, 1.33355655e+00,
                              2.77976190e+00],
                             [8.84722222e-01, 1.72927083e+00, 2.23108879e+00,
                              2.44305556e+00, 2.41805060e+00, 2.20895337e+00,
                              1.86864335e+00, 1.45000000e+00, 1.00590278e+00,
                              5.89231151e-01, 2.52864583e-01, 4.96825397e-02,
                              3.25644841e-02, 2.54389881e-01, 7.68038194e-01,
                              1.62638889e+00],
                             [3.00000000e+00, 2.64166667e+00, 2.30853175e+00,
                              2.00000000e+00, 1.71547619e+00, 1.45436508e+00,
                              1.21607143e+00, 1.00000000e+00, 8.05555556e-01,
                              6.32142857e-01, 4.79166667e-01, 3.46031746e-01,
                              2.32142857e-01, 1.36904762e-01, 5.97222222e-02,
                              0.00000000e+00]])
        np.testing.assert_allclose(xarr.values, expected)


annotation_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<product>
  <adsHeader>
    <missionId>S1B</missionId>
    <productType>GRD</productType>
    <polarisation>HH</polarisation>
    <mode>EW</mode>
    <swath>EW</swath>
    <startTime>2020-03-15T05:04:28.137817</startTime>
    <stopTime>2020-03-15T05:05:32.416171</stopTime>
    <absoluteOrbitNumber>20698</absoluteOrbitNumber>
    <missionDataTakeId>160707</missionDataTakeId>
    <imageNumber>001</imageNumber>
  </adsHeader>
  <imageAnnotation>
    <imageInformation>
      <productFirstLineUtcTime>2020-03-15T05:04:28.137817</productFirstLineUtcTime>
      <productLastLineUtcTime>2020-03-15T05:05:32.416171</productLastLineUtcTime>
      <ascendingNodeTime>2020-03-15T04:33:22.256260</ascendingNodeTime>
      <anchorTime>2020-03-15T05:04:28.320641</anchorTime>
      <productComposition>Slice</productComposition>
      <sliceNumber>1</sliceNumber>
      <sliceList count="3">
        <slice>
          <sliceNumber>1</sliceNumber>
          <sensingStartTime>2020-03-15T05:04:29.485847</sensingStartTime>
          <sensingStopTime>2020-03-15T05:05:36.317420</sensingStopTime>
        </slice>
        <slice>
          <sliceNumber>2</sliceNumber>
          <sensingStartTime>2020-03-15T05:05:30.253413</sensingStartTime>
          <sensingStopTime>2020-03-15T05:06:34.046608</sensingStopTime>
        </slice>
        <slice>
          <sliceNumber>3</sliceNumber>
          <sensingStartTime>2020-03-15T05:06:31.020979</sensingStartTime>
          <sensingStopTime>2020-03-15T05:07:31.775796</sensingStopTime>
        </slice>
      </sliceList>
      <slantRangeTime>4.955163637998161e-03</slantRangeTime>
      <pixelValue>Detected</pixelValue>
      <outputPixels>16 bit Unsigned Integer</outputPixels>
      <rangePixelSpacing>4.000000e+01</rangePixelSpacing>
      <azimuthPixelSpacing>4.000000e+01</azimuthPixelSpacing>
      <azimuthTimeInterval>5.998353361537205e-03</azimuthTimeInterval>
      <azimuthFrequency>3.425601970000000e+02</azimuthFrequency>
      <numberOfSamples>10</numberOfSamples>
      <numberOfLines>10</numberOfLines>
      <zeroDopMinusAcqTime>-1.366569000000000e+00</zeroDopMinusAcqTime>
      <incidenceAngleMidSwath>3.468272707039038e+01</incidenceAngleMidSwath>
      <imageStatistics>
        <outputDataMean>
          <re>4.873919e+02</re>
          <im>0.000000e+00</im>
        </outputDataMean>
        <outputDataStdDev>
          <re>2.451083e+02</re>
          <im>0.000000e+00</im>
        </outputDataStdDev>
      </imageStatistics>
    </imageInformation>
  </imageAnnotation>
  <geolocationGrid>
    <geolocationGridPointList count="4">
      <geolocationGridPoint>
        <azimuthTime>2018-02-12T03:24:58.493342</azimuthTime>
        <slantRangeTime>4.964462411376810e-03</slantRangeTime>
        <line>0</line>
        <pixel>0</pixel>
        <latitude>7.021017981690355e+01</latitude>
        <longitude>5.609684402205929e+01</longitude>
        <height>8.234046399593353e-04</height>
        <incidenceAngle>1.918318045731997e+01</incidenceAngle>
        <elevationAngle>1.720012646010728e+01</elevationAngle>
      </geolocationGridPoint>
      <geolocationGridPoint>
        <azimuthTime>2018-02-12T03:24:58.493342</azimuthTime>
        <slantRangeTime>4.964462411376810e-03</slantRangeTime>
        <line>0</line>
        <pixel>9</pixel>
        <latitude>7.021017981690355e+01</latitude>
        <longitude>5.609684402205929e+01</longitude>
        <height>8.234046399593353e-04</height>
        <incidenceAngle>1.918318045731997e+01</incidenceAngle>
        <elevationAngle>1.720012646010728e+01</elevationAngle>
      </geolocationGridPoint>
      <geolocationGridPoint>
        <azimuthTime>2018-02-12T03:24:58.493342</azimuthTime>
        <slantRangeTime>4.964462411376810e-03</slantRangeTime>
        <line>9</line>
        <pixel>0</pixel>
        <latitude>7.021017981690355e+01</latitude>
        <longitude>5.609684402205929e+01</longitude>
        <height>8.234046399593353e-04</height>
        <incidenceAngle>1.918318045731997e+01</incidenceAngle>
        <elevationAngle>1.720012646010728e+01</elevationAngle>
      </geolocationGridPoint>
      <geolocationGridPoint>
        <azimuthTime>2018-02-12T03:24:58.493342</azimuthTime>
        <slantRangeTime>4.964462411376810e-03</slantRangeTime>
        <line>9</line>
        <pixel>9</pixel>
        <latitude>7.021017981690355e+01</latitude>
        <longitude>5.609684402205929e+01</longitude>
        <height>8.234046399593353e-04</height>
        <incidenceAngle>1.918318045731997e+01</incidenceAngle>
        <elevationAngle>1.720012646010728e+01</elevationAngle>
      </geolocationGridPoint>
    </geolocationGridPointList>
  </geolocationGrid>
</product>
"""

noise_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<noise>
  <noiseRangeVectorList count="3">
    <noiseRangeVector>
      <azimuthTime>2020-03-15T05:04:28.137817</azimuthTime>
      <line>0</line>
      <pixel count="6">0 2 4 6 8 9</pixel>
      <noiseRangeLut count="6">0.00000e+00 2.00000e+00 4.00000e+00 6.00000e+00 8.00000e+00 9.00000e+00</noiseRangeLut>
    </noiseRangeVector>
    <noiseRangeVector>
      <azimuthTime>2020-03-15T05:04:28.137817</azimuthTime>
      <line>5</line>
      <pixel count="6">0 2 4 7 8 9</pixel>
      <noiseRangeLut count="6">0.00000e+00 2.00000e+00 4.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00</noiseRangeLut>
    </noiseRangeVector>
    <noiseRangeVector>
      <azimuthTime>2020-03-15T05:04:28.137817</azimuthTime>
      <line>9</line>
      <pixel count="6">0 2 5 7 8 9</pixel>
      <noiseRangeLut count="6">0.00000e+00 2.00000e+00 5.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00</noiseRangeLut>
    </noiseRangeVector>
  </noiseRangeVectorList>
  <noiseAzimuthVectorList count="8">
    <noiseAzimuthVector>
      <swath>IW1</swath>
      <firstAzimuthLine>0</firstAzimuthLine>
      <firstRangeSample>1</firstRangeSample>
      <lastAzimuthLine>1</lastAzimuthLine>
      <lastRangeSample>3</lastRangeSample>
      <line count="1">0</line>
      <noiseAzimuthLut count="1">1.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW1</swath>
      <firstAzimuthLine>2</firstAzimuthLine>
      <firstRangeSample>0</firstRangeSample>
      <lastAzimuthLine>9</lastAzimuthLine>
      <lastRangeSample>1</lastRangeSample>
      <line count="4">2 4 6 8</line>
      <noiseAzimuthLut count="4">2.000000e+00 2.000000e+00 2.000000e+00 2.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>2</firstAzimuthLine>
      <firstRangeSample>2</firstRangeSample>
      <lastAzimuthLine>4</lastAzimuthLine>
      <lastRangeSample>4</lastRangeSample>
      <line count="2">2 4</line>
      <noiseAzimuthLut count="2">3.000000e+00 3.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>2</firstAzimuthLine>
      <firstRangeSample>5</firstRangeSample>
      <lastAzimuthLine>4</lastAzimuthLine>
      <lastRangeSample>8</lastRangeSample>
      <line count="2">2 4</line>
      <noiseAzimuthLut count="2">4.000000e+00 4.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>5</firstAzimuthLine>
      <firstRangeSample>2</firstRangeSample>
      <lastAzimuthLine>7</lastAzimuthLine>
      <lastRangeSample>5</lastRangeSample>
      <line count="2">5 6</line>
      <noiseAzimuthLut count="2">5.000000e+00 5.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>5</firstAzimuthLine>
      <firstRangeSample>6</firstRangeSample>
      <lastAzimuthLine>7</lastAzimuthLine>
      <lastRangeSample>9</lastRangeSample>
      <line count="2">5 6</line>
      <noiseAzimuthLut count="2">6.000000e+00 6.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>8</firstAzimuthLine>
      <firstRangeSample>2</firstRangeSample>
      <lastAzimuthLine>9</lastAzimuthLine>
      <lastRangeSample>6</lastRangeSample>
      <line count="1">8</line>
      <noiseAzimuthLut count="1">7.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>8</firstAzimuthLine>
      <firstRangeSample>7</firstRangeSample>
      <lastAzimuthLine>9</lastAzimuthLine>
      <lastRangeSample>9</lastRangeSample>
      <line count="1">8</line>
      <noiseAzimuthLut count="1">8.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
  </noiseAzimuthVectorList>
</noise>
"""

noise_xml_with_holes = b"""<?xml version="1.0" encoding="UTF-8"?>
<noise>
  <noiseRangeVectorList count="3">
    <noiseRangeVector>
      <azimuthTime>2020-03-15T05:04:28.137817</azimuthTime>
      <line>0</line>
      <pixel count="6">0 2 4 6 8 9</pixel>
      <noiseRangeLut count="6">0.00000e+00 2.00000e+00 4.00000e+00 6.00000e+00 8.00000e+00 9.00000e+00</noiseRangeLut>
    </noiseRangeVector>
    <noiseRangeVector>
      <azimuthTime>2020-03-15T05:04:28.137817</azimuthTime>
      <line>5</line>
      <pixel count="6">0 2 4 7 8 9</pixel>
      <noiseRangeLut count="6">0.00000e+00 2.00000e+00 4.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00</noiseRangeLut>
    </noiseRangeVector>
    <noiseRangeVector>
      <azimuthTime>2020-03-15T05:04:28.137817</azimuthTime>
      <line>9</line>
      <pixel count="6">0 2 5 7 8 9</pixel>
      <noiseRangeLut count="6">0.00000e+00 2.00000e+00 5.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00</noiseRangeLut>
    </noiseRangeVector>
  </noiseRangeVectorList>
  <noiseAzimuthVectorList count="12">
    <noiseAzimuthVector>
      <swath>IW1</swath>
      <firstAzimuthLine>0</firstAzimuthLine>
      <firstRangeSample>3</firstRangeSample>
      <lastAzimuthLine>2</lastAzimuthLine>
      <lastRangeSample>5</lastRangeSample>
      <line count="1">0</line>
      <noiseAzimuthLut count="1">1.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW1</swath>
      <firstAzimuthLine>1</firstAzimuthLine>
      <firstRangeSample>0</firstRangeSample>
      <lastAzimuthLine>5</lastAzimuthLine>
      <lastRangeSample>1</lastRangeSample>
      <line count="4">2 4 5</line>
      <noiseAzimuthLut count="4">2.000000e+00 2.000000e+00 2.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>2</firstAzimuthLine>
      <firstRangeSample>8</firstRangeSample>
      <lastAzimuthLine>4</lastAzimuthLine>
      <lastRangeSample>9</lastRangeSample>
      <line count="2">2 4</line>
      <noiseAzimuthLut count="2">3.000000e+00 3.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>3</firstAzimuthLine>
      <firstRangeSample>2</firstRangeSample>
      <lastAzimuthLine>5</lastAzimuthLine>
      <lastRangeSample>3</lastRangeSample>
      <line count="2">3 5</line>
      <noiseAzimuthLut count="2">4.000000e+00 4.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>3</firstAzimuthLine>
      <firstRangeSample>4</firstRangeSample>
      <lastAzimuthLine>4</lastAzimuthLine>
      <lastRangeSample>5</lastRangeSample>
      <line count="2">3 4</line>
      <noiseAzimuthLut count="2">5.000000e+00 5.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>4</firstAzimuthLine>
      <firstRangeSample>6</firstRangeSample>
      <lastAzimuthLine>4</lastAzimuthLine>
      <lastRangeSample>7</lastRangeSample>
      <line count="2">4</line>
      <noiseAzimuthLut count="2">6.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>5</firstAzimuthLine>
      <firstRangeSample>4</firstRangeSample>
      <lastAzimuthLine>7</lastAzimuthLine>
      <lastRangeSample>6</lastRangeSample>
      <line count="1">5 7</line>
      <noiseAzimuthLut count="1">7.000000e+00 7.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>5</firstAzimuthLine>
      <firstRangeSample>7</firstRangeSample>
      <lastAzimuthLine>7</lastAzimuthLine>
      <lastRangeSample>9</lastRangeSample>
      <line count="1">6</line>
      <noiseAzimuthLut count="1">8.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>6</firstAzimuthLine>
      <firstRangeSample>0</firstRangeSample>
      <lastAzimuthLine>7</lastAzimuthLine>
      <lastRangeSample>3</lastRangeSample>
      <line count="2">6 7</line>
      <noiseAzimuthLut count="2">9.000000e+00 9.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>8</firstAzimuthLine>
      <firstRangeSample>0</firstRangeSample>
      <lastAzimuthLine>9</lastAzimuthLine>
      <lastRangeSample>0</lastRangeSample>
      <line count="2">8</line>
      <noiseAzimuthLut count="2">10.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW2</swath>
      <firstAzimuthLine>8</firstAzimuthLine>
      <firstRangeSample>2</firstRangeSample>
      <lastAzimuthLine>9</lastAzimuthLine>
      <lastRangeSample>3</lastRangeSample>
      <line count="1">8 9</line>
      <noiseAzimuthLut count="1">11.000000e+00 11.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
    <noiseAzimuthVector>
      <swath>IW3</swath>
      <firstAzimuthLine>8</firstAzimuthLine>
      <firstRangeSample>4</firstRangeSample>
      <lastAzimuthLine>8</lastAzimuthLine>
      <lastRangeSample>5</lastRangeSample>
      <line count="1">8</line>
      <noiseAzimuthLut count="1">12.000000e+00</noiseAzimuthLut>
    </noiseAzimuthVector>
  </noiseAzimuthVectorList>
</noise>
"""


calibration_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<calibration>
  <adsHeader>
    <missionId>S1A</missionId>
    <productType>GRD</productType>
    <polarisation>VV</polarisation>
    <mode>IW</mode>
    <swath>IW</swath>
    <startTime>2018-02-12T03:24:58.493726</startTime>
    <stopTime>2018-02-12T03:25:01.493726</stopTime>
    <absoluteOrbitNumber>20568</absoluteOrbitNumber>
    <missionDataTakeId>144162</missionDataTakeId>
    <imageNumber>001</imageNumber>
  </adsHeader>
  <calibrationInformation>
    <absoluteCalibrationConstant>1.000000e+00</absoluteCalibrationConstant>
  </calibrationInformation>
  <calibrationVectorList count="4">
    <calibrationVector>
      <azimuthTime>2018-02-12T03:24:58.493726</azimuthTime>
      <line>0</line>
      <pixel count="6">0 2 4 6 8 9</pixel>
      <sigmaNought count="6">1.894274e+03 1.788593e+03 1.320240e+03 1.277968e+03 1.277968e+03 1.277968e+03</sigmaNought>
      <betaNought count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</betaNought>
      <gamma count="6">1.840695e+03 1.718649e+03 1.187203e+03 1.185249e+03 1.183303e+03 1.181365e+03</gamma>
      <dn count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</dn>
    </calibrationVector>
    <calibrationVector>
      <azimuthTime>2018-02-12T03:24:59.493726</azimuthTime>
      <line>3</line>
      <pixel count="6">0 2 4 6 8 9</pixel>
      <sigmaNought count="6">1.894274e+03 1.788593e+03 1.320240e+03 1.277968e+03 1.277968e+03 1.277968e+03</sigmaNought>
      <betaNought count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</betaNought>
      <gamma count="6">1.840695e+03 1.718649e+03 1.187203e+03 1.185249e+03 1.183303e+03 1.181365e+03</gamma>
      <dn count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</dn>
    </calibrationVector>
    <calibrationVector>
      <azimuthTime>2018-02-12T03:25:00.493726</azimuthTime>
      <line>6</line>
      <pixel count="6">0 2 4 6 8 9</pixel>
      <sigmaNought count="6">1.894274e+03 1.788593e+03 1.320240e+03 1.277968e+03 1.277968e+03 1.277968e+03</sigmaNought>
      <betaNought count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</betaNought>
      <gamma count="6">1.840695e+03 1.718649e+03 1.187203e+03 1.185249e+03 1.183303e+03 1.181365e+03</gamma>
      <dn count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</dn>
    </calibrationVector>
    <calibrationVector>
      <azimuthTime>2018-02-12T03:25:01.493726</azimuthTime>
      <line>9</line>
      <pixel count="6">0 2 4 6 8 9</pixel>
      <sigmaNought count="6">1.894274e+03 1.788593e+03 1.320240e+03 1.277968e+03 1.277968e+03 1.277968e+03</sigmaNought>
      <betaNought count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</betaNought>
      <gamma count="6">1.840695e+03 1.718649e+03 1.187203e+03 1.185249e+03 1.183303e+03 1.181365e+03</gamma>
      <dn count="6">1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03 1.0870e+03</dn>
    </calibrationVector>
  </calibrationVectorList>
</calibration>
"""


class TestSAFEXMLNoise(unittest.TestCase):
    """Test the SAFE XML Noise file handler."""

    def setUp(self):
        """Set up the test case."""
        filename_info = dict(start_time=None, end_time=None, polarization="vv")
        self.annotation_fh = SAFEXMLAnnotation(BytesIO(annotation_xml), filename_info, mock.MagicMock())
        self.noise_fh = SAFEXMLNoise(BytesIO(noise_xml), filename_info, mock.MagicMock(), self.annotation_fh)

        self.expected_azimuth_noise = np.array([[np.nan, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                                [np.nan, 1, 1, 1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                                [2, 2, 3, 3, 3, 4, 4, 4, 4, np.nan],
                                                [2, 2, 3, 3, 3, 4, 4, 4, 4, np.nan],
                                                [2, 2, 3, 3, 3, 4, 4, 4, 4, np.nan],
                                                [2, 2, 5, 5, 5, 5, 6, 6, 6, 6],
                                                [2, 2, 5, 5, 5, 5, 6, 6, 6, 6],
                                                [2, 2, 5, 5, 5, 5, 6, 6, 6, 6],
                                                [2, 2, 7, 7, 7, 7, 7, 8, 8, 8],
                                                [2, 2, 7, 7, 7, 7, 7, 8, 8, 8],
                                                ])

        self.expected_range_noise = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                              ])

        self.noise_fh_with_holes = SAFEXMLNoise(BytesIO(noise_xml_with_holes), filename_info, mock.MagicMock(),
                                                self.annotation_fh)
        self.expected_azimuth_noise_with_holes = np.array(
            [[np.nan, np.nan, np.nan, 1, 1, 1, np.nan, np.nan, np.nan, np.nan],
             [2, 2, np.nan, 1, 1, 1, np.nan, np.nan, np.nan, np.nan],
             [2, 2, np.nan, 1, 1, 1, np.nan, np.nan, 3, 3],
             [2, 2, 4, 4, 5, 5, np.nan, np.nan, 3, 3],
             [2, 2, 4, 4, 5, 5, 6, 6, 3, 3],
             [2, 2, 4, 4, 7, 7, 7, 8, 8, 8],
             [9, 9, 9, 9, 7, 7, 7, 8, 8, 8],
             [9, 9, 9, 9, 7, 7, 7, 8, 8, 8],
             [10, np.nan, 11, 11, 12, 12, np.nan, np.nan, np.nan, np.nan],
             [10, np.nan, 11, 11, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
             ])

    def test_azimuth_noise_array(self):
        """Test reading the azimuth-noise array."""
        res = self.noise_fh.azimuth_noise_reader.read_azimuth_noise_array()
        np.testing.assert_array_equal(res, self.expected_azimuth_noise)

    def test_azimuth_noise_array_with_holes(self):
        """Test reading the azimuth-noise array."""
        res = self.noise_fh_with_holes.azimuth_noise_reader.read_azimuth_noise_array()
        np.testing.assert_array_equal(res, self.expected_azimuth_noise_with_holes)

    def test_range_noise_array(self):
        """Test reading the range-noise array."""
        res = self.noise_fh.read_range_noise_array(chunks=5)
        np.testing.assert_allclose(res, self.expected_range_noise)

    def test_get_noise_dataset(self):
        """Test using get_dataset for the noise."""
        query = DataQuery(name="noise", polarization="vv")
        res = self.noise_fh.get_dataset(query, {})
        np.testing.assert_allclose(res, self.expected_azimuth_noise * self.expected_range_noise)

    def test_get_noise_dataset_has_right_chunk_size(self):
        """Test using get_dataset for the noise has right chunk size in result."""
        query = DataQuery(name="noise", polarization="vv")
        res = self.noise_fh.get_dataset(query, {}, chunks=3)
        assert res.data.chunksize == (3, 3)


class Calibration(Enum):
    """Calibration levels."""

    gamma = 1
    sigma_nought = 2
    beta_nought = 3
    dn = 4


class TestSAFEXMLCalibration(unittest.TestCase):
    """Test the SAFE XML Calibration file handler."""

    def setUp(self):
        """Set up the test case."""
        filename_info = dict(start_time=None, end_time=None, polarization="vv")
        self.annotation_fh = SAFEXMLAnnotation(BytesIO(annotation_xml), filename_info, mock.MagicMock())
        self.calibration_fh = SAFEXMLCalibration(BytesIO(calibration_xml),
                                                 filename_info,
                                                 mock.MagicMock(),
                                                 self.annotation_fh)

        self.expected_gamma = np.array([[1840.695, 1779.672, 1718.649, 1452.926, 1187.203, 1186.226,
                                         1185.249, 1184.276, 1183.303, 1181.365]]) * np.ones((10, 1))

    def test_dn_calibration_array(self):
        """Test reading the dn calibration array."""
        expected_dn = np.ones((10, 10)) * 1087
        res = self.calibration_fh.get_calibration(Calibration.dn, chunks=5)
        np.testing.assert_allclose(res, expected_dn)

    def test_beta_calibration_array(self):
        """Test reading the beta calibration array."""
        expected_beta = np.ones((10, 10)) * 1087
        res = self.calibration_fh.get_calibration(Calibration.beta_nought, chunks=5)
        np.testing.assert_allclose(res, expected_beta)

    def test_sigma_calibration_array(self):
        """Test reading the sigma calibration array."""
        expected_sigma = np.array([[1894.274, 1841.4335, 1788.593, 1554.4165, 1320.24, 1299.104,
                                    1277.968, 1277.968, 1277.968, 1277.968]]) * np.ones((10, 1))
        res = self.calibration_fh.get_calibration(Calibration.sigma_nought, chunks=5)
        np.testing.assert_allclose(res, expected_sigma)

    def test_gamma_calibration_array(self):
        """Test reading the gamma calibration array."""
        res = self.calibration_fh.get_calibration(Calibration.gamma, chunks=5)
        np.testing.assert_allclose(res, self.expected_gamma)

    def test_get_calibration_dataset(self):
        """Test using get_dataset for the calibration."""
        query = DataQuery(name="gamma", polarization="vv")
        res = self.calibration_fh.get_dataset(query, {})
        np.testing.assert_allclose(res, self.expected_gamma)

    def test_get_calibration_dataset_has_right_chunk_size(self):
        """Test using get_dataset for the calibration yields array with right chunksize."""
        query = DataQuery(name="gamma", polarization="vv")
        res = self.calibration_fh.get_dataset(query, {}, chunks=3)
        assert res.data.chunksize == (3, 3)
        np.testing.assert_allclose(res, self.expected_gamma)

    def test_get_calibration_constant(self):
        """Test getting the calibration constant."""
        query = DataQuery(name="calibration_constant", polarization="vv")
        res = self.calibration_fh.get_dataset(query, {})
        assert res == 1


class TestSAFEXMLAnnotation(unittest.TestCase):
    """Test the SAFE XML Annotation file handler."""

    def setUp(self):
        """Set up the test case."""
        filename_info = dict(start_time=None, end_time=None, polarization="vv")
        self.annotation_fh = SAFEXMLAnnotation(BytesIO(annotation_xml), filename_info, mock.MagicMock())

    def test_incidence_angle(self):
        """Test reading the incidence angle."""
        query = DataQuery(name="incidence_angle", polarization="vv")
        res = self.annotation_fh.get_dataset(query, {})
        np.testing.assert_allclose(res, 19.18318046)
