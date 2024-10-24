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

import os
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pytest
import yaml

geotiepoints = pytest.importorskip("geotiepoints", "1.7.5")

from satpy._config import PACKAGE_CONFIG_PATH  # noqa: E402
from satpy.dataset import DataQuery  # noqa: E402
from satpy.dataset.dataid import DataID  # noqa: E402
from satpy.readers.sar_c_safe import Calibrator, Denoiser, SAFEXMLAnnotation  # noqa: E402

rasterio = pytest.importorskip("rasterio")


dirname_suffix = "20190201T024655_20190201T024720_025730_02DC2A_AE07"
filename_suffix = "20190201t024655-20190201t024720-025730-02dc2a"

START_TIME = datetime(2019, 2, 1, 2, 46, 55)
END_TIME = datetime(2019, 2, 1, 2, 47, 20)

@pytest.fixture(scope="module")
def granule_directory(tmp_path_factory):
  """Create a granule directory."""
  data_dir = tmp_path_factory.mktemp("data")
  gdir = data_dir / f"S1A_IW_GRDH_1SDV_{dirname_suffix}.SAFE"
  os.mkdir(gdir)
  return gdir


@pytest.fixture(scope="module")
def annotation_file(granule_directory):
  """Create an annotation file."""
  ann_dir = granule_directory / "annotation"
  os.makedirs(ann_dir, exist_ok=True)
  annotation_file = ann_dir / f"s1a-iw-grd-vv-{filename_suffix}-001.xml"
  with open(annotation_file, "wb") as fd:
      fd.write(annotation_xml)
  return annotation_file


@pytest.fixture(scope="module")
def annotation_filehandler(annotation_file):
  """Create an annotation filehandler."""
  filename_info = dict(start_time=START_TIME, end_time=END_TIME, polarization="vv")
  return SAFEXMLAnnotation(annotation_file, filename_info, None)


@pytest.fixture(scope="module")
def calibration_file(granule_directory):
  """Create a calibration file."""
  cal_dir = granule_directory / "annotation" / "calibration"
  os.makedirs(cal_dir, exist_ok=True)
  calibration_file = cal_dir / f"calibration-s1a-iw-grd-vv-{filename_suffix}-001.xml"
  with open(calibration_file, "wb") as fd:
      fd.write(calibration_xml)
  return Path(calibration_file)

@pytest.fixture(scope="module")
def calibration_filehandler(calibration_file, annotation_filehandler):
  """Create a calibration filehandler."""
  filename_info = dict(start_time=START_TIME, end_time=END_TIME, polarization="vv")
  return Calibrator(calibration_file,
                    filename_info,
                    None,
                    image_shape=annotation_filehandler.image_shape)

@pytest.fixture(scope="module")
def noise_file(granule_directory):
  """Create a noise file."""
  noise_dir = granule_directory / "annotation" / "calibration"
  os.makedirs(noise_dir, exist_ok=True)
  noise_file = noise_dir / f"noise-s1a-iw-grd-vv-{filename_suffix}-001.xml"
  with open(noise_file, "wb") as fd:
      fd.write(noise_xml)
  return noise_file


@pytest.fixture(scope="module")
def noise_filehandler(noise_file, annotation_filehandler):
  """Create a noise filehandler."""
  filename_info = dict(start_time=START_TIME, end_time=END_TIME, polarization="vv")
  return Denoiser(noise_file, filename_info, None, image_shape=annotation_filehandler.image_shape)


@pytest.fixture(scope="module")
def noise_with_holes_filehandler(annotation_filehandler, tmpdir_factory):
  """Create a noise filehandler from data with holes."""
  filename_info = dict(start_time=START_TIME, end_time=END_TIME, polarization="vv")
  noise_xml_file = tmpdir_factory.mktemp("data").join("noise_with_holes.xml")
  with open(noise_xml_file, "wb") as fd:
    fd.write(noise_xml_with_holes)
  noise_filehandler = Denoiser(noise_xml_file,
                               filename_info, None,
                               image_shape=annotation_filehandler.image_shape)
  return noise_filehandler



@pytest.fixture(scope="module")
def measurement_file(granule_directory):
  """Create a tiff measurement file."""
  GCP = rasterio.control.GroundControlPoint

  gcps = [GCP(0, 0, 0, 0, 0),
          GCP(0, 3, 1, 0, 0),
          GCP(3, 0, 0, 1, 0),
          GCP(3, 3, 1, 1, 0),
          GCP(0, 7, 2, 0, 0),
          GCP(3, 7, 2, 1, 0),
          GCP(7, 7, 2, 2, 0),
          GCP(7, 3, 1, 2, 0),
          GCP(7, 0, 0, 2, 0),
          GCP(0, 15, 3, 0, 0),
          GCP(3, 15, 3, 1, 0),
          GCP(7, 15, 3, 2, 0),
          GCP(15, 15, 3, 3, 0),
          GCP(15, 7, 2, 3, 0),
          GCP(15, 3, 1, 3, 0),
          GCP(15, 0, 0, 3, 0),
          ]
  Z = np.linspace(0, 30000, 100, dtype=np.uint16).reshape((10, 10))
  m_dir = granule_directory / "measurement"
  os.makedirs(m_dir, exist_ok=True)
  filename = m_dir / f"s1a-iw-grd-vv-{filename_suffix}-001.tiff"
  with rasterio.open(
    filename,
    "w",
    driver="GTiff",
    height=Z.shape[0],
    width=Z.shape[1],
    count=1,
    dtype=Z.dtype,
    crs="+proj=latlong",
    gcps=gcps) as dst:
      dst.write(Z, 1)
  return Path(filename)


@pytest.fixture(scope="module")
def measurement_filehandler(measurement_file, noise_filehandler, calibration_filehandler):
  """Create a measurement filehandler."""
  filename_info = {"mission_id": "S1A", "dataset_name": "foo", "start_time": START_TIME, "end_time": END_TIME,
                   "polarization": "vv"}
  filetype_info = None
  from satpy.readers.sar_c_safe import SAFEGRD
  filehandler =  SAFEGRD(measurement_file,
                         filename_info,
                         filetype_info,
                         calibration_filehandler,
                         noise_filehandler)
  return filehandler



expected_longitudes = np.array([[-0., 0.54230055, 0.87563228, 1., 0.91541479,
                                 0.62184442, 0.26733714, -0., -0.18015287, -0.27312165],
                                [1.0883956 , 1.25662247, 1.34380634, 1.34995884, 1.2750712 ,
                                 1.11911385, 0.9390845 , 0.79202785, 0.67796547, 0.59691204],
                                [1.75505196, 1.74123364, 1.71731849, 1.68330292, 1.63918145,
                                 1.58494674, 1.52376394, 1.45880655, 1.39007883, 1.31758574],
                                [2., 1.99615628, 1.99615609, 2., 2.00768917,
                                 2.0192253 , 2.02115051, 2.        , 1.95576762, 1.88845002],
                                [1.82332931, 2.02143515, 2.18032829, 2.30002491, 2.38053511,
                                 2.4218612 , 2.43113105, 2.41546985, 2.37487052, 2.3093278 ],
                                [1.22479001, 1.81701462, 2.26984318, 2.58335874, 2.75765719,
                                 2.79279164, 2.75366973, 2.70519769, 2.64737395, 2.58019762],
                                [0.51375081, 1.53781389, 2.3082042 , 2.82500549, 3.0885147 ,
                                 3.09893859, 2.98922885, 2.89232293, 2.8082302 , 2.7369586 ],
                                [0., 1.33889733, 2.33891557, 3., 3.32266837,
                                 3.30731797, 3.1383157 , 3., 2.8923933 , 2.81551297],
                                [-0.31638932, 1.22031759, 2.36197571, 3.10836734, 3.46019271,
                                 3.41800603, 3.20098223, 3.02826595, 2.89989242, 2.81588745],
                                [-0.43541441, 1.18211505, 2.37738272, 3.1501186 , 3.50112948,
                                 3.43104055, 3.17724665, 2.97712796, 2.83072911, 2.73808164]])


class Calibration(Enum):
    """Calibration levels."""

    gamma = 1
    sigma_nought = 2
    beta_nought = 3
    dn = 4


class TestSAFEGRD:
    """Test the SAFE GRD file handler."""

    def test_read_calibrated_natural(self, measurement_filehandler):
        """Test the calibration routines."""
        calibration = Calibration.sigma_nought
        xarr = measurement_filehandler.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                   calibration=calibration, quantity="natural"), info=dict())
        expected = np.array([[np.nan, 0.02707529], [2.55858416, 3.27611055]], dtype=np.float32)
        np.testing.assert_allclose(xarr.values[:2, :2], expected, rtol=2e-7)
        assert xarr.dtype == np.float32
        assert xarr.compute().dtype == np.float32

    def test_read_calibrated_dB(self, measurement_filehandler):
        """Test the calibration routines."""
        calibration = Calibration.sigma_nought
        xarr = measurement_filehandler.get_dataset(DataQuery(name="measurement", polarization="vv",
                                                   calibration=calibration, quantity="dB"), info=dict())
        expected = np.array([[np.nan, -15.674268], [4.079997, 5.153585]], dtype=np.float32)
        np.testing.assert_allclose(xarr.values[:2, :2], expected, rtol=1e-6)
        assert xarr.dtype == np.float32
        assert xarr.compute().dtype == np.float32

    def test_read_lon_lats(self, measurement_filehandler):
        """Test reading lons and lats."""
        query = DataQuery(name="longitude", polarization="vv")
        xarr = measurement_filehandler.get_dataset(query, info=dict())
        np.testing.assert_allclose(xarr.values, expected_longitudes)
        assert xarr.dtype == np.float64
        assert xarr.compute().dtype == np.float64


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


class TestSAFEXMLNoise:
    """Test the SAFE XML Noise file handler."""

    def setup_method(self):
        """Set up the test case."""
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

    def test_azimuth_noise_array(self, noise_filehandler):
        """Test reading the azimuth-noise array."""
        res = noise_filehandler.azimuth_noise_reader.read_azimuth_noise_array()
        np.testing.assert_array_equal(res, self.expected_azimuth_noise)

    def test_azimuth_noise_array_with_holes(self, noise_with_holes_filehandler):
        """Test reading the azimuth-noise array."""
        res = noise_with_holes_filehandler.azimuth_noise_reader.read_azimuth_noise_array()
        np.testing.assert_array_equal(res, self.expected_azimuth_noise_with_holes)

    def test_range_noise_array(self, noise_filehandler):
        """Test reading the range-noise array."""
        res = noise_filehandler.read_range_noise_array(chunks=5)
        np.testing.assert_allclose(res, self.expected_range_noise)

    def test_get_noise_dataset(self, noise_filehandler):
        """Test using get_dataset for the noise."""
        query = DataQuery(name="noise", polarization="vv")
        res = noise_filehandler.get_dataset(query, {})
        np.testing.assert_allclose(res, self.expected_azimuth_noise * self.expected_range_noise)
        assert res.dtype == np.float32
        assert res.compute().dtype == np.float32

    def test_get_noise_dataset_has_right_chunk_size(self, noise_filehandler):
        """Test using get_dataset for the noise has right chunk size in result."""
        query = DataQuery(name="noise", polarization="vv")
        res = noise_filehandler.get_dataset(query, {}, chunks=3)
        assert res.data.chunksize == (3, 3)


class TestSAFEXMLCalibration:
    """Test the SAFE XML Calibration file handler."""

    def setup_method(self):
      """Set up testing."""
      self.expected_gamma = np.array([[1840.695, 1779.672, 1718.649, 1452.926, 1187.203, 1186.226,
                                1185.249, 1184.276, 1183.303, 1181.365]]) * np.ones((10, 1))


    def test_dn_calibration_array(self, calibration_filehandler):
        """Test reading the dn calibration array."""
        expected_dn = np.ones((10, 10)) * 1087
        res = calibration_filehandler.get_calibration(Calibration.dn, chunks=5)
        np.testing.assert_allclose(res, expected_dn)
        assert res.dtype == np.float32
        assert res.compute().dtype == np.float32

    def test_beta_calibration_array(self, calibration_filehandler):
        """Test reading the beta calibration array."""
        expected_beta = np.ones((10, 10)) * 1087
        res = calibration_filehandler.get_calibration(Calibration.beta_nought, chunks=5)
        np.testing.assert_allclose(res, expected_beta)
        assert res.dtype == np.float32
        assert res.compute().dtype == np.float32

    def test_sigma_calibration_array(self, calibration_filehandler):
        """Test reading the sigma calibration array."""
        expected_sigma = np.array([[1894.274, 1841.4335, 1788.593, 1554.4165, 1320.24, 1299.104,
                                    1277.968, 1277.968, 1277.968, 1277.968]]) * np.ones((10, 1))
        res = calibration_filehandler.get_calibration(Calibration.sigma_nought, chunks=5)
        np.testing.assert_allclose(res, expected_sigma)
        assert res.dtype == np.float32
        assert res.compute().dtype == np.float32

    def test_gamma_calibration_array(self, calibration_filehandler):
        """Test reading the gamma calibration array."""
        res = calibration_filehandler.get_calibration(Calibration.gamma, chunks=5)
        np.testing.assert_allclose(res, self.expected_gamma)
        assert res.dtype == np.float32
        assert res.compute().dtype == np.float32

    def test_get_calibration_dataset(self, calibration_filehandler):
        """Test using get_dataset for the calibration."""
        query = DataQuery(name="gamma", polarization="vv")
        res = calibration_filehandler.get_dataset(query, {})
        np.testing.assert_allclose(res, self.expected_gamma)
        assert res.dtype == np.float32
        assert res.compute().dtype == np.float32

    def test_get_calibration_dataset_has_right_chunk_size(self, calibration_filehandler):
        """Test using get_dataset for the calibration yields array with right chunksize."""
        query = DataQuery(name="gamma", polarization="vv")
        res = calibration_filehandler.get_dataset(query, {}, chunks=3)
        assert res.data.chunksize == (3, 3)
        np.testing.assert_allclose(res, self.expected_gamma)

    def test_get_calibration_constant(self, calibration_filehandler):
        """Test getting the calibration constant."""
        query = DataQuery(name="calibration_constant", polarization="vv")
        res = calibration_filehandler.get_dataset(query, {})
        assert res == 1
        assert type(res) is np.float32


def test_incidence_angle(annotation_filehandler):
  """Test reading the incidence angle in an annotation file."""
  query = DataQuery(name="incidence_angle", polarization="vv")
  res = annotation_filehandler.get_dataset(query, {})
  np.testing.assert_allclose(res, 19.18318046)
  assert res.dtype == np.float32
  assert res.compute().dtype == np.float32


def test_reading_from_reader(measurement_file, calibration_file, noise_file, annotation_file):
  """Test reading using the reader defined in the config."""
  with open(Path(PACKAGE_CONFIG_PATH) / "readers" / "sar-c_safe.yaml") as fd:
    config = yaml.load(fd, Loader=yaml.UnsafeLoader)
  reader_class = config["reader"]["reader"]
  reader = reader_class(config)

  files = [measurement_file, calibration_file, noise_file, annotation_file]
  reader.create_storage_items(files)
  query = DataQuery(name="measurement", polarization="vv",
                    calibration="sigma_nought", quantity="dB")
  query = DataID(reader._id_keys, **query.to_dict())
  dataset_dict = reader.load([query])
  array = dataset_dict["measurement"]
  np.testing.assert_allclose(array.attrs["area"].lons, expected_longitudes)
  expected_db = np.array([[np.nan, -15.674268], [4.079997, 5.153585]])
  np.testing.assert_allclose(array.values[:2, :2], expected_db, rtol=1e-6)
  assert array.dtype == np.float32
  assert array.compute().dtype == np.float32


def test_filename_filtering_from_reader(measurement_file, calibration_file, noise_file, annotation_file, tmp_path):
  """Test that filenames get filtered before filehandlers are created."""
  with open(Path(PACKAGE_CONFIG_PATH) / "readers" / "sar-c_safe.yaml") as fd:
    config = yaml.load(fd, Loader=yaml.UnsafeLoader)
  reader_class = config["reader"]["reader"]
  filter_parameters = {"start_time": datetime(2019, 2, 1, 0, 0, 0),
                       "end_time": datetime(2019, 2, 1, 12, 0, 0)}
  reader = reader_class(config, filter_parameters)

  spurious_file = (tmp_path / "S1A_IW_GRDH_1SDV_20190202T024655_20190202T024720_025730_02DC2A_AE07.SAFE" /
                   "measurement" /
                   "s1a-iw-grd-vv-20190202t024655-20190202t024720-025730-02dc2a-001.tiff")


  files = [spurious_file, measurement_file, calibration_file, noise_file, annotation_file]

  files = reader.filter_selected_filenames(files)
  assert spurious_file not in files
  try:
    reader.create_storage_items(files)
  except rasterio.RasterioIOError as err:
     pytest.fail(str(err))


def test_swath_def_contains_gcps_and_bounding_box(measurement_file, calibration_file, noise_file, annotation_file):
  """Test reading using the reader defined in the config."""
  with open(Path(PACKAGE_CONFIG_PATH) / "readers" / "sar-c_safe.yaml") as fd:
    config = yaml.load(fd, Loader=yaml.UnsafeLoader)
  reader_class = config["reader"]["reader"]
  reader = reader_class(config)

  files = [measurement_file, calibration_file, noise_file, annotation_file]
  reader.create_storage_items(files)
  query = DataQuery(name="measurement", polarization="vv",
                    calibration="sigma_nought", quantity="dB")
  query = DataID(reader._id_keys, **query.to_dict())
  dataset_dict = reader.load([query])
  array = dataset_dict["measurement"]
  assert array.attrs["area"].attrs["gcps"] is not None
  assert array.attrs["area"].attrs["bounding_box"] is not None
