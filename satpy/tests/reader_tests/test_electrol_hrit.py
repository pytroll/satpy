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
"""The HRIT electrol reader tests package."""

import datetime
import numpy as np
import dask.array as da
from xarray import DataArray

from satpy.readers.electrol_hrit import (recarray2dict, prologue,
                                         HRITGOMSPrologueFileHandler,
                                         HRITGOMSEpilogueFileHandler,
                                         HRITGOMSFileHandler,
                                         satellite_status,
                                         image_acquisition,
                                         epilogue)

import unittest
from unittest import mock

# Simplify some type selections
f64_t = np.float64
i32_t = np.int32
u32_t = np.uint32


class Testrecarray2dict(unittest.TestCase):
    """Test the function that converts numpy record arrays into dicts for use within SatPy."""
    def test_fun(self):
        inner_st = np.dtype([('test_str', '<S20'), ('test_int', 'i4')])
        outer_st = np.dtype([('test_sec', inner_st), ('test_flt', 'f4')])

        inner_da = np.array([('Testing', 10)], dtype=inner_st)
        outer_da = np.array([(inner_da, 1.45)], dtype=outer_st)

        expected = {'test_sec': {'test_str': np.array([b'Testing'], dtype='<S20'),
                                 'test_int': np.array([10], dtype=np.int32)},
                    'test_flt': np.array([1.45], dtype=np.float32)}
        self.assertEqual(expected, recarray2dict(outer_da))


class TestHRITGOMSProFileHandler(unittest.TestCase):
    """Test the HRIT Prologue FileHandler."""
    # Below are variable definitions used in testing the prologue reader.
    # These values are taken from a typical ELECTRO-L HRIT scene
    test_sat_status = {'TagType': 2,
                       'TagLength': 292,
                       'SatelliteID': 19002,
                       'SatelliteName': b'ELECTRO',
                       'NominalLongitude': 1.3264,
                       'SatelliteCondition': 1,
                       'TimeOffset': 0.}

    test_img_acq = {'TagType': np.repeat(3, 10).astype(u32_t),
                    'TagLength': np.repeat(24, 10).astype(u32_t),
                    'Status': np.repeat(2, 10).astype(u32_t),
                    'StartDelay': np.repeat(9119019, 10).astype(np.int32),
                    'Cel': np.repeat(0., 10)}

    test_calib = np.full((10, 1024), 50, dtype=np.int32)

    test_pro = {'SatelliteStatus': test_sat_status,
                'ImageAcquisition': test_img_acq,
                'ImageCalibration': test_calib}

    @mock.patch('satpy.readers.electrol_hrit.np.fromfile')
    @mock.patch('satpy.readers.electrol_hrit.HRITFileHandler.__init__')
    def test_init(self, new_fh_init, fromfile):
        """Setup the hrit file handler for testing."""
        new_fh_init.return_value.filename = 'filename'
        HRITGOMSPrologueFileHandler.filename = 'filename'
        HRITGOMSPrologueFileHandler.mda = {'total_header_length': 1}

        # Set up the test data to use within the prologue reader
        tss = np.array([(2, 292, 19002, 'ELECTRO', 1.3264, 1, 0.)],
                       dtype=satellite_status)
        tia = np.tile(np.array([(3, 24, 2, 9119019, 0.)],
                      dtype=image_acquisition), (1, 10))
        tc = np.full((10, 1024), 50)
        rtv = np.array([(tss, tia, tc)], dtype=prologue)
        # Pretend to return this when reading a fake prologue
        fromfile.return_value = rtv
        m = mock.mock_open()

        with mock.patch('satpy.readers.electrol_hrit.open',
                        m, create=True) as newopen:
            newopen.return_value.__enter__.return_value.seek.return_value = 1
            test_t = datetime.datetime(2018, 1, 1, 0, 0)
            self.reader = HRITGOMSPrologueFileHandler(
                             'filename', {'platform_shortname': 'GOMS2',
                                          'start_time': test_t,
                                          'service': 'test_service'},
                             {'filetype': 'info'})

        # assertDictEqual doesn't seem to work for dicts containing dicts,
        # so we must compare some items individually
        self.assertDictEqual(self.test_pro['SatelliteStatus'],
                             self.reader.prologue['SatelliteStatus'])
        prop = 'ImageAcquisition'
        for key in self.reader.prologue[prop]:
            np.testing.assert_array_equal(self.test_pro[prop][key],
                                          self.reader.prologue[prop][key])
        np.testing.assert_array_equal(self.test_pro['ImageCalibration'],
                                      self.reader.prologue['ImageCalibration'])


class TestHRITGOMSEpiFileHandler(unittest.TestCase):
    '''Test the HRIT Epilogue FileHandler.'''

    @mock.patch('satpy.readers.electrol_hrit.np.fromfile')
    @mock.patch('satpy.readers.electrol_hrit.HRITFileHandler.__init__')
    def test_init(self, new_fh_init, fromfile):
        """Setup the hrit file handler for testing."""
        new_fh_init.return_value.filename = 'filename'
        HRITGOMSEpilogueFileHandler.filename = 'filename'
        HRITGOMSEpilogueFileHandler.mda = {'total_header_length': 1}

        # Set up the test data to use within the epilogue reader
        rtv = np.ones((1,), dtype=epilogue)
        # Pretend to return this when reading a fake epilogue
        fromfile.return_value = rtv
        m = mock.mock_open()

        with mock.patch('satpy.readers.electrol_hrit.open',
                        m, create=True) as newopen:
            newopen.return_value.__enter__.return_value.seek.return_value = 1
            test_t = datetime.datetime(2018, 1, 1, 0, 0)
            self.reader = HRITGOMSEpilogueFileHandler(
                             'filename', {'platform_shortname': 'GOMS2',
                                          'start_time': test_t,
                                          'service': 'test_service'},
                             {'filetype': 'info'})

            epi = self.reader.epilogue

            # We don't check everything in the epilogue (too many nested dicts)
            # but rather check the epilogue is returned as a dict and that two
            # representative data fields are as we expect.
            self.assertIsInstance(epi, dict)
            np.testing.assert_array_equal(
                epi['RadiometricProcessing']['RPSummary']['IsOptic'], np.ones(10))
            np.testing.assert_array_equal(
                epi['GeometricProcessing']['TimeProcessing'], np.ones(10))


class resser:
    attrs = {}
    calibration = 'counts'


@mock.patch('satpy.readers.electrol_hrit.HRITGOMSFileHandler.__init__', return_value=None)
@mock.patch('satpy.readers.electrol_hrit.HRITFileHandler.get_dataset', return_value={})
class TestHRITGOMSFileHandler(unittest.TestCase):
    '''A test of the ELECTRO-L main file handler functions'''

    @mock.patch('satpy.readers.electrol_hrit.HRITGOMSFileHandler.calibrate', return_value=resser())
    def test_get_dataset(self, *mocks):
        fh = HRITGOMSFileHandler()
        fh.platform_name = 'Electro'
        fh.mda = {'projection_parameters': {'SSP_longitude': 0.0},
                  'orbital_parameters': {'satellite_nominal_longitude': 0.5}}
        info = {'units': 'm', 'standard_name': 'electro', 'wavelength': 5.0}
        output = fh.get_dataset(resser(), info)

        # Check that 'calibrate' is called
        mocks[1].assert_called()

        # Check that the correct attributes are returned
        attrs_exp = info.copy()
        attrs_exp.update({'orbital_parameters': {'satellite_nominal_longitude': 0.5,
                                                 'satellite_nominal_latitude': 0.0,
                                                 'projection_longitude': 0.0,
                                                 'projection_latitude': 0.0,
                                                 'projection_altitude': 35785831.00},
                          'platform_name': 'Electro',
                          'sensor': 'msu-gs'})
        self.assertDictContainsSubset(attrs_exp, output.attrs)

    def test_calibrate(self, *mocks):
        lut = np.linspace(1e6, 1.6e6, num=1024).astype(np.int32)
        lut = np.tile(lut, (10, 1))
        fh = HRITGOMSFileHandler()
        fh.prologue = {'ImageCalibration': lut}
        fh.chid = 1

        # Set up test input data

        counts = DataArray(da.linspace(1, 1023, 25, chunks=5,
                                       dtype=np.uint16).reshape(5, 5))

        # Test that calibration fails if given a silly mode
        self.assertRaises(NotImplementedError, fh.calibrate, counts,
                          'nonsense')

        # Test that 'counts' calibration returns identical values to input
        out = fh.calibrate(counts, 'counts')
        self.assertTrue(np.all(out.values == counts.values))

        # Test that 'radiance' calibrates successfully
        out = fh.calibrate(counts, 'radiance')
        self.assertTrue(np.allclose(out.values, lut[0, counts]/1000.))

        # Test that 'brightness_temperature' calibrates successfully
        out = fh.calibrate(counts, 'brightness_temperature')
        self.assertTrue(np.allclose(out.values, lut[0, counts]/1000.))

    def test_get_area_def(self, *mocks):

        example_area_ext = (-5566748.0802, -1854249.1809,
                            5570748.6178, 2000.2688)

        fh = HRITGOMSFileHandler()
        fh.mda = {'cfac': 10231753, 'lfac': 10231753,
                  'coff': 1392.0, 'loff': 0.0, 'number_of_lines': 464,
                  'number_of_columns': 2784,
                  'projection_parameters': {'SSP_longitude': 0.0}}
        area = fh.get_area_def(True)

        self.assertTrue(np.allclose(np.array(area.area_extent),
                        np.array(example_area_ext)))
