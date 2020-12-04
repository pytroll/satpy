#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""The HRIT msg reader tests package."""

import unittest
from unittest import mock
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers.seviri_l1b_hrit import (HRITMSGFileHandler, HRITMSGPrologueFileHandler, HRITMSGEpilogueFileHandler,
                                           NoValidOrbitParams, pad_data)
from satpy.readers.seviri_base import CHANNEL_NAMES, VIS_CHANNELS
from satpy.tests.utils import make_dataid
import satpy.tests.reader_tests.test_seviri_l1b_hrit_setup as setup

from numpy import testing as npt


class TestHRITMSGBase(unittest.TestCase):
    """Baseclass for SEVIRI HRIT reader tests."""

    def assert_attrs_equal(self, attrs, attrs_exp):
        """Assert equality of dataset attributes."""
        # Test attributes (just check if raw metadata is there and then remove
        # it before checking the remaining attributes)
        self.assertIn('raw_metadata', attrs)
        attrs.pop('raw_metadata')
        self.assertDictEqual(attrs, attrs_exp)


class TestHRITMSGFileHandlerHRV(TestHRITMSGBase):
    """Test the HRITFileHandler."""

    def setUp(self):
        """Set up the hrit file handler for testing HRV."""
        prologue = setup.get_fake_prologue()
        epilogue = {
            'ImageProductionStats': {
                'ActualL15CoverageHRV': {
                    'LowerSouthLineActual': 1,
                    'LowerNorthLineActual': 8256,
                    'LowerEastColumnActual': 2877,
                    'LowerWestColumnActual': 8444,
                    'UpperSouthLineActual': 8257,
                    'UpperNorthLineActual': 11136,
                    'UpperEastColumnActual': 1805,
                    'UpperWestColumnActual': 7372
                }
            }
        }
        self.start_time = datetime(2016, 3, 3, 0, 0)
        self.nlines = 464
        mda = setup.get_fake_mda(
            nlines=self.nlines, ncols=5568, start_time=self.start_time
        )
        mda.update({
            'segment_sequence_number': 18,
            'planned_start_segment_number': 1
        })
        filename_info = setup.get_fake_filename_info(self.start_time)
        self.reader = setup.get_fake_file_handler(
            filename_info, mda, prologue, epilogue
        )
        self.reader.fill_hrv = True

    @mock.patch('satpy.readers.hrit_base.np.memmap')
    def test_read_hrv_band(self, memmap):
        """Test reading the hrv band."""
        nbits = self.reader.mda['number_of_bits_per_pixel']
        memmap.return_value = np.random.randint(0, 256,
                                                size=int((464 * 5568 * nbits) / 8),
                                                dtype=np.uint8)
        res = self.reader.read_band('HRV', None)
        self.assertEqual(res.shape, (464, 5568))

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset(self, calibrate, parent_get_dataset):
        """Test getting the hrv dataset."""
        key = make_dataid(name='HRV', calibration='reflectance')
        info = setup.get_fake_dataset_info()

        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = xr.DataArray(data=np.zeros((464, 5568)), dims=('y', 'x'))
        res = self.reader.get_dataset(key, info)
        self.assertEqual(res.shape, (464, 11136))

        # Test method calls
        parent_get_dataset.assert_called_with(key, info)
        calibrate.assert_called_with(parent_get_dataset(), key['calibration'])

        self.assert_attrs_equal(res.attrs, setup.get_attrs_exp())
        np.testing.assert_equal(
            res['acq_time'],
            setup.get_acq_time_exp(self.start_time, self.nlines)
        )

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset_non_fill(self, calibrate, parent_get_dataset):
        """Test getting a non-filled hrv dataset."""
        key = make_dataid(name='HRV', calibration='reflectance')
        key.name = 'HRV'
        info = setup.get_fake_dataset_info()
        self.reader.fill_hrv = False
        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = xr.DataArray(data=np.zeros((464, 5568)), dims=('y', 'x'))
        res = self.reader.get_dataset(key, info)
        self.assertEqual(res.shape, (464, 5568))

        # Test method calls
        parent_get_dataset.assert_called_with(key, info)
        calibrate.assert_called_with(parent_get_dataset(), key['calibration'])

        self.assert_attrs_equal(res.attrs, setup.get_attrs_exp())
        np.testing.assert_equal(
            res['acq_time'],
            setup.get_acq_time_exp(self.start_time, self.nlines)
        )

    def test_pad_data(self):
        """Test the hrv padding."""
        data = xr.DataArray(data=np.zeros((1, 10)), dims=('y', 'x'))
        east_bound = 4
        west_bound = 13
        final_size = (1, 20)
        res = pad_data(data, final_size, east_bound, west_bound)
        expected = np.array([[np.nan, np.nan, np.nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
        np.testing.assert_allclose(res, expected)

        east_bound = 3
        self.assertRaises(IndexError, pad_data, data, final_size, east_bound, west_bound)

    def test_get_area_def(self):
        """Test getting the area def."""
        from pyresample.utils import proj4_radius_parameters
        area = self.reader.get_area_def(make_dataid(name='HRV'))
        self.assertEqual(area.area_extent,
                         (-45561979844414.07, -3720765401003.719, 45602912357076.38, 77771774058.38356))
        proj_dict = area.proj_dict
        a, b = proj4_radius_parameters(proj_dict)
        self.assertEqual(a, 6378169.0)
        self.assertEqual(b, 6356583.8)
        self.assertEqual(proj_dict['h'], 35785831.0)
        self.assertEqual(proj_dict['lon_0'], 44.0)
        self.assertEqual(proj_dict['proj'], 'geos')
        self.assertEqual(proj_dict['units'], 'm')
        self.reader.fill_hrv = False
        area = self.reader.get_area_def(make_dataid(name='HRV'))
        npt.assert_allclose(area.defs[0].area_extent,
                            (-22017598561055.01, -2926674655354.9604, 23564847539690.22, 77771774058.38356))
        npt.assert_allclose(area.defs[1].area_extent,
                            (-30793529275853.656, -3720765401003.719, 14788916824891.568, -2926674655354.9604))


class TestHRITMSGFileHandler(TestHRITMSGBase):
    """Test the HRITFileHandler."""

    def setUp(self):
        """Set up the hrit file handler for testing."""
        prologue = setup.get_fake_prologue()
        epilogue = {}
        self.start_time = datetime(2016, 3, 3, 0, 0)
        self.nlines = 464
        mda = setup.get_fake_mda(
            nlines=self.nlines, ncols=3712, start_time=self.start_time)
        filename_info = setup.get_fake_filename_info(self.start_time)
        self.reader = setup.get_fake_file_handler(
            filename_info, mda, prologue, epilogue
        )

    def test_get_area_def(self):
        """Test getting the area def."""
        from pyresample.utils import proj4_radius_parameters
        area = self.reader.get_area_def(make_dataid(name='VIS006'))
        proj_dict = area.proj_dict
        a, b = proj4_radius_parameters(proj_dict)
        self.assertEqual(a, 6378169.0)
        self.assertEqual(b, 6356583.8)
        self.assertEqual(proj_dict['h'], 35785831.0)
        self.assertEqual(proj_dict['lon_0'], 44.0)
        self.assertEqual(proj_dict['proj'], 'geos')
        self.assertEqual(proj_dict['units'], 'm')
        self.assertEqual(area.area_extent,
                         (-77771774058.38356, -3720765401003.719,
                          30310525626438.438, 77771774058.38356))

        # Data shifted by 1.5km to N-W
        self.reader.mda['offset_corrected'] = False
        area = self.reader.get_area_def(make_dataid(name='VIS006'))
        self.assertEqual(area.area_extent,
                         (-77771772558.38356, -3720765402503.719,
                          30310525627938.438, 77771772558.38356))

    @mock.patch('satpy.readers.hrit_base.np.memmap')
    def test_read_band(self, memmap):
        """Test reading a band."""
        nbits = self.reader.mda['number_of_bits_per_pixel']
        memmap.return_value = np.random.randint(0, 256,
                                                size=int((464 * 3712 * nbits) / 8),
                                                dtype=np.uint8)
        res = self.reader.read_band('VIS006', None)
        self.assertEqual(res.shape, (464, 3712))

    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', return_value=None)
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler._get_header', autospec=True)
    @mock.patch('satpy.readers.seviri_base.SEVIRICalibrationHandler._convert_to_radiance')
    def test_calibrate(self, _convert_to_radiance, get_header, *mocks):
        """Test selection of calibration coefficients."""
        shp = (10, 10)
        counts = xr.DataArray(np.zeros(shp))
        nominal_gain = np.arange(1, 13)
        nominal_offset = np.arange(-1, -13, -1)
        gsics_gain = np.arange(0.1, 1.3, 0.1)
        gsics_offset = np.arange(-0.1, -1.3, -0.1)

        # Mock prologue & epilogue
        pro = mock.MagicMock(prologue={'RadiometricProcessing': {
            'Level15ImageCalibration': {'CalSlope': nominal_gain,
                                        'CalOffset': nominal_offset},
            'MPEFCalFeedback': {'GSICSCalCoeff': gsics_gain,
                                'GSICSOffsetCount': gsics_offset}
        }})
        epi = mock.MagicMock(epilogue=None)

        # Mock header readout
        mda = {'image_segment_line_quality': {'line_validity': np.zeros(shp[0]),
                                              'line_radiometric_quality': np.zeros(shp[0]),
                                              'line_geometric_quality': np.zeros(shp[0])}}

        def get_header_patched(self):
            self.mda = mda

        get_header.side_effect = get_header_patched

        # Test selection of calibration coefficients
        #
        # a) Default: Nominal calibration
        reader = HRITMSGFileHandler(filename=None, filename_info=None, filetype_info=None,
                                    prologue=pro, epilogue=epi)
        for ch_id, ch_name in CHANNEL_NAMES.items():
            reader.channel_name = ch_name
            reader.mda['spectral_channel_id'] = ch_id
            reader.calibrate(data=counts, calibration='radiance')
            _convert_to_radiance.assert_called_with(mock.ANY, nominal_gain[ch_id - 1],
                                                    nominal_offset[ch_id - 1])

        # b) GSICS calibration for IR channels, nominal calibration for VIS channels
        reader = HRITMSGFileHandler(filename=None, filename_info=None, filetype_info=None,
                                    prologue=pro, epilogue=epi, calib_mode='GSICS')
        for ch_id, ch_name in CHANNEL_NAMES.items():
            if ch_name in VIS_CHANNELS:
                gain, offset = nominal_gain[ch_id - 1], nominal_offset[ch_id - 1]
            else:
                gain, offset = gsics_gain[ch_id - 1], gsics_offset[ch_id - 1]
                offset = offset * gain

            reader.channel_name = ch_name
            reader.mda['spectral_channel_id'] = ch_id
            reader.calibrate(data=counts, calibration='radiance')
            _convert_to_radiance.assert_called_with(mock.ANY, gain, offset)

        # c) External calibration coefficients for selected channels, GSICS coefs for remaining
        #    IR channels, nominal coefs for remaining VIS channels
        coefs = {'VIS006': {'gain': 1.234, 'offset': -0.1},
                 'IR_108': {'gain': 2.345, 'offset': -0.2}}
        reader = HRITMSGFileHandler(filename=None, filename_info=None, filetype_info=None,
                                    prologue=pro, epilogue=epi, ext_calib_coefs=coefs,
                                    calib_mode='GSICS')
        for ch_id, ch_name in CHANNEL_NAMES.items():
            if ch_name in coefs:
                gain, offset = coefs[ch_name]['gain'], coefs[ch_name]['offset']
            elif ch_name not in VIS_CHANNELS:
                gain, offset = gsics_gain[ch_id - 1], gsics_offset[ch_id - 1]
                offset = offset * gain
            else:
                gain, offset = nominal_gain[ch_id - 1], nominal_offset[ch_id - 1]

            reader.channel_name = ch_name
            reader.mda['spectral_channel_id'] = ch_id
            reader.calibrate(data=counts, calibration='radiance')
            _convert_to_radiance.assert_called_with(mock.ANY, gain, offset)

        # d) Invalid mode
        self.assertRaises(ValueError, HRITMSGFileHandler, filename=None, filename_info=None,
                          filetype_info=None, prologue=pro, epilogue=epi, calib_mode='invalid')

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset(self, calibrate, parent_get_dataset):
        """Test getting the dataset."""
        key = make_dataid(name='VIS006', calibration='reflectance')
        info = setup.get_fake_dataset_info()

        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = xr.DataArray(
            data=np.zeros((self.reader.mda['number_of_lines'],
                           self.reader.mda['number_of_columns'])),
            dims=('y', 'x'))

        res = self.reader.get_dataset(key, info)

        # Test method calls
        parent_get_dataset.assert_called_with(key, info)
        calibrate.assert_called_with(parent_get_dataset(), key['calibration'])

        self.assert_attrs_equal(res.attrs, setup.get_attrs_exp())
        np.testing.assert_equal(
            res['acq_time'],
            setup.get_acq_time_exp(self.start_time, self.nlines)
        )

    def test_get_raw_mda(self):
        """Test provision of raw metadata."""
        self.reader.mda = {'segment': 1, 'loff': 123}
        self.reader.prologue_.reduce = lambda max_size: {'prologue': 1}
        self.reader.epilogue_.reduce = lambda max_size: {'epilogue': 1}
        expected = {'prologue': 1, 'epilogue': 1, 'segment': 1}
        self.assertDictEqual(self.reader._get_raw_mda(), expected)

        # Make sure _get_raw_mda() doesn't modify the original dictionary
        self.assertIn('loff', self.reader.mda)

    def test_get_header(self):
        """Test getting the header."""
        # Make sure that the actual satellite position is only included if available
        self.reader.mda['orbital_parameters'] = {}
        self.reader.prologue_.get_satpos.return_value = 1, 2, 3
        self.reader._get_header()
        self.assertIn('satellite_actual_longitude', self.reader.mda['orbital_parameters'])

        self.reader.mda['orbital_parameters'] = {}
        self.reader.prologue_.get_satpos.return_value = None, None, None
        self.reader._get_header()
        self.assertNotIn('satellite_actual_longitude', self.reader.mda['orbital_parameters'])


class TestHRITMSGPrologueFileHandler(unittest.TestCase):
    """Test the HRIT prologue file handler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler.__init__', return_value=None)
    def setUp(self, *mocks):
        """Set up the test case."""
        self.reader = HRITMSGPrologueFileHandler()
        self.reader.satpos = None
        self.reader.prologue = {
            'GeometricProcessing': {
                'EarthModel': {
                    'EquatorialRadius': 6378.169,
                    'NorthPolarRadius': 6356.5838,
                    'SouthPolarRadius': 6356.5838
                }
            },
            'ImageAcquisition': {
                'PlannedAcquisitionTime': {
                    'TrueRepeatCycleStart': datetime(2006, 1, 1, 12, 15, 9, 304888)
                }
            },
            'SatelliteStatus': {
                'Orbit': {
                    'OrbitPolynomial': {
                        'StartTime': np.array([
                            [datetime(2006, 1, 1, 6), datetime(2006, 1, 1, 12), datetime(2006, 1, 1, 18),
                             datetime(1958, 1, 1, 0)]]),
                        'EndTime': np.array([
                            [datetime(2006, 1, 1, 12), datetime(2006, 1, 1, 18), datetime(2006, 1, 2, 0),
                             datetime(1958, 1, 1, 0)]]),
                        'X': [np.zeros(8),
                              [8.41607082e+04, 2.94319260e+00, 9.86748617e-01, -2.70135453e-01,
                               -3.84364650e-02, 8.48718433e-03, 7.70548174e-04, -1.44262718e-04],
                              np.zeros(8)],
                        'Y': [np.zeros(8),
                              [-5.21170255e+03, 5.12998948e+00, -1.33370453e+00, -3.09634144e-01,
                               6.18232793e-02, 7.50505681e-03, -1.35131011e-03, -1.12054405e-04],
                              np.zeros(8)],
                        'Z': [np.zeros(8),
                              [-6.51293855e+02, 1.45830459e+02, 5.61379400e+01, -3.90970565e+00,
                               -7.38137565e-01, 3.06131644e-02, 3.82892428e-03, -1.12739309e-04],
                              np.zeros(8)],
                    }
                }
            }
        }
        self.reader._reduced = None

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler.read_prologue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def test_extra_kwargs(self, init, *mocks):
        """Test whether the prologue file handler accepts extra keyword arguments."""
        def init_patched(self, *args, **kwargs):
            self.mda = {}
        init.side_effect = init_patched

        HRITMSGPrologueFileHandler(filename=None,
                                   filename_info={'service': ''},
                                   filetype_info=None,
                                   ext_calib_coefs={},
                                   mda_max_array_size=123,
                                   calib_mode='nominal')

    def test_find_orbit_coefs(self):
        """Test identification of orbit coefficients."""
        # Contiguous validity intervals (that's the norm)
        self.assertEqual(self.reader._find_orbit_coefs(), 1)

        # No interval enclosing the given timestamp ...
        # a) closest interval should be selected (if not too far away)
        self.reader.prologue['SatelliteStatus'] = {
            'Orbit': {
                'OrbitPolynomial': {
                    'StartTime': np.array([
                        [datetime(2006, 1, 1, 10), datetime(2006, 1, 1, 13)]]),
                    'EndTime': np.array([
                        [datetime(2006, 1, 1, 12), datetime(2006, 1, 1, 18)]])
                }
            }
        }
        self.assertEqual(self.reader._find_orbit_coefs(), 0)

        # b) closest interval too far away
        self.reader.prologue['SatelliteStatus'] = {
            'Orbit': {
                'OrbitPolynomial': {
                    'StartTime': np.array([
                        [datetime(2006, 1, 1, 0), datetime(2006, 1, 1, 18)]]),
                    'EndTime': np.array([
                        [datetime(2006, 1, 1, 4), datetime(2006, 1, 1, 22)]])
                }
            }
        }
        self.assertRaises(NoValidOrbitParams, self.reader._find_orbit_coefs)

        # Overlapping intervals -> most recent interval should be selected
        self.reader.prologue['SatelliteStatus'] = {
            'Orbit': {
                'OrbitPolynomial': {
                    'StartTime': np.array([
                        [datetime(2006, 1, 1, 6), datetime(2006, 1, 1, 10)]]),
                    'EndTime': np.array([
                        [datetime(2006, 1, 1, 13), datetime(2006, 1, 1, 18)]])
                }
            }
        }
        self.assertEqual(self.reader._find_orbit_coefs(), 1)

        # No valid coefficients at all
        self.reader.prologue['SatelliteStatus'] = {
            'Orbit': {
                'OrbitPolynomial': {
                    'StartTime': np.array([
                        [datetime(1958, 1, 1, 0), datetime(1958, 1, 1)]]),
                    'EndTime': np.array([
                        [datetime(1958, 1, 1, 0), datetime(1958, 1, 1)]])
                }
            }
        }
        self.assertRaises(NoValidOrbitParams, self.reader._find_orbit_coefs)

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler._find_orbit_coefs')
    def test_get_satpos_cart(self, find_orbit_coefs):
        """Test satellite position in cartesian coordinates."""
        find_orbit_coefs.return_value = 1
        x, y, z = self.reader._get_satpos_cart()
        self.assertTrue(np.allclose([x, y, z], [42078421.37095518, -2611352.744615312, -419828.9699940758]))

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler._get_satpos_cart')
    def test_get_satpos(self, get_satpos_cart):
        """Test satellite position in spherical coordinates."""
        get_satpos_cart.return_value = [42078421.37095518, -2611352.744615312, -419828.9699940758]
        lon, lat, dist = self.reader.get_satpos()
        self.assertTrue(np.allclose(lon, lat, dist), [-3.5511754052132387, -0.5711189258409902, 35783328.146167226])

        # Test cache
        self.reader.get_satpos()
        self.assertEqual(get_satpos_cart.call_count, 1)

        # No valid coefficients
        self.reader.satpos = None  # reset cache
        get_satpos_cart.side_effect = NoValidOrbitParams
        self.reader.prologue['ImageAcquisition']['PlannedAcquisitionTime'][
            'TrueRepeatCycleStart'] = datetime(2000, 1, 1)
        self.assertTupleEqual(self.reader.get_satpos(), (None, None, None))

    def test_get_earth_radii(self):
        """Test readout of earth radii."""
        earth_model = self.reader.prologue['GeometricProcessing']['EarthModel']
        earth_model['EquatorialRadius'] = 2
        earth_model['NorthPolarRadius'] = 1
        earth_model['SouthPolarRadius'] = 2
        a, b = self.reader.get_earth_radii()
        self.assertEqual(a, 2000)
        self.assertEqual(b, 1500)

    @mock.patch('satpy.readers.seviri_l1b_hrit.utils.reduce_mda')
    def test_reduce(self, reduce_mda):
        """Test metadata reduction."""
        reduce_mda.return_value = 'reduced'

        # Set buffer
        self.assertEqual(self.reader.reduce(123), 'reduced')
        reduce_mda.assert_called()

        # Read buffer
        reduce_mda.reset_mock()
        self.reader._reduced = 'red'
        self.assertEqual(self.reader.reduce(123), 'red')
        reduce_mda.assert_not_called()


class TestHRITMSGEpilogueFileHandler(unittest.TestCase):
    """Test the HRIT epilogue file handler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGEpilogueFileHandler.read_epilogue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def setUp(self, init, *mocks):
        """Set up the test case."""
        def init_patched(self, *args, **kwargs):
            self.mda = {}

        init.side_effect = init_patched

        self.reader = HRITMSGEpilogueFileHandler(filename=None,
                                                 filename_info={'service': ''},
                                                 filetype_info=None,
                                                 calib_mode='nominal')

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGEpilogueFileHandler.read_epilogue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def test_extra_kwargs(self, init, *mocks):
        """Test whether the epilogue file handler accepts extra keyword arguments."""
        def init_patched(self, *args, **kwargs):
            self.mda = {}

        init.side_effect = init_patched

        HRITMSGEpilogueFileHandler(filename=None,
                                   filename_info={'service': ''},
                                   filetype_info=None,
                                   ext_calib_coefs={},
                                   mda_max_array_size=123,
                                   calib_mode='nominal')

    @mock.patch('satpy.readers.seviri_l1b_hrit.utils.reduce_mda')
    def test_reduce(self, reduce_mda):
        """Test metadata reduction."""
        reduce_mda.return_value = 'reduced'

        # Set buffer
        self.assertEqual(self.reader.reduce(123), 'reduced')
        reduce_mda.assert_called()

        # Read buffer
        reduce_mda.reset_mock()
        self.reader._reduced = 'red'
        self.assertEqual(self.reader.reduce(123), 'red')
        reduce_mda.assert_not_called()
