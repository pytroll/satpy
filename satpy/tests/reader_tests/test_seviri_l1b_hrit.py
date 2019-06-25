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
"""The HRIT msg reader tests package.
"""

import sys
from datetime import datetime

import numpy as np
import xarray as xr

from satpy.readers.seviri_l1b_hrit import (HRITMSGFileHandler, HRITMSGPrologueFileHandler, HRITMSGEpilogueFileHandler,
                                           NoValidOrbitParams)
from satpy.readers.seviri_base import CHANNEL_NAMES, VIS_CHANNELS
from satpy.dataset import DatasetID

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


def new_get_hd(instance, hdr_info):
    instance.mda = {'spectral_channel_id': 'bla'}
    instance.mda.setdefault('number_of_bits_per_pixel', 10)

    instance.mda['projection_parameters'] = {'a': 6378169.00,
                                             'b': 6356583.80,
                                             'h': 35785831.00,
                                             'SSP_longitude': 0.0}
    instance.mda['orbital_parameters'] = {}
    instance.mda['total_header_length'] = 12


class TestHRITMSGFileHandler(unittest.TestCase):
    """Test the HRITFileHandler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.np.fromfile')
    def setUp(self, fromfile):
        """Setup the hrit file handler for testing."""
        m = mock.mock_open()
        fromfile.return_value = np.array([(1, 2)], dtype=[('total_header_length', int),
                                                          ('hdr_id', int)])

        with mock.patch('satpy.readers.hrit_base.open', m, create=True) as newopen:
            with mock.patch('satpy.readers.seviri_l1b_hrit.CHANNEL_NAMES'):
                with mock.patch.object(HRITMSGFileHandler, '_get_hd', new=new_get_hd):
                    newopen.return_value.__enter__.return_value.tell.return_value = 1
                    prologue = mock.MagicMock()
                    prologue.prologue = {"SatelliteStatus": {"SatelliteDefinition": {"SatelliteId": 324,
                                                                                     "NominalLongitude": 47}},
                                         'GeometricProcessing': {'EarthModel': {'TypeOfEarthModel': 2,
                                                                                'NorthPolarRadius': 10,
                                                                                'SouthPolarRadius': 10,
                                                                                'EquatorialRadius': 10}},
                                         'ImageDescription': {'ProjectionDescription': {'LongitudeOfSSP': 0.0}}}
                    prologue.get_satpos.return_value = None, None, None
                    prologue.get_earth_radii.return_value = None, None

                    self.reader = HRITMSGFileHandler(
                        'filename',
                        {'platform_shortname': 'MSG3',
                         'start_time': datetime(2016, 3, 3, 0, 0),
                         'service': 'MSG'},
                        {'filetype': 'info'},
                        prologue,
                        mock.MagicMock())
                    ncols = 3712
                    nlines = 464
                    nbits = 10
                    self.reader.mda['number_of_bits_per_pixel'] = nbits
                    self.reader.mda['number_of_lines'] = nlines
                    self.reader.mda['number_of_columns'] = ncols
                    self.reader.mda['data_field_length'] = nlines * ncols * nbits
                    self.reader.mda['cfac'] = 5
                    self.reader.mda['lfac'] = 5
                    self.reader.mda['coff'] = 10
                    self.reader.mda['loff'] = 10
                    self.reader.mda['projection_parameters'] = {}
                    self.reader.mda['projection_parameters']['a'] = 6378169.0
                    self.reader.mda['projection_parameters']['b'] = 6356583.8
                    self.reader.mda['projection_parameters']['h'] = 35785831.0
                    self.reader.mda['projection_parameters']['SSP_longitude'] = 44
                    self.reader.mda['projection_parameters']['SSP_latitude'] = 0.0
                    self.reader.mda['orbital_parameters'] = {}
                    self.reader.mda['orbital_parameters']['satellite_nominal_longitude'] = 47
                    self.reader.mda['orbital_parameters']['satellite_nominal_latitude'] = 0.0
                    self.reader.mda['orbital_parameters']['satellite_actual_longitude'] = 47.5
                    self.reader.mda['orbital_parameters']['satellite_actual_latitude'] = -0.5
                    self.reader.mda['orbital_parameters']['satellite_actual_altitude'] = 35783328

                    tline = np.zeros(nlines, dtype=[('days', '>u2'), ('milliseconds', '>u4')])
                    tline['days'][1:-1] = 21246 * np.ones(nlines-2)  # 2016-03-03
                    tline['milliseconds'][1:-1] = np.arange(nlines-2)
                    self.reader.mda['image_segment_line_quality'] = {'line_mean_acquisition': tline}

    def test_get_xy_from_linecol(self):
        """Test get_xy_from_linecol."""
        x__, y__ = self.reader.get_xy_from_linecol(0, 0, (10, 10), (5, 5))
        self.assertEqual(-131072, x__)
        self.assertEqual(131072, y__)
        x__, y__ = self.reader.get_xy_from_linecol(10, 10, (10, 10), (5, 5))
        self.assertEqual(0, x__)
        self.assertEqual(0, y__)
        x__, y__ = self.reader.get_xy_from_linecol(20, 20, (10, 10), (5, 5))
        self.assertEqual(131072, x__)
        self.assertEqual(-131072, y__)

    def test_get_area_extent(self):
        res = self.reader.get_area_extent((20, 20), (10, 10), (5, 5), 33)
        exp = (-71717.44995740513, -79266.655216079365,
               79266.655216079365, 71717.44995740513)
        self.assertTupleEqual(res, exp)

    def test_get_area_def(self):
        area = self.reader.get_area_def(DatasetID('VIS006'))
        self.assertEqual(area.proj_dict, {'a': 6378169.0,
                                          'b': 6356583.8,
                                          'h': 35785831.0,
                                          'lon_0': 44.0,
                                          'proj': 'geos',
                                          'units': 'm'})
        self.assertEqual(area.area_extent,
                         (-77771774058.38356, -3720765401003.719,
                          30310525626438.438, 77771774058.38356))

        # Data shifted by 1.5km to N-W
        self.reader.mda['offset_corrected'] = False
        area = self.reader.get_area_def(DatasetID('VIS006'))
        self.assertEqual(area.area_extent,
                         (-77771772558.38356, -3720765402503.719,
                          30310525627938.438, 77771772558.38356))

    @mock.patch('satpy.readers.hrit_base.np.memmap')
    def test_read_band(self, memmap):
        nbits = self.reader.mda['number_of_bits_per_pixel']
        memmap.return_value = np.random.randint(0, 256,
                                                size=int((464 * 3712 * nbits) / 8),
                                                dtype=np.uint8)
        res = self.reader.read_band('VIS006', None)
        self.assertEqual(res.compute().shape, (464, 3712))

    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', return_value=None)
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler._get_header', autospec=True)
    @mock.patch('satpy.readers.seviri_base.SEVIRICalibrationHandler._convert_to_radiance')
    def test_calibrate(self, _convert_to_radiance, get_header, *mocks):
        """Test selection of calibration coefficients"""
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
            if ch_name in coefs.keys():
                gain, offset = coefs[ch_name]['gain'], coefs[ch_name]['offset']
            elif ch_name not in VIS_CHANNELS:
                gain, offset = gsics_gain[ch_id - 1], gsics_offset[ch_id - 1]
            else:
                gain, offset = nominal_gain[ch_id - 1], nominal_offset[ch_id - 1]

            reader.channel_name = ch_name
            reader.mda['spectral_channel_id'] = ch_id
            reader.calibrate(data=counts, calibration='radiance')
            _convert_to_radiance.assert_called_with(mock.ANY, gain, offset)

        # d) Invalid mode
        self.assertRaises(ValueError, HRITMSGFileHandler, filename=None, filename_info=None,
                          filetype_info=None, prologue=pro, epilogue=epi, calib_mode='invalid')

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler._get_timestamps')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset(self, calibrate, parent_get_dataset, _get_timestamps):
        key = mock.MagicMock(calibration='calibration')
        info = {'units': 'units', 'wavelength': 'wavelength', 'standard_name': 'standard_name'}
        timestamps = np.array([1, 2, 3], dtype='datetime64[ns]')

        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = xr.DataArray(data=np.zeros((3, 3)), dims=('y', 'x'))
        _get_timestamps.return_value = timestamps

        res = self.reader.get_dataset(key, info)

        # Test method calls
        parent_get_dataset.assert_called_with(key, info)
        calibrate.assert_called_with(parent_get_dataset(), key.calibration)

        # Test attributes (just check if raw metadata is there and then remove it before checking the remaining
        # attributes)
        attrs_exp = info.copy()
        attrs_exp.update({
            'platform_name': self.reader.platform_name,
            'sensor': 'seviri',
            'satellite_longitude': self.reader.mda['projection_parameters']['SSP_longitude'],
            'satellite_latitude': self.reader.mda['projection_parameters']['SSP_latitude'],
            'satellite_altitude': self.reader.mda['projection_parameters']['h'],
            'orbital_parameters': {'projection_longitude': 44,
                                   'projection_latitude': 0.,
                                   'projection_altitude': 35785831.0,
                                   'satellite_nominal_longitude': 47,
                                   'satellite_nominal_latitude': 0.0,
                                   'satellite_actual_longitude': 47.5,
                                   'satellite_actual_latitude': -0.5,
                                   'satellite_actual_altitude': 35783328},
            'georef_offset_corrected': self.reader.mda['offset_corrected']
        })
        self.assertIn('raw_metadata', res.attrs)
        res.attrs.pop('raw_metadata')
        self.assertDictEqual(attrs_exp, res.attrs)

        # Test timestamps
        self.assertTrue(np.all(res['acq_time'] == timestamps))
        self.assertEqual(res['acq_time'].attrs['long_name'], 'Mean scanline acquisition time')

    def test_get_raw_mda(self):
        """Test provision of raw metadata"""
        self.reader.mda = {'segment': 1, 'loff': 123}
        self.reader.prologue_.reduce = lambda max_size: {'prologue': 1}
        self.reader.epilogue_.reduce = lambda max_size: {'epilogue': 1}
        expected = {'prologue': 1, 'epilogue': 1, 'segment': 1}
        self.assertDictEqual(self.reader._get_raw_mda(), expected)

        # Make sure _get_raw_mda() doesn't modify the original dictionary
        self.assertIn('loff', self.reader.mda)

    def test_get_timestamps(self):
        tline = self.reader._get_timestamps()

        # First and last scanline have invalid timestamps (space)
        self.assertTrue(np.isnat(tline[0]))
        self.assertTrue(np.isnat(tline[-1]))

        # Test remaining lines
        year = tline.astype('datetime64[Y]').astype(int) + 1970
        month = tline.astype('datetime64[M]').astype(int) % 12 + 1
        day = (tline.astype('datetime64[D]') - tline.astype('datetime64[M]') + 1).astype(int)
        msec = (tline - tline.astype('datetime64[D]')).astype(int)
        self.assertTrue(np.all(year[1:-1] == 2016))
        self.assertTrue(np.all(month[1:-1] == 3))
        self.assertTrue(np.all(day[1:-1] == 3))
        self.assertTrue(np.all(msec[1:-1] == np.arange(len(tline) - 2)))


class TestHRITMSGPrologueFileHandler(unittest.TestCase):
    """Test the HRIT prologue file handler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler.__init__', return_value=None)
    def setUp(self, *mocks):
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
                            [datetime(2006, 1, 1, 6), datetime(2006, 1, 1, 12), datetime(2006, 1, 1, 18)]]),
                        'EndTime': np.array([
                            [datetime(2006, 1, 1, 12), datetime(2006, 1, 1, 18), datetime(2006, 1, 2, 0)]]),
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
        """Test whether the prologue file handler accepts extra keyword arguments"""
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
        """Test identification of orbit coefficients"""

        self.assertEqual(self.reader._find_orbit_coefs(), 1)

        # No interval enclosing the given timestamp
        self.reader.prologue['ImageAcquisition']['PlannedAcquisitionTime'][
            'TrueRepeatCycleStart'] = datetime(2000, 1, 1)
        self.assertRaises(NoValidOrbitParams, self.reader._find_orbit_coefs)

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler._find_orbit_coefs')
    def test_get_satpos_cart(self, find_orbit_coefs):
        """Test satellite position in cartesian coordinates"""
        find_orbit_coefs.return_value = 1
        x, y, z = self.reader._get_satpos_cart()
        self.assertTrue(np.allclose([x, y, z], [42078421.37095518, -2611352.744615312, -419828.9699940758]))

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler._get_satpos_cart')
    def test_get_satpos(self, get_satpos_cart):
        """Test satellite position in spherical coordinates"""
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
        """Test readout of earth radii"""
        earth_model = self.reader.prologue['GeometricProcessing']['EarthModel']
        earth_model['EquatorialRadius'] = 2
        earth_model['NorthPolarRadius'] = 1
        earth_model['SouthPolarRadius'] = 2
        a, b = self.reader.get_earth_radii()
        self.assertEqual(a, 2000)
        self.assertEqual(b, 1500)

    @mock.patch('satpy.readers.seviri_l1b_hrit.utils.reduce_mda')
    def test_reduce(self, reduce_mda):
        """Test metadata reduction"""
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
        """Test whether the epilogue file handler accepts extra keyword arguments"""

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
        """Test metadata reduction"""
        reduce_mda.return_value = 'reduced'

        # Set buffer
        self.assertEqual(self.reader.reduce(123), 'reduced')
        reduce_mda.assert_called()

        # Read buffer
        reduce_mda.reset_mock()
        self.reader._reduced = 'red'
        self.assertEqual(self.reader.reduce(123), 'red')
        reduce_mda.assert_not_called()


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    tests = [TestHRITMSGFileHandler, TestHRITMSGPrologueFileHandler, TestHRITMSGEpilogueFileHandler]
    for test in tests:
        mysuite.addTest(loader.loadTestsFromTestCase(test))
    return mysuite


if __name__ == '__main__':
    unittest.main()
