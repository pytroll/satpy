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
"""The abi_l1b reader tests package.
"""

import unittest
try:
    from unittest import mock
except ImportError:
    import mock

import warnings
import numpy as np
import dask.array as da
from datetime import datetime
from pyresample.geometry import AreaDefinition
from satpy.readers.ahi_hsd import AHIHSDFileHandler
from satpy.readers.utils import get_geostationary_mask


class TestAHIHSDNavigation(unittest.TestCase):
    """Test the AHI HSD reader navigation."""

    @mock.patch('satpy.readers.ahi_hsd.np2str')
    @mock.patch('satpy.readers.ahi_hsd.np.fromfile')
    def test_region(self, fromfile, np2str):
        """Test region navigation."""
        np2str.side_effect = lambda x: x
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_hsd.open', m, create=True):
            fh = AHIHSDFileHandler(None, {'segment_number': 1, 'total_segments': 1}, None)
            fh.proj_info = {'CFAC': 40932549,
                            'COFF': -591.5,
                            'LFAC': 40932549,
                            'LOFF': 5132.5,
                            'blocklength': 127,
                            'coeff_for_sd': 1737122264.0,
                            'distance_from_earth_center': 42164.0,
                            'earth_equatorial_radius': 6378.137,
                            'earth_polar_radius': 6356.7523,
                            'hblock_number': 3,
                            'req2_rpol2': 1.006739501,
                            'req2_rpol2_req2': 0.0066943844,
                            'resampling_size': 4,
                            'resampling_types': 0,
                            'rpol2_req2': 0.993305616,
                            'spare': '',
                            'sub_lon': 140.7}

            fh.data_info = {'blocklength': 50,
                            'compression_flag_for_data': 0,
                            'hblock_number': 2,
                            'number_of_bits_per_pixel': 16,
                            'number_of_columns': 1000,
                            'number_of_lines': 1000,
                            'spare': ''}

            area_def = fh.get_area_def(None)
            self.assertEqual(area_def.proj_dict, {'a': 6378137.0, 'b': 6356752.3,
                                                  'h': 35785863.0, 'lon_0': 140.7,
                                                  'proj': 'geos', 'units': 'm'})

            self.assertEqual(area_def.area_extent, (592000.0038256244, 4132000.026701824,
                                                    1592000.0102878278, 5132000.033164027))

    @mock.patch('satpy.readers.ahi_hsd.np2str')
    @mock.patch('satpy.readers.ahi_hsd.np.fromfile')
    def test_segment(self, fromfile, np2str):
        """Test segment navigation."""
        np2str.side_effect = lambda x: x
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_hsd.open', m, create=True):
            fh = AHIHSDFileHandler(None, {'segment_number': 8, 'total_segments': 10}, None)
            fh.proj_info = {'CFAC': 40932549,
                            'COFF': 5500.5,
                            'LFAC': 40932549,
                            'LOFF': 5500.5,
                            'blocklength': 127,
                            'coeff_for_sd': 1737122264.0,
                            'distance_from_earth_center': 42164.0,
                            'earth_equatorial_radius': 6378.137,
                            'earth_polar_radius': 6356.7523,
                            'hblock_number': 3,
                            'req2_rpol2': 1.006739501,
                            'req2_rpol2_req2': 0.0066943844,
                            'resampling_size': 4,
                            'resampling_types': 0,
                            'rpol2_req2': 0.993305616,
                            'spare': '',
                            'sub_lon': 140.7}

            fh.data_info = {'blocklength': 50,
                            'compression_flag_for_data': 0,
                            'hblock_number': 2,
                            'number_of_bits_per_pixel': 16,
                            'number_of_columns': 11000,
                            'number_of_lines': 1100,
                            'spare': ''}

            area_def = fh.get_area_def(None)
            self.assertEqual(area_def.proj_dict, {'a': 6378137.0, 'b': 6356752.3,
                                                  'h': 35785863.0, 'lon_0': 140.7,
                                                  'proj': 'geos', 'units': 'm'})

            self.assertEqual(area_def.area_extent, (-5500000.035542117, -3300000.021325271,
                                                    5500000.035542117, -2200000.0142168473))


class TestAHIHSDFileHandler(unittest.TestCase):
    @mock.patch('satpy.readers.ahi_hsd.np2str')
    @mock.patch('satpy.readers.ahi_hsd.np.fromfile')
    def setUp(self, fromfile, np2str):
        """Create a test file handler."""
        np2str.side_effect = lambda x: x
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_hsd.open', m, create=True):
            # Check if file handler raises exception for invalid calibration mode
            with self.assertRaises(ValueError):
                fh = AHIHSDFileHandler(None, {'segment_number': 8, 'total_segments': 10}, None, calib_mode='BAD_MODE')

            fh = AHIHSDFileHandler(None, {'segment_number': 8, 'total_segments': 10}, None)
            fh.proj_info = {'CFAC': 40932549,
                            'COFF': 5500.5,
                            'LFAC': 40932549,
                            'LOFF': 5500.5,
                            'blocklength': 127,
                            'coeff_for_sd': 1737122264.0,
                            'distance_from_earth_center': 42164.0,
                            'earth_equatorial_radius': 6378.137,
                            'earth_polar_radius': 6356.7523,
                            'hblock_number': 3,
                            'req2_rpol2': 1.006739501,
                            'req2_rpol2_req2': 0.0066943844,
                            'resampling_size': 4,
                            'resampling_types': 0,
                            'rpol2_req2': 0.993305616,
                            'spare': '',
                            'sub_lon': 140.7}
            fh.nav_info = {'SSP_longitude': 140.66,
                           'SSP_latitude': 0.03,
                           'distance_earth_center_to_satellite': 42165.04,
                           'nadir_longitude': 140.67,
                           'nadir_latitude': 0.04}
            fh.data_info = {'blocklength': 50,
                            'compression_flag_for_data': 0,
                            'hblock_number': 2,
                            'number_of_bits_per_pixel': 16,
                            'number_of_columns': 11000,
                            'number_of_lines': 1100,
                            'spare': ''}
            fh.basic_info = {
                'observation_start_time': np.array([58413.12523839]),
                'observation_end_time': np.array([58413.12562439]),
                'observation_timeline': np.array([300]),
            }

            self.fh = fh

    def test_time_properties(self):
        """Test start/end/scheduled time properties."""
        self.assertEqual(self.fh.start_time, datetime(2018, 10, 22, 3, 0, 20, 596896))
        self.assertEqual(self.fh.end_time, datetime(2018, 10, 22, 3, 0, 53, 947296))
        self.assertEqual(self.fh.scheduled_time, datetime(2018, 10, 22, 3, 0, 0, 0))

    @mock.patch('satpy.readers.ahi_hsd.AHIHSDFileHandler.__init__',
                return_value=None)
    def test_calibrate(self, *mocks):
        """Test calibration"""
        def_cali = [-0.0037, 15.20]
        upd_cali = [-0.0074, 30.40]
        bad_cali = [0.0, 0.0]
        fh = AHIHSDFileHandler()
        fh.calib_mode = 'NOMINAL'
        fh._header = {
            'block5': {'band_number': [5],
                       'gain_count2rad_conversion': [def_cali[0]],
                       'offset_count2rad_conversion': [def_cali[1]],
                       'central_wave_length': [10.4073], },
            'calibration': {'coeff_rad2albedo_conversion': [0.0019255],
                            'speed_of_light': [299792458.0],
                            'planck_constant': [6.62606957e-34],
                            'boltzmann_constant': [1.3806488e-23],
                            'c0_rad2tb_conversion': [-0.116127314574],
                            'c1_rad2tb_conversion': [1.00099153832],
                            'c2_rad2tb_conversion': [-1.76961091571e-06],
                            'cali_gain_count2rad_conversion': [upd_cali[0]],
                            'cali_offset_count2rad_conversion': [upd_cali[1]]},
        }

        # Counts
        self.assertEqual(fh.calibrate(data=123, calibration='counts'),
                         123)

        # Radiance
        counts = da.array(np.array([[0., 1000.],
                                    [2000., 5000.]]))
        rad_exp = np.array([[15.2, 11.5],
                            [7.8, 0]])
        rad = fh.calibrate(data=counts, calibration='radiance')
        self.assertTrue(np.allclose(rad, rad_exp))

        # Brightness Temperature
        bt_exp = np.array([[330.978979, 310.524688],
                           [285.845017, np.nan]])
        bt = fh.calibrate(data=counts, calibration='brightness_temperature')
        np.testing.assert_allclose(bt, bt_exp)

        # Reflectance
        refl_exp = np.array([[2.92676, 2.214325],
                             [1.50189, 0.]])
        refl = fh.calibrate(data=counts, calibration='reflectance')
        self.assertTrue(np.allclose(refl, refl_exp))

        # Updated calibration
        # Standard operation
        fh.calib_mode = 'UPDATE'
        rad_exp = np.array([[30.4, 23.0],
                            [15.6, 0.]])
        rad = fh.calibrate(data=counts, calibration='radiance')
        self.assertTrue(np.allclose(rad, rad_exp))

        # Case for no updated calibration available (older data)
        fh._header = {
            'block5': {'band_number': [5],
                       'gain_count2rad_conversion': [def_cali[0]],
                       'offset_count2rad_conversion': [def_cali[1]],
                       'central_wave_length': [10.4073], },
            'calibration': {'coeff_rad2albedo_conversion': [0.0019255],
                            'speed_of_light': [299792458.0],
                            'planck_constant': [6.62606957e-34],
                            'boltzmann_constant': [1.3806488e-23],
                            'c0_rad2tb_conversion': [-0.116127314574],
                            'c1_rad2tb_conversion': [1.00099153832],
                            'c2_rad2tb_conversion': [-1.76961091571e-06],
                            'cali_gain_count2rad_conversion': [bad_cali[0]],
                            'cali_offset_count2rad_conversion': [bad_cali[1]]},
        }
        rad = fh.calibrate(data=counts, calibration='radiance')
        rad_exp = np.array([[15.2, 11.5],
                            [7.8, 0]])
        self.assertTrue(np.allclose(rad, rad_exp))

    @mock.patch('satpy.readers.ahi_hsd.AHIHSDFileHandler._read_header')
    @mock.patch('satpy.readers.ahi_hsd.AHIHSDFileHandler._read_data')
    @mock.patch('satpy.readers.ahi_hsd.AHIHSDFileHandler._mask_invalid')
    @mock.patch('satpy.readers.ahi_hsd.AHIHSDFileHandler.calibrate')
    def test_read_band(self, calibrate, *mocks):
        # Test masking of space pixels
        nrows = 25
        ncols = 100
        self.fh.area = AreaDefinition('test', 'test', 'test',
                                      {'a': '6378137.0', 'b': '6356752.3', 'h': '35785863.0', 'lon_0': '140.7',
                                       'proj': 'geos', 'units': 'm'},
                                      ncols, nrows,
                                      [-5499999.901174725, -4399999.92093978, 5499999.901174725, -3299999.9407048346])
        calibrate.return_value = np.ones((nrows, ncols))
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_hsd.open', m, create=True):
            im = self.fh.read_band(info=mock.MagicMock(), key=mock.MagicMock())
            # Note: Within the earth's shape get_geostationary_mask() is True but the numpy.ma mask
            # is False
            mask = im.to_masked_array().mask
            ref_mask = np.logical_not(get_geostationary_mask(self.fh.area).compute())
            self.assertTrue(np.all(mask == ref_mask))

            # Test attributes
            orb_params_exp = {'projection_longitude': 140.7,
                              'projection_latitude': 0.,
                              'projection_altitude': 35785863.0,
                              'satellite_actual_longitude': 140.66,
                              'satellite_actual_latitude': 0.03,
                              'nadir_longitude': 140.67,
                              'nadir_latitude': 0.04}
            self.assertDictContainsSubset(orb_params_exp, im.attrs['orbital_parameters'])
            self.assertTrue(np.isclose(im.attrs['orbital_parameters']['satellite_actual_altitude'], 35786903.00581372))

            # Test if masking space pixels disables with appropriate flag
            self.fh.mask_space = False
            with mock.patch('satpy.readers.ahi_hsd.AHIHSDFileHandler._mask_space') as mask_space:
                self.fh.read_band(info=mock.MagicMock(), key=mock.MagicMock())
                mask_space.assert_not_called()

    def test_blocklen_error(self, *mocks):
        open_name = '%s.open' % __name__
        fpos = 50
        with mock.patch(open_name, create=True) as mock_open:
            with mock_open(mock.MagicMock(), 'r') as fp_:
                # Expected and actual blocklength match
                fp_.tell.return_value = 50
                with warnings.catch_warnings(record=True) as w:
                    self.fh._check_fpos(fp_, fpos, 0, 'header 1')
                    self.assertTrue(len(w) == 0)

                # Expected and actual blocklength do not match
                fp_.tell.return_value = 100
                with warnings.catch_warnings(record=True) as w:
                    self.fh._check_fpos(fp_, fpos, 0, 'header 1')
                    self.assertTrue(len(w) > 0)

    @mock.patch('satpy.readers.ahi_hsd.AHIHSDFileHandler._check_fpos')
    def test_read_header(self, *mocks):
        nhdr = [
            {'blocklength': 0},
            {'blocklength': 0},
            {'blocklength': 0},
            {'blocklength': 0},
            {'blocklength': 0, 'band_number': [4]},
            {'blocklength': 0},
            {'blocklength': 0},
            {'blocklength': 0},
            {'blocklength': 0, 'numof_correction_info_data': [1]},
            {'blocklength': 0},
            {'blocklength': 0, 'number_of_observation_times': [1]},
            {'blocklength': 0},
            {'blocklength': 0, 'number_of_error_info_data': [1]},
            {'blocklength': 0},
            {'blocklength': 0}]
        with mock.patch('numpy.fromfile', side_effect=nhdr):
            self.fh._read_header(mock.MagicMock())


def suite():
    """The test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestAHIHSDNavigation))
    mysuite.addTest(loader.loadTestsFromTestCase(TestAHIHSDFileHandler))
    return mysuite


if __name__ == '__main__':
    unittest.main()
