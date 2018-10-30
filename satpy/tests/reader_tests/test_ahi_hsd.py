# -*- coding: utf-8 -*-

# Copyright (c) 2018 The Pytroll Crew

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""The abi_l1b reader tests package.
"""

import unittest
try:
    from unittest import mock
except ImportError:
    import mock

import numpy as np
from datetime import datetime
from satpy.readers.ahi_hsd import AHIHSDFileHandler


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


def suite():
    """The test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestAHIHSDNavigation))
    mysuite.addTest(loader.loadTestsFromTestCase(TestAHIHSDFileHandler))
    return mysuite


if __name__ == '__main__':
    unittest.main()
