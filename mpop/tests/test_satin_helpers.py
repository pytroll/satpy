#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Author(s):
# 
#   Panu Lahtinen <panu.lahtinen@fmi.fi
# 
# This file is part of mpop.
# 
# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

import unittest

'''Integration testing of
 - :mod:`mpop.satin.helper_functions`
'''

class TestSatinHelpers(unittest.TestCase):
    '''Class for testing mpop.satin'''

    def test_area_def_names_to_extent(self):
        '''Test conversion of area definition names to maximal area
        extent.'''

        from mpop.satin.helper_functions import area_def_names_to_extent
        from pyresample.geometry import AreaDefinition

        # MSG3 proj4 string from
        #  xrit.sat.load(..., only_metadata=True).proj4_params
        proj4_str = 'proj=geos lon_0=0.00 lat_0=0.00 ' \
            'a=6378169.00 b=6356583.80 h=35785831.00'

        # MSG3 maximum extent
        msg_extent = (-5567248.07, -5570248.48, 5570248.48, 5567248.07)

        euro4 = AreaDefinition('euro4', 'Euro 4 km area', 'ps60n',
                               {'ellps': 'bessel',
                                'lat_0': '90',
                                'lat_ts': '60',
                                'lon_0': '14',
                                'proj': 'stere'},
                               1024, 1024,
                               (-2717181.7304994687, -5571048.14031214,
                                 1378818.2695005313, -1475048.1403121399))

        eurol = AreaDefinition('eurol', 'Euro 3.0 km area', 'ps60wgs84',
                               {'ellps': 'WGS84',
                                'lat_0': '90',
                                'lat_ts': '60',
                                'lon_0': '0',
                                'proj': 'stere'},
                               2560, 2048,
                               (-3780000.0, -7644000.0,
                                 3900000.0, -1500000.0))

        area_defs = [eurol, euro4]

        correct_values = [-3182169.2947746729, 1935792.2825874905,
                           3263159.1568166986, 5388866.0573552148]

        max_extent = area_def_names_to_extent(area_defs,
                                              proj4_str,
                                              msg_extent)

        for i in range(len(max_extent)):
            self.assertAlmostEqual(max_extent[i], correct_values[i], 2)

        # Test for case with single area definition
        # Two of the area corner points is outside the satellite view,
        # so one of the extent values ('right' or 'east' border) is
        # replaced with the default value

        afghanistan = AreaDefinition('afghanistan', 'Afghanistan', 'merc',
                                     {'a': '6370997.0',
                                      'lat_0': '35',
                                      'lat_ts': '35',
                                      'lon_0': '67.5',
                                      'proj': 'merc'},
                                     1600, 1600,
                                     (-1600000.0, 1600000.0,
                                       1600000.0, 4800000.0))

        area_defs = afghanistan

        correct_values = [3053894.9120365814, 1619269.0985270864,
                          5570248.48, 4155907.3122006715]

        max_extent = area_def_names_to_extent(area_defs,
                                              proj4_str,
                                              msg_extent)

        for i in range(len(max_extent)):
            self.assertAlmostEqual(max_extent[i], correct_values[i], 2)

        self.assertEqual(max_extent[2], msg_extent[2])

def suite():
    """The test suite for test_satin_helpers.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestSatinHelpers))

    return mysuite
