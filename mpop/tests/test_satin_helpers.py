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
import mpop.satin

'''Integration testing of
 - :mod:`mpop.satin`
'''

class TestReaders(unittest.TestCase):
    '''Class for testing mpop.satin'''

    def test_area_def_names_to_extent(self):
        '''Test conversion of area definition names to maximal area
        extent.'''

        # MSG3 proj4 string from 
        #  xrit.sat.load(..., only_metadata=True).proj4_params
        proj4_str = 'proj=geos lon_0=0.00 lat_0=0.00 ' \
            'a=6378169.00 b=6356583.80 h=35785831.00'

        # MSG3 maximum extent
        msg_extent=(-5567248.07, -5570248.48, 5570248.48, 5567248.07)

        area_def_names = ['eurol', 'euro4']

        correct_values = [-3182169.2947746729, 1935792.2825874905,
                           3263159.1568166986, 5388866.0573552148]

        max_extent = \
            mpop.satin.helper_functions.area_def_names_to_extent(area_def_names,
                                                                 proj4_str,
                                                                 msg_extent)

    
        for i in range(len(max_extent)):
            self.assertAlmostEqual(max_extent[i], correct_values[i], 2)
