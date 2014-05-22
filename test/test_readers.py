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

    def test_lonlat_to_geo_extent(self):
        '''Test conversion of longitudes and latitudes to area extent.'''

        # MSG3 proj4 string from 
        #  xrit.sat.load(..., only_metadata=True).proj4_params
        proj4_str = 'proj=geos lon_0=0.00 lat_0=0.00 ' \
            'a=6378169.00 b=6356583.80 h=35785831.00'

        # MSG3 maximum extent
        max_extent=(-5567248.07, -5570248.48, 
                     5570248.48, 5567248.07)

        # Few area extents in longitudes/latitudes
        area_extents_ll = [[-68.328121068060341, # left longitude
                             18.363816196771392, # down latitude
                             74.770372053870972, # right longitude
                             75.66494585661934], # up latitude
                           # all corners outside Earth's disc
                           [1e30, 1e30, 1e30, 1e30]
                           ]

        # And corresponding correct values in GEO projection
        geo_extents = [[-5010596.02, 1741593.72, 5570248.48, 5567248.07],
                       [-5567248.07, -5570248.48, 5570248.48, 5567248.07]]

        for i in range(len(area_extents_ll)):
            res = mpop.satin.mipp_xrit.lonlat_to_geo_extent(area_extents_ll[i],
                                                            proj4_str,
                                                            max_extent=\
                                                                max_extent)
            for j in range(len(res)):
                self.assertAlmostEqual(res[j], geo_extents[i][j], 2)
