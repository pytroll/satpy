#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author(s):
#
#   Martin Raspaud <martin.raspaud@smhi.se>
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

import os
import sys
import unittest

import satpy.readers.yaml_reader as yr


class TestUtils(unittest.TestCase):

    def test_get_filebase(self):
        pattern = '{mission_id:3s}_OL_{processing_level:1s}_{datatype_id:_<6s}_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_{creation_time:%Y%m%dT%H%M%S}_{duration:4d}_{cycle:3d}_{relative_orbit:3d}_{frame:4d}_{centre:3s}_{mode:1s}_{timeliness:2s}_{collection:3s}.SEN3/geo_coordinates.nc'
        filename = '/home/a001673/data/satellite/Sentinel-3/S3A_OL_1_EFR____20161020T081224_20161020T081524_20161020T102406_0179_010_078_2340_SVL_O_NR_002.SEN3/Oa05_radiance.nc'
        expected = 'S3A_OL_1_EFR____20161020T081224_20161020T081524_20161020T102406_0179_010_078_2340_SVL_O_NR_002.SEN3/Oa05_radiance.nc'
        self.assertEqual(yr.get_filebase(filename, pattern), expected)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestUtils))

    return mysuite


if __name__ == "__main__":
    unittest.main()
