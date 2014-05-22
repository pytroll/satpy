#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014 Martin Raspaud

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

"""The tests package.
"""

from mpop.tests import (test_satin_helpers,
                        #test_pp_core, # crash
                        test_channel,
                        test_image,
                        test_geo_image,
                        #test_mipp,
                        test_projector,
                        #test_satellites,
                        test_scene,
                        #test_seviri,
                        #test_viirs_sdr,
                        #test_visir,
                        )

import unittest

def suite():
    """The global test suite.
    """

    mysuite = unittest.TestSuite()
    mysuite.addTests(test_satin_helpers.suite())
    #mysuite.addTests(test_pp_core.suite())
    mysuite.addTests(test_channel.suite())
    mysuite.addTests(test_image.suite())
    mysuite.addTests(test_geo_image.suite())
    #mysuite.addTests(test_mipp.suite())
    mysuite.addTests(test_projector.suite())
    #mysuite.addTests(test_satellites.suite())
    mysuite.addTests(test_scene.suite())
    #mysuite.addTests(test_seviri.suite())
    #mysuite.addTests(test_viirs_sdr.suite())
    #mysuite.addTests(test_visir.suite())

    return mysuite
