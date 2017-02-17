#!/usr/bin/python
# Copyright (c) 2015.
#

# Author(s):
#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
"""

import unittest
import numpy as np
from satpy import dataset
import mock
from satpy.writers import to_image, show

class TestWritersModule(unittest.TestCase):

    def test_to_image_1D(self):
        """
        Conversion to image
        """
        # 1D
        p = dataset.Dataset(np.arange(25))
        self.assertRaises(ValueError, to_image, p)

    @mock.patch('satpy.writers.Image')
    def test_to_image_2D(self, mock_geoimage):
        """
        Conversion to image
        """
        # 2D
        data = np.arange(25).reshape((5, 5))
        p = dataset.Dataset(data, mode="L", fill_value=0, palette=[0, 1, 2, 3, 4, 5])
        to_image(p)
        np.testing.assert_array_equal(data, mock_geoimage.call_args[0][0][0])
        mock_geoimage.reset_mock()

    @mock.patch('satpy.writers.Image')
    def test_to_image_3D(self, mock_geoimage):
        """
        Conversion to image
        """
        # 3D
        data = np.arange(75).reshape((3, 5, 5))
        p = dataset.Dataset(data)
        to_image(p)
        np.testing.assert_array_equal(data[0], mock_geoimage.call_args[0][0][0])
        np.testing.assert_array_equal(data[1], mock_geoimage.call_args[0][0][1])
        np.testing.assert_array_equal(data[2], mock_geoimage.call_args[0][0][2])

    @mock.patch('satpy.writers.get_enhanced_image')
    def test_show(self, mock_get_image):
        data = np.arange(25).reshape((5, 5))
        p = dataset.Dataset(data)
        show(p)
        self.assertTrue(mock_get_image.return_value.show.called)

    def test_show_unloaded(self):
        p = dataset.Dataset([])
        self.assertRaises(ValueError, show, p)


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestWritersModule))

    return my_suite
