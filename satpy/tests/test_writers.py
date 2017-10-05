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

import mock
import numpy as np
import xarray as xr
import yaml

from satpy.writers import show, to_image


class TestWritersModule(unittest.TestCase):

    def test_to_image_1D(self):
        """
        Conversion to image
        """
        # 1D
        p = xr.DataArray(np.arange(25), dims=['y'])
        self.assertRaises(ValueError, to_image, p)

    @mock.patch('satpy.writers.Image')
    def test_to_image_2D(self, mock_geoimage):
        """
        Conversion to image
        """
        # 2D
        data = np.arange(25).reshape((5, 5))
        p = xr.DataArray(data, attrs=dict(
            mode="L", fill_value=0, palette=[0, 1, 2, 3, 4, 5]),
            dims=['y', 'x'])
        to_image(p)

        np.testing.assert_array_equal(
            data, mock_geoimage.call_args[0][0][0])
        mock_geoimage.reset_mock()

    @mock.patch('satpy.writers.Image')
    def test_to_image_3D(self, mock_geoimage):
        """
        Conversion to image
        """
        # 3D
        data = np.arange(75).reshape((3, 5, 5))
        p = xr.DataArray(data, dims=['bands', 'y', 'x'])
        p['bands'] = ['R', 'G', 'B']
        to_image(p)
        np.testing.assert_array_equal(data[0], mock_geoimage.call_args[0][0][0])
        np.testing.assert_array_equal(data[1], mock_geoimage.call_args[0][0][1])
        np.testing.assert_array_equal(data[2], mock_geoimage.call_args[0][0][2])

    @mock.patch('satpy.writers.get_enhanced_image')
    def test_show(self, mock_get_image):
        data = np.arange(25).reshape((5, 5))
        p = xr.DataArray(data, dims=['y', 'x'])
        show(p)
        self.assertTrue(mock_get_image.return_value.show.called)


class TestEnhancer(unittest.TestCase):

    def test_basic_init_no_args(self):
        from satpy.writers import Enhancer
        e = Enhancer()
        self.assertIsNotNone(e.enhancement_tree)

    def test_basic_init_no_enh(self):
        from satpy.writers import Enhancer
        e = Enhancer(enhancement_config_file=False)
        self.assertIsNone(e.enhancement_tree)

    def test_basic_init_provided_enh(self):
        from satpy.writers import Enhancer
        e = Enhancer(enhancement_config_file=["""enhancements:
  enh1:
    standard_name: toa_bidirectional_reflectance
    operations:
    - name: stretch
      method: &stretchfun !!python/name:satpy.enhancements.stretch ''
      kwargs: {stretch: linear}
"""])
        self.assertIsNotNone(e.enhancement_tree)

    def test_init_nonexistent_enh_file(self):
        from satpy.writers import Enhancer
        self.assertRaises(
            ValueError, Enhancer, enhancement_config_file="is_not_a_valid_filename_?.yaml")


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestWritersModule))
    my_suite.addTest(loader.loadTestsFromTestCase(TestEnhancer))

    return my_suite
