#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015

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

"""test projectable objects.
"""

import unittest
from mpop import projectable
import numpy as np
import sys
import mock

class TestDataset(unittest.TestCase):
    """
    Test the dataset class
    """

    def test_copy(self):
        """
        Test copying a dataset
        """
        ds = projectable.Dataset(np.arange(8), foo="bar")
        ds_copy = ds.copy(False)
        self.assert_(ds_copy.data is ds.data)
        if sys.version >= "2.7":
            self.assertDictEqual(ds.info, ds_copy.info)
        ds_copy = ds.copy(True)
        self.assert_(ds_copy.data is not ds.data
                     and all(ds.data == ds_copy.data))
        if sys.version >= "2.7":
            self.assertDictEqual(ds.info, ds_copy.info)
        ds_copy = ds.copy()
        self.assert_(ds_copy.data is not ds.data
                     and all(ds.data == ds_copy.data))
        if sys.version >= "2.7":
            self.assertDictEqual(ds.info, ds_copy.info)

    def test_arithmetics(self):
        """
        Test the arithmetic functions
        """
        ds = projectable.Dataset(np.arange(1, 25), foo="bar")
        ref = np.arange(1, 25)
        ds2 = ds + 1
        ds3 = projectable.Dataset(np.linspace(0, 1, 24), foo="bar")
        np.testing.assert_array_equal((ds + 1).data, ref + 1)
        np.testing.assert_array_equal((1 + ds).data, ref + 1)
        np.testing.assert_array_equal((ds + ds).data, ref * 2)
        np.testing.assert_array_equal((ds * 2).data, ref * 2)
        np.testing.assert_array_equal((2 * ds).data, ref * 2)
        np.testing.assert_array_equal((ds * ds).data, ref ** 2)
        np.testing.assert_array_equal((ds - 1).data, ref - 1)
        np.testing.assert_array_equal((1 - ds).data, 1 - ref)
        np.testing.assert_array_equal((ds - ds).data, np.zeros_like(ref))
        np.testing.assert_array_equal((ds / 2).data, ref / 2)
        np.testing.assert_array_equal((2 / ds).data, 2 / ref)
        np.testing.assert_array_equal((ds / ds).data, np.ones_like(ref))
        np.testing.assert_array_equal((-ds).data, -ref)
        np.testing.assert_array_equal((abs(ds)).data, abs(ref))
        np.testing.assert_array_equal((ds ** 2).data, ref ** 2)
        np.testing.assert_array_equal((ds ** ds3).data, ref ** ds3.data)

    def test_str_repr(self):
        ds = projectable.Dataset(np.arange(1, 25), foo="bar")
        ds_str = str(ds)
        ds_repr = repr(ds)

class TestProjectable(unittest.TestCase):
    """
    Test the projectable class
    """
    def test_init(self):
        """
        Test initialization
        """
        self.assert_('name' in projectable.Projectable().info)

    def test_isloaded(self):
        """
        Test isloaded method
        """
        self.assertFalse(projectable.Projectable().is_loaded())
        self.assertTrue(projectable.Projectable(data=1).is_loaded())

    def test_to_image_1D(self):
        """
        Conversion to image
        """
        # 1D
        p = projectable.Projectable(np.arange(25))
        self.assertRaises(ValueError, p.to_image)

    def test_str(self):
        # FIXME: Is there a better way to fake the area?
        class FakeArea(object):
            name = "fake_area"

        # Normal situation
        p = projectable.Projectable(np.arange(25),
                                    sensor="fake_sensor",
                                    wavelength_range=500,
                                    resolution=250,
                                    fake_attr="fakeattr",
                                    area=FakeArea(),
                                    )
        p_str = str(p)

        # Not loaded data
        p = projectable.Projectable()
        p_str = str(p)
        self.assertTrue("not loaded" in p_str)

        # Data that doesn't have a shape
        p = projectable.Projectable(data=tuple())
        p_str = str(p)

    @mock.patch('mpop.projectable.GeoImage')
    def test_to_image_2D(self, mock_geoimage):
        """
        Conversion to image
        """
        # 2D
        data = np.arange(25).reshape((5, 5))
        p = projectable.Projectable(data)
        p.to_image()
        np.testing.assert_array_equal(data, mock_geoimage.call_args[0][0][0])
        mock_geoimage.reset_mock()

    @mock.patch('mpop.projectable.GeoImage')
    def test_to_image_3D(self, mock_geoimage):
        """
        Conversion to image
        """
        # 3D
        data = np.arange(75).reshape((3, 5, 5))
        p = projectable.Projectable(data)
        p.to_image()
        np.testing.assert_array_equal(data[0], mock_geoimage.call_args[0][0][0])
        np.testing.assert_array_equal(data[1], mock_geoimage.call_args[0][0][1])
        np.testing.assert_array_equal(data[2], mock_geoimage.call_args[0][0][2])

    @mock.patch('mpop.projectable.GeoImage')
    def test_show_show(self, mock_geoimage):
        data = np.arange(25).reshape((5, 5))
        p = projectable.Projectable(data)
        p.show()
        self.assertTrue(mock_geoimage.return_value.show.called)

    @mock.patch('mpop.projectable.GeoImage')
    def test_show_save(self, mock_geoimage):
        data = np.arange(25).reshape((5, 5))
        p = projectable.Projectable(data)
        filename = "whatever"
        p.show(filename)
        mock_geoimage.return_value.save.assert_called_once_with(filename)

    def test_show_unloaded(self):
        p = projectable.Projectable()
        self.assertRaises(ValueError, p.show)

    @mock.patch('mpop.projectable.resample_kd_tree_nearest')
    def test_resample_2D(self, mock_resampler):
        data = np.arange(25).reshape((5, 5))
        mock_resampler.return_value = data
        p = projectable.Projectable(data)
        source_area = "here"
        destination_area = "there"
        p.info["area"] = source_area
        res = p.resample(destination_area)
        mock_resampler.assert_called_once_with(source_area, data, destination_area)
        self.assertTrue(isinstance(res, projectable.Projectable))
        np.testing.assert_array_equal(res.data, mock_resampler.return_value)

    @mock.patch('mpop.projectable.resample_kd_tree_nearest')
    def test_resample_3D(self, mock_resampler):
        data = np.arange(75).reshape((3, 5, 5))
        mock_resampler.return_value = np.rollaxis(data, 0, 3)
        p = projectable.Projectable(data)
        source_area = "here"
        destination_area = "there"
        p.info["area"] = source_area
        res = p.resample(destination_area)
        self.assertTrue(mock_resampler.called)
        self.assertEqual(mock_resampler.call_args[0][0], source_area)
        np.testing.assert_array_equal(np.rollaxis(data, 0, 3), mock_resampler.call_args[0][1])
        self.assertEqual(mock_resampler.call_args[0][2], destination_area)
        self.assertTrue(isinstance(res, projectable.Projectable))
        np.testing.assert_array_equal(res.data, data)


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestDataset))
    my_suite.addTest(loader.loadTestsFromTestCase(TestProjectable))

    return my_suite
