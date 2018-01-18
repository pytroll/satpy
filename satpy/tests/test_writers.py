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

import os
import errno
import shutil
import unittest
import numpy as np
from satpy import dataset
from satpy.writers import to_image, show

try:
    from unittest import mock
except ImportError:
    import mock


def mkdir_p(path):
    if not path or path == '.':
        return

    # Use for python 2.7 compatibility
    # When python 2.7 support is dropped just use
    # `os.makedirs(path, exist_ok=True)`
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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


class TestEnhancer(unittest.TestCase):
    """Test basic `Enhancer` functionality with builtin configs"""
    def test_basic_init_no_args(self):
        """Test Enhancer init with no arguments passed"""
        from satpy.writers import Enhancer
        e = Enhancer()
        self.assertIsNotNone(e.enhancement_tree)

    def test_basic_init_no_enh(self):
        """Test Enhancer init requesting no enhancements"""
        from satpy.writers import Enhancer
        e = Enhancer(enhancement_config_file=False)
        self.assertIsNone(e.enhancement_tree)

    def test_basic_init_provided_enh(self):
        """Test Enhancer init with string enhancement configs"""
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
        """Test Enhancer init with a nonexistent enhancement configuration file"""
        from satpy.writers import Enhancer
        self.assertRaises(ValueError, Enhancer, enhancement_config_file="is_not_a_valid_filename_?.yaml")


class TestEnhancerUserConfigs(unittest.TestCase):
    """Test `Enhancer` functionality when user's custom configurations are present"""

    TEST_CONFIGS = {
        'test_sensor.yaml': """
sensor_name: visir/test_sensor
enhancements:
  test1_default:
    name: test1
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch ''
      kwargs: {stretch: linear, cutoffs: [0., 0.]}
        
        """,
        'enhancements/test_sensor.yaml': """
sensor_name: visir/test_sensor
enhancements:
  test1_kelvin:
    name: test1
    units: kelvin
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch ''
      kwargs: {stretch: crude, min_stretch: 0, max_stretch: 20}

        """,
        'test_sensor2.yaml': """
sensor_name: visir/test_sensor2


        """,
        'enhancements/test_sensor2.yaml': """
sensor_name: visir/test_sensor2

        """,
    }

    @classmethod
    def setUpClass(cls):
        """Create fake user configurations"""
        for fn, content in cls.TEST_CONFIGS.items():
            base_dir = os.path.dirname(fn)
            mkdir_p(base_dir)
            with open(fn, 'w') as f:
                f.write(content)

    @classmethod
    def tearDownClass(cls):
        """Remove fake user configurations"""
        for fn, content in cls.TEST_CONFIGS.items():
            base_dir = os.path.dirname(fn)
            if base_dir not in ['.', ''] and os.path.isdir(base_dir):
                shutil.rmtree(base_dir)
            elif os.path.isfile(fn):
                os.remove(fn)

    def test_enhance_with_sensor_no_entry(self):
        """Test enhancing an image that has no configuration sections"""
        from satpy.writers import Enhancer, get_enhanced_image
        from satpy import Dataset
        ds = Dataset(np.arange(1, 11.).reshape((2, 5)), sensor='test_sensor2', mode='L')
        e = Enhancer()
        self.assertIsNotNone(e.enhancement_tree)
        get_enhanced_image(ds, enhancer=e)
        self.assertSetEqual(set(e.sensor_enhancement_configs),
                            {'test_sensor2.yaml', 'enhancements/test_sensor2.yaml'})

    def test_enhance_with_sensor_entry(self):
        """Test enhancing an image with a configuration section"""
        from satpy.writers import Enhancer, get_enhanced_image
        from satpy import Dataset
        ds = Dataset(np.arange(1, 11.).reshape((2, 5)),
                     name='test1', sensor='test_sensor', mode='L')
        e = Enhancer()
        self.assertIsNotNone(e.enhancement_tree)
        img = get_enhanced_image(ds, enhancer=e)
        self.assertSetEqual(set(e.sensor_enhancement_configs),
                            {'test_sensor.yaml', 'enhancements/test_sensor.yaml'})
        np.testing.assert_almost_equal(img.channels[0].max(), 1.)

    def test_enhance_with_sensor_entry2(self):
        """Test enhancing an image with a more detailed configuration section"""
        from satpy.writers import Enhancer, get_enhanced_image
        from satpy import Dataset
        ds = Dataset(np.arange(1, 11.).reshape((2, 5)),
                     name='test1', units='kelvin', sensor='test_sensor', mode='L')
        e = Enhancer()
        self.assertIsNotNone(e.enhancement_tree)
        img = get_enhanced_image(ds, enhancer=e)
        self.assertSetEqual(set(e.sensor_enhancement_configs),
                            {'test_sensor.yaml', 'enhancements/test_sensor.yaml'})
        np.testing.assert_almost_equal(img.channels[0].max(), 0.5)


def suite():
    """The test suite for test_projector.
    """
    loader = unittest.TestLoader()
    my_suite = unittest.TestSuite()
    my_suite.addTest(loader.loadTestsFromTestCase(TestWritersModule))
    my_suite.addTest(loader.loadTestsFromTestCase(TestEnhancer))
    my_suite.addTest(loader.loadTestsFromTestCase(TestEnhancerUserConfigs))

    return my_suite
