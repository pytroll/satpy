#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010, 2011, 2012, 2014, 2015.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

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
"""Unit tests for scene.py.
"""

import os
import sys

import mock

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

# clear the config dir environment variable so it doesn't interfere
os.environ.pop("PPP_CONFIG_DIR", None)


class TestScene(unittest.TestCase):
    """
    Test the scene class
    """

    def test_init(self):
        import satpy.scene
        with mock.patch('satpy.scene.Scene._compute_metadata_from_readers') as cmfr:
            with mock.patch('satpy.scene.Scene.create_reader_instances') as cri:
                test_mda = {'test1': 'value1', 'test2': 'value2'}
                cmfr.return_value = test_mda
                test_mda_2 = {'test3': 'value3', 'test4': 'value4'}
                scene = satpy.scene.Scene(filenames='bla',
                                          base_dir='bli',
                                          reader='blo',
                                          **test_mda_2)
                cri.assert_called_once_with(filenames='bla',
                                            base_dir='bli',
                                            reader='blo')
                self.assertDictContainsSubset(test_mda, scene.info)
                self.assertDictContainsSubset(test_mda_2, scene.info)

    def test_create_reader_instances_with_filenames(self):
        import satpy.scene
        filenames = ["bla", "foo", "bar"]
        sensors = None
        reader_name = None
        with mock.patch('satpy.scene.Scene._compute_metadata_from_readers'):
            with mock.patch('satpy.scene.ReaderFinder') as findermock:
                scene = satpy.scene.Scene(filenames=filenames)
                findermock.assert_called_once_with(ppp_config_dir=mock.ANY,
                                                   base_dir=None,
                                                   area=None,
                                                   end_time=None,
                                                   start_time=None)
                findermock.return_value.assert_called_once_with(
                    reader=reader_name,
                    sensor=sensors,
                    filenames=filenames)

    def test_init_with_empty_filenames(self):
        from satpy.scene import Scene
        filenames = []
        Scene(filenames=filenames)

    def test_create_reader_instances_with_sensor(self):
        import satpy.scene
        sensors = ["bla", "foo", "bar"]
        filenames = None
        reader_name = None
        with mock.patch('satpy.scene.Scene._compute_metadata_from_readers'):
            with mock.patch('satpy.scene.ReaderFinder') as findermock:
                scene = satpy.scene.Scene(sensor=sensors)
                findermock.assert_called_once_with(ppp_config_dir=mock.ANY,
                                                   base_dir=None,
                                                   area=None,
                                                   end_time=None,
                                                   start_time=None)
                findermock.return_value.assert_called_once_with(
                    reader=reader_name,
                    sensor=sensors,
                    filenames=filenames)

    def test_create_reader_instances_with_sensor_and_filenames(self):
        import satpy.scene
        sensors = ["bla", "foo", "bar"]
        filenames = ["1", "2", "3"]
        reader_name = None
        with mock.patch('satpy.scene.Scene._compute_metadata_from_readers'):
            with mock.patch('satpy.scene.ReaderFinder') as findermock:
                scene = satpy.scene.Scene(sensor=sensors, filenames=filenames)
                findermock.assert_called_once_with(ppp_config_dir=mock.ANY,
                                                   base_dir=None,
                                                   area=None,
                                                   end_time=None,
                                                   start_time=None)
                findermock.return_value.assert_called_once_with(
                    reader=reader_name,
                    sensor=sensors,
                    filenames=filenames)

    def test_create_reader_instances_with_reader(self):
        from satpy.scene import Scene
        reader = "foo"
        filenames = ["1", "2", "3"]
        sensors = None
        with mock.patch('satpy.scene.Scene._compute_metadata_from_readers'):
            with mock.patch('satpy.scene.ReaderFinder') as findermock:
                scene = Scene(reader=reader, filenames=filenames)
                findermock.assert_called_once_with(ppp_config_dir=mock.ANY,
                                                   base_dir=None,
                                                   area=None,
                                                   end_time=None,
                                                   start_time=None)
                findermock.return_value.assert_called_once_with(
                    reader=reader,
                    sensor=sensors,
                    filenames=filenames)

    def test_init_alone(self):
        from satpy.scene import Scene
        from satpy.config import PACKAGE_CONFIG_PATH
        scn = Scene()
        self.assertEqual(scn.ppp_config_dir, PACKAGE_CONFIG_PATH)

    def test_init_with_ppp_config_dir(self):
        from satpy.scene import Scene
        scn = Scene(ppp_config_dir="foo")
        self.assertEqual(scn.ppp_config_dir, 'foo')

    def test_iter(self):
        from satpy import Scene, Projectable
        import numpy as np
        scene = Scene()
        scene["1"] = Projectable(np.arange(5))
        scene["2"] = Projectable(np.arange(5))
        scene["3"] = Projectable(np.arange(5))
        for x in scene:
            self.assertIsInstance(x, Projectable)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestScene))

    return mysuite

if __name__ == "__main__":
    unittest.main()
