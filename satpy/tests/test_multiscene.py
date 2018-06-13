#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018.
#
# Author(s):
#
#   David Hoese <david.hoese@ssec.wisc.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for multiscene.py.
"""

import os
import sys
import tempfile
import shutil

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

DEFAULT_SHAPE = (5, 10)


def _fake_get_enhanced_image(img):
    from trollimage.xrimage import XRImage
    return XRImage(img)


def _create_test_area(proj_str=None, shape=DEFAULT_SHAPE, extents=None):
    """Create a test area definition."""
    from pyresample.geometry import AreaDefinition
    from pyresample.utils import proj4_str_to_dict
    if proj_str is None:
        proj_str = '+proj=lcc +datum=WGS84 +ellps=WGS84 +lon_0=-95. ' \
                   '+lat_0=25 +lat_1=25 +units=m +no_defs'
    proj_dict = proj4_str_to_dict(proj_str)
    extents = extents or (-1000., -1500., 1000., 1500.)

    return AreaDefinition(
        'test',
        'test',
        'test',
        proj_dict,
        x_size=shape[1],
        y_size=shape[0],
        area_extent=extents
    )


def _create_test_dataset(name, shape=DEFAULT_SHAPE, area=None):
    """Create a test DataArray object."""
    import xarray as xr
    import dask.array as da
    import numpy as np

    return xr.DataArray(
        da.zeros(shape, dtype=np.float32, chunks=shape), dims=('y', 'x'),
        attrs={'name': name, 'area': area})


def _create_test_scenes(num_scenes=2, shape=DEFAULT_SHAPE, area=None):
    """Helper to create some test scenes."""
    from satpy import Scene
    ds1 = _create_test_dataset('ds1', shape=shape, area=area)
    ds2 = _create_test_dataset('ds2', shape=shape, area=area)
    scenes = []
    for scn_idx in range(num_scenes):
        scn = Scene()
        scn['ds1'] = ds1
        scn['ds2'] = ds2
        scenes.append(scn)
    return scenes


class TestMultiScene(unittest.TestCase):
    """Test basic functionality of MultiScene."""

    def test_init_empty(self):
        """Test creating a multiscene with no children."""
        from satpy import MultiScene
        mscn = MultiScene()

    def test_init_children(self):
        """Test creating a multiscene with children."""
        from satpy import MultiScene
        scenes = _create_test_scenes()
        mscn = MultiScene(scenes)

    def test_properties(self):
        """Test basic properties/attributes of the MultiScene."""
        from satpy import MultiScene, DatasetID

        area = _create_test_area()
        scenes = _create_test_scenes(area=area)
        ds1_id = DatasetID(name='ds1')
        ds2_id = DatasetID(name='ds2')
        ds3_id = DatasetID(name='ds3')
        ds4_id = DatasetID(name='ds4')

        # Add a dataset to only one of the Scenes
        scenes[1]['ds3'] = _create_test_dataset('ds3')
        mscn = MultiScene(scenes)

        self.assertSetEqual(mscn.loaded_dataset_ids,
                            {ds1_id, ds2_id, ds3_id})
        self.assertSetEqual(mscn.shared_dataset_ids, {ds1_id, ds2_id})
        self.assertTrue(mscn.all_same_area)

        bigger_area = _create_test_area(shape=(20, 40))
        scenes[0]['ds4'] = _create_test_dataset('ds4', shape=(20, 40),
                                                     area=bigger_area)

        self.assertSetEqual(mscn.loaded_dataset_ids,
                            {ds1_id, ds2_id, ds3_id, ds4_id})
        self.assertSetEqual(mscn.shared_dataset_ids, {ds1_id, ds2_id})
        self.assertFalse(mscn.all_same_area)


class TestMultiSceneSave(unittest.TestCase):
    """Test saving a MultiScene to various formats."""

    def setUp(self):
        """Create temporary directory to save files to"""
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test"""
        try:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    @mock.patch('satpy.multiscene.get_enhanced_image', _fake_get_enhanced_image)
    def test_save_mp4(self):
        """Save a series of fake scenes to an mp4 video."""
        from satpy import MultiScene, DatasetID

        area = _create_test_area()
        scenes = _create_test_scenes(area=area)
        ds1_id = DatasetID(name='ds1')
        ds2_id = DatasetID(name='ds2')
        ds3_id = DatasetID(name='ds3')
        ds4_id = DatasetID(name='ds4')

        # Add a dataset to only one of the Scenes
        scenes[1]['ds3'] = _create_test_dataset('ds3')
        mscn = MultiScene(scenes)
        fn = os.path.join(self.base_dir, 'test_save_mp4.mp4')
        mscn.save(fn)


def suite():
    """The test suite for test_multiscene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestMultiScene))
    mysuite.addTest(loader.loadTestsFromTestCase(TestMultiSceneSave))

    return mysuite


if __name__ == "__main__":
    unittest.main()
