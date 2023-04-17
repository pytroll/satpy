#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018-2023 Satpy developers
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
"""Unit tests for saving animations using Multiscene."""

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from satpy.tests.multiscene_tests.test_utils import (
    _create_test_area,
    _create_test_dataset,
    _create_test_scenes,
    _fake_get_enhanced_image,
)


class TestMultiSceneSave(unittest.TestCase):
    """Test saving a MultiScene to various formats."""

    def setUp(self):
        """Create temporary directory to save files to."""
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        try:
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    @mock.patch('satpy.multiscene.get_enhanced_image', _fake_get_enhanced_image)
    def test_save_mp4_distributed(self):
        """Save a series of fake scenes to an mp4 video."""
        from satpy import MultiScene
        area = _create_test_area()
        scenes = _create_test_scenes(area=area)

        # Add a dataset to only one of the Scenes
        scenes[1]['ds3'] = _create_test_dataset('ds3')
        # Add a start and end time
        for ds_id in ['ds1', 'ds2', 'ds3']:
            scenes[1][ds_id].attrs['start_time'] = datetime(2018, 1, 2)
            scenes[1][ds_id].attrs['end_time'] = datetime(2018, 1, 2, 12)
            if ds_id == 'ds3':
                continue
            scenes[0][ds_id].attrs['start_time'] = datetime(2018, 1, 1)
            scenes[0][ds_id].attrs['end_time'] = datetime(2018, 1, 1, 12)

        mscn = MultiScene(scenes)
        fn = os.path.join(
            self.base_dir,
            'test_save_mp4_{name}_{start_time:%Y%m%d_%H}_{end_time:%Y%m%d_%H}.mp4')
        writer_mock = mock.MagicMock()
        client_mock = mock.MagicMock()
        client_mock.compute.side_effect = lambda x: tuple(v.compute() for v in x)
        client_mock.gather.side_effect = lambda x: x
        with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer:
            get_writer.return_value = writer_mock
            # force order of datasets by specifying them
            mscn.save_animation(fn, client=client_mock, datasets=['ds1', 'ds2', 'ds3'])

        # 2 saves for the first scene + 1 black frame
        # 3 for the second scene
        self.assertEqual(writer_mock.append_data.call_count, 3 + 3)
        filenames = [os.path.basename(args[0][0]) for args in get_writer.call_args_list]
        self.assertEqual(filenames[0], 'test_save_mp4_ds1_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[1], 'test_save_mp4_ds2_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[2], 'test_save_mp4_ds3_20180102_00_20180102_12.mp4')

        # Test no distributed client found
        mscn = MultiScene(scenes)
        fn = os.path.join(
            self.base_dir,
            'test_save_mp4_{name}_{start_time:%Y%m%d_%H}_{end_time:%Y%m%d_%H}.mp4')
        writer_mock = mock.MagicMock()
        client_mock = mock.MagicMock()
        client_mock.compute.side_effect = lambda x: tuple(v.compute() for v in x)
        client_mock.gather.side_effect = lambda x: x
        with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer, \
                mock.patch('satpy.multiscene.get_client', mock.Mock(side_effect=ValueError("No client"))):
            get_writer.return_value = writer_mock
            # force order of datasets by specifying them
            mscn.save_animation(fn, datasets=['ds1', 'ds2', 'ds3'])

        # 2 saves for the first scene + 1 black frame
        # 3 for the second scene
        self.assertEqual(writer_mock.append_data.call_count, 3 + 3)
        filenames = [os.path.basename(args[0][0]) for args in get_writer.call_args_list]
        self.assertEqual(filenames[0], 'test_save_mp4_ds1_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[1], 'test_save_mp4_ds2_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[2], 'test_save_mp4_ds3_20180102_00_20180102_12.mp4')

    @mock.patch('satpy.multiscene.get_enhanced_image', _fake_get_enhanced_image)
    def test_save_mp4_no_distributed(self):
        """Save a series of fake scenes to an mp4 video when distributed isn't available."""
        from satpy import MultiScene
        area = _create_test_area()
        scenes = _create_test_scenes(area=area)

        # Add a dataset to only one of the Scenes
        scenes[1]['ds3'] = _create_test_dataset('ds3')
        # Add a start and end time
        for ds_id in ['ds1', 'ds2', 'ds3']:
            scenes[1][ds_id].attrs['start_time'] = datetime(2018, 1, 2)
            scenes[1][ds_id].attrs['end_time'] = datetime(2018, 1, 2, 12)
            if ds_id == 'ds3':
                continue
            scenes[0][ds_id].attrs['start_time'] = datetime(2018, 1, 1)
            scenes[0][ds_id].attrs['end_time'] = datetime(2018, 1, 1, 12)

        mscn = MultiScene(scenes)
        fn = os.path.join(
            self.base_dir,
            'test_save_mp4_{name}_{start_time:%Y%m%d_%H}_{end_time:%Y%m%d_%H}.mp4')
        writer_mock = mock.MagicMock()
        client_mock = mock.MagicMock()
        client_mock.compute.side_effect = lambda x: tuple(v.compute() for v in x)
        client_mock.gather.side_effect = lambda x: x
        with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer, \
                mock.patch('satpy.multiscene.get_client', None):
            get_writer.return_value = writer_mock
            # force order of datasets by specifying them
            mscn.save_animation(fn, datasets=['ds1', 'ds2', 'ds3'])

        # 2 saves for the first scene + 1 black frame
        # 3 for the second scene
        self.assertEqual(writer_mock.append_data.call_count, 3 + 3)
        filenames = [os.path.basename(args[0][0]) for args in get_writer.call_args_list]
        self.assertEqual(filenames[0], 'test_save_mp4_ds1_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[1], 'test_save_mp4_ds2_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[2], 'test_save_mp4_ds3_20180102_00_20180102_12.mp4')

    @mock.patch('satpy.multiscene.get_enhanced_image', _fake_get_enhanced_image)
    def test_save_datasets_simple(self):
        """Save a series of fake scenes to an PNG images."""
        from satpy import MultiScene
        area = _create_test_area()
        scenes = _create_test_scenes(area=area)

        # Add a dataset to only one of the Scenes
        scenes[1]['ds3'] = _create_test_dataset('ds3')
        # Add a start and end time
        for ds_id in ['ds1', 'ds2', 'ds3']:
            scenes[1][ds_id].attrs['start_time'] = datetime(2018, 1, 2)
            scenes[1][ds_id].attrs['end_time'] = datetime(2018, 1, 2, 12)
            if ds_id == 'ds3':
                continue
            scenes[0][ds_id].attrs['start_time'] = datetime(2018, 1, 1)
            scenes[0][ds_id].attrs['end_time'] = datetime(2018, 1, 1, 12)

        mscn = MultiScene(scenes)
        client_mock = mock.MagicMock()
        client_mock.compute.side_effect = lambda x: tuple(v for v in x)
        client_mock.gather.side_effect = lambda x: x
        with mock.patch('satpy.multiscene.Scene.save_datasets') as save_datasets:
            save_datasets.return_value = [True]  # some arbitrary return value
            # force order of datasets by specifying them
            mscn.save_datasets(base_dir=self.base_dir, client=False, datasets=['ds1', 'ds2', 'ds3'],
                               writer='simple_image')

        # 2 for each scene
        self.assertEqual(save_datasets.call_count, 2)

    @mock.patch('satpy.multiscene.get_enhanced_image', _fake_get_enhanced_image)
    def test_save_datasets_distributed_delayed(self):
        """Test distributed save for writers returning delayed obejcts e.g. simple_image."""
        from dask.delayed import Delayed

        from satpy import MultiScene
        area = _create_test_area()
        scenes = _create_test_scenes(area=area)

        # Add a dataset to only one of the Scenes
        scenes[1]['ds3'] = _create_test_dataset('ds3')
        # Add a start and end time
        for ds_id in ['ds1', 'ds2', 'ds3']:
            scenes[1][ds_id].attrs['start_time'] = datetime(2018, 1, 2)
            scenes[1][ds_id].attrs['end_time'] = datetime(2018, 1, 2, 12)
            if ds_id == 'ds3':
                continue
            scenes[0][ds_id].attrs['start_time'] = datetime(2018, 1, 1)
            scenes[0][ds_id].attrs['end_time'] = datetime(2018, 1, 1, 12)

        mscn = MultiScene(scenes)
        client_mock = mock.MagicMock()
        client_mock.compute.side_effect = lambda x: tuple(v for v in x)
        client_mock.gather.side_effect = lambda x: x
        future_mock = mock.MagicMock()
        future_mock.__class__ = Delayed
        with mock.patch('satpy.multiscene.Scene.save_datasets') as save_datasets:
            save_datasets.return_value = [future_mock]  # some arbitrary return value
            # force order of datasets by specifying them
            mscn.save_datasets(base_dir=self.base_dir, client=client_mock, datasets=['ds1', 'ds2', 'ds3'],
                               writer='simple_image')

        # 2 for each scene
        self.assertEqual(save_datasets.call_count, 2)

    @mock.patch('satpy.multiscene.get_enhanced_image', _fake_get_enhanced_image)
    def test_save_datasets_distributed_source_target(self):
        """Test distributed save for writers returning sources and targets e.g. geotiff writer."""
        import dask.array as da

        from satpy import MultiScene
        area = _create_test_area()
        scenes = _create_test_scenes(area=area)

        # Add a dataset to only one of the Scenes
        scenes[1]['ds3'] = _create_test_dataset('ds3')
        # Add a start and end time
        for ds_id in ['ds1', 'ds2', 'ds3']:
            scenes[1][ds_id].attrs['start_time'] = datetime(2018, 1, 2)
            scenes[1][ds_id].attrs['end_time'] = datetime(2018, 1, 2, 12)
            if ds_id == 'ds3':
                continue
            scenes[0][ds_id].attrs['start_time'] = datetime(2018, 1, 1)
            scenes[0][ds_id].attrs['end_time'] = datetime(2018, 1, 1, 12)

        mscn = MultiScene(scenes)
        client_mock = mock.MagicMock()
        client_mock.compute.side_effect = lambda x: tuple(v for v in x)
        client_mock.gather.side_effect = lambda x: x
        source_mock = mock.MagicMock()
        source_mock.__class__ = da.Array
        target_mock = mock.MagicMock()
        with mock.patch('satpy.multiscene.Scene.save_datasets') as save_datasets:
            save_datasets.return_value = [(source_mock, target_mock)]  # some arbitrary return value
            # force order of datasets by specifying them
            with self.assertRaises(NotImplementedError):
                mscn.save_datasets(base_dir=self.base_dir, client=client_mock, datasets=['ds1', 'ds2', 'ds3'],
                                   writer='geotiff')

    def test_crop(self):
        """Test the crop method."""
        import numpy as np
        from pyresample.geometry import AreaDefinition
        from xarray import DataArray

        from satpy import MultiScene, Scene
        scene1 = Scene()
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927,
                       5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        x_size = 3712
        y_size = 3712
        area_def = AreaDefinition(
            'test', 'test', 'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )
        area_def2 = AreaDefinition(
            'test2', 'test2', 'test2', proj_dict,
            x_size // 2,
            y_size // 2,
            area_extent,
        )
        scene1["1"] = DataArray(np.zeros((y_size, x_size)))
        scene1["2"] = DataArray(np.zeros((y_size, x_size)), dims=('y', 'x'))
        scene1["3"] = DataArray(np.zeros((y_size, x_size)), dims=('y', 'x'),
                                attrs={'area': area_def})
        scene1["4"] = DataArray(np.zeros((y_size // 2, x_size // 2)), dims=('y', 'x'),
                                attrs={'area': area_def2})
        mscn = MultiScene([scene1])

        # by lon/lat bbox
        new_mscn = mscn.crop(ll_bbox=(-20., -5., 0, 0))
        new_scn1 = list(new_mscn.scenes)[0]
        self.assertIn('1', new_scn1)
        self.assertIn('2', new_scn1)
        self.assertIn('3', new_scn1)
        self.assertTupleEqual(new_scn1['1'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['2'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['3'].shape, (184, 714))
        self.assertTupleEqual(new_scn1['4'].shape, (92, 357))


@mock.patch('satpy.multiscene.get_enhanced_image')
def test_save_mp4(smg, tmp_path):
    """Save a series of fake scenes to an mp4 video."""
    from satpy import MultiScene
    area = _create_test_area()
    scenes = _create_test_scenes(area=area)
    smg.side_effect = _fake_get_enhanced_image

    # Add a dataset to only one of the Scenes
    scenes[1]['ds3'] = _create_test_dataset('ds3')
    # Add a start and end time
    for ds_id in ['ds1', 'ds2', 'ds3']:
        scenes[1][ds_id].attrs['start_time'] = datetime(2018, 1, 2)
        scenes[1][ds_id].attrs['end_time'] = datetime(2018, 1, 2, 12)
        if ds_id == 'ds3':
            continue
        scenes[0][ds_id].attrs['start_time'] = datetime(2018, 1, 1)
        scenes[0][ds_id].attrs['end_time'] = datetime(2018, 1, 1, 12)

    mscn = MultiScene(scenes)
    fn = str(tmp_path /
             'test_save_mp4_{name}_{start_time:%Y%m%d_%H}_{end_time:%Y%m%d_%H}.mp4')
    writer_mock = mock.MagicMock()
    with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer:
        get_writer.return_value = writer_mock
        # force order of datasets by specifying them
        mscn.save_animation(fn, datasets=['ds1', 'ds2', 'ds3'], client=False)

    # 2 saves for the first scene + 1 black frame
    # 3 for the second scene
    assert writer_mock.append_data.call_count == 3 + 3
    filenames = [os.path.basename(args[0][0]) for args in get_writer.call_args_list]
    assert filenames[0] == 'test_save_mp4_ds1_20180101_00_20180102_12.mp4'
    assert filenames[1] == 'test_save_mp4_ds2_20180101_00_20180102_12.mp4'
    assert filenames[2] == 'test_save_mp4_ds3_20180102_00_20180102_12.mp4'

    # make sure that not specifying datasets still saves all of them
    fn = str(tmp_path /
             'test_save_mp4_{name}_{start_time:%Y%m%d_%H}_{end_time:%Y%m%d_%H}.mp4')
    writer_mock = mock.MagicMock()
    with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer:
        get_writer.return_value = writer_mock
        # force order of datasets by specifying them
        mscn.save_animation(fn, client=False)
    # the 'ds3' dataset isn't known to the first scene so it doesn't get saved
    # 2 for first scene, 2 for second scene
    assert writer_mock.append_data.call_count == 2 + 2
    assert "test_save_mp4_ds1_20180101_00_20180102_12.mp4" in filenames
    assert "test_save_mp4_ds2_20180101_00_20180102_12.mp4" in filenames
    assert "test_save_mp4_ds3_20180102_00_20180102_12.mp4" in filenames

    # test decorating and enhancing

    fn = str(tmp_path /
             'test-{name}_{start_time:%Y%m%d_%H}_{end_time:%Y%m%d_%H}-rich.mp4')
    writer_mock = mock.MagicMock()
    with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer:
        get_writer.return_value = writer_mock
        mscn.save_animation(
            fn, client=False,
            enh_args={"decorate": {
                "decorate": [{
                    "text": {
                        "txt":
                            "Test {start_time:%Y-%m-%d %H:%M} - "
                            "{end_time:%Y-%m-%d %H:%M}"}}]}})
    assert writer_mock.append_data.call_count == 2 + 2
    assert ("2018-01-02" in smg.call_args_list[-1][1]
            ["decorate"]["decorate"][0]["text"]["txt"])
