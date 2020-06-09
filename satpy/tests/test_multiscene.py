#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Unit tests for multiscene.py."""

import os
import tempfile
import shutil
from datetime import datetime
import unittest
from unittest import mock

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
        shape[1],
        shape[0],
        extents
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
    """Create some test scenes for various test cases."""
    from satpy import Scene
    ds1 = _create_test_dataset('ds1', shape=shape, area=area)
    ds2 = _create_test_dataset('ds2', shape=shape, area=area)
    scenes = []
    for _ in range(num_scenes):
        scn = Scene()
        scn['ds1'] = ds1.copy()
        scn['ds2'] = ds2.copy()
        scenes.append(scn)
    return scenes


class TestMultiScene(unittest.TestCase):
    """Test basic functionality of MultiScene."""

    def test_init_empty(self):
        """Test creating a multiscene with no children."""
        from satpy import MultiScene
        MultiScene()

    def test_init_children(self):
        """Test creating a multiscene with children."""
        from satpy import MultiScene
        scenes = _create_test_scenes()
        MultiScene(scenes)

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

    def test_from_files(self):
        """Test creating a multiscene from multiple files."""
        from satpy import MultiScene
        input_files = [
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171502203_e20171171504576_c20171171505018.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171507203_e20171171509576_c20171171510018.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171512203_e20171171514576_c20171171515017.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171517203_e20171171519577_c20171171520019.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171522203_e20171171524576_c20171171525020.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171527203_e20171171529576_c20171171530017.nc",
        ]
        with mock.patch('satpy.multiscene.Scene') as scn_mock:
            mscn = MultiScene.from_files(input_files, reader='abi_l1b')
            self.assertTrue(len(mscn.scenes), 6)
            calls = [mock.call(filenames={'abi_l1b': [in_file]}) for in_file in input_files]
            scn_mock.assert_has_calls(calls)

    def test_group(self):
        from satpy import Scene, MultiScene, DatasetID

        ds1 = _create_test_dataset(name='ds1')
        ds2 = _create_test_dataset(name='ds2')
        ds3 = _create_test_dataset(name='ds3')
        ds4 = _create_test_dataset(name='ds4')
        scene1 = Scene()
        scene1['ds1'] = ds1
        scene1['ds2'] = ds2
        scene2 = Scene()
        scene2['ds3'] = ds3
        scene2['ds4'] = ds4

        multi_scene = MultiScene([scene1, scene2])
        groups = {DatasetID(name='odd', wavelength=(1, 2, 3)): ['ds1', 'ds3'],
                  DatasetID(name='even', wavelength=(2, 3, 4)): ['ds2', 'ds4']}
        multi_scene.group(groups)

        self.assertSetEqual(multi_scene.shared_dataset_ids, set(groups.keys()))

    def test_add_group_aliases(self):
        """Test adding group aliases."""
        import xarray as xr
        import numpy as np
        import types

        from satpy.multiscene import add_group_aliases
        from satpy import DatasetID
        from satpy import Scene

        # Define test scenes
        ds_id1 = DatasetID('ds1', wavelength=(10.7, 10.8, 10.9))
        ds_id2 = DatasetID('ds2', wavelength=(1.9, 2.0, 2.1))
        ds_id3 = DatasetID('ds3', wavelength=(10.8, 10.9, 11.0))
        ds_id31 = DatasetID('ds31', polarization='H')

        scene1 = Scene()
        scene1[ds_id1] = xr.DataArray([1])
        scene2 = Scene()
        scene2[ds_id2] = xr.DataArray([2])
        scene3 = Scene()
        scene3[ds_id3] = xr.DataArray([3])
        scene3[ds_id31] = xr.DataArray([4])
        scenes = [scene1, scene2, scene3]

        # Define groups
        g1 = DatasetID(name='g1', wavelength=(10, 11, 12))
        g2 = DatasetID(name='g2', wavelength=(1, 2, 3), polarization='V')
        groups = {g1: ['ds1', 'ds3'], g2: ['ds2']}

        # Test adding aliases
        with_aliases = add_group_aliases(iter(scenes), groups)
        self.assertIsInstance(with_aliases, types.GeneratorType)
        with_aliases = list(with_aliases)
        self.assertSetEqual(set(with_aliases[0].keys()), {g1, ds_id1})
        self.assertSetEqual(set(with_aliases[1].keys()), {g2, ds_id2})
        self.assertSetEqual(set(with_aliases[2].keys()), {g1, ds_id3, ds_id31})

        np.testing.assert_array_equal(with_aliases[0]['g1'].values, [1])
        np.testing.assert_array_equal(with_aliases[0]['ds1'].values, [1])
        np.testing.assert_array_equal(with_aliases[1]['g2'].values, [2])
        np.testing.assert_array_equal(with_aliases[1]['ds2'].values, [2])
        np.testing.assert_array_equal(with_aliases[2]['g1'].values, [3])
        np.testing.assert_array_equal(with_aliases[2]['ds3'].values, [3])
        np.testing.assert_array_equal(with_aliases[2]['ds31'].values, [4])

        # Make sure that modifying the result doesn't modify the original
        self.assertNotIn(g1, scene1)

        # Adding an alias for multiple datasets in one scene should fail
        gen = add_group_aliases([scene3], {g1: ['ds3', 'ds31']})
        self.assertRaises(ValueError, list, gen)


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
    def test_save_mp4(self):
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
        with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer:
            get_writer.return_value = writer_mock
            # force order of datasets by specifying them
            mscn.save_animation(fn, datasets=['ds1', 'ds2', 'ds3'], client=False)

        # 2 saves for the first scene + 1 black frame
        # 3 for the second scene
        self.assertEqual(writer_mock.append_data.call_count, 3 + 3)
        filenames = [os.path.basename(args[0][0]) for args in get_writer.call_args_list]
        self.assertEqual(filenames[0], 'test_save_mp4_ds1_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[1], 'test_save_mp4_ds2_20180101_00_20180102_12.mp4')
        self.assertEqual(filenames[2], 'test_save_mp4_ds3_20180102_00_20180102_12.mp4')

        # make sure that not specifying datasets still saves all of them
        fn = os.path.join(
            self.base_dir,
            'test_save_mp4_{name}_{start_time:%Y%m%d_%H}_{end_time:%Y%m%d_%H}.mp4')
        writer_mock = mock.MagicMock()
        with mock.patch('satpy.multiscene.imageio.get_writer') as get_writer:
            get_writer.return_value = writer_mock
            # force order of datasets by specifying them
            mscn.save_animation(fn, client=False)
        # the 'ds3' dataset isn't known to the first scene so it doesn't get saved
        # 2 for first scene, 2 for second scene
        self.assertEqual(writer_mock.append_data.call_count, 2 + 2)
        self.assertIn('test_save_mp4_ds1_20180101_00_20180102_12.mp4', filenames)
        self.assertIn('test_save_mp4_ds2_20180101_00_20180102_12.mp4', filenames)
        self.assertIn('test_save_mp4_ds3_20180102_00_20180102_12.mp4', filenames)

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
    def test_save_datasets_distributed(self):
        """Save a series of fake scenes to an PNG images using dask distributed."""
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
        with mock.patch('satpy.multiscene.Scene.save_datasets') as save_datasets:
            save_datasets.return_value = [future_mock]  # some arbitrary return value
            # force order of datasets by specifying them
            mscn.save_datasets(base_dir=self.base_dir, client=client_mock, datasets=['ds1', 'ds2', 'ds3'],
                               writer='simple_image')

        # 2 for each scene
        self.assertEqual(save_datasets.call_count, 2)

    def test_crop(self):
        """Test the crop method."""
        from satpy import Scene, MultiScene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        import numpy as np
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


class TestBlendFuncs(unittest.TestCase):
    """Test individual functions used for blending."""

    def setUp(self):
        """Set up test data."""
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                              2, 2, [-200, -200, 200, 200])
        ds1 = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                           attrs={'start_time': datetime(2018, 1, 1, 0, 0, 0), 'area': area})
        self.ds1 = ds1
        ds2 = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                           attrs={'start_time': datetime(2018, 1, 1, 1, 0, 0), 'area': area})
        self.ds2 = ds2
        ds3 = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'time'),
                           attrs={'start_time': datetime(2018, 1, 1, 0, 0, 0), 'area': area})
        self.ds3 = ds3
        ds4 = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'time'),
                           attrs={'start_time': datetime(2018, 1, 1, 1, 0, 0), 'area': area})
        self.ds4 = ds4

    def test_stack(self):
        """Test the 'stack' function."""
        from satpy.multiscene import stack
        res = stack([self.ds1, self.ds2])
        self.assertTupleEqual(self.ds1.shape, res.shape)

    def test_timeseries(self):
        """Test the 'timeseries' function."""
        from satpy.multiscene import timeseries
        import xarray as xr
        res = timeseries([self.ds1, self.ds2])
        res2 = timeseries([self.ds3, self.ds4])
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res2, xr.DataArray)
        self.assertTupleEqual((2, self.ds1.shape[0], self.ds1.shape[1]), res.shape)
        self.assertTupleEqual((self.ds3.shape[0], self.ds3.shape[1]+self.ds4.shape[1]), res2.shape)
