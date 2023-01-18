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
import shutil
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import pytest
import xarray as xr

from satpy import DataQuery
from satpy.dataset.dataid import DataID, ModifierTuple, WavelengthRange

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path

DEFAULT_SHAPE = (5, 10)

local_id_keys_config = {'name': {
    'required': True,
},
    'wavelength': {
    'type': WavelengthRange,
},
    'resolution': None,
    'calibration': {
    'enum': [
        'reflectance',
        'brightness_temperature',
        'radiance',
        'counts'
    ]
},
    'polarization': None,
    'level': None,
    'modifiers': {
    'required': True,
    'default': ModifierTuple(),
    'type': ModifierTuple,
},
}


def make_dataid(**items):
    """Make a data id."""
    return DataID(local_id_keys_config, **items)


def _fake_get_enhanced_image(img, enhance=None, overlay=None, decorate=None):
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
    import dask.array as da
    import numpy as np
    import xarray as xr

    return xr.DataArray(
        da.zeros(shape, dtype=np.float32, chunks=shape), dims=('y', 'x'),
        attrs={'name': name, 'area': area, '_satpy_id_keys': local_id_keys_config})


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
        from satpy import MultiScene

        area = _create_test_area()
        scenes = _create_test_scenes(area=area)
        ds1_id = make_dataid(name='ds1')
        ds2_id = make_dataid(name='ds2')
        ds3_id = make_dataid(name='ds3')
        ds4_id = make_dataid(name='ds4')

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
        input_files_abi = [
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171502203_e20171171504576_c20171171505018.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171507203_e20171171509576_c20171171510018.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171512203_e20171171514576_c20171171515017.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171517203_e20171171519577_c20171171520019.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171522203_e20171171524576_c20171171525020.nc",
            "OR_ABI-L1b-RadC-M3C01_G16_s20171171527203_e20171171529576_c20171171530017.nc",
            ]
        input_files_glm = [
            "OR_GLM-L2-GLMC-M3_G16_s20171171500000_e20171171501000_c20380190314080.nc",
            "OR_GLM-L2-GLMC-M3_G16_s20171171501000_e20171171502000_c20380190314080.nc",
            "OR_GLM-L2-GLMC-M3_G16_s20171171502000_e20171171503000_c20380190314080.nc",
            "OR_GLM-L2-GLMC-M3_G16_s20171171503000_e20171171504000_c20380190314080.nc",
            "OR_GLM-L2-GLMC-M3_G16_s20171171504000_e20171171505000_c20380190314080.nc",
            "OR_GLM-L2-GLMC-M3_G16_s20171171505000_e20171171506000_c20380190314080.nc",
            "OR_GLM-L2-GLMC-M3_G16_s20171171506000_e20171171507000_c20380190314080.nc",
            "OR_GLM-L2-GLMC-M3_G16_s20171171507000_e20171171508000_c20380190314080.nc",
        ]
        with mock.patch('satpy.multiscene.Scene') as scn_mock:
            mscn = MultiScene.from_files(
                    input_files_abi,
                    reader='abi_l1b',
                    scene_kwargs={"reader_kwargs": {}})
            assert len(mscn.scenes) == 6
            calls = [mock.call(
                filenames={'abi_l1b': [in_file_abi]},
                reader_kwargs={})
                for in_file_abi in input_files_abi]
            scn_mock.assert_has_calls(calls)

            scn_mock.reset_mock()
            with pytest.warns(DeprecationWarning):
                mscn = MultiScene.from_files(
                        input_files_abi + input_files_glm,
                        reader=('abi_l1b', "glm_l2"),
                        group_keys=["start_time"],
                        ensure_all_readers=True,
                        time_threshold=30)
            assert len(mscn.scenes) == 2
            calls = [mock.call(
                filenames={'abi_l1b': [in_file_abi], 'glm_l2': [in_file_glm]})
                for (in_file_abi, in_file_glm) in
                zip(input_files_abi[0:2],
                    [input_files_glm[2]] + [input_files_glm[7]])]
            scn_mock.assert_has_calls(calls)
            scn_mock.reset_mock()
            mscn = MultiScene.from_files(
                    input_files_abi + input_files_glm,
                    reader=('abi_l1b', "glm_l2"),
                    group_keys=["start_time"],
                    ensure_all_readers=False,
                    time_threshold=30)
            assert len(mscn.scenes) == 12


class TestMultiSceneGrouping:
    """Test dataset grouping in MultiScene."""

    @pytest.fixture
    def scene1(self):
        """Create first test scene."""
        from satpy import Scene
        scene = Scene()
        dsid1 = make_dataid(
            name="ds1",
            resolution=123,
            wavelength=(1, 2, 3),
            polarization="H"
        )
        scene[dsid1] = _create_test_dataset(name='ds1')
        dsid2 = make_dataid(
            name="ds2",
            resolution=456,
            wavelength=(4, 5, 6),
            polarization="V"
        )
        scene[dsid2] = _create_test_dataset(name='ds2')
        return scene

    @pytest.fixture
    def scene2(self):
        """Create second test scene."""
        from satpy import Scene
        scene = Scene()
        dsid1 = make_dataid(
            name="ds3",
            resolution=123.1,
            wavelength=(1.1, 2.1, 3.1),
            polarization="H"
        )
        scene[dsid1] = _create_test_dataset(name='ds3')
        dsid2 = make_dataid(
            name="ds4",
            resolution=456.1,
            wavelength=(4.1, 5.1, 6.1),
            polarization="V"
        )
        scene[dsid2] = _create_test_dataset(name='ds4')
        return scene

    @pytest.fixture
    def multi_scene(self, scene1, scene2):
        """Create small multi scene for testing."""
        from satpy import MultiScene
        return MultiScene([scene1, scene2])

    @pytest.fixture
    def groups(self):
        """Get group definitions for the MultiScene."""
        return {
            DataQuery(name='odd'): ['ds1', 'ds3'],
            DataQuery(name='even'): ['ds2', 'ds4']
        }

    def test_multi_scene_grouping(self, multi_scene, groups, scene1):
        """Test grouping a MultiScene."""
        multi_scene.group(groups)
        shared_ids_exp = {make_dataid(name="odd"), make_dataid(name="even")}
        assert multi_scene.shared_dataset_ids == shared_ids_exp
        assert DataQuery(name='odd') not in scene1
        xr.testing.assert_allclose(multi_scene.scenes[0]["ds1"], scene1["ds1"])

    def test_fails_to_add_multiple_datasets_from_the_same_scene_to_a_group(self, multi_scene):
        """Test that multiple datasets from the same scene in one group fails."""
        groups = {DataQuery(name='mygroup'): ['ds1', 'ds2']}
        multi_scene.group(groups)
        with pytest.raises(ValueError):
            next(multi_scene.scenes)


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


class TestBlendFuncs(unittest.TestCase):
    """Test individual functions used for blending."""

    def setUp(self):
        """Set up test data."""
        from datetime import datetime

        import dask.array as da
        import xarray as xr
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
        import xarray as xr

        from satpy.multiscene import timeseries
        res = timeseries([self.ds1, self.ds2])
        res2 = timeseries([self.ds3, self.ds4])
        self.assertIsInstance(res, xr.DataArray)
        self.assertIsInstance(res2, xr.DataArray)
        self.assertTupleEqual((2, self.ds1.shape[0], self.ds1.shape[1]), res.shape)
        self.assertTupleEqual((self.ds3.shape[0], self.ds3.shape[1]+self.ds4.shape[1]), res2.shape)


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
