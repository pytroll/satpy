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

"""Unit tests for the Multiscene object."""

import unittest
from unittest import mock

import pytest
import xarray as xr

from satpy import DataQuery
from satpy.tests.multiscene_tests.test_utils import _create_test_area, _create_test_dataset, _create_test_scenes
from satpy.tests.utils import make_dataid


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
