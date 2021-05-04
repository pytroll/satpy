#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010-2019 Satpy developers
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
"""Unit tests for scene.py."""

import os
import unittest
from unittest import mock
import string
import random

import satpy
from satpy import Scene
from satpy.tests.utils import (default_id_keys_config, make_cid, make_dataid,
                               make_dsq, spy_decorator,
                               FAKE_FILEHANDLER_START, FAKE_FILEHANDLER_END)

import numpy as np
import xarray as xr
import pytest


TEST_ETC_DIR = os.path.join(os.path.dirname(__file__), 'etc')


def _check_comp19_deps_are_loaded(scene):
    # comp19 required resampling to produce so we should have its 3 deps
    # 1. comp13
    # 2. ds5
    # 3. ds2
    loaded_ids = list(scene.keys())
    assert len(loaded_ids) == 3
    for name in ('comp13', 'ds5', 'ds2'):
        assert any(x['name'] == name for x in loaded_ids)


class TestScene:
    """Test the scene class."""

    def setup_method(self):
        """Set config_path to point to test 'etc' directory."""
        self.old_config_path = satpy.config.get('config_path')
        satpy.config.set(config_path=[TEST_ETC_DIR])

    def teardown_method(self):
        """Restore previous 'config_path' setting."""
        satpy.config.set(config_path=self.old_config_path)

    def test_init(self):
        """Test scene initialization."""
        with mock.patch('satpy.scene.Scene._create_reader_instances') as cri:
            cri.return_value = {}
            Scene(filenames=['bla'], reader='blo')
            cri.assert_called_once_with(filenames=['bla'], reader='blo',
                                        reader_kwargs=None)

    def test_init_str_filename(self):
        """Test initializing with a single string as filenames."""
        pytest.raises(ValueError, Scene, reader='blo', filenames='test.nc')

    def test_start_end_times(self):
        """Test start and end times for a scene."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        assert scene.start_time == FAKE_FILEHANDLER_START
        assert scene.end_time == FAKE_FILEHANDLER_END

    def test_init_preserve_reader_kwargs(self):
        """Test that the initialization preserves the kwargs."""
        cri = spy_decorator(Scene._create_reader_instances)
        with mock.patch('satpy.scene.Scene._create_reader_instances', cri):
            reader_kwargs = {'calibration_type': 'gsics'}
            scene = Scene(filenames=['fake1_1.txt'],
                          reader='fake1',
                          filter_parameters={'area': 'euron1'},
                          reader_kwargs=reader_kwargs)
            assert reader_kwargs is not cri.mock.call_args[1]['reader_kwargs']
            assert scene.start_time == FAKE_FILEHANDLER_START
            assert scene.end_time == FAKE_FILEHANDLER_END

    def test_init_alone(self):
        """Test simple initialization."""
        from satpy.scene import Scene
        scn = Scene()
        assert not scn._readers, 'Empty scene should not load any readers'

    def test_init_no_files(self):
        """Test that providing an empty list of filenames fails."""
        from satpy.scene import Scene
        pytest.raises(ValueError, Scene, reader='viirs_sdr', filenames=[])

    def test_create_reader_instances_with_filenames(self):
        """Test creating a reader providing filenames."""
        filenames = ["bla", "foo", "bar"]
        reader_name = None
        with mock.patch('satpy.scene.Scene._compute_metadata_from_readers') as md:
            md.return_value = {'sensor': {'sensor'}}
            with mock.patch('satpy.scene.load_readers') as findermock:
                Scene(filenames=filenames)
                findermock.assert_called_once_with(
                    filenames=filenames,
                    reader=reader_name,
                    reader_kwargs=None,
                )

    def test_init_with_empty_filenames(self):
        """Test initialization with empty filename list."""
        from satpy.scene import Scene
        filenames = []
        Scene(filenames=filenames)

    def test_init_with_fsfile(self):
        """Test initialisation with FSFile objects."""
        from satpy.readers import FSFile

        # We should not mock _create_reader_instances here, because in
        # https://github.com/pytroll/satpy/issues/1605 satpy fails with
        # TypeError within that method if passed an FSFile instance.
        # Instead rely on the ValueError that satpy raises if no readers
        # are found.

        # Choose random filename that doesn't exist.  Not using tempfile here,
        # because tempfile creates files and we don't want that here.
        fsf = FSFile("".join(random.choices(string.printable, k=50)))
        with pytest.raises(ValueError, match="No supported files found"):
            Scene(filenames=[fsf], reader=[])

    # TODO: Rewrite this test for the 'find_files_and_readers' function
    # def test_create_reader_instances_with_sensor(self):
    #     import satpy.scene
    #     sensors = ["bla", "foo", "bar"]
    #     filenames = None
    #     reader_name = None
    #     with mock.patch('satpy.scene.Scene._compute_metadata_from_readers'):
    #         with mock.patch('satpy.scene.load_readers') as findermock:
    #             scene = satpy.scene.Scene(sensor=sensors)
    #             findermock.assert_called_once_with(
    #                 ppp_config_dir=mock.ANY,
    #                 reader=reader_name,
    #                 filenames=filenames,
    #                 reader_kwargs=None,
    #             )

    # def test_create_reader_instances_with_sensor_and_filenames(self):
    #     import satpy.scene
    #     sensors = ["bla", "foo", "bar"]
    #     filenames = ["1", "2", "3"]
    #     reader_name = None
    #     with mock.patch('satpy.scene.Scene._compute_metadata_from_readers'):
    #         with mock.patch('satpy.scene.load_readers') as findermock:
    #             scene = satpy.scene.Scene(sensor=sensors, filenames=filenames)
    #             findermock.assert_called_once_with(
    #                 ppp_config_dir=mock.ANY,
    #                 reader=reader_name,
    #                 sensor=sensors,
    #                 filenames=filenames,
    #                 reader_kwargs=None,
    #             )

    def test_create_reader_instances_with_reader(self):
        """Test createring a reader instance providing the reader name."""
        from satpy.scene import Scene
        reader = "foo"
        filenames = ["1", "2", "3"]
        with mock.patch('satpy.scene.load_readers') as findermock:
            findermock.return_value = {}
            Scene(reader=reader, filenames=filenames)
            findermock.assert_called_once_with(reader=reader,
                                               filenames=filenames,
                                               reader_kwargs=None,
                                               )

    def test_create_reader_instances_with_reader_kwargs(self):
        """Test creating a reader instance with reader kwargs."""
        from satpy.readers.yaml_reader import FileYAMLReader
        reader_kwargs = {'calibration_type': 'gsics'}
        filter_parameters = {'area': 'euron1'}
        reader_kwargs2 = {'calibration_type': 'gsics', 'filter_parameters': filter_parameters}

        rinit = spy_decorator(FileYAMLReader.create_filehandlers)
        with mock.patch('satpy.readers.yaml_reader.FileYAMLReader.create_filehandlers', rinit):
            scene = Scene(filenames=['fake1_1.txt'],
                          reader='fake1',
                          filter_parameters={'area': 'euron1'},
                          reader_kwargs=reader_kwargs)
            del scene
            assert reader_kwargs == rinit.mock.call_args[1]['fh_kwargs']
            rinit.mock.reset_mock()
            scene = Scene(filenames=['fake1_1.txt'],
                          reader='fake1',
                          reader_kwargs=reader_kwargs2)
            assert reader_kwargs == rinit.mock.call_args[1]['fh_kwargs']
            del scene

    def test_create_multiple_reader_different_kwargs(self):
        """Test passing different kwargs to different readers."""
        from satpy.readers import load_reader
        with satpy.config.set(config_path=[TEST_ETC_DIR]), \
             mock.patch.object(satpy.readers, 'load_reader', wraps=load_reader) as lr:
            Scene(filenames={"fake1_1ds": ["fake1_1ds_1.txt"],
                             "fake2_1ds": ["fake2_1ds_1.txt"]},
                  reader_kwargs={
                      "fake1_1ds": {"mouth": "omegna"},
                      "fake2_1ds": {"mouth": "varallo"}
                  })
            lr.assert_has_calls([
                    mock.call([os.path.join(TEST_ETC_DIR, 'readers', 'fake1_1ds.yaml')], mouth="omegna"),
                    mock.call([os.path.join(TEST_ETC_DIR, 'readers', 'fake2_1ds.yaml')], mouth="varallo")])

    def test_iter(self):
        """Test iteration over the scene."""
        from satpy import Scene
        from xarray import DataArray
        scene = Scene()
        scene["1"] = DataArray(np.arange(5))
        scene["2"] = DataArray(np.arange(5))
        scene["3"] = DataArray(np.arange(5))
        for x in scene:
            assert isinstance(x, DataArray)

    def test_iter_by_area_swath(self):
        """Test iterating by area on a swath."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import SwathDefinition
        scene = Scene()
        sd = SwathDefinition(lons=np.arange(5), lats=np.arange(5))
        scene["1"] = DataArray(np.arange(5), attrs={'area': sd})
        scene["2"] = DataArray(np.arange(5), attrs={'area': sd})
        scene["3"] = DataArray(np.arange(5))
        for area_obj, ds_list in scene.iter_by_area():
            ds_list_names = set(ds['name'] for ds in ds_list)
            if area_obj is sd:
                assert ds_list_names == {'1', '2'}
            else:
                assert area_obj is None
                assert ds_list_names == {'3'}

    def test_bad_setitem(self):
        """Test setting an item wrongly."""
        from satpy import Scene
        scene = Scene()
        pytest.raises(ValueError, scene.__setitem__, '1', np.arange(5))

    def test_setitem(self):
        """Test setting an item."""
        from satpy import Scene
        from satpy.tests.utils import make_dataid
        import xarray as xr
        scene = Scene()
        scene["1"] = ds1 = xr.DataArray(np.arange(5))
        expected_id = make_cid(**ds1.attrs)
        assert set(scene._datasets.keys()) == {expected_id}
        assert set(scene._wishlist) == {expected_id}

        did = make_dataid(name='oranges')
        scene[did] = ds1
        assert 'oranges' in scene
        nparray = np.arange(5*5).reshape(5, 5)
        with pytest.raises(ValueError):
            scene['apples'] = nparray
        assert 'apples' not in scene
        did = make_dataid(name='apples')
        scene[did] = nparray
        assert 'apples' in scene

    def test_getitem(self):
        """Test __getitem__ with names only."""
        from satpy import Scene
        from xarray import DataArray
        scene = Scene()
        scene["1"] = ds1 = DataArray(np.arange(5))
        scene["2"] = ds2 = DataArray(np.arange(5))
        scene["3"] = ds3 = DataArray(np.arange(5))
        assert scene['1'] is ds1
        assert scene['2'] is ds2
        assert scene['3'] is ds3
        pytest.raises(KeyError, scene.__getitem__, '4')
        assert scene.get('3') is ds3
        assert scene.get('4') is None

    def test_getitem_modifiers(self):
        """Test __getitem__ with names and modifiers."""
        from satpy import Scene
        from xarray import DataArray

        # Return least modified item
        scene = Scene()
        scene['1'] = ds1_m0 = DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1',))
              ] = ds1_m1 = DataArray(np.arange(5))
        assert scene['1'] is ds1_m0
        assert len(list(scene.keys())) == 2

        scene = Scene()
        scene['1'] = ds1_m0 = DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1',))
              ] = ds1_m1 = DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1', 'mod2'))
              ] = ds1_m2 = DataArray(np.arange(5))
        assert scene['1'] is ds1_m0
        assert len(list(scene.keys())) == 3

        scene = Scene()
        scene[make_dataid(name='1', modifiers=('mod1', 'mod2'))
              ] = ds1_m2 = DataArray(np.arange(5))
        scene[make_dataid(name='1', modifiers=('mod1',))
              ] = ds1_m1 = DataArray(np.arange(5))
        assert scene['1'] is ds1_m1
        assert scene[make_dataid(name='1', modifiers=('mod1', 'mod2'))] is ds1_m2
        pytest.raises(KeyError, scene.__getitem__,
                      make_dataid(name='1', modifiers=tuple()))
        assert len(list(scene.keys())) == 2

    def test_getitem_slices(self):
        """Test __getitem__ with slices."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition, SwathDefinition
        from pyresample.utils import proj4_str_to_dict
        scene1 = Scene()
        scene2 = Scene()
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            200,
            400,
            (-1000., -1500., 1000., 1500.),
        )
        swath_def = SwathDefinition(lons=np.zeros((5, 10)),
                                    lats=np.zeros((5, 10)))
        scene1["1"] = scene2["1"] = DataArray(np.zeros((5, 10)))
        scene1["2"] = scene2["2"] = DataArray(np.zeros((5, 10)),
                                              dims=('y', 'x'))
        scene1["3"] = DataArray(np.zeros((5, 10)), dims=('y', 'x'),
                                attrs={'area': area_def})
        anc_vars = [DataArray(np.ones((5, 10)), attrs={'name': 'anc_var',
                                                       'area': area_def})]
        attrs = {'ancillary_variables': anc_vars, 'area': area_def}
        scene1["3a"] = DataArray(np.zeros((5, 10)),
                                 dims=('y', 'x'),
                                 attrs=attrs)
        scene2["4"] = DataArray(np.zeros((5, 10)), dims=('y', 'x'),
                                attrs={'area': swath_def})
        anc_vars = [DataArray(np.ones((5, 10)), attrs={'name': 'anc_var',
                                                       'area': swath_def})]
        attrs = {'ancillary_variables': anc_vars, 'area': swath_def}
        scene2["4a"] = DataArray(np.zeros((5, 10)),
                                 dims=('y', 'x'),
                                 attrs=attrs)
        new_scn1 = scene1[2:5, 2:8]
        new_scn2 = scene2[2:5, 2:8]
        for new_scn in [new_scn1, new_scn2]:
            # datasets without an area don't get sliced
            assert new_scn['1'].shape == (5, 10)
            assert new_scn['2'].shape == (5, 10)

        assert new_scn1['3'].shape == (3, 6)
        assert 'area' in new_scn1['3'].attrs
        assert new_scn1['3'].attrs['area'].shape == (3, 6)
        assert new_scn1['3a'].shape == (3, 6)
        a_var = new_scn1['3a'].attrs['ancillary_variables'][0]
        assert a_var.shape == (3, 6)

        assert new_scn2['4'].shape == (3, 6)
        assert 'area' in new_scn2['4'].attrs
        assert new_scn2['4'].attrs['area'].shape == (3, 6)
        assert new_scn2['4a'].shape == (3, 6)
        a_var = new_scn2['4a'].attrs['ancillary_variables'][0]
        assert a_var.shape == (3, 6)

    def test_crop(self):
        """Test the crop method."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        scene1 = Scene()
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927,
                       5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        x_size = 3712
        y_size = 3712
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )
        area_def2 = AreaDefinition(
            'test2',
            'test2',
            'test2',
            proj_dict,
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

        # by area
        crop_area = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            (area_extent[0] + 10000., area_extent[1] + 500000.,
             area_extent[2] - 10000., area_extent[3] - 500000.)
        )
        new_scn1 = scene1.crop(crop_area)
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert '3' in new_scn1
        assert new_scn1['1'].shape == (y_size, x_size)
        assert new_scn1['2'].shape == (y_size, x_size)
        assert new_scn1['3'].shape == (3380, 3708)
        assert new_scn1['4'].shape == (1690, 1854)

        # by lon/lat bbox
        new_scn1 = scene1.crop(ll_bbox=(-20., -5., 0, 0))
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert '3' in new_scn1
        assert new_scn1['1'].shape == (y_size, x_size)
        assert new_scn1['2'].shape == (y_size, x_size)
        assert new_scn1['3'].shape == (184, 714)
        assert new_scn1['4'].shape == (92, 357)

        # by x/y bbox
        new_scn1 = scene1.crop(xy_bbox=(-200000., -100000., 0, 0))
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert '3' in new_scn1
        assert new_scn1['1'].shape == (y_size, x_size)
        assert new_scn1['2'].shape == (y_size, x_size)
        assert new_scn1['3'].shape == (36, 70)
        assert new_scn1['4'].shape == (18, 35)

    def test_crop_epsg_crs(self):
        """Test the crop method when source area uses an EPSG code."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition

        scene1 = Scene()
        area_extent = (699960.0, 5390220.0, 809760.0, 5500020.0)
        x_size = 3712
        y_size = 3712
        area_def = AreaDefinition(
            'test', 'test', 'test',
            "EPSG:32630",
            x_size,
            y_size,
            area_extent,
        )
        scene1["1"] = DataArray(np.zeros((y_size, x_size)), dims=('y', 'x'),
                                attrs={'area': area_def})
        # by x/y bbox
        new_scn1 = scene1.crop(xy_bbox=(719695.7781587119, 5427887.407618969, 725068.1609052602, 5433708.364368956))
        assert '1' in new_scn1
        assert new_scn1['1'].shape == (198, 182)

    def test_crop_rgb(self):
        """Test the crop method on multi-dimensional data."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        scene1 = Scene()
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927,
                       5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        x_size = 3712
        y_size = 3712
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )
        area_def2 = AreaDefinition(
            'test2',
            'test2',
            'test2',
            proj_dict,
            x_size // 2,
            y_size // 2,
            area_extent,
            )
        scene1["1"] = DataArray(np.zeros((3, y_size, x_size)), dims=('bands', 'y', 'x'), attrs={'area': area_def})
        scene1["2"] = DataArray(np.zeros((y_size // 2, 3, x_size // 2)), dims=('y', 'bands', 'x'),
                                attrs={'area': area_def2})

        # by lon/lat bbox
        new_scn1 = scene1.crop(ll_bbox=(-20., -5., 0, 0))
        assert '1' in new_scn1
        assert '2' in new_scn1
        assert 'bands' in new_scn1['1'].dims
        assert 'bands' in new_scn1['2'].dims
        assert new_scn1['1'].shape == (3, 184, 714)
        assert new_scn1['2'].shape == (92, 3, 357)

    def test_contains(self):
        """Test contains."""
        from satpy import Scene
        from xarray import DataArray
        scene = Scene()
        scene["1"] = DataArray(np.arange(5), attrs={'wavelength': (0.1, 0.2, 0.3),
                                                    '_satpy_id_keys': default_id_keys_config})
        assert '1' in scene
        assert 0.15 in scene
        assert '2' not in scene
        assert 0.31 not in scene

        scene = Scene()
        scene['blueberry'] = DataArray(np.arange(5))
        scene['blackberry'] = DataArray(np.arange(5))
        scene['strawberry'] = DataArray(np.arange(5))
        scene['raspberry'] = DataArray(np.arange(5))
        #  deepcode ignore replace~keys~list~compare: This is on purpose
        assert make_cid(name='blueberry') in scene.keys()
        assert make_cid(name='blueberry') in scene
        assert 'blueberry' in scene
        assert 'blueberry' not in scene.keys()

    def test_delitem(self):
        """Test deleting an item."""
        from satpy import Scene
        from xarray import DataArray
        scene = Scene()
        scene["1"] = DataArray(np.arange(5), attrs={'wavelength': (0.1, 0.2, 0.3),
                                                    '_satpy_id_keys': default_id_keys_config})
        scene["2"] = DataArray(np.arange(5), attrs={'wavelength': (0.4, 0.5, 0.6),
                                                    '_satpy_id_keys': default_id_keys_config})
        scene["3"] = DataArray(np.arange(5), attrs={'wavelength': (0.7, 0.8, 0.9),
                                                    '_satpy_id_keys': default_id_keys_config})
        del scene['1']
        del scene['3']
        del scene[0.45]
        assert not scene._wishlist
        assert not list(scene._datasets.keys())
        pytest.raises(KeyError, scene.__delitem__, 0.2)

    def test_all_datasets_no_readers(self):
        """Test all datasets with no reader."""
        from satpy import Scene
        scene = Scene()
        pytest.raises(KeyError, scene.all_dataset_ids, reader_name='fake')
        id_list = scene.all_dataset_ids()
        assert id_list == []
        # no sensors are loaded so we shouldn't get any comps either
        id_list = scene.all_dataset_ids(composites=True)
        assert id_list == []

    def test_all_dataset_names_no_readers(self):
        """Test all dataset names with no reader."""
        from satpy import Scene
        scene = Scene()
        pytest.raises(KeyError, scene.all_dataset_names, reader_name='fake')
        name_list = scene.all_dataset_names()
        assert name_list == []
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.all_dataset_names(composites=True)
        assert name_list == []

    def test_available_dataset_no_readers(self):
        """Test the available datasets without a reader."""
        from satpy import Scene
        scene = Scene()
        pytest.raises(
            KeyError, scene.available_dataset_ids, reader_name='fake')
        name_list = scene.available_dataset_ids()
        assert name_list == []
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_ids(composites=True)
        assert name_list == []

    def test_available_dataset_names_no_readers(self):
        """Test the available dataset names without a reader."""
        from satpy import Scene
        scene = Scene()
        pytest.raises(
            KeyError, scene.available_dataset_names, reader_name='fake')
        name_list = scene.available_dataset_names()
        assert name_list == []
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_names(composites=True)
        assert name_list == []


class TestFinestCoarsestArea:
    """Test the Scene logic for finding the finest and coarsest area."""

    def setup_method(self):
        """Set common variables."""
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        self.scene = Scene()
        self.scene["1"] = DataArray(np.arange(10).reshape((2, 5)),
                                    attrs={'wavelength': (0.1, 0.2, 0.3)})
        self.ds1 = self.scene["1"]

        self.scene["2"] = DataArray(np.arange(40).reshape((4, 10)),
                                    attrs={'wavelength': (0.4, 0.5, 0.6)})
        self.ds2 = self.scene["2"]

        self.scene["3"] = DataArray(np.arange(40).reshape((4, 10)),
                                    attrs={'wavelength': (0.7, 0.8, 0.9)})
        self.ds3 = self.scene["3"]

        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        self.area_def1 = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        self.area_def2 = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            200,
            400,
            (-1000., -1500., 1000., 1500.),
        )

    def test_coarsest_finest_area_upright_area(self):
        """Test 'coarsest_area' and 'finest_area' methods for upright areas."""
        self.ds1.attrs['area'] = self.area_def1
        self.ds2.attrs['area'] = self.area_def2
        self.ds3.attrs['area'] = self.area_def2
        assert self.scene.coarsest_area() is self.area_def1
        assert self.scene.finest_area() is self.area_def2
        assert self.scene.coarsest_area(['2', '3']) is self.area_def2

    def test_coarsest_finest_area_flipped_area(self):
        """Test 'coarsest_area' and 'finest_area' methods for flipped areas with negative pixel sizes."""
        area_def1_flipped = self.area_def1.copy(area_extent=tuple([-1*ae for ae in self.area_def1.area_extent]))
        area_def2_flipped = self.area_def2.copy(area_extent=tuple([-1*ae for ae in self.area_def2.area_extent]))
        self.ds1.attrs['area'] = area_def1_flipped
        self.ds2.attrs['area'] = area_def2_flipped
        self.ds3.attrs['area'] = area_def2_flipped
        assert self.scene.coarsest_area() is area_def1_flipped
        assert self.scene.finest_area() is area_def2_flipped
        assert self.scene.coarsest_area(['2', '3']) is area_def2_flipped


class TestSceneAvailableDatasets:
    """Test the Scene's handling of various dependencies."""

    def setup_method(self):
        """Set config_path to point to test 'etc' directory."""
        self.old_config_path = satpy.config.get('config_path')
        satpy.config.set(config_path=[TEST_ETC_DIR])

    def teardown_method(self):
        """Restore previous 'config_path' setting."""
        satpy.config.set(config_path=self.old_config_path)

    def test_all_datasets_one_reader(self):
        """Test all datasets for one reader."""
        scene = Scene(filenames=['fake1_1.txt'],
                      reader='fake1')
        id_list = scene.all_dataset_ids()
        # 20 data products + 6 lon/lat products
        num_reader_ds = 21 + 6
        assert len(id_list) == num_reader_ds
        id_list = scene.all_dataset_ids(composites=True)
        assert len(id_list) == num_reader_ds + 29

    def test_all_datasets_multiple_reader(self):
        """Test all datasets for multiple readers."""
        scene = Scene(filenames={'fake1_1ds': ['fake1_1ds_1.txt'],
                                 'fake2_1ds': ['fake2_1ds_1.txt']})
        id_list = scene.all_dataset_ids()
        assert len(id_list) == 2
        id_list = scene.all_dataset_ids(composites=True)
        # ds1 and ds2 => 2
        # composites that use these two datasets => 11
        assert len(id_list) == 2 + 11

    def test_available_datasets_one_reader(self):
        """Test the available datasets for one reader."""
        scene = Scene(filenames=['fake1_1ds_1.txt'],
                      reader='fake1_1ds')
        id_list = scene.available_dataset_ids()
        assert len(id_list) == 1
        id_list = scene.available_dataset_ids(composites=True)
        # ds1, comp1, comp14, comp16, static_image, comp26
        assert len(id_list) == 6

    def test_available_composite_ids_missing_available(self):
        """Test available_composite_ids when a composites dep is missing."""
        scene = Scene(filenames=['fake1_1ds_1.txt'],
                      reader='fake1_1ds')
        assert 'comp2' not in scene.available_composite_names()


class TestSceneLoading:
    """Test the Scene objects `.load` method."""

    def setup_method(self):
        """Set config_path to point to test 'etc' directory."""
        self.old_config_path = satpy.config.get('config_path')
        satpy.config.set(config_path=[TEST_ETC_DIR])

    def teardown_method(self):
        """Restore previous 'config_path' setting."""
        satpy.config.set(config_path=self.old_config_path)

    def test_load_str(self):
        """Test passing a string to Scene.load."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(TypeError, scene.load, 'ds1')

    def test_load_no_exist(self):
        """Test loading a dataset that doesn't exist."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(KeyError, scene.load, ['im_a_dataset_that_doesnt_exist'])

    def test_load_no_exist2(self):
        """Test loading a dataset that doesn't exist then another load."""
        from satpy.readers.yaml_reader import FileYAMLReader
        load_mock = spy_decorator(FileYAMLReader.load)
        with mock.patch.object(FileYAMLReader, 'load', load_mock):
            lmock = load_mock.mock
            scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
            scene.load(['ds9_fail_load'])
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 0
            lmock.assert_called_once_with(
                set([make_dataid(name='ds9_fail_load', wavelength=(1.0, 1.1, 1.2))]))

            scene.load(['ds1'])
            loaded_ids = list(scene._datasets.keys())
            assert lmock.call_count == 2
            # most recent call should have only been ds1
            lmock.assert_called_with(set([
                make_dataid(name='ds1', resolution=250,
                            calibration='reflectance', modifiers=tuple())]))
            assert len(loaded_ids) == 1

    def test_load_ds1_no_comps(self):
        """Test loading one dataset with no loaded compositors."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_dataid(name='ds1', resolution=250, calibration='reflectance', modifiers=tuple())

    def test_load_ds1_load_twice(self):
        """Test loading one dataset with no loaded compositors."""
        from satpy.readers.yaml_reader import FileYAMLReader
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds1'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_dataid(name='ds1', resolution=250, calibration='reflectance', modifiers=tuple())

        load_mock = spy_decorator(FileYAMLReader.load)
        with mock.patch.object(FileYAMLReader, 'load', load_mock):
            lmock = load_mock.mock
            scene.load(['ds1'])
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 1
            assert loaded_ids[0] == make_dataid(name='ds1',
                                                resolution=250,
                                                calibration='reflectance',
                                                modifiers=tuple())
            assert not lmock.called, ("Reader.load was called again when "
                                      "loading something that's already "
                                      "loaded")

    def test_load_ds1_unknown_modifier(self):
        """Test loading one dataset with no loaded compositors."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(KeyError, scene.load,
                      [make_dataid(name='ds1', modifiers=('_fake_bad_mod_',))])

    def test_load_ds4_cal(self):
        """Test loading a dataset that has two calibration variations."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds4'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['calibration'] == 'reflectance'

    def test_load_ds5_best_resolution(self):
        """Test loading a dataset has multiple resolutions available."""
        scene = Scene(filenames=['fake1_1.txt', 'fake1_highres_1.txt'], reader='fake1')
        scene.load(['ds5'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'ds5'
        assert loaded_ids[0]['resolution'] == 250

    def test_load_ds5_multiple_resolution(self):
        """Test loading a dataset has multiple resolutions available with different resolutions."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds5'], resolution=1000)
        scene.load(['ds5'], resolution=500)
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0]['name'] == 'ds5'
        assert loaded_ids[0]['resolution'] == 500
        assert loaded_ids[1]['name'] == 'ds5'
        assert loaded_ids[1]['resolution'] == 1000

    def test_load_ds5_resolution_list(self):
        """Test loading a dataset has multiple resolutions available with different resolutions."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds5'], resolution=[500, 1000])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'ds5'
        assert loaded_ids[0]['resolution'] == 500

    def test_load_ds5_empty_modifiers(self):
        """Test loading a dataset has multiple resolutions available with different resolutions."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([make_dsq(name='ds5', modifiers=tuple())])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'ds5'

    def test_load_ds5_missing_best_resolution(self):
        """Test loading a dataset that has multiple resolutions but the best isn't available."""
        # only the 500m is available
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds5'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'ds5'
        assert loaded_ids[0]['resolution'] == 500

    def test_load_ds6_wl(self):
        """Test loading a dataset by wavelength."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([0.22])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'ds6'

    def test_load_ds9_fail_load(self):
        """Test loading a dataset that will fail during load."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ds9_fail_load'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 0

    def test_load_comp1(self):
        """Test loading a composite with one required prereq."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp1'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp1')

    def test_load_comp4(self):
        """Test loading a composite that depends on a composite."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp4'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp4')

    def test_load_multiple_resolutions(self):
        """Test loading a dataset has multiple resolutions available with different resolutions."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        comp25 = make_cid(name='comp25', resolution=1000)
        scene[comp25] = 'bla'
        scene.load(['comp25'], resolution=500)

        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0]['name'] == 'comp25'
        assert loaded_ids[0]['resolution'] == 500
        assert loaded_ids[1]['name'] == 'comp25'
        assert loaded_ids[1]['resolution'] == 1000

    def test_load_same_subcomposite(self):
        """Test loading a composite and one of it's subcomposites at the same time."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp24', 'comp25'], resolution=500)
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0]['name'] == 'comp24'
        assert loaded_ids[0]['resolution'] == 500
        assert loaded_ids[1]['name'] == 'comp25'
        assert loaded_ids[1]['resolution'] == 500

    def test_load_comp5(self):
        """Test loading a composite that has an optional prerequisite."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp5'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp5')

    def test_load_comp6(self):
        """Test loading a composite that has an optional composite prerequisite."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp6'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp6')

    def test_load_comp8(self):
        """Test loading a composite that has a non-existent prereq."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        pytest.raises(KeyError, scene.load, ['comp8'])

    def test_load_comp9(self):
        """Test loading a composite that has a non-existent optional prereq."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp9'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp9')

    def test_load_comp10(self):
        """Test loading a composite that depends on a modified dataset."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp10'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp10')

    def test_load_comp11(self):
        """Test loading a composite that depends all wavelengths."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp11'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp11')

    def test_load_comp12(self):
        """Test loading a composite that depends all wavelengths that get modified."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp12'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp12')

    def test_load_comp13(self):
        """Test loading a composite that depends on a modified dataset where the resolution changes."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp13'])
        loaded_ids = list(scene.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp13')

    def test_load_comp14(self):
        """Test loading a composite that updates the DataID during generation."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp14'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'comp14'

    def test_load_comp15(self):
        """Test loading a composite whose prerequisites can't be loaded.

        Note that the prereq exists in the reader, but fails in loading.

        """
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp15'])
        loaded_ids = list(scene._datasets.keys())
        assert not loaded_ids

    def test_load_comp16(self):
        """Test loading a composite whose opt prereq can't be loaded.

        Note that the prereq exists in the reader, but fails in loading

        """
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp16'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['name'] == 'comp16'

    def test_load_comp17(self):
        """Test loading a composite that depends on a composite that won't load."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp17'])
        loaded_ids = list(scene._datasets.keys())
        assert not loaded_ids

    def test_load_comp18(self):
        """Test loading a composite that depends on a incompatible area modified dataset."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt', 'fake1_highres_1.txt'], reader='fake1')
        scene.load(['comp18'])
        loaded_ids = list(scene._datasets.keys())
        # depends on:
        #   ds3
        #   ds4 (mod1, mod3)
        #   ds5 (mod1, incomp_areas)
        # We should end up with ds3, ds4 (mod1, mod3), ds5 (mod1), and ds1
        # for the incomp_areas modifier
        assert len(loaded_ids) == 4  # the 1 dependencies
        assert 'ds3' in scene._datasets
        assert make_dataid(name='ds4', calibration='reflectance',
                           modifiers=('mod1', 'mod3')) in scene._datasets
        assert make_dataid(name='ds5', resolution=250,
                           modifiers=('mod1',)) in scene._datasets

    def test_load_comp18_2(self):
        """Test loading a composite that depends on a incompatible area modified dataset.

        Specifically a modified dataset where the modifier has optional
        dependencies.

        """
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt', 'fake1_highres_1.txt'], reader='fake1')
        scene.load(['comp18_2'])
        loaded_ids = list(scene._datasets.keys())
        # depends on:
        #   ds3
        #   ds4 (mod1, mod3)
        #   ds5 (mod1, incomp_areas_opt)
        # We should end up with ds3, ds4 (mod1, mod3), ds5 (mod1), and ds1
        # and ds2 for the incomp_areas_opt modifier
        assert len(loaded_ids) == 5  # the 1 dependencies
        assert 'ds3' in scene._datasets
        assert 'ds2' in scene._datasets
        assert make_dataid(name='ds4', calibration='reflectance',
                           modifiers=('mod1', 'mod3')) in scene._datasets
        assert make_dataid(name='ds5', resolution=250,
                           modifiers=('mod1',)) in scene._datasets

    def test_load_comp19(self):
        """Test loading a composite that shares a dep with a dependency.

        More importantly test that loading a dependency that depends on
        the same dependency as this composite (a sibling dependency) and
        that sibling dependency includes a modifier. This test makes sure
        that the Node in the dependency tree is the exact same node.

        """
        # Check dependency tree nodes
        # initialize the dep tree without loading the data
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene._dependency_tree.populate_with_keys({'comp19'})

        this_node = scene._dependency_tree['comp19']
        shared_dep_id = make_dataid(name='ds5', modifiers=('res_change',))
        shared_dep_expected_node = scene._dependency_tree[shared_dep_id]
        # get the node for the first dep in the prereqs list of the
        # comp13 node
        shared_dep_node = scene._dependency_tree['comp13'].data[1][0]
        shared_dep_node2 = this_node.data[1][0]
        assert shared_dep_expected_node is shared_dep_node
        assert shared_dep_expected_node is shared_dep_node2

        scene.load(['comp19'])

        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp19')

    def test_load_multiple_comps(self):
        """Test loading multiple composites."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6',
                    'comp7', 'comp9', 'comp10'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 9

    def test_load_multiple_comps_separate(self):
        """Test loading multiple composites, one at a time."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp10'])
        scene.load(['comp9'])
        scene.load(['comp7'])
        scene.load(['comp6'])
        scene.load(['comp5'])
        scene.load(['comp4'])
        scene.load(['comp3'])
        scene.load(['comp2'])
        scene.load(['comp1'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 9

    def test_load_modified(self):
        """Test loading a modified dataset."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([make_dsq(name='ds1', modifiers=('mod1', 'mod2'))])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0]['modifiers'] == ('mod1', 'mod2')

    def test_load_multiple_modified(self):
        """Test loading multiple modified datasets."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load([
            make_dataid(name='ds1', modifiers=('mod1', 'mod2')),
            make_dataid(name='ds2', modifiers=('mod2', 'mod1')),
        ])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        for i in loaded_ids:
            if i['name'] == 'ds1':
                assert i['modifiers'] == ('mod1', 'mod2')
            else:
                assert i['name'] == 'ds2'
                assert i['modifiers'] == ('mod2', 'mod1')

    def test_load_dataset_after_composite(self):
        """Test load composite followed by other datasets."""
        from satpy.readers.yaml_reader import FileYAMLReader
        from satpy.tests.utils import FakeCompositor
        load_mock = spy_decorator(FileYAMLReader.load)
        comp_mock = spy_decorator(FakeCompositor.__call__)
        with mock.patch.object(FileYAMLReader, 'load', load_mock),\
                mock.patch.object(FakeCompositor, '__call__', comp_mock):
            lmock = load_mock.mock
            scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
            scene.load(['comp3'])
            assert lmock.call_count == 1
            scene.load(['ds1'])
            assert lmock.call_count == 2
            scene.load(['ds1'])
            # we should only load from the file twice
            assert lmock.call_count == 2
            # we should only generate the composite once
            assert comp_mock.mock.call_count == 1
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 2

    def test_load_dataset_after_composite2(self):
        """Test load complex composite followed by other datasets."""
        from satpy.readers.yaml_reader import FileYAMLReader
        from satpy.tests.utils import FakeCompositor, FakeModifier
        load_mock = spy_decorator(FileYAMLReader.load)
        comp_mock = spy_decorator(FakeCompositor.__call__)
        mod_mock = spy_decorator(FakeModifier.__call__)
        with mock.patch.object(FileYAMLReader, 'load', load_mock), \
             mock.patch.object(FakeCompositor, '__call__', comp_mock), \
             mock.patch.object(FakeModifier, '__call__', mod_mock):
            lmock = load_mock.mock
            scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
            scene.load(['comp10'])
            assert lmock.call_count == 1
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 1
            with mock.patch.object(scene, '_generate_composites_nodes_from_loaded_datasets',
                                   wraps=scene._generate_composites_nodes_from_loaded_datasets) as m:
                scene.load(['ds1'])
                assert lmock.call_count == 2
                loaded_ids = list(scene._datasets.keys())
                assert len(loaded_ids) == 2
                # this is the unmodified ds1
                assert make_dataid(
                    name='ds1', resolution=250, calibration='reflectance', modifiers=tuple()
                ) in loaded_ids
                # m.assert_called_once_with(set([scene._dependency_tree['ds1']]))
                m.assert_called_once_with(set())
            with mock.patch.object(scene, '_generate_composites_nodes_from_loaded_datasets',
                                   wraps=scene._generate_composites_nodes_from_loaded_datasets) as m:
                scene.load(['ds1'])
                assert lmock.call_count == 2
                loaded_ids = list(scene._datasets.keys())
                assert len(loaded_ids) == 2
                # this is the unmodified ds1
                assert make_dataid(
                    name='ds1', resolution=250, calibration='reflectance', modifiers=tuple()
                ) in loaded_ids
                m.assert_called_once_with(set())
            # we should only generate the comp10 composite once but comp2 was also generated
            assert comp_mock.mock.call_count == 1 + 1
            # Create the modded ds1 at comp10, then load the umodified version
            # again
            assert mod_mock.mock.call_count == 1
            loaded_ids = list(scene._datasets.keys())
            assert len(loaded_ids) == 2

    def test_load_comp20(self):
        """Test loading composite with optional modifier dependencies."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp20'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp20')

    def test_load_comp21(self):
        """Test loading composite with bad optional modifier dependencies."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp21'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp21')

    def test_load_comp22(self):
        """Test loading composite with only optional modifier dependencies."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp22'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='comp22')

    def test_load_green(self):
        """Test loading ahi_green."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['ahi_green'])
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 1
        assert loaded_ids[0] == make_cid(name='ahi_green')

    def test_no_generate_comp10(self):
        """Test generating a composite after loading."""
        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp10'], generate=False)
        assert any(ds_id['name'] == 'comp10' for ds_id in scene._wishlist)
        assert 'comp10' not in scene._datasets
        # two dependencies should have been loaded
        assert len(scene._datasets) == 2
        assert len(scene.missing_datasets) == 1

        scene._generate_composites_from_loaded_datasets()
        assert any(ds_id['name'] == 'comp10' for ds_id in scene._wishlist)
        assert 'comp10' in scene._datasets
        assert not scene.missing_datasets

    def test_modified_with_wl_dep(self):
        """Test modifying a dataset with a modifier with modified deps.

        More importantly test that loading the modifiers dependency at the
        same time as the original modified dataset that the dependency tree
        nodes are unique and that DataIDs.

        """
        from satpy.dataset.dataid import WavelengthRange

        # Check dependency tree nodes
        # initialize the dep tree without loading the data
        ds1_mod_id = make_dsq(name='ds1', modifiers=('mod_wl',))
        ds3_mod_id = make_dsq(name='ds3', modifiers=('mod_wl',))

        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene._dependency_tree.populate_with_keys({ds1_mod_id, ds3_mod_id})

        ds1_mod_node = scene._dependency_tree[ds1_mod_id]
        ds3_mod_node = scene._dependency_tree[ds3_mod_id]
        ds1_mod_dep_node = ds1_mod_node.data[1][1]
        ds3_mod_dep_node = ds3_mod_node.data[1][1]
        # mod_wl depends on the this node:
        ds6_modded_node = scene._dependency_tree[make_dataid(name='ds6', modifiers=('mod1',))]
        # this dep should be full qualified with name and wavelength
        assert ds6_modded_node.name['name'] is not None
        assert isinstance(ds6_modded_node.name['wavelength'], WavelengthRange)
        # the node should be shared between everything that uses it
        assert ds1_mod_dep_node is ds3_mod_dep_node
        assert ds1_mod_dep_node is ds6_modded_node

        # it is fine that an optional prereq doesn't exist
        scene.load([ds1_mod_id, ds3_mod_id])

        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert ds1_mod_id in scene._datasets
        assert ds3_mod_id in scene._datasets

    def test_load_comp11_and_23(self):
        """Test loading two composites that depend on similar wavelengths."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        # mock the available comps/mods in the compositor loader
        avail_comps = scene.available_composite_ids()
        assert make_cid(name='comp11') in avail_comps
        assert make_cid(name='comp23') in avail_comps
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp11', 'comp23'])
        comp11_node = scene._dependency_tree['comp11']
        comp23_node = scene._dependency_tree['comp23']
        assert comp11_node.data[1][-1].name['name'] == 'ds10'
        assert comp23_node.data[1][0].name['name'] == 'ds8'
        loaded_ids = list(scene._datasets.keys())
        assert len(loaded_ids) == 2
        assert 'comp11' in scene
        assert 'comp23' in scene

    def test_load_too_many(self):
        """Test dependency tree if too many reader keys match."""
        scene = Scene(filenames=['fake3_1.txt'], reader='fake3')
        avail_comps = scene.available_composite_ids()
        # static image => 1
        assert len(avail_comps) == 1
        pytest.raises(KeyError, scene.load, [0.21])

    def test_available_comps_no_deps(self):
        """Test Scene available composites when composites don't have a dependency."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        all_comp_ids = scene.available_composite_ids()
        assert make_cid(name='static_image') in all_comp_ids
        available_comp_ids = scene.available_composite_ids()
        assert make_cid(name='static_image') in available_comp_ids


class TestSceneResampling:
    """Test resampling a Scene to another Scene object."""

    def setup_method(self):
        """Set config_path to point to test 'etc' directory."""
        self.old_config_path = satpy.config.get('config_path')
        satpy.config.set(config_path=[TEST_ETC_DIR])

    def teardown_method(self):
        """Restore previous 'config_path' setting."""
        satpy.config.set(config_path=self.old_config_path)

    def _fake_resample_dataset(self, dataset, dest_area, **kwargs):
        """Return copy of dataset pretending it was resampled."""
        return dataset.copy()

    def _fake_resample_dataset_force_20x20(self, dataset, dest_area, **kwargs):
        """Return copy of dataset pretending it was resampled to (20, 20) shape."""
        data = np.zeros((20, 20))
        attrs = dataset.attrs.copy()
        attrs['area'] = dest_area
        return xr.DataArray(
            data,
            dims=('y', 'x'),
            attrs=attrs,
        )

    @mock.patch('satpy.scene.resample_dataset')
    @pytest.mark.parametrize('datasets', [
        None,
        ('comp13', 'ds5', 'ds2'),
    ])
    def test_resample_scene_copy(self, rs, datasets):
        """Test that the Scene is properly copied during resampling.

        The Scene that is created as a copy of the original Scene should not
        be able to affect the original Scene object.

        """
        from pyresample.geometry import AreaDefinition
        rs.side_effect = self._fake_resample_dataset_force_20x20

        proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                    '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_str, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        scene = Scene(filenames=['fake1_1.txt', 'fake1_highres_1.txt'], reader='fake1')

        scene.load(['comp19'])
        new_scene = scene.resample(area_def, datasets=datasets)
        new_scene['new_ds'] = new_scene['comp19'].copy()

        scene.load(['ds1'])

        comp19_node = scene._dependency_tree['comp19']
        ds5_mod_id = make_dataid(name='ds5', modifiers=('res_change',))
        ds5_node = scene._dependency_tree[ds5_mod_id]
        comp13_node = scene._dependency_tree['comp13']

        assert comp13_node.data[1][0] is comp19_node.data[1][0]
        assert comp13_node.data[1][0] is ds5_node
        pytest.raises(KeyError, scene._dependency_tree.__getitem__, 'new_ds')

        # comp19 required resampling to produce so we should have its 3 deps
        # 1. comp13
        # 2. ds5
        # 3. ds2
        # Then we loaded ds1 separately so we should have
        # 4. ds1
        loaded_ids = list(scene.keys())
        assert len(loaded_ids) == 4
        for name in ('comp13', 'ds5', 'ds2', 'ds1'):
            assert any(x['name'] == name for x in loaded_ids)

        loaded_ids = list(new_scene.keys())
        assert len(loaded_ids) == 2
        assert loaded_ids[0] == make_cid(name='comp19')
        assert loaded_ids[1] == make_cid(name='new_ds')

    @mock.patch('satpy.scene.resample_dataset')
    def test_resample_scene_preserves_requested_dependencies(self, rs):
        """Test that the Scene is properly copied during resampling.

        The Scene that is created as a copy of the original Scene should not
        be able to affect the original Scene object.

        """
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict

        rs.side_effect = self._fake_resample_dataset
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')

        # Set PYTHONHASHSEED to 0 in the interpreter to test as intended (comp26 comes before comp14)
        scene.load(['comp26', 'comp14'], generate=False)
        scene.resample(area_def, unload=True)
        new_scene_2 = scene.resample(area_def, unload=True)

        assert 'comp14' not in scene
        assert 'comp26' not in scene
        assert 'comp14' in new_scene_2
        assert 'comp26' in new_scene_2
        assert 'ds1' not in new_scene_2  # unloaded

    @mock.patch('satpy.scene.resample_dataset')
    def test_resample_reduce_data_toggle(self, rs):
        """Test that the Scene can be reduced or not reduced during resampling."""
        from pyresample.geometry import AreaDefinition
        import dask.array as da
        import xarray as xr

        rs.side_effect = self._fake_resample_dataset_force_20x20
        proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                    '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
        target_area = AreaDefinition('test', 'test', 'test', proj_str, 4, 4, (-1000., -1500., 1000., 1500.))
        area_def = AreaDefinition('test', 'test', 'test', proj_str, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        get_area_slices = area_def.get_area_slices
        get_area_slices.return_value = (slice(0, 3, None), slice(0, 3, None))
        area_def_big = AreaDefinition('test', 'test', 'test', proj_str, 10, 10, (-1000., -1500., 1000., 1500.))
        area_def_big.get_area_slices = mock.MagicMock()
        get_area_slices_big = area_def_big.get_area_slices
        get_area_slices_big.return_value = (slice(0, 6, None), slice(0, 6, None))

        # Test that data reduction can be disabled
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp19'])
        scene['comp19'].attrs['area'] = area_def
        scene['comp19_big'] = xr.DataArray(
            da.zeros((10, 10)), dims=('y', 'x'),
            attrs=scene['comp19'].attrs.copy())
        scene['comp19_big'].attrs['area'] = area_def_big
        scene['comp19_copy'] = scene['comp19'].copy()
        orig_slice_data = scene._slice_data
        # we force the below order of processing to test that success isn't
        # based on data of the same resolution being processed together
        test_order = [
            make_cid(**scene['comp19'].attrs),
            make_cid(**scene['comp19_big'].attrs),
            make_cid(**scene['comp19_copy'].attrs),
        ]
        with mock.patch('satpy.scene.Scene._slice_data') as slice_data, \
                mock.patch('satpy.dataset.dataset_walker') as ds_walker:
            ds_walker.return_value = test_order
            slice_data.side_effect = orig_slice_data
            scene.resample(target_area, reduce_data=False)
            assert not slice_data.called
            assert not get_area_slices.called
            scene.resample(target_area)
            assert slice_data.called_once
            assert get_area_slices.called_once
            scene.resample(target_area, reduce_data=True)
            # 2 times for each dataset
            # once for default (reduce_data=True)
            # once for kwarg forced to `True`
            assert slice_data.call_count == 2 * 3
            assert get_area_slices.called_once

    def test_resample_ancillary(self):
        """Test that the Scene reducing data does not affect final output."""
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')

        scene.load(['comp19', 'comp20'])
        scene['comp19'].attrs['area'] = area_def
        scene['comp19'].attrs['ancillary_variables'] = [scene['comp20']]
        scene['comp20'].attrs['area'] = area_def

        dst_area = AreaDefinition('dst', 'dst', 'dst',
                                  proj_dict,
                                  2, 2,
                                  (-1000., -1500., 0., 0.),
                                  )
        new_scene = scene.resample(dst_area)
        assert new_scene['comp20'] is new_scene['comp19'].attrs['ancillary_variables'][0]

    def test_resample_reduce_data(self):
        """Test that the Scene reducing data does not affect final output."""
        from pyresample.geometry import AreaDefinition
        proj_str = ('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                    '+lon_0=-95. +lat_0=25 +lat_1=25 +units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_str, 20, 20, (-1000., -1500., 1000., 1500.))
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')

        scene.load(['comp19'])
        scene['comp19'].attrs['area'] = area_def
        dst_area = AreaDefinition('dst', 'dst', 'dst',
                                  proj_str,
                                  20, 20,
                                  (-1000., -1500., 0., 0.),
                                  )
        new_scene1 = scene.resample(dst_area, reduce_data=False)
        new_scene2 = scene.resample(dst_area)
        new_scene3 = scene.resample(dst_area, reduce_data=True)
        assert new_scene1['comp19'].shape == (20, 20, 3)
        assert new_scene2['comp19'].shape == (20, 20, 3)
        assert new_scene3['comp19'].shape == (20, 20, 3)

    @mock.patch('satpy.scene.resample_dataset')
    def test_no_generate_comp10(self, rs):
        """Test generating a composite after loading."""
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict

        rs.side_effect = self._fake_resample_dataset
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            200,
            400,
            (-1000., -1500., 1000., 1500.),
        )

        # it is fine that an optional prereq doesn't exist
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        scene.load(['comp10'], generate=False)
        assert any(ds_id['name'] == 'comp10' for ds_id in scene._wishlist)
        assert 'comp10' not in scene
        # two dependencies should have been loaded
        assert len(scene._datasets) == 2
        assert len(scene.missing_datasets) == 1

        new_scn = scene.resample(area_def, generate=False)
        assert 'comp10' not in scene
        # two dependencies should have been loaded
        assert len(scene._datasets) == 2
        assert len(scene.missing_datasets) == 1

        new_scn._generate_composites_from_loaded_datasets()
        assert any(ds_id['name'] == 'comp10' for ds_id in new_scn._wishlist)
        assert 'comp10' in new_scn
        assert not new_scn.missing_datasets

        # try generating them right away
        new_scn = scene.resample(area_def)
        assert any(ds_id['name'] == 'comp10' for ds_id in new_scn._wishlist)
        assert 'comp10' in new_scn
        assert not new_scn.missing_datasets

    def test_comps_need_resampling_optional_mod_deps(self):
        """Test that a composite with complex dependencies.

        This is specifically testing the case where a compositor depends on
        multiple resolution prerequisites which themselves are composites.
        These sub-composites depend on data with a modifier that only has
        optional dependencies. This is a very specific use case and is the
        simplest way to present the problem (so far).

        The general issue is that the Scene loading creates the "ds13"
        dataset which already has one modifier on it. The "comp27"
        composite requires resampling so its 4 prerequisites + the
        requested "ds13" (from the reader which includes mod1 modifier)
        remain. If the DependencyTree is not copied properly in this
        situation then the new Scene object will have the composite
        dependencies without resolution in its dep tree, but have
        the DataIDs with the resolution in the dataset dictionary.
        This all results in the Scene trying to regenerate composite
        dependencies that aren't needed which fail.

        """
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        # should require resampling
        scene.load(['comp27', 'ds13'])
        assert 'comp27' not in scene
        assert 'ds13' in scene

        new_scene = scene.resample(resampler='native')
        assert len(list(new_scene.keys())) == 2
        assert 'comp27' in new_scene
        assert 'ds13' in new_scene


class TestSceneSaving(unittest.TestCase):
    """Test the Scene's saving method."""

    def setUp(self):
        """Create temporary directory to save files to."""
        import tempfile
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_save_datasets_default(self):
        """Save a dataset using 'save_datasets'."""
        from satpy.scene import Scene
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime(2018, 1, 1, 0, 0, 0)}
        )
        scn = Scene()
        scn['test'] = ds1
        scn.save_datasets(base_dir=self.base_dir)
        assert os.path.isfile(os.path.join(self.base_dir, 'test_20180101_000000.tif'))

    def test_save_datasets_by_ext(self):
        """Save a dataset using 'save_datasets' with 'filename'."""
        from satpy.scene import Scene
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime(2018, 1, 1, 0, 0, 0)}
        )
        scn = Scene()
        scn['test'] = ds1

        from satpy.writers.simple_image import PillowWriter
        save_image_mock = spy_decorator(PillowWriter.save_image)
        with mock.patch.object(PillowWriter, 'save_image', save_image_mock):
            scn.save_datasets(base_dir=self.base_dir, filename='{name}.png')
        save_image_mock.mock.assert_called_once()
        assert os.path.isfile(os.path.join(self.base_dir, 'test.png'))

    def test_save_datasets_bad_writer(self):
        """Save a dataset using 'save_datasets' and a bad writer."""
        from satpy.scene import Scene
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime.utcnow()}
        )
        scn = Scene()
        scn['test'] = ds1
        pytest.raises(ValueError,
                      scn.save_datasets,
                      writer='_bad_writer_',
                      base_dir=self.base_dir)

    def test_save_datasets_missing_wishlist(self):
        """Calling 'save_datasets' with no valid datasets."""
        from satpy.scene import Scene
        scn = Scene()
        scn._wishlist.add(make_cid(name='true_color'))
        pytest.raises(RuntimeError,
                      scn.save_datasets,
                      writer='geotiff',
                      base_dir=self.base_dir)
        pytest.raises(KeyError,
                      scn.save_datasets,
                      datasets=['no_exist'])

    def test_save_dataset_default(self):
        """Save a dataset using 'save_dataset'."""
        from satpy.scene import Scene
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        ds1 = xr.DataArray(
            da.zeros((100, 200), chunks=50),
            dims=('y', 'x'),
            attrs={'name': 'test',
                   'start_time': datetime(2018, 1, 1, 0, 0, 0)}
        )
        scn = Scene()
        scn['test'] = ds1
        scn.save_dataset('test', base_dir=self.base_dir)
        assert os.path.isfile(os.path.join(self.base_dir, 'test_20180101_000000.tif'))


class TestSceneConversions(unittest.TestCase):
    """Test Scene conversion to geoviews, xarray, etc."""

    def test_geoviews_basic_with_area(self):
        """Test converting a Scene to geoviews with an AreaDefinition."""
        from satpy import Scene
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import AreaDefinition
        scn = Scene()
        area = AreaDefinition('test', 'test', 'test',
                              {'proj': 'geos', 'lon_0': -95.5, 'h': 35786023.0},
                              2, 2, [-200, -200, 200, 200])
        scn['ds1'] = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                                  attrs={'start_time': datetime(2018, 1, 1),
                                         'area': area})
        gv_obj = scn.to_geoviews()
        # we assume that if we got something back, geoviews can use it
        assert gv_obj is not None

    def test_geoviews_basic_with_swath(self):
        """Test converting a Scene to geoviews with a SwathDefinition."""
        from satpy import Scene
        import xarray as xr
        import dask.array as da
        from datetime import datetime
        from pyresample.geometry import SwathDefinition
        scn = Scene()
        lons = xr.DataArray(da.zeros((2, 2)))
        lats = xr.DataArray(da.zeros((2, 2)))
        area = SwathDefinition(lons, lats)
        scn['ds1'] = xr.DataArray(da.zeros((2, 2), chunks=-1), dims=('y', 'x'),
                                  attrs={'start_time': datetime(2018, 1, 1),
                                         'area': area})
        gv_obj = scn.to_geoviews()
        # we assume that if we got something back, geoviews can use it
        assert gv_obj is not None


class TestSceneAggregation(unittest.TestCase):
    """Test the scene's aggregate method."""

    def test_aggregate(self):
        """Test the aggregate method."""
        x_size = 3712
        y_size = 3712

        scene1 = self._create_test_data(x_size, y_size)

        scene2 = scene1.aggregate(func='sum', x=2, y=2)
        expected_aggregated_shape = (y_size / 2, x_size / 2)
        self._check_aggregation_results(expected_aggregated_shape, scene1, scene2, x_size, y_size)

    @staticmethod
    def _create_test_data(x_size, y_size):
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        scene1 = Scene()
        area_extent = (-5570248.477339745, -5561247.267842293, 5567248.074173927,
                       5570248.477339745)
        proj_dict = {'a': 6378169.0, 'b': 6356583.8, 'h': 35785831.0,
                     'lon_0': 0.0, 'proj': 'geos', 'units': 'm'}
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )
        scene1["1"] = DataArray(np.ones((y_size, x_size)), attrs={'_satpy_id_keys': default_id_keys_config})
        scene1["2"] = DataArray(np.ones((y_size, x_size)), dims=('y', 'x'),
                                attrs={'_satpy_id_keys': default_id_keys_config})
        scene1["3"] = DataArray(np.ones((y_size, x_size)), dims=('y', 'x'),
                                attrs={'area': area_def, '_satpy_id_keys': default_id_keys_config})
        scene1["4"] = DataArray(np.ones((y_size, x_size)), dims=('y', 'x'),
                                attrs={'area': area_def, 'standard_name': 'backscatter',
                                       '_satpy_id_keys': default_id_keys_config})
        return scene1

    def _check_aggregation_results(self, expected_aggregated_shape, scene1, scene2, x_size, y_size):
        assert scene1['1'] is scene2['1']
        assert scene1['2'] is scene2['2']
        np.testing.assert_allclose(scene2['3'].data, 4)
        assert scene2['1'].shape == (y_size, x_size)
        assert scene2['2'].shape == (y_size, x_size)
        assert scene2['3'].shape == expected_aggregated_shape
        assert 'standard_name' in scene2['4'].attrs
        assert scene2['4'].attrs['standard_name'] == 'backscatter'

    def test_aggregate_with_boundary(self):
        """Test aggregation with boundary argument."""
        x_size = 3711
        y_size = 3711

        scene1 = self._create_test_data(x_size, y_size)

        with pytest.raises(ValueError):
            scene1.aggregate(func='sum', x=2, y=2, boundary='exact')

        scene2 = scene1.aggregate(func='sum', x=2, y=2, boundary='trim')
        expected_aggregated_shape = (y_size // 2, x_size // 2)
        self._check_aggregation_results(expected_aggregated_shape, scene1, scene2, x_size, y_size)
