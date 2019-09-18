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
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock

# clear the config dir environment variable so it doesn't interfere
os.environ.pop("PPP_CONFIG_DIR", None)


class TestScene(unittest.TestCase):
    """Test the scene class."""

    def test_init(self):
        """Test scene initialization."""
        import satpy.scene
        with mock.patch('satpy.scene.Scene.create_reader_instances') as cri:
            cri.return_value = {}
            satpy.scene.Scene(filenames=['bla'], reader='blo')
            cri.assert_called_once_with(filenames=['bla'], reader='blo',
                                        reader_kwargs=None)

    def test_init_str_filename(self):
        """Test initializing with a single string as filenames."""
        import satpy.scene
        self.assertRaises(ValueError, satpy.scene.Scene, reader='blo', filenames='test.nc')

    def test_init_with_sensor(self):
        """Test initializing with a sensor."""
        import satpy.scene
        from satpy.tests.utils import FakeReader
        with mock.patch('satpy.scene.Scene.create_reader_instances') as cri:
            cri.return_value = {
                'fake_reader': FakeReader('fake_reader', sensor_name='fake_sensor'),
            }
            scene = satpy.scene.Scene(filenames=['bla'],
                                      base_dir='bli',
                                      sensor='fake_sensor')
            self.assertIsInstance(scene.attrs['sensor'], set)
            scene = satpy.scene.Scene(filenames=['bla'],
                                      base_dir='bli',
                                      sensor=['fake_sensor'])
            self.assertIsInstance(scene.attrs['sensor'], set)

    def test_start_end_times(self):
        """Test start and end times for a scene."""
        import satpy.scene
        from satpy.tests.utils import FakeReader
        from datetime import datetime
        with mock.patch('satpy.scene.Scene.create_reader_instances') as cri:
            r = FakeReader('fake_reader',
                           start_time=datetime(2017, 1, 1, 0, 0, 0),
                           end_time=datetime(2017, 1, 1, 1, 0, 0),
                           )
            cri.return_value = {'fake_reader': r}
            scene = satpy.scene.Scene(filenames=['bla'],
                                      base_dir='bli',
                                      sensor='fake_sensor')
            self.assertEqual(scene.start_time, r.start_time)
            self.assertEqual(scene.end_time, r.end_time)

    def test_init_preserve_reader_kwargs(self):
        """Test that the initialization preserves the kwargs."""
        import satpy.scene
        from satpy.tests.utils import FakeReader
        from datetime import datetime
        with mock.patch('satpy.scene.Scene.create_reader_instances') as cri:
            r = FakeReader('fake_reader',
                           start_time=datetime(2017, 1, 1, 0, 0, 0),
                           end_time=datetime(2017, 1, 1, 1, 0, 0),
                           )
            cri.return_value = {'fake_reader': r}
            reader_kwargs = {'calibration_type': 'gsics'}
            scene = satpy.scene.Scene(filenames=['bla'],
                                      base_dir='bli',
                                      sensor='fake_sensor',
                                      filter_parameters={'area': 'euron1'},
                                      reader_kwargs=reader_kwargs)
            self.assertIsNot(reader_kwargs, cri.call_args[1]['reader_kwargs'])
            self.assertEqual(scene.start_time, r.start_time)
            self.assertEqual(scene.end_time, r.end_time)

    def test_init_alone(self):
        """Test simple initialization."""
        from satpy.scene import Scene
        from satpy.config import PACKAGE_CONFIG_PATH
        scn = Scene()
        self.assertEqual(scn.ppp_config_dir, PACKAGE_CONFIG_PATH)
        self.assertFalse(scn.readers, 'Empty scene should not load any readers')

    def test_init_no_files(self):
        """Test that providing an empty list of filenames fails."""
        from satpy.scene import Scene
        self.assertRaises(ValueError, Scene, reader='viirs_sdr', filenames=[])

    def test_init_with_ppp_config_dir(self):
        """Test initializing with a ppp_config_dir."""
        from satpy.scene import Scene
        scn = Scene(ppp_config_dir="foo")
        self.assertEqual(scn.ppp_config_dir, 'foo')

    def test_create_reader_instances_with_filenames(self):
        """Test creating a reader providing filenames."""
        import satpy.scene
        filenames = ["bla", "foo", "bar"]
        reader_name = None
        with mock.patch('satpy.scene.Scene._compute_metadata_from_readers') as md:
            md.return_value = {'sensor': {'sensor'}}
            with mock.patch('satpy.scene.load_readers') as findermock:
                satpy.scene.Scene(filenames=filenames)
                findermock.assert_called_once_with(
                    filenames=filenames,
                    reader=reader_name,
                    reader_kwargs=None,
                    ppp_config_dir=mock.ANY
                )

    def test_init_with_empty_filenames(self):
        """Test initialization with empty filename list."""
        from satpy.scene import Scene
        filenames = []
        Scene(filenames=filenames)

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
            findermock.assert_called_once_with(ppp_config_dir=mock.ANY,
                                               reader=reader,
                                               filenames=filenames,
                                               reader_kwargs=None,
                                               )

    def test_create_reader_instances_with_reader_kwargs(self):
        """Test creating a reader instance with reader kwargs."""
        import satpy.scene
        from satpy.tests.utils import FakeReader
        from datetime import datetime
        filenames = ["1", "2", "3"]
        reader_kwargs = {'calibration_type': 'gsics'}
        filter_parameters = {'area': 'euron1'}
        reader_kwargs2 = {'calibration_type': 'gsics', 'filter_parameters': filter_parameters}

        with mock.patch('satpy.readers.load_reader') as lr_mock:
            r = FakeReader('fake_reader',
                           start_time=datetime(2017, 1, 1, 0, 0, 0),
                           end_time=datetime(2017, 1, 1, 1, 0, 0),
                           )
            lr_mock.return_value = r
            r.select_files_from_pathnames = mock.MagicMock()
            r.select_files_from_pathnames.return_value = filenames
            r.create_filehandlers = mock.MagicMock()
            scene = satpy.scene.Scene(filenames=['bla'],
                                      base_dir='bli',
                                      sensor='fake_sensor',
                                      filter_parameters={'area': 'euron1'},
                                      reader_kwargs=reader_kwargs)
            del scene
            self.assertDictEqual(reader_kwargs, r.create_filehandlers.call_args[1]['fh_kwargs'])
            scene = satpy.scene.Scene(filenames=['bla'],
                                      base_dir='bli',
                                      sensor='fake_sensor',
                                      reader_kwargs=reader_kwargs2)
            self.assertDictEqual(reader_kwargs, r.create_filehandlers.call_args[1]['fh_kwargs'])
            del scene

    def test_iter(self):
        """Test iteration over the scene."""
        from satpy import Scene
        from xarray import DataArray
        import numpy as np
        scene = Scene()
        scene["1"] = DataArray(np.arange(5))
        scene["2"] = DataArray(np.arange(5))
        scene["3"] = DataArray(np.arange(5))
        for x in scene:
            self.assertIsInstance(x, DataArray)

    def test_iter_by_area_swath(self):
        """Test iterating by area on a swath."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import SwathDefinition
        import numpy as np
        scene = Scene()
        sd = SwathDefinition(lons=np.arange(5), lats=np.arange(5))
        scene["1"] = DataArray(np.arange(5), attrs={'area': sd})
        scene["2"] = DataArray(np.arange(5), attrs={'area': sd})
        scene["3"] = DataArray(np.arange(5))
        for area_obj, ds_list in scene.iter_by_area():
            ds_list_names = set(ds.name for ds in ds_list)
            if area_obj is sd:
                self.assertSetEqual(ds_list_names, {'1', '2'})
            else:
                self.assertIsNone(area_obj)
                self.assertSetEqual(ds_list_names, {'3'})

    def test_bad_setitem(self):
        """Test setting an item wrongly."""
        from satpy import Scene
        import numpy as np
        scene = Scene()
        self.assertRaises(ValueError, scene.__setitem__, '1', np.arange(5))

    def test_setitem(self):
        """Test setting an item."""
        from satpy import Scene, DatasetID
        import numpy as np
        import xarray as xr
        scene = Scene()
        scene["1"] = ds1 = xr.DataArray(np.arange(5))
        expected_id = DatasetID.from_dict(ds1.attrs)
        self.assertSetEqual(set(scene.datasets.keys()), {expected_id})
        self.assertSetEqual(set(scene.wishlist), {expected_id})

    def test_getitem(self):
        """Test __getitem__ with names only."""
        from satpy import Scene
        from xarray import DataArray
        import numpy as np
        scene = Scene()
        scene["1"] = ds1 = DataArray(np.arange(5))
        scene["2"] = ds2 = DataArray(np.arange(5))
        scene["3"] = ds3 = DataArray(np.arange(5))
        self.assertIs(scene['1'], ds1)
        self.assertIs(scene['2'], ds2)
        self.assertIs(scene['3'], ds3)
        self.assertRaises(KeyError, scene.__getitem__, '4')
        self.assertIs(scene.get('3'), ds3)
        self.assertIs(scene.get('4'), None)

    def test_getitem_modifiers(self):
        """Test __getitem__ with names and modifiers."""
        from satpy import Scene, DatasetID
        from xarray import DataArray
        import numpy as np

        # Return least modified item
        scene = Scene()
        scene['1'] = ds1_m0 = DataArray(np.arange(5))
        scene[DatasetID(name='1', modifiers=('mod1',))
              ] = ds1_m1 = DataArray(np.arange(5))
        self.assertIs(scene['1'], ds1_m0)
        self.assertEqual(len(list(scene.keys())), 2)

        scene = Scene()
        scene['1'] = ds1_m0 = DataArray(np.arange(5))
        scene[DatasetID(name='1', modifiers=('mod1',))
              ] = ds1_m1 = DataArray(np.arange(5))
        scene[DatasetID(name='1', modifiers=('mod1', 'mod2'))
              ] = ds1_m2 = DataArray(np.arange(5))
        self.assertIs(scene['1'], ds1_m0)
        self.assertEqual(len(list(scene.keys())), 3)

        scene = Scene()
        scene[DatasetID(name='1', modifiers=('mod1', 'mod2'))
              ] = ds1_m2 = DataArray(np.arange(5))
        scene[DatasetID(name='1', modifiers=('mod1',))
              ] = ds1_m1 = DataArray(np.arange(5))
        self.assertIs(scene['1'], ds1_m1)
        self.assertIs(scene[DatasetID('1', modifiers=('mod1', 'mod2'))], ds1_m2)
        self.assertRaises(KeyError, scene.__getitem__,
                          DatasetID(name='1', modifiers=tuple()))
        self.assertEqual(len(list(scene.keys())), 2)

    def test_getitem_slices(self):
        """Test __getitem__ with slices."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition, SwathDefinition
        from pyresample.utils import proj4_str_to_dict
        import numpy as np
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
            self.assertTupleEqual(new_scn['1'].shape, (5, 10))
            self.assertTupleEqual(new_scn['2'].shape, (5, 10))

        self.assertTupleEqual(new_scn1['3'].shape, (3, 6))
        self.assertIn('area', new_scn1['3'].attrs)
        self.assertTupleEqual(new_scn1['3'].attrs['area'].shape, (3, 6))
        self.assertTupleEqual(new_scn1['3a'].shape, (3, 6))
        a_var = new_scn1['3a'].attrs['ancillary_variables'][0]
        self.assertTupleEqual(a_var.shape, (3, 6))

        self.assertTupleEqual(new_scn2['4'].shape, (3, 6))
        self.assertIn('area', new_scn2['4'].attrs)
        self.assertTupleEqual(new_scn2['4'].attrs['area'].shape, (3, 6))
        self.assertTupleEqual(new_scn2['4a'].shape, (3, 6))
        a_var = new_scn2['4a'].attrs['ancillary_variables'][0]
        self.assertTupleEqual(a_var.shape, (3, 6))

    def test_crop(self):
        """Test the crop method."""
        from satpy import Scene
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
        self.assertIn('1', new_scn1)
        self.assertIn('2', new_scn1)
        self.assertIn('3', new_scn1)
        self.assertTupleEqual(new_scn1['1'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['2'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['3'].shape, (3380, 3708))
        self.assertTupleEqual(new_scn1['4'].shape, (1690, 1854))

        # by lon/lat bbox
        new_scn1 = scene1.crop(ll_bbox=(-20., -5., 0, 0))
        self.assertIn('1', new_scn1)
        self.assertIn('2', new_scn1)
        self.assertIn('3', new_scn1)
        self.assertTupleEqual(new_scn1['1'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['2'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['3'].shape, (184, 714))
        self.assertTupleEqual(new_scn1['4'].shape, (92, 357))

        # by x/y bbox
        new_scn1 = scene1.crop(xy_bbox=(-200000., -100000., 0, 0))
        self.assertIn('1', new_scn1)
        self.assertIn('2', new_scn1)
        self.assertIn('3', new_scn1)
        self.assertTupleEqual(new_scn1['1'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['2'].shape, (y_size, x_size))
        self.assertTupleEqual(new_scn1['3'].shape, (36, 70))
        self.assertTupleEqual(new_scn1['4'].shape, (18, 35))

    def test_crop_rgb(self):
        """Test the crop method on multi-dimensional data."""
        from satpy import Scene
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
        self.assertIn('1', new_scn1)
        self.assertIn('2', new_scn1)
        self.assertIn('bands', new_scn1['1'].dims)
        self.assertIn('bands', new_scn1['2'].dims)
        self.assertTupleEqual(new_scn1['1'].shape, (3, 184, 714))
        self.assertTupleEqual(new_scn1['2'].shape, (92, 3, 357))

    def test_aggregate(self):
        """Test the aggregate method."""
        if (sys.version_info < (3, 0)):
            self.skipTest("Not implemented in python 2 (xarray).")
        from satpy import Scene
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
            'test',
            'test',
            'test',
            proj_dict,
            x_size,
            y_size,
            area_extent,
        )

        scene1["1"] = DataArray(np.ones((y_size, x_size)))
        scene1["2"] = DataArray(np.ones((y_size, x_size)), dims=('y', 'x'))
        scene1["3"] = DataArray(np.ones((y_size, x_size)), dims=('y', 'x'),
                                attrs={'area': area_def})

        scene2 = scene1.aggregate(func='sum', x=2, y=2)
        self.assertIs(scene1['1'], scene2['1'])
        self.assertIs(scene1['2'], scene2['2'])
        np.testing.assert_allclose(scene2['3'].data, 4)
        self.assertTupleEqual(scene2['1'].shape, (y_size, x_size))
        self.assertTupleEqual(scene2['2'].shape, (y_size, x_size))
        self.assertTupleEqual(scene2['3'].shape, (y_size / 2, x_size / 2))

    def test_contains(self):
        """Test contains."""
        from satpy import Scene
        from xarray import DataArray
        import numpy as np
        scene = Scene()
        scene["1"] = DataArray(np.arange(5), attrs={'wavelength': (0.1, 0.2, 0.3)})
        self.assertTrue('1' in scene)
        self.assertTrue(0.15 in scene)
        self.assertFalse('2' in scene)
        self.assertFalse(0.31 in scene)

    def test_delitem(self):
        """Test deleting an item."""
        from satpy import Scene
        from xarray import DataArray
        import numpy as np
        scene = Scene()
        scene["1"] = DataArray(np.arange(5), attrs={'wavelength': (0.1, 0.2, 0.3)})
        scene["2"] = DataArray(np.arange(5), attrs={'wavelength': (0.4, 0.5, 0.6)})
        scene["3"] = DataArray(np.arange(5), attrs={'wavelength': (0.7, 0.8, 0.9)})
        del scene['1']
        del scene['3']
        del scene[0.45]
        self.assertEqual(len(scene.wishlist), 0)
        self.assertEqual(len(scene.datasets.keys()), 0)
        self.assertRaises(KeyError, scene.__delitem__, 0.2)

    def test_min_max_area(self):
        """Test 'min_area' and 'max_area' methods."""
        from satpy import Scene
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        import numpy as np
        scene = Scene()
        scene["1"] = ds1 = DataArray(np.arange(10).reshape((2, 5)),
                                     attrs={'wavelength': (0.1, 0.2, 0.3)})
        scene["2"] = ds2 = DataArray(np.arange(40).reshape((4, 10)),
                                     attrs={'wavelength': (0.4, 0.5, 0.6)})
        scene["3"] = ds3 = DataArray(np.arange(40).reshape((4, 10)),
                                     attrs={'wavelength': (0.7, 0.8, 0.9)})
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def1 = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            100,
            200,
            (-1000., -1500., 1000., 1500.),
        )
        area_def2 = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            200,
            400,
            (-1000., -1500., 1000., 1500.),
        )
        ds1.attrs['area'] = area_def1
        ds2.attrs['area'] = area_def2
        ds3.attrs['area'] = area_def2
        self.assertIs(scene.min_area(), area_def1)
        self.assertIs(scene.max_area(), area_def2)
        self.assertIs(scene.min_area(['2', '3']), area_def2)

    def test_all_datasets_no_readers(self):
        """Test all datasets with no reader."""
        from satpy import Scene
        scene = Scene()
        self.assertRaises(KeyError, scene.all_dataset_ids, reader_name='fake')
        id_list = scene.all_dataset_ids()
        self.assertListEqual(id_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        id_list = scene.all_dataset_ids(composites=True)
        self.assertListEqual(id_list, [])

    def test_all_dataset_names_no_readers(self):
        """Test all dataset names with no reader."""
        from satpy import Scene
        scene = Scene()
        self.assertRaises(KeyError, scene.all_dataset_names, reader_name='fake')
        name_list = scene.all_dataset_names()
        self.assertListEqual(name_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.all_dataset_names(composites=True)
        self.assertListEqual(name_list, [])

    def test_available_dataset_no_readers(self):
        """Test the available datasets without a reader."""
        from satpy import Scene
        scene = Scene()
        self.assertRaises(
            KeyError, scene.available_dataset_ids, reader_name='fake')
        name_list = scene.available_dataset_ids()
        self.assertListEqual(name_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_ids(composites=True)
        self.assertListEqual(name_list, [])

    def test_available_dataset_names_no_readers(self):
        """Test the available dataset names without a reader."""
        from satpy import Scene
        scene = Scene()
        self.assertRaises(
            KeyError, scene.available_dataset_names, reader_name='fake')
        name_list = scene.available_dataset_names()
        self.assertListEqual(name_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_names(composites=True)
        self.assertListEqual(name_list, [])

    def test_available_composites_no_datasets(self):
        """Test the available composites with no datasets."""
        from satpy import Scene
        scene = Scene()
        id_list = scene.available_composite_ids(available_datasets=[])
        self.assertListEqual(id_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        id_list = scene.available_composite_names(available_datasets=[])
        self.assertListEqual(id_list, [])

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_all_datasets_one_reader(self, cri, cl):
        """Test all datasets for one reader."""
        from satpy import Scene
        from satpy.tests.utils import FakeReader, test_composites
        r = FakeReader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': r}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = Scene(filenames=['bla'],
                      base_dir='bli',
                      reader='fake_reader')
        # patch the cpl
        scene.cpl.compositors = comps
        scene.cpl.modifiers = mods
        id_list = scene.all_dataset_ids()
        self.assertEqual(len(id_list), len(r.all_ids))
        id_list = scene.all_dataset_ids(composites=True)
        self.assertEqual(len(id_list),
                         len(r.all_ids) + len(scene.cpl.compositors['fake_sensor'].keys()))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_all_datasets_multiple_reader(self, cri, cl):
        """Test all datasets for multiple readers."""
        from satpy import Scene
        from satpy.tests.utils import FakeReader, test_composites
        r = FakeReader('fake_reader', 'fake_sensor', datasets=['ds1'])
        r2 = FakeReader(
            'fake_reader2', 'fake_sensor2', datasets=['ds2'])
        cri.return_value = {'fake_reader': r, 'fake_reader2': r2}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = Scene(filenames=['bla'],
                      base_dir='bli',
                      reader='fake_reader')
        # patch the cpl
        scene.cpl.compositors = comps
        scene.cpl.modifiers = mods
        id_list = scene.all_dataset_ids()
        self.assertEqual(len(id_list), 2)
        id_list = scene.all_dataset_ids(composites=True)
        self.assertEqual(len(id_list),
                         2 + len(scene.cpl.compositors['fake_sensor'].keys()))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_available_datasets_one_reader(self, cri, cl):
        """Test the available datasets for one reader."""
        from satpy import Scene
        from satpy.tests.utils import FakeReader, test_composites
        r = FakeReader('fake_reader', 'fake_sensor', datasets=['ds1'])
        cri.return_value = {'fake_reader': r}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = Scene(filenames=['bla'],
                      base_dir='bli',
                      reader='fake_reader')
        # patch the cpl
        scene.cpl.compositors = comps
        scene.cpl.modifiers = mods
        id_list = scene.available_dataset_ids()
        self.assertEqual(len(id_list), 1)
        id_list = scene.available_dataset_ids(composites=True)
        # ds1, comp1, comp14, comp16, static_image
        self.assertEqual(len(id_list), 5)

    def test_available_composite_ids_bad_available(self):
        """Test the available composite ids."""
        from satpy import Scene
        scn = Scene()
        self.assertRaises(ValueError, scn.available_composite_ids,
                          available_datasets=['bad'])

    def test_available_composite_names_bad_available(self):
        """Test the available composite names."""
        from satpy import Scene
        scn = Scene()
        self.assertRaises(
            ValueError, scn.available_composite_names, available_datasets=['bad'])


class TestSceneLoading(unittest.TestCase):
    """Test the Scene objects `.load` method."""

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_no_exist(self, cri, cl):
        """Test loading a dataset that doesn't exist."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        self.assertRaises(KeyError, scene.load, [
                          'im_a_dataset_that_doesnt_exist'])

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_no_exist2(self, cri, cl):
        """Test loading a dataset that doesn't exist then another load."""
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID, Scene
        r = FakeReader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': r}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = Scene(filenames=['bla'],
                      base_dir='bli',
                      reader='fake_reader')
        scene.load(['ds9_fail_load'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 0)
        r.load.assert_called_once_with(
            set([DatasetID(name='ds9_fail_load', wavelength=(1.0, 1.1, 1.2))]))

        scene.load(['ds1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(r.load.call_count, 2)
        # most recent call should have only been ds1
        r.load.assert_called_with(set([DatasetID(name='ds1')]))
        self.assertEqual(len(loaded_ids), 1)

    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds1_no_comps(self, cri):
        """Test loading one dataset with no loaded compositors."""
        import satpy.scene
        from satpy.tests.utils import FakeReader
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='ds1')))

    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds1_load_twice(self, cri):
        """Test loading one dataset with no loaded compositors."""
        import satpy.scene
        from satpy.tests.utils import FakeReader
        from satpy import DatasetID
        r = FakeReader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': r}
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='ds1')))

        with mock.patch.object(r, 'load') as m:
            scene.load(['ds1'])
            loaded_ids = list(scene.datasets.keys())
            self.assertEqual(len(loaded_ids), 1)
            self.assertTupleEqual(
                tuple(loaded_ids[0]), tuple(DatasetID(name='ds1')))
            self.assertFalse(
                m.called, "Reader.load was called again when loading something that's already loaded")

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds1_unknown_modifier(self, cri, cl):
        """Test loading one dataset with no loaded compositors."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        self.assertRaises(KeyError, scene.load,
                          [DatasetID(name='ds1', modifiers=('_fake_bad_mod_',))])

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds4_cal(self, cri, cl):
        """Test loading a dataset that has two calibration variations."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds4'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].calibration, 'reflectance')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds5_best_resolution(self, cri, cl):
        """Test loading a dataset has multiple resolutions available."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds5'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].name, 'ds5')
        self.assertEqual(loaded_ids[0].resolution, 250)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds5_multiple_resolution(self, cri, cl):
        """Test loading a dataset has multiple resolutions available with different resolutions."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds5'], resolution=1000)
        scene.load(['ds5'], resolution=500)
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        self.assertEqual(loaded_ids[0].name, 'ds5')
        self.assertEqual(loaded_ids[0].resolution, 500)
        self.assertEqual(loaded_ids[1].name, 'ds5')
        self.assertEqual(loaded_ids[1].resolution, 1000)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds5_missing_best_resolution(self, cri, cl):
        """Test loading a dataset that has multiple resolutions but the best isn't available."""
        import satpy.scene
        from satpy import DatasetID
        from satpy.tests.utils import FakeReader, test_composites

        # only the 500m is available
        available_datasets = [DatasetID('ds5', resolution=500)]
        cri.return_value = {
            'fake_reader': FakeReader(
                'fake_reader', 'fake_sensor', datasets=['ds5'],
                available_datasets=available_datasets),
        }
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds5'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].name, 'ds5')
        self.assertEqual(loaded_ids[0].resolution, 500)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds6_wl(self, cri, cl):
        """Test loading a dataset by wavelength."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load([0.22])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].name, 'ds6')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds9_fail_load(self, cri, cl):
        """Test loading a dataset that will fail during load."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds9_fail_load'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 0)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp1(self, cri, cl):
        """Test loading a composite with one required prereq."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp1')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp4(self, cri, cl):
        """Test loading a composite that depends on a composite."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp4'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp4')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_multiple_resolutions(self, cri, cl):
        """Test loading a dataset has multiple resolutions available with different resolutions."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        comp25 = DatasetID(name='comp25', resolution=1000)
        scene[comp25] = 'bla'
        scene.load(['comp25'], resolution=500)

        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        self.assertEqual(loaded_ids[0].name, 'comp25')
        self.assertEqual(loaded_ids[0].resolution, 500)
        self.assertEqual(loaded_ids[1].name, 'comp25')
        self.assertEqual(loaded_ids[1].resolution, 1000)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_same_subcomposite(self, cri, cl):
        """Test loading a composite and one of it's subcomposites at the same time."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')

        scene.load(['comp24', 'comp25'], resolution=500)
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        self.assertEqual(loaded_ids[0].name, 'comp24')
        self.assertEqual(loaded_ids[0].resolution, 500)
        self.assertEqual(loaded_ids[1].name, 'comp25')
        self.assertEqual(loaded_ids[1].resolution, 500)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp5(self, cri, cl):
        """Test loading a composite that has an optional prerequisite."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp5'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp5')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp6(self, cri, cl):
        """Test loading a composite that has an optional composite prerequisite."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp6'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp6')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp8(self, cri, cl):
        """Test loading a composite that has a non-existent prereq."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        self.assertRaises(KeyError, scene.load, ['comp8'])

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp9(self, cri, cl):
        """Test loading a composite that has a non-existent optional prereq."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp9'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp9')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp10(self, cri, cl):
        """Test loading a composite that depends on a modified dataset."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp10'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp10')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp11(self, cri, cl):
        """Test loading a composite that depends all wavelengths."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp11'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp11')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp12(self, cri, cl):
        """Test loading a composite that depends all wavelengths that get modified."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp12'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp12')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp13(self, cri, cl):
        """Test loading a composite that depends on a modified dataset where the resolution changes."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp13'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp13')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp14(self, cri, cl):
        """Test loading a composite that updates the DatasetID during generation."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp14'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].name, 'comp14')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp15(self, cri, cl):
        """Test loading a composite whose prerequisites can't be loaded.

        Note that the prereq exists in the reader, but fails in loading.

        """
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp15'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 0)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp16(self, cri, cl):
        """Test loading a composite whose opt prereq can't be loaded.

        Note that the prereq exists in the reader, but fails in loading

        """
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp16'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].name, 'comp16')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp17(self, cri, cl):
        """Test loading a composite that depends on a composite that won't load."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp17'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 0)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp18(self, cri, cl):
        """Test loading a composite that depends on a incompatible area modified dataset."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp18'])
        loaded_ids = list(scene.datasets.keys())
        # depends on:
        #   ds3
        #   ds4 (mod1, mod3)
        #   ds5 (mod1, incomp_areas)
        # We should end up with ds3, ds4 (mod1, mod3), ds5 (mod1), and ds1
        # for the incomp_areas modifier
        self.assertEqual(len(loaded_ids), 4)  # the 1 dependencies
        self.assertIn('ds3', scene.datasets)
        self.assertIn(DatasetID(name='ds4', calibration='reflectance',
                                modifiers=('mod1', 'mod3')),
                      scene.datasets)
        self.assertIn(DatasetID(name='ds5', resolution=250,
                                modifiers=('mod1',)),
                      scene.datasets)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp18_2(self, cri, cl):
        """Test loading a composite that depends on a incompatible area modified dataset.

        Specifically a modified dataset where the modifier has optional
        dependencies.

        """
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp18_2'])
        loaded_ids = list(scene.datasets.keys())
        # depends on:
        #   ds3
        #   ds4 (mod1, mod3)
        #   ds5 (mod1, incomp_areas_opt)
        # We should end up with ds3, ds4 (mod1, mod3), ds5 (mod1), and ds1
        # and ds2 for the incomp_areas_opt modifier
        self.assertEqual(len(loaded_ids), 5)  # the 1 dependencies
        self.assertIn('ds3', scene.datasets)
        self.assertIn('ds2', scene.datasets)
        self.assertIn(DatasetID(name='ds4', calibration='reflectance',
                                modifiers=('mod1', 'mod3')),
                      scene.datasets)
        self.assertIn(DatasetID(name='ds5', resolution=250,
                                modifiers=('mod1',)),
                      scene.datasets)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp19(self, cri, cl):
        """Test loading a composite that shares a dep with a dependency.

        More importantly test that loading a dependency that depends on
        the same dependency as this composite (a sibling dependency) and
        that sibling dependency includes a modifier. This test makes sure
        that the Node in the dependency tree is the exact same node.

        """
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')

        # Check dependency tree nodes
        # initialize the dep tree without loading the data
        scene.dep_tree.find_dependencies({'comp19'})
        this_node = scene.dep_tree['comp19']
        shared_dep_id = DatasetID(name='ds5', modifiers=('res_change',))
        shared_dep_expected_node = scene.dep_tree[shared_dep_id]
        # get the node for the first dep in the prereqs list of the
        # comp13 node
        shared_dep_node = scene.dep_tree['comp13'].data[1][0]
        shared_dep_node2 = this_node.data[1][0]
        self.assertIs(shared_dep_expected_node, shared_dep_node)
        self.assertIs(shared_dep_expected_node, shared_dep_node2)

        # it is fine that an optional prereq doesn't exist
        scene.load(['comp19'])

        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp19')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_multiple_comps(self, cri, cl):
        """Test loading multiple composites."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6',
                    'comp7', 'comp9', 'comp10'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 9)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_multiple_comps_separate(self, cri, cl):
        """Test loading multiple composites, one at a time."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp10'])
        scene.load(['comp9'])
        scene.load(['comp7'])
        scene.load(['comp6'])
        scene.load(['comp5'])
        scene.load(['comp4'])
        scene.load(['comp3'])
        scene.load(['comp2'])
        scene.load(['comp1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 9)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_modified(self, cri, cl):
        """Test loading a modified dataset."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load([DatasetID(name='ds1', modifiers=('mod1', 'mod2'))])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(loaded_ids[0].modifiers, ('mod1', 'mod2'))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_multiple_modified(self, cri, cl):
        """Test loading multiple modified datasets."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load([
            DatasetID(name='ds1', modifiers=('mod1', 'mod2')),
            DatasetID(name='ds2', modifiers=('mod2', 'mod1')),
        ])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        for i in loaded_ids:
            if i.name == 'ds1':
                self.assertTupleEqual(i.modifiers, ('mod1', 'mod2'))
            else:
                self.assertEqual(i.name, 'ds2')
                self.assertTupleEqual(i.modifiers, ('mod2', 'mod1'))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_dataset_after_composite(self, cri, cl):
        """Test load composite followed by other datasets."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        r = FakeReader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': r}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp3'])
        self.assertEqual(r.load.call_count, 1)
        scene.load(['ds1'])
        self.assertEqual(r.load.call_count, 2)
        scene.load(['ds1'])
        # we should only load from the file twice
        self.assertEqual(r.load.call_count, 2)
        # we should only generate the composite once
        self.assertEqual(comps['fake_sensor'][
                         'comp3'].side_effect.call_count, 1)
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_dataset_after_composite2(self, cri, cl):
        """Test load complex composite followed by other datasets."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        r = FakeReader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': r}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp10'])
        self.assertEqual(r.load.call_count, 1)
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        with mock.patch.object(scene, '_read_composites', wraps=scene._read_composites) as m:
            scene.load(['ds1'])
            self.assertEqual(r.load.call_count, 2)
            loaded_ids = list(scene.datasets.keys())
            self.assertEqual(len(loaded_ids), 2)
            # this is the unmodified ds1
            self.assertIn(DatasetID(name='ds1'), loaded_ids)
            # m.assert_called_once_with(set([scene.dep_tree['ds1']]))
            m.assert_called_once_with(set())
        with mock.patch.object(scene, '_read_composites', wraps=scene._read_composites) as m:
            scene.load(['ds1'])
            self.assertEqual(r.load.call_count, 2)
            loaded_ids = list(scene.datasets.keys())
            self.assertEqual(len(loaded_ids), 2)
            # this is the unmodified ds1
            self.assertIn(DatasetID(name='ds1'), loaded_ids)
            m.assert_called_once_with(set())
        # we should only generate the composite once
        self.assertEqual(comps['fake_sensor'][
                         'comp10'].side_effect.call_count, 1)
        # Create the modded ds1 at comp10, then load the numodified version
        # again
        self.assertEqual(comps['fake_sensor']['ds1']._call_mock.call_count, 1)
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp20(self, cri, cl):
        """Test loading composite with optional modifier dependencies."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp20'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp20')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp21(self, cri, cl):
        """Test loading composite with bad optional modifier dependencies."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp21'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp21')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp22(self, cri, cl):
        """Test loading composite with only optional modifier dependencies."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp22'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 1)
        self.assertTupleEqual(
            tuple(loaded_ids[0]), tuple(DatasetID(name='comp22')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_no_generate_comp10(self, cri, cl):
        """Test generating a composite after loading."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp10'], generate=False)
        self.assertTrue(any(ds_id == 'comp10' for ds_id in scene.wishlist))
        self.assertNotIn('comp10', scene.datasets)
        # two dependencies should have been loaded
        self.assertEqual(len(scene.datasets), 2)
        self.assertEqual(len(scene.missing_datasets), 1)

        scene.generate_composites()
        self.assertTrue(any(ds_id == 'comp10' for ds_id in scene.wishlist))
        self.assertIn('comp10', scene.datasets)
        self.assertEqual(len(scene.missing_datasets), 0)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_modified_with_wl_dep(self, cri, cl):
        """Test modifying a dataset with a modifier with modified deps.

        More importantly test that loading the modifiers dependency at the
        same time as the original modified dataset that the dependency tree
        nodes are unique and that DatasetIDs.

        """
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')

        # Check dependency tree nodes
        # initialize the dep tree without loading the data
        ds1_mod_id = DatasetID(name='ds1', modifiers=('mod_wl',))
        ds3_mod_id = DatasetID(name='ds3', modifiers=('mod_wl',))
        scene.dep_tree.find_dependencies({ds1_mod_id, ds3_mod_id})
        ds1_mod_node = scene.dep_tree[ds1_mod_id]
        ds3_mod_node = scene.dep_tree[ds3_mod_id]
        ds1_mod_dep_node = ds1_mod_node.data[1][1]
        ds3_mod_dep_node = ds3_mod_node.data[1][1]
        # mod_wl depends on the this node:
        ds6_modded_node = scene.dep_tree[DatasetID(name='ds6', modifiers=('mod1',))]
        # this dep should be full qualified with name and wavelength
        self.assertIsNotNone(ds6_modded_node.name.name)
        self.assertIsNotNone(ds6_modded_node.name.wavelength)
        self.assertEqual(len(ds6_modded_node.name.wavelength), 3)
        # the node should be shared between everything that uses it
        self.assertIs(ds1_mod_dep_node, ds3_mod_dep_node)
        self.assertIs(ds1_mod_dep_node, ds6_modded_node)

        # it is fine that an optional prereq doesn't exist
        scene.load([ds1_mod_id, ds3_mod_id])

        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        self.assertIn(ds1_mod_id, scene.datasets)
        self.assertIn(ds3_mod_id, scene.datasets)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors', autospec=True)
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp11_and_23(self, cri, cl):
        """Test loading two composites that depend on similar wavelengths."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID, DatasetDict
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')

        def _test(self, sensor_names):
            if not self.compositors:
                self.compositors = comps
                self.modifiers = mods
            new_comps = {}
            new_mods = {}
            for sn in sensor_names:
                new_comps[sn] = DatasetDict(
                    self.compositors[sn].copy())
                new_mods[sn] = self.modifiers[sn].copy()
            return new_comps, new_mods

        cl.side_effect = _test
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')
        # mock the available comps/mods in the compositor loader
        avail_comps = scene.available_composite_ids()
        self.assertIn(DatasetID(name='comp11'), avail_comps)
        self.assertIn(DatasetID(name='comp23'), avail_comps)
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp11', 'comp23'])
        comp11_node = scene.dep_tree['comp11']
        comp23_node = scene.dep_tree['comp23']
        self.assertEqual(comp11_node.data[1][-1].name.name, 'ds10')
        self.assertEqual(comp23_node.data[1][0].name.name, 'ds8')
        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        self.assertIn('comp11', scene.datasets)
        self.assertIn('comp23', scene.datasets)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors', autospec=True)
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_too_many(self, cri, cl):
        """Test dependency tree if too many reader keys match."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        datasets = [DatasetID(name='duplicate1', wavelength=(0.1, 0.2, 0.3)),
                    DatasetID(name='duplicate2', wavelength=(0.1, 0.2, 0.3))]
        reader = FakeReader('fake_reader', 'fake_sensor', datasets=datasets,
                            filter_datasets=False)
        cri.return_value = {'fake_reader': reader}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'], base_dir='bli', reader='fake_reader')
        # mock the available comps/mods in the compositor loader
        avail_comps = scene.available_composite_ids()
        self.assertEqual(len(avail_comps), 0)
        self.assertRaises(KeyError, scene.load, [0.21])

    @mock.patch('satpy.composites.CompositorLoader.load_compositors', autospec=True)
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_available_comps_no_deps(self, cri, cl):
        """Test Scene available composites when composites don't have a dependency."""
        from satpy.tests.utils import FakeReader, test_composites
        import satpy.scene
        from satpy.readers import DatasetDict
        from satpy import DatasetID

        def _test(self, sensor_names):
            if not self.compositors:
                self.compositors = comps
                self.modifiers = mods
            new_comps = {}
            new_mods = {}
            for sn in sensor_names:
                new_comps[sn] = DatasetDict(
                    self.compositors[sn].copy())
                new_mods[sn] = self.modifiers[sn].copy()
            return new_comps, new_mods

        # fancy magic to make sure the CompositorLoader thinks it has comps
        cl.side_effect = _test

        reader = FakeReader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': reader}
        comps, mods = test_composites('fake_sensor')
        scene = satpy.scene.Scene(filenames=['bla'], base_dir='bli', reader='fake_reader')
        all_comp_ids = scene.available_composite_ids()
        self.assertIn(DatasetID(name='static_image'), all_comp_ids)
        available_comp_ids = scene.available_composite_ids()
        self.assertIn(DatasetID(name='static_image'), available_comp_ids)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_empty_node_copy(self, cri, cl):
        """Test copying a dependency tree while preserving the empty node identical."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')

        # Check dependency tree nodes
        # initialize the dep tree without loading the data
        scene.dep_tree.find_dependencies({'comp19'})
        sc2 = scene.copy()
        self.assertIs(scene.dep_tree.children[0].children[0].children[1], scene.dep_tree.empty_node)
        self.assertIs(scene.dep_tree.children[0].children[0].children[1], sc2.dep_tree.empty_node)
        self.assertIs(sc2.dep_tree.children[0].children[0].children[1], scene.dep_tree.empty_node)
        self.assertIs(sc2.dep_tree.children[0].children[0].children[1], sc2.dep_tree.empty_node)


class TestSceneResampling(unittest.TestCase):
    """Test resampling a Scene to another Scene object."""

    def _fake_resample_dataset(self, dataset, dest_area, **kwargs):
        """Return copy of dataset pretending it was resampled."""
        return dataset.copy()

    @mock.patch('satpy.scene.resample_dataset')
    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_resample_scene_copy(self, cri, cl, rs):
        """Test that the Scene is properly copied during resampling.

        The Scene that is created as a copy of the original Scene should not
        be able to affect the original Scene object.

        """
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        rs.side_effect = self._fake_resample_dataset

        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')

        scene.load(['comp19'])
        new_scene = scene.resample(area_def)
        new_scene['new_ds'] = new_scene['comp19'].copy()

        scene.load(['ds1'])

        comp19_node = scene.dep_tree['comp19']
        ds5_mod_id = DatasetID(name='ds5', modifiers=('res_change',))
        ds5_node = scene.dep_tree[ds5_mod_id]
        comp13_node = scene.dep_tree['comp13']

        self.assertIs(comp13_node.data[1][0], comp19_node.data[1][0])
        self.assertIs(comp13_node.data[1][0], ds5_node)
        self.assertRaises(KeyError, scene.dep_tree.__getitem__, 'new_ds')

        loaded_ids = list(scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp19')))
        self.assertTupleEqual(tuple(loaded_ids[1]), tuple(DatasetID(name='ds1')))

        loaded_ids = list(new_scene.datasets.keys())
        self.assertEqual(len(loaded_ids), 2)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp19')))
        self.assertTupleEqual(tuple(loaded_ids[1]), tuple(DatasetID(name='new_ds')))

    @mock.patch('satpy.scene.resample_dataset')
    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_resample_reduce_data_toggle(self, cri, cl, rs):
        """Test that the Scene can be reduced or not reduced during resampling."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from satpy import DatasetID
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        import dask.array as da
        import xarray as xr
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        rs.side_effect = self._fake_resample_dataset

        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        target_area = AreaDefinition('test', 'test', 'test', proj_dict, 4, 4, (-1000., -1500., 1000., 1500.))
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        area_def.get_area_slices = mock.MagicMock()
        get_area_slices = area_def.get_area_slices
        get_area_slices.return_value = (slice(0, 3, None), slice(0, 3, None))
        area_def_big = AreaDefinition('test', 'test', 'test', proj_dict, 10, 10, (-1000., -1500., 1000., 1500.))
        area_def_big.get_area_slices = mock.MagicMock()
        get_area_slices_big = area_def_big.get_area_slices
        get_area_slices_big.return_value = (slice(0, 6, None), slice(0, 6, None))

        # Test that data reduction can be disabled
        scene = satpy.scene.Scene(filenames=['bla'], base_dir='bli', reader='fake_reader')
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
            DatasetID.from_dict(scene['comp19'].attrs),
            DatasetID.from_dict(scene['comp19_big'].attrs),
            DatasetID.from_dict(scene['comp19_copy'].attrs),
        ]
        with mock.patch('satpy.scene.Scene._slice_data') as slice_data, \
                mock.patch('satpy.dataset.dataset_walker') as ds_walker:
            ds_walker.return_value = test_order
            slice_data.side_effect = orig_slice_data
            scene.resample(target_area, reduce_data=False)
            self.assertFalse(slice_data.called)
            self.assertFalse(get_area_slices.called)
            scene.resample(target_area)
            self.assertTrue(slice_data.called_once)
            self.assertTrue(get_area_slices.called_once)
            scene.resample(target_area, reduce_data=True)
            # 2 times for each dataset
            # once for default (reduce_data=True)
            # once for kwarg forced to `True`
            self.assertEqual(slice_data.call_count, 2 * 3)
            self.assertTrue(get_area_slices.called_once)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_resample_ancillary(self, cri, cl):
        """Test that the Scene reducing data does not affect final output."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)

        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        scene = satpy.scene.Scene(filenames=['bla'], base_dir='bli', reader='fake_reader')

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
        self.assertIs(new_scene['comp20'], new_scene['comp19'].attrs['ancillary_variables'][0])

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_resample_reduce_data(self, cri, cl):
        """Test that the Scene reducing data does not affect final output."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)

        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        area_def = AreaDefinition('test', 'test', 'test', proj_dict, 5, 5, (-1000., -1500., 1000., 1500.))
        scene = satpy.scene.Scene(filenames=['bla'], base_dir='bli', reader='fake_reader')

        scene.load(['comp19'])
        scene['comp19'].attrs['area'] = area_def
        dst_area = AreaDefinition('dst', 'dst', 'dst',
                                  proj_dict,
                                  2, 2,
                                  (-1000., -1500., 0., 0.),
                                  )
        new_scene1 = scene.resample(dst_area, reduce_data=False)
        new_scene2 = scene.resample(dst_area)
        new_scene3 = scene.resample(dst_area, reduce_data=True)
        self.assertTupleEqual(new_scene1['comp19'].shape, (2, 2, 3))
        self.assertTupleEqual(new_scene2['comp19'].shape, (2, 2, 3))
        self.assertTupleEqual(new_scene3['comp19'].shape, (2, 2, 3))

    @mock.patch('satpy.scene.resample_dataset')
    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_no_generate_comp10(self, cri, cl, rs):
        """Test generating a composite after loading."""
        import satpy.scene
        from satpy.tests.utils import FakeReader, test_composites
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
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
        cri.return_value = {'fake_reader': FakeReader(
            'fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames=['bla'],
                                  base_dir='bli',
                                  reader='fake_reader')

        # it is fine that an optional prereq doesn't exist
        scene.load(['comp10'], generate=False)
        self.assertTrue(any(ds_id == 'comp10' for ds_id in scene.wishlist))
        self.assertNotIn('comp10', scene.datasets)
        # two dependencies should have been loaded
        self.assertEqual(len(scene.datasets), 2)
        self.assertEqual(len(scene.missing_datasets), 1)

        new_scn = scene.resample(area_def, generate=False)
        self.assertNotIn('comp10', scene.datasets)
        # two dependencies should have been loaded
        self.assertEqual(len(scene.datasets), 2)
        self.assertEqual(len(scene.missing_datasets), 1)

        new_scn.generate_composites()
        self.assertTrue(any(ds_id == 'comp10' for ds_id in new_scn.wishlist))
        self.assertIn('comp10', new_scn.datasets)
        self.assertEqual(len(new_scn.missing_datasets), 0)

        # try generating them right away
        new_scn = scene.resample(area_def)
        self.assertTrue(any(ds_id == 'comp10' for ds_id in new_scn.wishlist))
        self.assertIn('comp10', new_scn.datasets)
        self.assertEqual(len(new_scn.missing_datasets), 0)


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
        self.assertTrue(os.path.isfile(
            os.path.join(self.base_dir, 'test_20180101_000000.tif')))

    def test_save_datasets_by_ext(self):
        """Save a dataset using 'save_datasets' with 'filename'."""
        from satpy.scene import Scene
        from satpy.tests.utils import spy_decorator
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
        self.assertTrue(os.path.isfile(
            os.path.join(self.base_dir, 'test.png')))

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
        self.assertRaises(ValueError,
                          scn.save_datasets,
                          writer='_bad_writer_',
                          base_dir=self.base_dir)

    def test_save_datasets_missing_wishlist(self):
        """Calling 'save_datasets' with no valid datasets."""
        from satpy.scene import Scene, DatasetID
        scn = Scene()
        scn.wishlist.add(DatasetID(name='true_color'))
        self.assertRaises(RuntimeError,
                          scn.save_datasets,
                          writer='geotiff',
                          base_dir=self.base_dir)
        self.assertRaises(KeyError,
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
        self.assertTrue(os.path.isfile(
            os.path.join(self.base_dir, 'test_20180101_000000.tif')))


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
        self.assertIsNotNone(gv_obj)

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
        self.assertIsNotNone(gv_obj)


def suite():
    """Test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestScene))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSceneLoading))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSceneResampling))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSceneSaving))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSceneConversions))

    return mysuite


if __name__ == "__main__":
    unittest.main()
