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
                                            reader='blo',
                                            reader_kwargs=None)
                self.assertDictContainsSubset(test_mda, scene.info)
                self.assertDictContainsSubset(test_mda_2, scene.info)

    def test_init_with_sensor(self):
        import satpy.scene
        from satpy.tests.utils import create_fake_reader
        with mock.patch('satpy.scene.Scene.create_reader_instances') as cri:
            cri.return_value = {
                'fake_reader': create_fake_reader('fake_reader', sensor_name='fake_sensor'),
            }
            scene = satpy.scene.Scene(filenames='bla',
                                      base_dir='bli',
                                      sensor='fake_sensor')
            self.assertIsInstance(scene.info['sensor'], set)
            scene = satpy.scene.Scene(filenames='bla',
                                      base_dir='bli',
                                      sensor=['fake_sensor'])
            self.assertIsInstance(scene.info['sensor'], set)

    def test_start_end_times(self):
        import satpy.scene
        from satpy.tests.utils import create_fake_reader
        from datetime import datetime
        with mock.patch('satpy.scene.Scene.create_reader_instances') as cri:
            r = create_fake_reader('fake_reader',
                                   start_time=datetime(2017, 1, 1, 0, 0, 0),
                                   end_time=datetime(2017, 1, 1, 1, 0, 0),
                                   )
            cri.return_value = {'fake_reader': r}
            scene = satpy.scene.Scene(filenames='bla',
                                      base_dir='bli',
                                      sensor='fake_sensor')
            self.assertEqual(scene.start_time, r.start_time)
            self.assertEqual(scene.end_time, r.end_time)

    def test_init_alone(self):
        from satpy.scene import Scene
        from satpy.config import PACKAGE_CONFIG_PATH
        scn = Scene()
        self.assertEqual(scn.ppp_config_dir, PACKAGE_CONFIG_PATH)
        self.assertFalse(scn.readers, 'Empty scene should not load any readers')

    def test_init_with_ppp_config_dir(self):
        from satpy.scene import Scene
        scn = Scene(ppp_config_dir="foo")
        self.assertEqual(scn.ppp_config_dir, 'foo')

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
                    sensor=set(),
                    filenames=filenames,
                    reader_kwargs=None,
                )

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
                    filenames=filenames,
                    reader_kwargs=None
                )

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
                    filenames=filenames,
                    reader_kwargs=None,
                )

    def test_create_reader_instances_with_reader(self):
        from satpy.scene import Scene
        reader = "foo"
        filenames = ["1", "2", "3"]
        sensors = set()
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
                    filenames=filenames,
                    reader_kwargs=None,
                )

    def test_iter(self):
        from satpy import Scene, Dataset
        import numpy as np
        scene = Scene()
        scene["1"] = Dataset(np.arange(5))
        scene["2"] = Dataset(np.arange(5))
        scene["3"] = Dataset(np.arange(5))
        for x in scene:
            self.assertIsInstance(x, Dataset)

    def test_iter_by_area_swath(self):
        from satpy import Scene, Dataset
        from pyresample.geometry import SwathDefinition
        import numpy as np
        scene = Scene()
        sd = SwathDefinition(lons=np.arange(5), lats=np.arange(5))
        scene["1"] = Dataset(np.arange(5), area=sd)
        scene["2"] = Dataset(np.arange(5), area=sd)
        scene["3"] = Dataset(np.arange(5))
        for area_obj, ds_list in scene.iter_by_area():
            ds_list_names = set(ds.name for ds in ds_list)
            if area_obj is sd:
                self.assertSetEqual(ds_list_names, {'1', '2'})
            else:
                self.assertIsNone(area_obj)
                self.assertSetEqual(ds_list_names, {'3'})

    def test_bad_setitem(self):
        from satpy import Scene, Dataset
        import numpy as np
        scene = Scene()
        self.assertRaises(ValueError, scene.__setitem__, '1', np.arange(5))

    def test_setitem(self):
        from satpy import Scene, Dataset
        import numpy as np
        scene = Scene()
        scene["1"] = ds1 = Dataset(np.arange(5))
        self.assertSetEqual(set(scene.datasets.keys()), {ds1.id})
        self.assertSetEqual(set(scene.wishlist), {ds1.id})

    def test_getitem(self):
        from satpy import Scene, Dataset
        import numpy as np
        scene = Scene()
        scene["1"] = ds1 = Dataset(np.arange(5))
        scene["2"] = ds2 = Dataset(np.arange(5))
        scene["3"] = ds3 = Dataset(np.arange(5))
        self.assertIs(scene['1'], ds1)
        self.assertIs(scene['2'], ds2)
        self.assertIs(scene['3'], ds3)
        self.assertRaises(KeyError, scene.__getitem__, '4')

    def test_contains(self):
        from satpy import Scene, Dataset
        import numpy as np
        scene = Scene()
        scene["1"] = ds1 = Dataset(np.arange(5), wavelength=(0.1, 0.2, 0.3))
        self.assertTrue('1' in scene)
        self.assertTrue(0.15 in scene)
        self.assertFalse('2' in scene)
        self.assertFalse(0.31 in scene)

    def test_delitem(self):
        from satpy import Scene, Dataset
        import numpy as np
        scene = Scene()
        scene["1"] = ds1 = Dataset(np.arange(5), wavelength=(0.1, 0.2, 0.3))
        scene["2"] = ds2 = Dataset(np.arange(5), wavelength=(0.4, 0.5, 0.6))
        scene["3"] = ds3 = Dataset(np.arange(5), wavelength=(0.7, 0.8, 0.9))
        del scene['1']
        del scene['3']
        del scene[0.45]
        self.assertEquals(len(scene.wishlist), 0)
        self.assertEquals(len(scene.datasets.keys()), 0)
        self.assertRaises(KeyError, scene.__delitem__, 0.2)

    def test_all_datasets_no_readers(self):
        from satpy import Scene
        scene = Scene()
        self.assertRaises(KeyError, scene.all_dataset_ids, reader_name='fake')
        id_list = scene.all_dataset_ids()
        self.assertListEqual(id_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        id_list = scene.all_dataset_ids(composites=True)
        self.assertListEqual(id_list, [])

    def test_all_dataset_names_no_readers(self):
        from satpy import Scene
        scene = Scene()
        self.assertRaises(KeyError, scene.all_dataset_names, reader_name='fake')
        name_list = scene.all_dataset_names()
        self.assertListEqual(name_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.all_dataset_names(composites=True)
        self.assertListEqual(name_list, [])

    def test_available_dataset_no_readers(self):
        from satpy import Scene
        scene = Scene()
        self.assertRaises(KeyError, scene.available_dataset_ids, reader_name='fake')
        name_list = scene.available_dataset_ids()
        self.assertListEqual(name_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_ids(composites=True)
        self.assertListEqual(name_list, [])

    def test_available_dataset_names_no_readers(self):
        from satpy import Scene
        scene = Scene()
        self.assertRaises(KeyError, scene.available_dataset_names, reader_name='fake')
        name_list = scene.available_dataset_names()
        self.assertListEqual(name_list, [])
        # no sensors are loaded so we shouldn't get any comps either
        name_list = scene.available_dataset_names(composites=True)
        self.assertListEqual(name_list, [])

    def test_available_composites_no_datasets(self):
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
        from satpy import Scene
        from satpy.tests.utils import create_fake_reader, test_composites
        r = create_fake_reader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': r}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = Scene(filenames='bla',
                      base_dir='bli',
                      reader='fake_reader')
        # patch the cpl
        scene.cpl.compositors = comps
        scene.cpl.modifiers = mods
        id_list = scene.all_dataset_ids()
        self.assertEqual(len(id_list), len(r.datasets))
        id_list = scene.all_dataset_ids(composites=True)
        self.assertEqual(len(id_list),
                         len(r.datasets) + len(scene.cpl.compositors['fake_sensor'].keys()))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_all_datasets_multiple_reader(self, cri, cl):
        from satpy import Scene
        from satpy.tests.utils import create_fake_reader, test_composites
        r = create_fake_reader('fake_reader', 'fake_sensor', datasets=['ds1'])
        r2 = create_fake_reader('fake_reader2', 'fake_sensor2', datasets=['ds2'])
        cri.return_value = {'fake_reader': r, 'fake_reader2': r2}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = Scene(filenames='bla',
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
        from satpy import Scene
        from satpy.tests.utils import create_fake_reader, test_composites
        r = create_fake_reader('fake_reader', 'fake_sensor', datasets=['ds1'])
        cri.return_value = {'fake_reader': r}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = Scene(filenames='bla',
                      base_dir='bli',
                      reader='fake_reader')
        # patch the cpl
        scene.cpl.compositors = comps
        scene.cpl.modifiers = mods
        id_list = scene.available_dataset_ids()
        self.assertEqual(len(id_list), 1)
        id_list = scene.available_dataset_ids(composites=True)
        self.assertEqual(len(id_list), 4)

    def test_available_composite_ids_bad_available(self):
        from satpy import Scene
        scn = Scene()
        self.assertRaises(ValueError, scn.available_composite_ids, available_datasets=['bad'])

    def test_available_composite_names_bad_available(self):
        from satpy import Scene
        scn = Scene()
        self.assertRaises(ValueError, scn.available_composite_names, available_datasets=['bad'])


class TestSceneLoading(unittest.TestCase):
    """Test the Scene objects `.load` method
    """
    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_no_exist(self, cri, cl):
        """Test loading a dataset that doesn't exist"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        self.assertRaises(KeyError, scene.load, ['im_a_dataset_that_doesnt_exist'])

    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds1_no_comps(self, cri):
        """Test loading one dataset with no loaded compositors"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='ds1')))

    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds1_load_twice(self, cri):
        """Test loading one dataset with no loaded compositors"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader
        from satpy import DatasetID
        r = create_fake_reader('fake_reader', 'fake_sensor')
        cri.return_value = {'fake_reader': r}
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='ds1')))

        with mock.patch.object(r, 'load') as m:
            scene.load(['ds1'])
            loaded_ids = list(scene.datasets.keys())
            self.assertEquals(len(loaded_ids), 1)
            self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='ds1')))
            self.assertFalse(m.called, "Reader.load was called again when loading something that's already loaded")

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds4_cal(self, cri, cl):
        """Test loading a dataset that has two calibration variations"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds4'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertEquals(loaded_ids[0].calibration, 'reflectance')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds6_wl(self, cri, cl):
        """Test loading a dataset by wavelength"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load([0.22])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertEquals(loaded_ids[0].name, 'ds6')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_ds9_fail_load(self, cri, cl):
        """Test loading a dataset that will fail during load"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds9_fail_load'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 0)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp1(self, cri, cl):
        """Test loading a composite with one required prereq"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp1'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp1')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp4(self, cri, cl):
        """Test loading a composite that depends on a composite"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp4'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp4')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp5(self, cri, cl):
        """Test loading a composite that has an optional prerequisite"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp5'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp5')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp6(self, cri, cl):
        """Test loading a composite that has an optional composite prerequisite"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp6'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp6')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp8(self, cri, cl):
        """Test loading a composite that has a non-existent prereq"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        self.assertRaises(KeyError, scene.load, ['comp8'])

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp9(self, cri, cl):
        """Test loading a composite that has a non-existent optional prereq"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp9'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp9')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp10(self, cri, cl):
        """Test loading a composite that depends on a modified dataset"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp10'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp10')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp11(self, cri, cl):
        """Test loading a composite that depends all wavelengths"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp11'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp11')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp12(self, cri, cl):
        """Test loading a composite that depends all wavelengths that get modified"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp12'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp12')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp13(self, cri, cl):
        """Test loading a composite that depends on a modified dataset where the resolution changes"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp13'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='comp13')))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp14(self, cri, cl):
        """Test loading a composite that updates the DatasetID during generation"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp14'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].name, 'comp14')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp15(self, cri, cl):
        """Test loading a composite whose prerequisites can't be loaded

        Note that the prereq exists in the reader, but fails in loading.
        """
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp15'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 0)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp16(self, cri, cl):
        """Test loading a composite whose opt prereq can't be loaded

        Note that the prereq exists in the reader, but fails in loading
        """
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp16'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertEqual(loaded_ids[0].name, 'comp16')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp17(self, cri, cl):
        """Test loading a composite that depends on a composite that won't load
        """
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp17'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 0)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_comp18(self, cri, cl):
        """Test loading a composite that depends on a incompatible area modified dataset
        """
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        # it is fine that an optional prereq doesn't exist
        scene.load(['comp18'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)  # the 1 dependencies
        self.assertEquals(loaded_ids[0].name, 'ds1')

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_multiple_comps(self, cri, cl):
        """Test loading multiple composites"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6',
                    'comp7', 'comp9', 'comp10'])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 9)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_multiple_comps_separate(self, cri, cl):
        """Test loading multiple composites, one at a time"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
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
        self.assertEquals(len(loaded_ids), 9)

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_modified(self, cri, cl):
        """Test loading a modified dataset"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load([DatasetID(name='ds1', modifiers=('mod1', 'mod2'))])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(loaded_ids[0].modifiers, ('mod1', 'mod2'))

    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_load_multiple_modified(self, cri, cl):
        """Test loading multiple modified datasets"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load([
            DatasetID(name='ds1', modifiers=('mod1', 'mod2')),
            DatasetID(name='ds2', modifiers=('mod2', 'mod1')),
        ])
        loaded_ids = list(scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 2)
        for i in loaded_ids:
            if i.name == 'ds1':
                self.assertTupleEqual(i.modifiers, ('mod1', 'mod2'))
            else:
                self.assertEqual(i.name, 'ds2')
                self.assertTupleEqual(i.modifiers, ('mod2', 'mod1'))


class TestSceneResample(unittest.TestCase):
    """Test the `.resample` method of Scene

    Note this does not actually run the resampling algorithms. It only tests
    how the Scene handles dependencies and delayed composites surrounding
    resampling.
    """
    @mock.patch('satpy.composites.CompositorLoader.load_compositors')
    @mock.patch('satpy.scene.Scene.create_reader_instances')
    def test_resample_comp1(self, cri, cl):
        """Test loading and resampling a single dataset"""
        import satpy.scene
        from satpy.tests.utils import create_fake_reader, test_composites
        from satpy import DatasetID, Dataset
        cri.return_value = {'fake_reader': create_fake_reader('fake_reader', 'fake_sensor')}
        comps, mods = test_composites('fake_sensor')
        cl.return_value = (comps, mods)
        scene = satpy.scene.Scene(filenames='bla',
                                  base_dir='bli',
                                  reader='fake_reader')
        scene.load(['ds1'])
        with mock.patch.object(Dataset, 'resample', autospec=True) as r:
            r.side_effect = lambda self, x, **kwargs: self
            new_scene = scene.resample(None)  # None is our fake Area destination
        loaded_ids = list(new_scene.datasets.keys())
        self.assertEquals(len(loaded_ids), 1)
        self.assertTupleEqual(tuple(loaded_ids[0]), tuple(DatasetID(name='ds1')))


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestScene))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSceneLoading))
    mysuite.addTest(loader.loadTestsFromTestCase(TestSceneResample))

    return mysuite

if __name__ == "__main__":
    unittest.main()
