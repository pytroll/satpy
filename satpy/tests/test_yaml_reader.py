#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author(s):
#
#   Martin Raspaud <martin.raspaud@smhi.se>
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

"""Testing the yaml_reader module."""

import os
import random
import unittest
from datetime import datetime
from tempfile import mkdtemp

from mock import MagicMock, patch

import satpy.readers.yaml_reader as yr
from satpy.dataset import DATASET_KEYS, DatasetID


class FakeFH(object):

    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.get_bounding_box = MagicMock()


class TestUtils(unittest.TestCase):
    """Test the utility functions."""

    def test_get_filebase(self):
        """Check the get_filebase function."""
        pattern = ('{mission_id:3s}_OL_{processing_level:1s}_{datatype_id:_<6s'
                   '}_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_{cre'
                   'ation_time:%Y%m%dT%H%M%S}_{duration:4d}_{cycle:3d}_{relati'
                   've_orbit:3d}_{frame:4d}_{centre:3s}_{mode:1s}_{timeliness:'
                   '2s}_{collection:3s}.SEN3/geo_coordinates.nc')
        filename = ('/home/a001673/data/satellite/Sentinel-3/S3A_OL_1_EFR____2'
                    '0161020T081224_20161020T081524_20161020T102406_0179_010_0'
                    '78_2340_SVL_O_NR_002.SEN3/Oa05_radiance.nc')
        expected = ('S3A_OL_1_EFR____20161020T081224_20161020T081524_20161020T'
                    '102406_0179_010_078_2340_SVL_O_NR_002.SEN3/Oa05_radiance.'
                    'nc')
        self.assertEqual(yr.get_filebase(filename, pattern), expected)

    def test_match_filenames(self):
        """Check that matching filenames works."""
        pattern = ('{mission_id:3s}_OL_{processing_level:1s}_{datatype_id:_<6s'
                   '}_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_{cre'
                   'ation_time:%Y%m%dT%H%M%S}_{duration:4d}_{cycle:3d}_{relati'
                   've_orbit:3d}_{frame:4d}_{centre:3s}_{mode:1s}_{timeliness:'
                   '2s}_{collection:3s}.SEN3/geo_coordinates.nc')
        filenames = ['/home/a001673/data/satellite/Sentinel-3/S3A_OL_1_EFR____2'
                     '0161020T081224_20161020T081524_20161020T102406_0179_010_0'
                     '78_2340_SVL_O_NR_002.SEN3/Oa05_radiance.nc',
                     '/home/a001673/data/satellite/Sentinel-3/S3A_OL_1_EFR____2'
                     '0161020T081224_20161020T081524_20161020T102406_0179_010_0'
                     '78_2340_SVL_O_NR_002.SEN3/geo_coordinates.nc']
        expected = ('S3A_OL_1_EFR____20161020T081224_20161020T081524_20161020T'
                    '102406_0179_010_078_2340_SVL_O_NR_002.SEN3/geo_coordinates'
                    '.nc')
        self.assertEqual(yr.match_filenames(filenames, pattern),
                         ["/home/a001673/data/satellite/Sentinel-3/" +
                          expected])

    def test_listify_string(self):
        """Check listify_string."""
        self.assertEqual(yr.listify_string(None), [])
        self.assertEqual(yr.listify_string('some string'), ['some string'])
        self.assertEqual(yr.listify_string(['some', 'string']),
                         ['some', 'string'])


class TestFileFileYAMLReader(unittest.TestCase):
    """Test units from FileYAMLReader."""

    @patch('satpy.readers.yaml_reader.recursive_dict_update')
    @patch('satpy.readers.yaml_reader.yaml', spec=yr.yaml)
    def setUp(self, _, rec_up):  # pylint: disable=arguments-differ
        """Setup a reader instance with a fake config."""
        patterns = ['a{something:3s}.bla']
        res_dict = {'reader': {'name': 'fake',
                               'sensors': ['canon']},
                    'file_types': {'ftype1': {'name': 'ft1',
                                              'file_patterns': patterns}},
                    'datasets': {'ch1': {'name': 'ch01',
                                         'wavelength': [0.5, 0.6, 0.7],
                                         'calibration': 'reflectance',
                                         'file_type': 'ftype1',
                                         'coordinates': ['lons', 'lats']},
                                 'ch2': {'name': 'ch02',
                                         'wavelength': [0.7, 0.75, 0.8],
                                         'calibration': 'counts',
                                         'file_type': 'ftype1',
                                         'coordinates': ['lons', 'lats']},
                                 'lons': {'name': 'lons',
                                          'file_type': 'ftype2'},
                                 'lats': {'name': 'lats',
                                          'file_type': 'ftype2'}}}

        rec_up.return_value = res_dict
        self.config = res_dict
        self.reader = yr.FileYAMLReader([__file__],
                                        start_time=datetime(2000, 1, 1),
                                        end_time=datetime(2000, 1, 2))

    def test_all_dataset_ids(self):
        """Check that all datasets ids are returned."""
        self.assertSetEqual(set(self.reader.all_dataset_ids),
                            {DatasetID(name='ch02',
                                       wavelength=(0.7, 0.75, 0.8),
                                       resolution=None,
                                       polarization=None,
                                       calibration='counts',
                                       modifiers=()),
                             DatasetID(name='ch01',
                                       wavelength=(0.5, 0.6, 0.7),
                                       resolution=None,
                                       polarization=None,
                                       calibration='reflectance',
                                       modifiers=()),
                             DatasetID(name='lons',
                                       wavelength=None,
                                       resolution=None,
                                       polarization=None,
                                       calibration=None,
                                       modifiers=()),
                             DatasetID(name='lats',
                                       wavelength=None,
                                       resolution=None,
                                       polarization=None,
                                       calibration=None,
                                       modifiers=())})

    def test_all_dataset_names(self):
        """Get all dataset names."""
        self.assertSetEqual(self.reader.all_dataset_names,
                            set(['ch01', 'ch02', 'lons', 'lats']))

    def test_available_dataset_ids(self):
        """Get ids of the available datasets."""
        self.reader.file_handlers = ['ftype1']
        self.assertSetEqual(set(self.reader.available_dataset_ids),
                            {DatasetID(name='ch02',
                                       wavelength=(0.7, 0.75, 0.8),
                                       resolution=None,
                                       polarization=None,
                                       calibration='counts',
                                       modifiers=()),
                             DatasetID(name='ch01',
                                       wavelength=(0.5, 0.6, 0.7),
                                       resolution=None,
                                       polarization=None,
                                       calibration='reflectance',
                                       modifiers=())})

    def test_available_dataset_names(self):
        """Get ids of the available datasets."""
        self.reader.file_handlers = ['ftype1']
        self.assertSetEqual(set(self.reader.available_dataset_names),
                            set(["ch01", "ch02"]))

    def test_filter_fh_by_time(self):
        """Check filtering filehandlers by time."""
        fh0 = FakeFH(datetime(1999, 12, 30), datetime(1999, 12, 31))
        fh1 = FakeFH(datetime(1999, 12, 31, 10, 0),
                     datetime(2000, 1, 1, 12, 30))
        fh2 = FakeFH(datetime(2000, 1, 1, 10, 0),
                     datetime(2000, 1, 1, 12, 30))
        fh3 = FakeFH(datetime(2000, 1, 1, 12, 30),
                     datetime(2000, 1, 2, 12, 30))
        fh4 = FakeFH(datetime(2000, 1, 2, 12, 30),
                     datetime(2000, 1, 3, 12, 30))
        fh5 = FakeFH(datetime(1999, 12, 31, 10, 0),
                     datetime(2000, 1, 3, 12, 30))

        res = self.reader.filter_fh_by_time([fh0, fh1, fh2, fh3, fh4, fh5])
        self.assertSetEqual(set(res), set([fh1, fh2, fh3, fh5]))

    def test_filter_fh_by_area(self):
        """Check filtering filehandlers by area."""
        with patch.object(self.reader, 'check_file_covers_area',
                          side_effect=[True, False, True]):
            res = self.reader.filter_fh_by_area([1, 2, 3])
            self.assertSetEqual(set(res), set([1, 3]))

    @patch('satpy.resample.get_area_def')
    def test_file_covers_area(self, gad):
        """Test that area coverage is checked properly."""
        file_handler = FakeFH(datetime(1999, 12, 31, 10, 0),
                              datetime(2000, 1, 3, 12, 30))

        trollsched = MagicMock()
        adb = trollsched.boundary.AreaDefBoundary
        bnd = trollsched.boundary.Boundary

        modules = {'trollsched': trollsched,
                   'trollsched.boundary': trollsched.boundary}

        with patch.dict('sys.modules', modules):

            self.reader._area = True
            bnd.return_value.contour_poly.intersection.return_value = True
            adb.return_value.contour_poly.intersection.return_value = True
            res = self.reader.check_file_covers_area(file_handler)
            self.assertTrue(res)

            bnd.return_value.contour_poly.intersection.return_value = False
            adb.return_value.contour_poly.intersection.return_value = False
            res = self.reader.check_file_covers_area(file_handler)
            self.assertFalse(res)

            self.reader._area = False
            res = self.reader.check_file_covers_area(file_handler)
            self.assertTrue(res)

            file_handler.get_bounding_box.side_effect = NotImplementedError()
            self.reader._area = True
            res = self.reader.check_file_covers_area(file_handler)
            self.assertTrue(res)

    def test_start_end_time(self):
        """Check start and end time behaviours."""
        self.reader.file_handlers = {}

        def get_start_time():
            return self.reader.start_time
        self.assertRaises(RuntimeError, get_start_time)

        def get_end_time():
            return self.reader.end_time
        self.assertRaises(RuntimeError, get_end_time)

        fh0 = FakeFH(datetime(1999, 12, 30, 0, 0),
                     datetime(1999, 12, 31, 0, 0))
        fh1 = FakeFH(datetime(1999, 12, 31, 10, 0),
                     datetime(2000, 1, 1, 12, 30))
        fh2 = FakeFH(datetime(2000, 1, 1, 10, 0),
                     datetime(2000, 1, 1, 12, 30))
        fh3 = FakeFH(datetime(2000, 1, 1, 12, 30),
                     datetime(2000, 1, 2, 12, 30))
        fh4 = FakeFH(datetime(2000, 1, 2, 12, 30),
                     datetime(2000, 1, 3, 12, 30))
        fh5 = FakeFH(datetime(1999, 12, 31, 10, 0),
                     datetime(2000, 1, 3, 12, 30))

        self.reader.file_handlers = {
            '0': [fh1, fh2, fh3, fh4, fh5],
            '1': [fh0, fh1, fh2, fh3, fh4, fh5],
            '2': [fh2, fh3],
        }

        self.assertEqual(self.reader.start_time, datetime(1999, 12, 30, 0, 0))
        self.assertEqual(self.reader.end_time, datetime(2000, 1, 3, 12, 30))

    def test_select_from_pathnames(self):
        """Check select_files_from_pathnames."""
        filelist = ['a001.bla', 'a002.bla', 'abcd.bla', 'k001.bla', 'a003.bli']

        res = self.reader.select_files_from_pathnames(filelist)
        for expected in ['a001.bla', 'a002.bla', 'abcd.bla']:
            self.assertIn(expected, res)

        self.assertEqual(0, len(self.reader.select_files_from_pathnames([])))

    def test_select_from_directory(self):
        """Check select_files_from_directory."""
        filelist = ['a001.bla', 'a002.bla', 'abcd.bla', 'k001.bla', 'a003.bli']
        dpath = mkdtemp()
        for fname in filelist:
            with open(os.path.join(dpath, fname), 'w'):
                pass

        res = self.reader.select_files_from_directory(dpath)
        for expected in ['a001.bla', 'a002.bla', 'abcd.bla']:
            self.assertIn(os.path.join(dpath, expected), res)

        for fname in filelist:
            os.remove(os.path.join(dpath, fname))
        self.assertEqual(0,
                         len(self.reader.select_files_from_directory(dpath)))
        os.rmdir(dpath)

    def test_supports_sensor(self):
        """Check supports_sensor."""
        self.assertTrue(self.reader.supports_sensor('canon'))
        self.assertFalse(self.reader.supports_sensor('nikon'))

    def test_get_datasets_by_name(self):
        """Check getting datasets by name."""
        self.assertEqual(len(self.reader.get_ds_ids_by_name('ch01')), 1)

        res = self.reader.get_ds_ids_by_name('ch01')[0]
        for key, val in self.config['datasets']['ch1'].items():
            if isinstance(val, list):
                val = tuple(val)
            if key not in DATASET_KEYS:
                continue
            self.assertEqual(getattr(res, key), val)

        self.assertRaises(KeyError, self.reader.get_ds_ids_by_name, 'bla')

    def test_get_datasets_by_wl(self):
        """Check getting datasets by wavelength."""
        res = self.reader.get_ds_ids_by_wavelength(.6)
        self.assertEqual(len(res), 1)
        res = res[0]
        for key, val in self.config['datasets']['ch1'].items():
            if isinstance(val, list):
                val = tuple(val)
            if key not in DATASET_KEYS:
                continue
            self.assertEqual(getattr(res, key), val)

        res = self.reader.get_ds_ids_by_wavelength(.7)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0].name, 'ch02')

        res = self.reader.get_ds_ids_by_wavelength((.7, .75, .8))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, 'ch02')

        res = self.reader.get_ds_ids_by_wavelength([.7, .75, .8])
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, 'ch02')

        self.assertRaises(KeyError, self.reader.get_ds_ids_by_wavelength, 12)

    def test_get_datasets_by_id(self):
        """Check getting datasets by id."""
        from satpy.dataset import DatasetID
        dsid = DatasetID('ch01')
        res = self.reader.get_ds_ids_by_id(dsid)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, 'ch01')

        dsid = DatasetID(wavelength=.6)
        res = self.reader.get_ds_ids_by_id(dsid)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, 'ch01')

        dsid = DatasetID('ch01', .6)
        res = self.reader.get_ds_ids_by_id(dsid)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, 'ch01')

        dsid = DatasetID('ch01', .1)
        self.assertRaises(KeyError, self.reader.get_ds_ids_by_id, dsid)

    def test_get_best_calibration(self):
        """Test finding best calibration for datasets."""
        calibration = None
        self.assertListEqual(self.reader._get_best_calibration(calibration),
                             ["brightness_temperature", "reflectance",
                              'radiance', 'counts'])

        calibration = ["reflectance", "radiance"]
        self.assertListEqual(self.reader._get_best_calibration(calibration),
                             ["reflectance", 'radiance'])

        calibration = ["radiance", "reflectance"]
        self.assertListEqual(self.reader._get_best_calibration(calibration),
                             ["reflectance", 'radiance'])

    def test_dataset_with_calibration(self):
        """Test getting datasets with calibration."""
        calibration = ["radiance", "reflectance"]

        datasets = [yr.DatasetID(name=ds['name'],
                                 wavelength=ds.get("wavelength"),
                                 calibration=ds.get("calibration"))
                    for ds in self.reader.datasets.values()]
        ds = self.reader._ds_ids_with_best_calibration(datasets, calibration)
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=[0.5, 0.6, 0.7],
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=None)])

    def test_dfilter_from_key(self):
        """Test creating a dfilter from a key."""
        dfilter = None

        key = yr.DatasetID(name='bla', calibration='radiance')

        expected = {'polarization': None,
                    'modifiers': None,
                    'resolution': None,
                    'calibration': ['radiance']}
        self.assertDictEqual(expected,
                             self.reader.dfilter_from_key(dfilter, key))

        key = yr.DatasetID(name='bla', calibration='reflectance')

        expected = {'polarization': None,
                    'modifiers': None,
                    'resolution': None,
                    'calibration': ['reflectance']}
        self.assertDictEqual(expected,
                             self.reader.dfilter_from_key(dfilter, key))

        key = yr.DatasetID(name='bla', calibration='reflectance',
                           modifiers=('rayleigh_corrected'))

        expected = {'polarization': None,
                    'modifiers': ('rayleigh_corrected'),
                    'resolution': None,
                    'calibration': ['reflectance']}
        self.assertDictEqual(expected,
                             self.reader.dfilter_from_key(dfilter, key))

        dfilter = {'resolution': [1000]}

        key = yr.DatasetID(name='bla', calibration='radiance')

        expected = {'polarization': None,
                    'modifiers': None,
                    'resolution': [1000],
                    'calibration': ['radiance']}
        self.assertDictEqual(expected,
                             self.reader.dfilter_from_key(dfilter, key))

        key = yr.DatasetID(name='bla', calibration='reflectance')

        expected = {'polarization': None,
                    'modifiers': None,
                    'resolution': [1000],
                    'calibration': ['reflectance']}
        self.assertDictEqual(expected,
                             self.reader.dfilter_from_key(dfilter, key))

        key = yr.DatasetID(name='bla', calibration='reflectance',
                           modifiers=('rayleigh_corrected'))

        expected = {'polarization': None,
                    'modifiers': ('rayleigh_corrected'),
                    'resolution': [1000],
                    'calibration': ['reflectance']}
        self.assertDictEqual(expected,
                             self.reader.dfilter_from_key(dfilter, key))

    def test_filter_datasets(self):
        """Test filtering datasets."""
        datasets = [yr.DatasetID(name=ds['name'],
                                 wavelength=ds.get("wavelength"),
                                 calibration=ds.get("calibration"))
                    for ds in self.reader.datasets.values()]

        dfilter = {'polarization': None,
                   'modifiers': None,
                   'resolution': None,
                   'calibration': ['reflectance']}

        ds = self.reader.filter_ds_ids(datasets, dfilter)
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=[0.5, 0.6, 0.7],
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=None)])

    def test_datasets_from_any_key(self):
        """Test getting dataset from any key."""
        ds = self.reader._ds_ids_from_any_key('ch01')
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=(0.5, 0.6, 0.7),
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=())])

        ds = self.reader._ds_ids_from_any_key(0.51)
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=(0.5, 0.6, 0.7),
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=())])

        ds = self.reader._ds_ids_from_any_key(yr.DatasetID(name='ch01',
                                                           wavelength=0.51))
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=(0.5, 0.6, 0.7),
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=())])

    def test_get_dataset_key(self):
        """Test get_dataset_key."""
        ds = self.reader.get_dataset_key('ch01', aslist=True)
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=(0.5, 0.6, 0.7),
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=())])

        ds = self.reader.get_dataset_key(0.51, aslist=True)
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=(0.5, 0.6, 0.7),
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=())])

        ds = self.reader.get_dataset_key(yr.DatasetID(name='ch01',
                                                      wavelength=0.51),
                                         aslist=True)
        self.assertListEqual(ds,
                             [yr.DatasetID(name='ch01',
                                           wavelength=(0.5, 0.6, 0.7),
                                           resolution=None,
                                           polarization=None,
                                           calibration='reflectance',
                                           modifiers=())])

        ds = self.reader.get_dataset_key('ch01')
        self.assertEqual(ds,
                         yr.DatasetID(name='ch01',
                                      wavelength=(0.5, 0.6, 0.7),
                                      resolution=None,
                                      polarization=None,
                                      calibration='reflectance',
                                      modifiers=()))

        ds = self.reader.get_dataset_key(0.51)
        self.assertEqual(ds,
                         yr.DatasetID(name='ch01',
                                      wavelength=(0.5, 0.6, 0.7),
                                      resolution=None,
                                      polarization=None,
                                      calibration='reflectance',
                                      modifiers=()))

        ds = self.reader.get_dataset_key(yr.DatasetID(name='ch01',
                                                      wavelength=0.51))
        self.assertEqual(ds,
                         yr.DatasetID(name='ch01',
                                      wavelength=(0.5, 0.6, 0.7),
                                      resolution=None,
                                      polarization=None,
                                      calibration='reflectance',
                                      modifiers=()))

    def test_combine_area_extents(self):
        """Test combination of area extents."""
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 6, 3, 2)
        res = self.reader._combine_area_extents(area1, area2)
        self.assertListEqual(res, [1, 6, 3, 4])

        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        res = self.reader._combine_area_extents(area1, area2)
        self.assertListEqual(res, [1, 2, 3, 6])

    def test_append_area_defs_fail(self):
        """Fail appending areas."""
        from satpy.composites import IncompatibleAreas
        area1 = MagicMock()
        area1.proj_dict = {"proj": 'A'}
        area2 = MagicMock()
        area2.proj_dict = {'proj': 'B'}
        res = self.reader._combine_area_extents(area1, area2)
        self.assertRaises(IncompatibleAreas,
                          self.reader._append_area_defs, area1, area2)

    @patch('satpy.readers.yaml_reader.AreaDefinition')
    def test_append_area_defs(self, adef):
        """Test appending area definitions."""
        area1 = MagicMock()
        area1.area_extent = (1, 2, 3, 4)
        area1.proj_dict = {"proj": 'A'}
        area1.y_size = random.randrange(6425)

        area2 = MagicMock()
        area2.area_extent = (1, 4, 3, 6)
        area2.proj_dict = {"proj": 'A'}
        area2.y_size = random.randrange(6425)

        res = self.reader._append_area_defs(area1, area2)
        area_extent = [1, 2, 3, 6]
        y_size = area1.y_size + area2.y_size
        adef.assert_called_once_with(area1.area_id, area1.name, area1.proj_id,
                                     area1.proj_dict, area1.x_size, y_size,
                                     area_extent)

    def test_load_area_def(self):
        """Test loading the area def for the reader."""
        dsid = MagicMock()
        file_handlers = []
        items = random.randrange(2, 10)
        for i in range(items):
            file_handlers.append(MagicMock())
        with patch.object(self.reader, '_append_area_defs') as aad:
            final_area = self.reader._load_area_def(dsid, file_handlers)
            self.assertEqual(final_area, aad.return_value)
            self.assertEqual(len(aad.mock_calls), items)
            fh1 = file_handlers[0]
            aad.reset_mock()
            final_area = self.reader._load_area_def(dsid, file_handlers[:1])
            self.assertEqual(final_area, fh1.get_area_def.return_value)
            self.assertEqual(len(aad.mock_calls), 0)

    def test_preferred_filetype(self):
        """Test finding the preferred filetype."""

        self.reader.file_handlers = {'a': 'a', 'b': 'b', 'c': 'c'}
        self.assertEqual(self.reader._preferred_filetype(['c', 'a']), 'c')
        self.assertEqual(self.reader._preferred_filetype(['a', 'c']), 'a')
        self.assertEqual(self.reader._preferred_filetype(['d', 'e']), None)

    def test_get_coordinates_for_dataset_key(self):
        """Test getting coordinates for a key."""
        ds_id = DatasetID(name='ch01', wavelength=(0.5, 0.6, 0.7),
                          resolution=None, polarization=None,
                          calibration='reflectance', modifiers=())
        res = self.reader._get_coordinates_for_dataset_key(ds_id)
        self.assertListEqual(res,
                             [DatasetID(name='lons',
                                        wavelength=None,
                                        resolution=None,
                                        polarization=None,
                                        calibration=None,
                                        modifiers=()),
                              DatasetID(name='lats',
                                        wavelength=None,
                                        resolution=None,
                                        polarization=None,
                                        calibration=None,
                                        modifiers=())])

    def test_get_coordinates_for_dataset_key_without(self):
        """Test getting coordinates for a key without coordinates."""
        ds_id = DatasetID(name='lons',
                          wavelength=None,
                          resolution=None,
                          polarization=None,
                          calibration=None,
                          modifiers=())
        res = self.reader._get_coordinates_for_dataset_key(ds_id)
        self.assertListEqual(res, [])

    def test_get_coordinates_for_dataset_keys(self):
        """Test getting coordinates for keys."""
        ds_id1 = DatasetID(name='ch01', wavelength=(0.5, 0.6, 0.7),
                           resolution=None, polarization=None,
                           calibration='reflectance', modifiers=())
        ds_id2 = DatasetID(name='ch02', wavelength=(0.7, 0.75, 0.8),
                           resolution=None, polarization=None,
                           calibration='counts', modifiers=())
        lons = DatasetID(name='lons',  wavelength=None,
                         resolution=None, polarization=None,
                         calibration=None, modifiers=())
        lats = DatasetID(name='lats', wavelength=None,
                         resolution=None, polarization=None,
                         calibration=None, modifiers=())

        res = self.reader._get_coordinates_for_dataset_keys([ds_id1, ds_id2,
                                                             lons])
        expected = {ds_id1: [lons, lats], ds_id2: [lons, lats], lons: []}

        self.assertDictEqual(res, expected)

    def test_get_file_handlers(self):
        """Test getting filehandler to load a dataset."""
        ds_id1 = DatasetID(name='ch01', wavelength=(0.5, 0.6, 0.7),
                           resolution=None, polarization=None,
                           calibration='reflectance', modifiers=())
        self.reader.file_handlers = {'ftype1': 'bla'}

        self.assertEqual(self.reader._get_file_handlers(ds_id1), 'bla')

        lons = DatasetID(name='lons',  wavelength=None,
                         resolution=None, polarization=None,
                         calibration=None, modifiers=())
        self.assertEqual(self.reader._get_file_handlers(lons), None)


def suite():
    """The test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestUtils))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFileFileYAMLReader))

    return mysuite


if __name__ == "__main__":
    unittest.main()
