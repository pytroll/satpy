#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2019 Satpy developers
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

import satpy.readers.yaml_reader as yr
from satpy.readers.file_handlers import BaseFileHandler
from satpy.dataset import DatasetID

try:
    from unittest.mock import MagicMock, patch
except ImportError:
    from mock import MagicMock, patch


class FakeFH(BaseFileHandler):

    def __init__(self, start_time, end_time):
        super(FakeFH, self).__init__("", {}, {})
        self._start_time = start_time
        self._end_time = end_time
        self.get_bounding_box = MagicMock()
        fake_ds = MagicMock()
        fake_ds.return_value.dims = ['x', 'y']
        self.get_dataset = fake_ds
        self.combine_info = MagicMock()

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time


class TestUtils(unittest.TestCase):
    """Test the utility functions."""

    def test_get_filebase(self):
        """Check the get_filebase function."""
        base_dir = os.path.join(os.path.expanduser('~'), 'data',
                                'satellite', 'Sentinel-3')
        base_data = ('S3A_OL_1_EFR____20161020T081224_20161020T081524_'
                     '20161020T102406_0179_010_078_2340_SVL_O_NR_002.SEN3')
        base_dir = os.path.join(base_dir, base_data)
        pattern = ('{mission_id:3s}_OL_{processing_level:1s}_{datatype_id:_<6s'
                   '}_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_{cre'
                   'ation_time:%Y%m%dT%H%M%S}_{duration:4d}_{cycle:3d}_{relati'
                   've_orbit:3d}_{frame:4d}_{centre:3s}_{mode:1s}_{timeliness:'
                   '2s}_{collection:3s}.SEN3/geo_coordinates.nc')
        pattern = os.path.join(*pattern.split('/'))
        filename = os.path.join(base_dir, 'Oa05_radiance.nc')
        expected = os.path.join(base_data, 'Oa05_radiance.nc')
        self.assertEqual(yr.get_filebase(filename, pattern), expected)

    def test_match_filenames(self):
        """Check that matching filenames works."""
        # just a fake path for testing that doesn't have to exist
        base_dir = os.path.join(os.path.expanduser('~'), 'data',
                                'satellite', 'Sentinel-3')
        base_data = ('S3A_OL_1_EFR____20161020T081224_20161020T081524_'
                     '20161020T102406_0179_010_078_2340_SVL_O_NR_002.SEN3')
        base_dir = os.path.join(base_dir, base_data)
        pattern = ('{mission_id:3s}_OL_{processing_level:1s}_{datatype_id:_<6s'
                   '}_{start_time:%Y%m%dT%H%M%S}_{end_time:%Y%m%dT%H%M%S}_{cre'
                   'ation_time:%Y%m%dT%H%M%S}_{duration:4d}_{cycle:3d}_{relati'
                   've_orbit:3d}_{frame:4d}_{centre:3s}_{mode:1s}_{timeliness:'
                   '2s}_{collection:3s}.SEN3/geo_coordinates.nc')
        pattern = os.path.join(*pattern.split('/'))
        filenames = [os.path.join(base_dir, 'Oa05_radiance.nc'),
                     os.path.join(base_dir, 'geo_coordinates.nc')]
        expected = os.path.join(base_dir, 'geo_coordinates.nc')
        self.assertEqual(yr.match_filenames(filenames, pattern), [expected])

    def test_listify_string(self):
        """Check listify_string."""
        self.assertEqual(yr.listify_string(None), [])
        self.assertEqual(yr.listify_string('some string'), ['some string'])
        self.assertEqual(yr.listify_string(['some', 'string']),
                         ['some', 'string'])


class DummyReader(BaseFileHandler):
    def __init__(self, filename, filename_info, filetype_info):
        super(DummyReader, self).__init__(
            filename, filename_info, filetype_info)
        self._start_time = datetime(2000, 1, 1, 12, 1)
        self._end_time = datetime(2000, 1, 1, 12, 2)
        self.metadata = {}

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time


class TestFileFileYAMLReaderMultiplePatterns(unittest.TestCase):
    """Test units from FileYAMLReader with multiple readers."""

    @patch('satpy.readers.yaml_reader.recursive_dict_update')
    @patch('satpy.readers.yaml_reader.yaml', spec=yr.yaml)
    def setUp(self, _, rec_up):  # pylint: disable=arguments-differ
        """Setup a reader instance with a fake config."""
        patterns = ['a{something:3s}.bla',
                    'a0{something:2s}.bla']
        res_dict = {'reader': {'name': 'fake',
                               'sensors': ['canon']},
                    'file_types': {'ftype1': {'name': 'ft1',
                                              'file_patterns': patterns,
                                              'file_reader': DummyReader}},
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
                                        filter_parameters={
                                            'start_time': datetime(2000, 1, 1),
                                            'end_time': datetime(2000, 1, 2)})

    def test_select_from_pathnames(self):
        """Check select_files_from_pathnames."""
        filelist = ['a001.bla', 'a002.bla', 'abcd.bla', 'k001.bla', 'a003.bli']

        res = self.reader.select_files_from_pathnames(filelist)
        for expected in ['a001.bla', 'a002.bla', 'abcd.bla']:
            self.assertIn(expected, res)
        self.assertEqual(len(res), 3)

    def test_fn_items_for_ft(self):
        """Check filename_items_for_filetype."""
        filelist = ['a001.bla', 'a002.bla', 'abcd.bla', 'k001.bla', 'a003.bli']
        ft_info = self.config['file_types']['ftype1']
        fiter = self.reader.filename_items_for_filetype(filelist, ft_info)

        filenames = dict(fname for fname in fiter)
        self.assertEqual(len(filenames.keys()), 3)

    def test_create_filehandlers(self):
        """Check create_filehandlers."""
        filelist = ['a001.bla', 'a002.bla', 'a001.bla', 'a002.bla',
                    'abcd.bla', 'k001.bla', 'a003.bli']

        self.reader.create_filehandlers(filelist)
        self.assertEqual(len(self.reader.file_handlers['ftype1']), 3)


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
                                              'file_reader': BaseFileHandler,
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
                                        filter_parameters={
                                            'start_time': datetime(2000, 1, 1),
                                            'end_time': datetime(2000, 1, 2),
        })

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
        loadables = self.reader.select_files_from_pathnames(['a001.bla'])
        self.reader.create_filehandlers(loadables)
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
        loadables = self.reader.select_files_from_pathnames(['a001.bla'])
        self.reader.create_filehandlers(loadables)
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

        for idx, fh in enumerate([fh0, fh1, fh2, fh3, fh4, fh5]):
            res = self.reader.time_matches(fh.start_time, fh.end_time)
            # only the first one should be false
            self.assertEqual(res, idx not in [0, 4])

        for idx, fh in enumerate([fh0, fh1, fh2, fh3, fh4, fh5]):
            res = self.reader.time_matches(fh.start_time, None)
            self.assertEqual(res, idx not in [0, 1, 4, 5])

    @patch('satpy.readers.yaml_reader.get_area_def')
    @patch('satpy.readers.yaml_reader.AreaDefBoundary')
    @patch('satpy.readers.yaml_reader.Boundary')
    def test_file_covers_area(self, bnd, adb, gad):
        """Test that area coverage is checked properly."""
        file_handler = FakeFH(datetime(1999, 12, 31, 10, 0),
                              datetime(2000, 1, 3, 12, 30))

        self.reader.filter_parameters['area'] = True
        bnd.return_value.contour_poly.intersection.return_value = True
        adb.return_value.contour_poly.intersection.return_value = True
        res = self.reader.check_file_covers_area(file_handler, True)
        self.assertTrue(res)

        bnd.return_value.contour_poly.intersection.return_value = False
        adb.return_value.contour_poly.intersection.return_value = False
        res = self.reader.check_file_covers_area(file_handler, True)
        self.assertFalse(res)

        file_handler.get_bounding_box.side_effect = NotImplementedError()
        self.reader.filter_parameters['area'] = True
        res = self.reader.check_file_covers_area(file_handler, True)
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

    @patch('satpy.readers.yaml_reader.StackedAreaDefinition')
    def test_load_area_def(self, sad):
        """Test loading the area def for the reader."""
        dsid = MagicMock()
        file_handlers = []
        items = random.randrange(2, 10)
        for i in range(items):
            file_handlers.append(MagicMock())
        final_area = self.reader._load_area_def(dsid, file_handlers)
        self.assertEqual(final_area, sad.return_value.squeeze.return_value)

        args, kwargs = sad.call_args
        self.assertEqual(len(args), items)

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

    @patch('satpy.readers.yaml_reader.xr')
    def test_load_entire_dataset(self, xarray):
        """Check loading an entire dataset."""
        file_handlers = [FakeFH(None, None), FakeFH(None, None),
                         FakeFH(None, None), FakeFH(None, None)]

        proj = self.reader._load_dataset(None, {}, file_handlers)

        self.assertIs(proj, xarray.concat.return_value)


class TestFileFileYAMLReaderMultipleFileTypes(unittest.TestCase):
    """Test units from FileYAMLReader with multiple file types."""

    @patch('satpy.readers.yaml_reader.recursive_dict_update')
    @patch('satpy.readers.yaml_reader.yaml', spec=yr.yaml)
    def setUp(self, _, rec_up):  # pylint: disable=arguments-differ
        """Setup a reader instance with a fake config."""
        # Example: GOES netCDF data
        #   a) From NOAA CLASS: ftype1, including coordinates
        #   b) From EUMETSAT: ftype2, coordinates in extra file (ftype3)
        #
        # For test completeness add one channel (ch3) which is only available
        # in ftype1.
        patterns1 = ['a.nc']
        patterns2 = ['b.nc']
        patterns3 = ['geo.nc']
        res_dict = {'reader': {'name': 'fake',
                               'sensors': ['canon']},
                    'file_types': {'ftype1': {'name': 'ft1',
                                              'file_patterns': patterns1},
                                   'ftype2': {'name': 'ft2',
                                              'file_patterns': patterns2},
                                   'ftype3': {'name': 'ft3',
                                              'file_patterns': patterns3}},
                    'datasets': {'ch1': {'name': 'ch01',
                                         'wavelength': [0.5, 0.6, 0.7],
                                         'calibration': 'reflectance',
                                         'file_type': ['ftype1', 'ftype2'],
                                         'coordinates': ['lons', 'lats']},
                                 'ch2': {'name': 'ch02',
                                         'wavelength': [0.7, 0.75, 0.8],
                                         'calibration': 'counts',
                                         'file_type': ['ftype1', 'ftype2'],
                                         'coordinates': ['lons', 'lats']},
                                 'ch3': {'name': 'ch03',
                                         'wavelength': [0.8, 0.85, 0.9],
                                         'calibration': 'counts',
                                         'file_type': 'ftype1',
                                         'coordinates': ['lons', 'lats']},
                                 'lons': {'name': 'lons',
                                          'file_type': ['ftype1', 'ftype3']},
                                 'lats': {'name': 'lats',
                                          'file_type': ['ftype1', 'ftype3']}}}

        rec_up.return_value = res_dict
        self.config = res_dict
        self.reader = yr.FileYAMLReader([__file__])

    def test_update_ds_ids_from_file_handlers(self):
        """Test updating existing dataset IDs with information from the file"""
        from functools import partial
        orig_ids = self.reader.all_ids

        def available_datasets(self, configured_datasets=None):
            res = self.resolution
            # update previously configured datasets
            for is_avail, ds_info in (configured_datasets or []):
                if is_avail is not None:
                    yield is_avail, ds_info

                matches = self.file_type_matches(ds_info['file_type'])
                if matches and ds_info.get('resolution') != res:
                    new_info = ds_info.copy()
                    new_info['resolution'] = res
                    yield True, new_info
                elif is_avail is None:
                    yield is_avail, ds_info

        def file_type_matches(self, ds_ftype):
            if isinstance(ds_ftype, str) and ds_ftype == self.filetype_info['file_type']:
                return True
            elif self.filetype_info['file_type'] in ds_ftype:
                return True
            return None

        for ftype, resol in zip(('ftype1', 'ftype2'), (1, 2)):
            # need to copy this because the dataset infos will be modified
            _orig_ids = {key: val.copy() for key, val in orig_ids.items()}
            with patch.dict(self.reader.all_ids, _orig_ids, clear=True), \
                    patch.dict(self.reader.available_ids, {}, clear=True):
                # Add a file handler with resolution property
                fh = MagicMock(filetype_info={'file_type': ftype},
                               resolution=resol)
                fh.available_datasets = partial(available_datasets, fh)
                fh.file_type_matches = partial(file_type_matches, fh)
                self.reader.file_handlers = {
                    ftype: [fh]}

                # Update existing dataset IDs with resolution property from
                # the file handler
                self.reader.update_ds_ids_from_file_handlers()

                # Make sure the resolution property has been transferred
                # correctly from the file handler to the dataset ID
                for ds_id, ds_info in self.reader.all_ids.items():
                    file_types = ds_info['file_type']
                    if not isinstance(file_types, list):
                        file_types = [file_types]
                    expected = resol if ftype in file_types else None
                    self.assertEqual(expected, ds_id.resolution)


def suite():
    """The test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestUtils))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFileFileYAMLReader))
    mysuite.addTest(loader.loadTestsFromTestCase(
        TestFileFileYAMLReaderMultiplePatterns))
    mysuite.addTest(loader.loadTestsFromTestCase(
        TestFileFileYAMLReaderMultipleFileTypes))

    return mysuite


if __name__ == "__main__":
    unittest.main()
