#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022 Satpy developers
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
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import xarray as xr

import satpy.readers.yaml_reader as yr
from satpy._compat import cache
from satpy.dataset import DataQuery
from satpy.dataset.dataid import ModifierTuple
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.pmw_channels_definitions import FrequencyDoubleSideBand, FrequencyRange
from satpy.tests.utils import make_dataid

MHS_YAML_READER_DICT = {
    'reader': {'name': 'mhs_l1c_aapp',
               'description': 'AAPP l1c Reader for AMSU-B/MHS data',
               'sensors': ['mhs'],
               'default_channels': [1, 2, 3, 4, 5],
               'data_identification_keys': {'name': {'required': True},
                                            'frequency_double_sideband':
                                                {'type': FrequencyDoubleSideBand},
                                            'frequency_range': {'type': FrequencyRange},
                                            'resolution': None,
                                            'polarization': {'enum': ['H', 'V']},
                                            'calibration': {'enum': ['brightness_temperature'], 'transitive': True},
                                            'modifiers': {'required': True,
                                                          'default': [],
                                                          'type': ModifierTuple}},
               'config_files': ('satpy/etc/readers/mhs_l1c_aapp.yaml',)},
    'datasets': {'1': {'name': '1',
                       'frequency_range': {'central': 89.0, 'bandwidth': 2.8, 'unit': 'GHz'},
                       'polarization': 'V',
                       'resolution': 16000,
                       'calibration': {'brightness_temperature': {'standard_name': 'toa_brightness_temperature'}},
                       'coordinates': ['longitude', 'latitude'],
                       'file_type': 'mhs_aapp_l1c'},
                 '2': {'name': '2',
                       'frequency_range': {'central': 157.0, 'bandwidth': 2.8, 'unit': 'GHz'},
                       'polarization': 'V',
                       'resolution': 16000,
                       'calibration': {'brightness_temperature': {'standard_name': 'toa_brightness_temperature'}},
                       'coordinates': ['longitude', 'latitude'],
                       'file_type': 'mhs_aapp_l1c'},
                 '3': {'name': '3',
                       'frequency_double_sideband': {'unit': 'GHz',
                                                     'central': 183.31,
                                                     'side': 1.0,
                                                     'bandwidth': 1.0},
                       'polarization': 'V',
                       'resolution': 16000,
                       'calibration': {'brightness_temperature': {'standard_name': 'toa_brightness_temperature'}},
                       'coordinates': ['longitude', 'latitude'],
                       'file_type': 'mhs_aapp_l1c'}},
    'file_types': {'mhs_aapp_l1c': {'file_reader': BaseFileHandler,
                                    'file_patterns': [
                                        'mhsl1c_{platform_shortname}_{start_time:%Y%m%d_%H%M}_{orbit_number:05d}.l1c']}}}  # noqa


class FakeFH(BaseFileHandler):
    """Fake file handler class."""

    def __init__(self, start_time, end_time):
        """Initialize fake file handler."""
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
        """Return start time."""
        return self._start_time

    @property
    def end_time(self):
        """Return end time."""
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
        self.assertEqual(yr._get_filebase(filename, pattern), expected)

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
        self.assertEqual(yr._match_filenames(filenames, pattern), {expected})

    def test_match_filenames_windows_forward_slash(self):
        """Check that matching filenames works on Windows with forward slashes.

        This is common from Qt5 which internally uses forward slashes everywhere.

        """
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
        filenames = [os.path.join(base_dir, 'Oa05_radiance.nc').replace(os.sep, '/'),
                     os.path.join(base_dir, 'geo_coordinates.nc').replace(os.sep, '/')]
        expected = os.path.join(base_dir, 'geo_coordinates.nc').replace(os.sep, '/')
        self.assertEqual(yr._match_filenames(filenames, pattern), {expected})

    def test_listify_string(self):
        """Check listify_string."""
        self.assertEqual(yr.listify_string(None), [])
        self.assertEqual(yr.listify_string('some string'), ['some string'])
        self.assertEqual(yr.listify_string(['some', 'string']),
                         ['some', 'string'])


class DummyReader(BaseFileHandler):
    """Dummy reader instance."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the dummy reader."""
        super(DummyReader, self).__init__(
            filename, filename_info, filetype_info)
        self._start_time = datetime(2000, 1, 1, 12, 1)
        self._end_time = datetime(2000, 1, 1, 12, 2)
        self.metadata = {}

    @property
    def start_time(self):
        """Return start time."""
        return self._start_time

    @property
    def end_time(self):
        """Return end time."""
        return self._end_time


class TestFileFileYAMLReaderMultiplePatterns(unittest.TestCase):
    """Test units from FileYAMLReader with multiple readers."""

    def setUp(self):
        """Prepare a reader instance with a fake config."""
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

        self.config = res_dict
        self.reader = yr.FileYAMLReader(self.config,
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

    def test_serializable(self):
        """Check that a reader is serializable by dask.

        This ensures users are able to serialize a Scene object that contains
        readers.
        """
        from distributed.protocol import deserialize, serialize
        filelist = ['a001.bla', 'a002.bla', 'a001.bla', 'a002.bla',
                    'abcd.bla', 'k001.bla', 'a003.bli']

        self.reader.create_filehandlers(filelist)
        cloned_reader = deserialize(*serialize(self.reader))
        assert self.reader.file_handlers.keys() == cloned_reader.file_handlers.keys()
        assert self.reader.all_ids == cloned_reader.all_ids


class TestFileYAMLReaderWithCustomIDKey(unittest.TestCase):
    """Test units from FileYAMLReader with custom id_keys."""

    def setUp(self):
        """Set up the test case."""
        self.config = MHS_YAML_READER_DICT
        self.reader = yr.FileYAMLReader(MHS_YAML_READER_DICT,
                                        filter_parameters={
                                            'start_time': datetime(2000, 1, 1),
                                            'end_time': datetime(2000, 1, 2),
                                        })

    def test_custom_type_with_dict_contents_gets_parsed_correctly(self):
        """Test custom type with dictionary contents gets parsed correctly."""
        ds_ids = list(self.reader.all_dataset_ids)
        assert ds_ids[0]["frequency_range"] == FrequencyRange(89., 2.8, "GHz")

        assert ds_ids[2]["frequency_double_sideband"] == FrequencyDoubleSideBand(183.31, 1., 1., "GHz")


class TestFileFileYAMLReader(unittest.TestCase):
    """Test units from FileYAMLReader."""

    def setUp(self):
        """Prepare a reader instance with a fake config."""
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

        self.config = res_dict
        self.reader = yr.FileYAMLReader(res_dict,
                                        filter_parameters={
                                            'start_time': datetime(2000, 1, 1),
                                            'end_time': datetime(2000, 1, 2),
                                        })

    def test_deprecated_passing_config_files(self):
        """Test that we get an exception when config files are passed to inti."""
        self.assertRaises(ValueError, yr.FileYAMLReader, '/path/to/some/file.yaml')

    def test_all_data_ids(self):
        """Check that all datasets ids are returned."""
        for dataid in self.reader.all_dataset_ids:
            name = dataid['name'].replace('0', '')
            assert self.config['datasets'][name]['name'] == dataid['name']
            if 'wavelength' in self.config['datasets'][name]:
                assert self.config['datasets'][name]['wavelength'] == list(dataid['wavelength'])[:3]
            if 'calibration' in self.config['datasets'][name]:
                assert self.config['datasets'][name]['calibration'] == dataid['calibration']

    def test_all_dataset_names(self):
        """Get all dataset names."""
        self.assertSetEqual(self.reader.all_dataset_names,
                            set(['ch01', 'ch02', 'lons', 'lats']))

    def test_available_dataset_ids(self):
        """Get ids of the available datasets."""
        loadables = self.reader.select_files_from_pathnames(['a001.bla'])
        self.reader.create_filehandlers(loadables)
        self.assertSetEqual(set(self.reader.available_dataset_ids),
                            {make_dataid(name='ch02',
                                         wavelength=(0.7, 0.75, 0.8),
                                         calibration='counts',
                                         modifiers=()),
                             make_dataid(name='ch01',
                                         wavelength=(0.5, 0.6, 0.7),
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

        from fsspec.implementations.local import LocalFileSystem

        class Silly(LocalFileSystem):
            def glob(self, pattern):
                return ["/grocery/apricot.nc", "/grocery/aubergine.nc"]

        res = self.reader.select_files_from_directory(dpath, fs=Silly())
        self.assertEqual(
            res,
            {"/grocery/apricot.nc", "/grocery/aubergine.nc"})

    def test_supports_sensor(self):
        """Check supports_sensor."""
        self.assertTrue(self.reader.supports_sensor('canon'))
        self.assertFalse(self.reader.supports_sensor('nikon'))

    @patch('satpy.readers.yaml_reader.StackedAreaDefinition')
    def test_load_area_def(self, sad):
        """Test loading the area def for the reader."""
        dataid = MagicMock()
        file_handlers = []
        items = random.randrange(2, 10)
        for _i in range(items):
            file_handlers.append(MagicMock())
        final_area = self.reader._load_area_def(dataid, file_handlers)
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
        ds_q = DataQuery(name='ch01', wavelength=(0.5, 0.6, 0.7, 'µm'),
                         calibration='reflectance', modifiers=())
        res = self.reader._get_coordinates_for_dataset_key(ds_q)
        self.assertListEqual(res,
                             [make_dataid(name='lons'),
                              make_dataid(name='lats')])

    def test_get_coordinates_for_dataset_key_without(self):
        """Test getting coordinates for a key without coordinates."""
        ds_id = make_dataid(name='lons',
                            modifiers=())
        res = self.reader._get_coordinates_for_dataset_key(ds_id)
        self.assertListEqual(res, [])

    def test_get_coordinates_for_dataset_keys(self):
        """Test getting coordinates for keys."""
        ds_id1 = make_dataid(name='ch01', wavelength=(0.5, 0.6, 0.7),
                             calibration='reflectance', modifiers=())
        ds_id2 = make_dataid(name='ch02', wavelength=(0.7, 0.75, 0.8),
                             calibration='counts', modifiers=())
        lons = make_dataid(name='lons', modifiers=())
        lats = make_dataid(name='lats', modifiers=())

        res = self.reader._get_coordinates_for_dataset_keys([ds_id1, ds_id2,
                                                             lons])
        expected = {ds_id1: [lons, lats], ds_id2: [lons, lats], lons: []}

        self.assertDictEqual(res, expected)

    def test_get_file_handlers(self):
        """Test getting filehandler to load a dataset."""
        ds_id1 = make_dataid(name='ch01', wavelength=(0.5, 0.6, 0.7),
                             calibration='reflectance', modifiers=())
        self.reader.file_handlers = {'ftype1': 'bla'}

        self.assertEqual(self.reader._get_file_handlers(ds_id1), 'bla')

        lons = make_dataid(name='lons', modifiers=())
        self.assertEqual(self.reader._get_file_handlers(lons), None)

    @patch('satpy.readers.yaml_reader.xr')
    def test_load_entire_dataset(self, xarray):
        """Check loading an entire dataset."""
        file_handlers = [FakeFH(None, None), FakeFH(None, None),
                         FakeFH(None, None), FakeFH(None, None)]

        proj = self.reader._load_dataset(None, {}, file_handlers)

        self.assertIs(proj, xarray.concat.return_value)


class TestFileYAMLReaderLoading(unittest.TestCase):
    """Tests for FileYAMLReader.load."""

    def setUp(self):
        """Prepare a reader instance with a fake config."""
        patterns = ['a{something:3s}.bla']
        res_dict = {'reader': {'name': 'fake',
                               'sensors': ['canon']},
                    'file_types': {'ftype1': {'name': 'ft1',
                                              'file_reader': BaseFileHandler,
                                              'file_patterns': patterns}},
                    'datasets': {'ch1': {'name': 'ch01',
                                         'wavelength': [0.5, 0.6, 0.7],
                                         'calibration': 'reflectance',
                                         'file_type': 'ftype1'},
                                 }}

        self.config = res_dict
        self.reader = yr.FileYAMLReader(res_dict,
                                        filter_parameters={
                                            'start_time': datetime(2000, 1, 1),
                                            'end_time': datetime(2000, 1, 2),
                                        })
        fake_fh = FakeFH(None, None)
        self.lons = xr.DataArray(np.ones((2, 2)) * 2,
                                 dims=['y', 'x'],
                                 attrs={'standard_name': 'longitude',
                                        'name': 'longitude'})
        self.lats = xr.DataArray(np.ones((2, 2)) * 2,
                                 dims=['y', 'x'],
                                 attrs={'standard_name': 'latitude',
                                        'name': 'latitude'})
        self.data = None

        def _assign_array(dsid, *_args, **_kwargs):
            if dsid['name'] == 'longitude':
                return self.lons
            if dsid['name'] == 'latitude':
                return self.lats
            return self.data

        fake_fh.get_dataset.side_effect = _assign_array
        self.reader.file_handlers = {'ftype1': [fake_fh]}

    def test_load_dataset_with_builtin_coords(self):
        """Test loading a dataset with builtin coordinates."""
        self.data = xr.DataArray(np.ones((2, 2)),
                                 coords={'longitude': self.lons,
                                         'latitude': self.lats},
                                 dims=['y', 'x'])

        self._check_area_for_ch01()

    def test_load_dataset_with_builtin_coords_in_wrong_order(self):
        """Test loading a dataset with builtin coordinates in the wrong order."""
        self.data = xr.DataArray(np.ones((2, 2)),
                                 coords={'latitude': self.lats,
                                         'longitude': self.lons},
                                 dims=['y', 'x'])

        self._check_area_for_ch01()

    def _check_area_for_ch01(self):
        res = self.reader.load(['ch01'])
        assert 'area' in res['ch01'].attrs
        np.testing.assert_array_equal(res['ch01'].attrs['area'].lons, self.lons)
        np.testing.assert_array_equal(res['ch01'].attrs['area'].lats, self.lats)
        assert res['ch01'].attrs.get("reader") == "fake"


class TestFileFileYAMLReaderMultipleFileTypes(unittest.TestCase):
    """Test units from FileYAMLReader with multiple file types."""

    def setUp(self):
        """Prepare a reader instance with a fake config."""
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

        self.config = res_dict
        self.reader = yr.FileYAMLReader(self.config)

    def test_update_ds_ids_from_file_handlers(self):
        """Test updating existing dataset IDs with information from the file."""
        from functools import partial
        orig_ids = self.reader.all_ids

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
                    if ftype in file_types:
                        self.assertEqual(resol, ds_id['resolution'])


# Test methods


def available_datasets(self, configured_datasets=None):
    """Fake available_datasets for testing multiple file types."""
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
    """Fake file_type_matches for testing multiple file types."""
    if isinstance(ds_ftype, str) and ds_ftype == self.filetype_info['file_type']:
        return True
    if self.filetype_info['file_type'] in ds_ftype:
        return True
    return None


class TestGEOFlippableFileYAMLReader(unittest.TestCase):
    """Test GEOFlippableFileYAMLReader."""

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch.object(yr.FileYAMLReader, "_load_dataset_with_area")
    def test_load_dataset_with_area_for_single_areas(self, ldwa):
        """Test _load_dataset_with_area() for single area definitions."""
        from pyresample.geometry import AreaDefinition

        from satpy.readers.yaml_reader import GEOFlippableFileYAMLReader

        reader = GEOFlippableFileYAMLReader()

        dsid = MagicMock()
        coords = MagicMock()

        # create a dummy upright xarray
        original_area_extent = (-1500, -1000, 1500, 1000)
        original_array = np.arange(6).reshape((2, 3))

        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            {'proj': 'geos',
             'h': 35785831,
             'type': 'crs'},
            3,
            2,
            original_area_extent,
        )

        dummy_ds_xr = xr.DataArray(original_array,
                                   coords={'y': np.arange(2),
                                           'x': np.arange(3),
                                           'time': ("y", np.arange(2))},
                                   attrs={'area': area_def},
                                   dims=('y', 'x'))
        # assign the dummy xr as return for the super _load_dataset_with_area method
        ldwa.return_value = dummy_ds_xr

        # check no input, nothing should change
        res = reader._load_dataset_with_area(dsid, coords)
        np.testing.assert_equal(res.values, original_array)
        np.testing.assert_equal(res.attrs['area'].area_extent, original_area_extent)
        np.testing.assert_equal(res.coords['y'], np.arange(2))
        np.testing.assert_equal(res.coords['x'], np.arange(3))
        np.testing.assert_equal(res.coords['time'], np.arange(2))

        # check wrong input
        with self.assertRaises(ValueError):
            _ = reader._load_dataset_with_area(dsid, coords, 'wronginput')

        # check native orientation, nothing should change
        res = reader._load_dataset_with_area(dsid, coords, 'native')
        np.testing.assert_equal(res.values, original_array)
        np.testing.assert_equal(res.attrs['area'].area_extent, original_area_extent)
        np.testing.assert_equal(res.coords['y'], np.arange(2))
        np.testing.assert_equal(res.coords['x'], np.arange(3))
        np.testing.assert_equal(res.coords['time'], np.arange(2))

        # check upright orientation, nothing should change since area is already upright
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, original_array)
        np.testing.assert_equal(res.attrs['area'].area_extent, original_area_extent)
        np.testing.assert_equal(res.coords['y'], np.arange(2))
        np.testing.assert_equal(res.coords['x'], np.arange(3))
        np.testing.assert_equal(res.coords['time'], np.arange(2))

        # check that left-right image is flipped correctly
        dummy_ds_xr.attrs['area'] = area_def.copy(area_extent=(1500, -1000, -1500, 1000))
        ldwa.return_value = dummy_ds_xr.copy()
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, np.fliplr(original_array))
        np.testing.assert_equal(res.attrs['area'].area_extent, original_area_extent)
        np.testing.assert_equal(res.coords['y'], np.arange(2))
        np.testing.assert_equal(res.coords['x'], np.flip(np.arange(3)))
        np.testing.assert_equal(res.coords['time'], np.arange(2))

        # check that upside down image is flipped correctly
        dummy_ds_xr.attrs['area'] = area_def.copy(area_extent=(-1500, 1000, 1500, -1000))
        ldwa.return_value = dummy_ds_xr.copy()
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, np.flipud(original_array))
        np.testing.assert_equal(res.attrs['area'].area_extent, original_area_extent)
        np.testing.assert_equal(res.coords['y'], np.flip(np.arange(2)))
        np.testing.assert_equal(res.coords['x'], np.arange(3))
        np.testing.assert_equal(res.coords['time'], np.flip(np.arange(2)))

        # check different projection than geos, nothing should be changed
        area_def = AreaDefinition(
            'test',
            'test',
            'test',
            {'proj': 'lcc',
             'lat_1': 25.0,
             'type': 'crs'},
            3,
            2,
            original_area_extent,
        )

        dummy_ds_xr = xr.DataArray(original_array,
                                   dims=('y', 'x'),
                                   attrs={'area': area_def})
        ldwa.return_value = dummy_ds_xr
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, original_array)
        np.testing.assert_equal(res.attrs['area'].area_extent, original_area_extent)

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch.object(yr.FileYAMLReader, "_load_dataset_with_area")
    def test_load_dataset_with_area_for_stacked_areas(self, ldwa):
        """Test _load_dataset_with_area() for stacked area definitions."""
        from pyresample.geometry import AreaDefinition, StackedAreaDefinition

        from satpy.readers.yaml_reader import GEOFlippableFileYAMLReader

        reader = GEOFlippableFileYAMLReader()

        dsid = MagicMock()
        coords = MagicMock()

        # create a dummy upright xarray
        original_area_extents = [(-1500, -1000, 1500, 1000), (3000, 5000, 7000, 8000)]
        original_array = np.arange(12).reshape((4, 3))

        area_def0 = AreaDefinition(
            'test',
            'test',
            'test',
            {'proj': 'geos',
             'h': 35785831,
             'type': 'crs'},
            3,
            2,
            original_area_extents[0],
        )
        area_def1 = area_def0.copy(area_extent=original_area_extents[1])

        dummy_ds_xr = xr.DataArray(original_array,
                                   dims=('y', 'x'),
                                   coords={'y': np.arange(4),
                                           'x': np.arange(3),
                                           'time': ("y", np.arange(4))},
                                   attrs={'area': StackedAreaDefinition(area_def0, area_def1)})

        # check that left-right image is flipped correctly
        dummy_ds_xr.attrs['area'].defs[0] = area_def0.copy(area_extent=(1500, -1000, -1500, 1000))
        dummy_ds_xr.attrs['area'].defs[1] = area_def1.copy(area_extent=(7000, 5000, 3000, 8000))
        ldwa.return_value = dummy_ds_xr.copy()
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, np.fliplr(original_array))
        np.testing.assert_equal(res.attrs['area'].defs[0].area_extent, original_area_extents[0])
        np.testing.assert_equal(res.attrs['area'].defs[1].area_extent, original_area_extents[1])
        np.testing.assert_equal(res.coords['y'], np.arange(4))
        np.testing.assert_equal(res.coords['x'], np.flip(np.arange(3)))
        np.testing.assert_equal(res.coords['time'], np.arange(4))

        # check that upside down image is flipped correctly
        dummy_ds_xr.attrs['area'].defs[0] = area_def0.copy(area_extent=(-1500, 1000, 1500, -1000))
        dummy_ds_xr.attrs['area'].defs[1] = area_def1.copy(area_extent=(3000, 8000, 7000, 5000))
        ldwa.return_value = dummy_ds_xr.copy()
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, np.flipud(original_array))
        # note that the order of the stacked areadefs is flipped here, as expected
        np.testing.assert_equal(res.attrs['area'].defs[1].area_extent, original_area_extents[0])
        np.testing.assert_equal(res.attrs['area'].defs[0].area_extent, original_area_extents[1])
        np.testing.assert_equal(res.coords['y'], np.flip(np.arange(4)))
        np.testing.assert_equal(res.coords['x'], np.arange(3))
        np.testing.assert_equal(res.coords['time'], np.flip(np.arange(4)))

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch.object(yr.FileYAMLReader, "_load_dataset_with_area")
    def test_load_dataset_with_area_for_swath_def_data(self, ldwa):
        """Test _load_dataset_with_area() for swath definition data."""
        from pyresample.geometry import SwathDefinition

        from satpy.readers.yaml_reader import GEOFlippableFileYAMLReader

        reader = GEOFlippableFileYAMLReader()

        dsid = MagicMock()
        coords = MagicMock()

        # create a dummy upright xarray
        original_array = np.ones(3)
        dim = np.arange(3)
        lats = np.arange(3)
        lons = np.arange(3)

        swath_def = SwathDefinition(lons, lats)
        dummy_ds_xr = xr.DataArray(original_array,
                                   coords={'y': dim},
                                   attrs={'area': swath_def},
                                   dims=('y',))

        # assign the dummy xr as return for the super _load_dataset_with_area method
        ldwa.return_value = dummy_ds_xr

        # returned dataset should be unchanged since datasets with a swath definition are not flippable
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, original_array)

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch.object(yr.FileYAMLReader, "_load_dataset_with_area")
    def test_load_dataset_with_area_for_data_without_area(self, ldwa):
        """Test _load_dataset_with_area() for data wihtout area information."""
        from satpy.readers.yaml_reader import GEOFlippableFileYAMLReader

        reader = GEOFlippableFileYAMLReader()

        dsid = MagicMock()
        coords = MagicMock()

        # create a dummy upright xarray
        original_array = np.ones(3)
        dim = np.arange(3)

        dummy_ds_xr = xr.DataArray(original_array,
                                   coords={'y': dim},
                                   attrs={},
                                   dims=('y',))

        # assign the dummy xr as return for the super _load_dataset_with_area method
        ldwa.return_value = dummy_ds_xr

        # returned dataset should be unchanged since datasets without area information are not flippable
        res = reader._load_dataset_with_area(dsid, coords, 'NE')
        np.testing.assert_equal(res.values, original_array)


def _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, chk_pos_info):
    seg_area = MagicMock()
    seg_area.crs = 'some_crs'
    seg_area.area_extent = aex
    seg_area.shape = ashape
    get_area_def = MagicMock()
    get_area_def.return_value = seg_area

    get_segment_position_info = MagicMock()
    get_segment_position_info.return_value = chk_pos_info

    fh = MagicMock()
    filetype_info = {'expected_segments': expected_segments,
                     'file_type': 'filetype1'}
    filename_info = {'segment': segment}
    fh.filetype_info = filetype_info
    fh.filename_info = filename_info
    fh.get_area_def = get_area_def
    fh.get_segment_position_info = get_segment_position_info

    return fh, seg_area


class TestGEOSegmentYAMLReader(unittest.TestCase):
    """Test GEOSegmentYAMLReader."""

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch.object(yr.FileYAMLReader, "create_filehandlers")
    def test_get_expected_segments(self, cfh):
        """Test that expected segments can come from the filename."""
        from satpy.readers.yaml_reader import GEOSegmentYAMLReader
        reader = GEOSegmentYAMLReader()

        fake_fh = MagicMock()
        fake_fh.filename_info = {}
        fake_fh.filetype_info = {}
        cfh.return_value = {'ft1': [fake_fh]}

        # default (1)
        created_fhs = reader.create_filehandlers(['fake.nc'])
        es = created_fhs['ft1'][0].filetype_info['expected_segments']
        self.assertEqual(es, 1)

        # YAML defined for each file type
        fake_fh.filetype_info['expected_segments'] = 2
        created_fhs = reader.create_filehandlers(['fake.nc'])
        es = created_fhs['ft1'][0].filetype_info['expected_segments']
        self.assertEqual(es, 2)

        # defined both in the filename and the YAML metadata
        # YAML has priority
        fake_fh.filename_info = {'total_segments': 3}
        fake_fh.filetype_info = {'expected_segments': 2}
        created_fhs = reader.create_filehandlers(['fake.nc'])
        es = created_fhs['ft1'][0].filetype_info['expected_segments']
        self.assertEqual(es, 2)

        # defined in the filename
        fake_fh.filename_info = {'total_segments': 3}
        fake_fh.filetype_info = {}
        created_fhs = reader.create_filehandlers(['fake.nc'])
        es = created_fhs['ft1'][0].filetype_info['expected_segments']
        self.assertEqual(es, 3)

        # check correct FCI segment (aka chunk in the FCI world) number reading into segment
        fake_fh.filename_info = {'count_in_repeat_cycle': 5}
        created_fhs = reader.create_filehandlers(['fake.nc'])
        es = created_fhs['ft1'][0].filename_info['segment']
        self.assertEqual(es, 5)

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch('satpy.readers.yaml_reader.FileYAMLReader._load_dataset')
    @patch('satpy.readers.yaml_reader.xr')
    @patch('satpy.readers.yaml_reader._find_missing_segments')
    def test_load_dataset(self, mss, xr, parent_load_dataset):
        """Test _load_dataset()."""
        from satpy.readers.yaml_reader import GEOSegmentYAMLReader
        reader = GEOSegmentYAMLReader()

        # Projectable is None
        mss.return_value = [0, 0, 0, False, None]
        with self.assertRaises(KeyError):
            res = reader._load_dataset(None, None, None)
        # Failure is True
        mss.return_value = [0, 0, 0, True, 0]
        with self.assertRaises(KeyError):
            res = reader._load_dataset(None, None, None)

        # Setup input, and output of mocked functions
        counter = 9
        expected_segments = 8
        seg = MagicMock(dims=['y', 'x'])
        slice_list = expected_segments * [seg, ]
        failure = False
        projectable = MagicMock()
        mss.return_value = (counter, expected_segments, slice_list,
                            failure, projectable)
        empty_segment = MagicMock()
        xr.full_like.return_value = empty_segment
        concat_slices = MagicMock()
        xr.concat.return_value = concat_slices
        dataid = MagicMock()
        ds_info = MagicMock()
        file_handlers = MagicMock()

        # No missing segments
        res = reader._load_dataset(dataid, ds_info, file_handlers)
        self.assertTrue(res.attrs is file_handlers[0].combine_info.return_value)
        self.assertTrue(empty_segment not in slice_list)

        # One missing segment in the middle
        slice_list[4] = None
        counter = 8
        mss.return_value = (counter, expected_segments, slice_list,
                            failure, projectable)
        res = reader._load_dataset(dataid, ds_info, file_handlers)
        self.assertTrue(slice_list[4] is empty_segment)

        # The last segment is missing
        slice_list = expected_segments * [seg, ]
        slice_list[-1] = None
        counter = 8
        mss.return_value = (counter, expected_segments, slice_list,
                            failure, projectable)
        res = reader._load_dataset(dataid, ds_info, file_handlers)
        self.assertTrue(slice_list[-1] is empty_segment)

        # The last two segments are missing
        slice_list = expected_segments * [seg, ]
        slice_list[-1] = None
        counter = 7
        mss.return_value = (counter, expected_segments, slice_list,
                            failure, projectable)
        res = reader._load_dataset(dataid, ds_info, file_handlers)
        self.assertTrue(slice_list[-1] is empty_segment)
        self.assertTrue(slice_list[-2] is empty_segment)

        # The first segment is missing
        slice_list = expected_segments * [seg, ]
        slice_list[0] = None
        counter = 9
        mss.return_value = (counter, expected_segments, slice_list,
                            failure, projectable)
        res = reader._load_dataset(dataid, ds_info, file_handlers)
        self.assertTrue(slice_list[0] is empty_segment)

        # The first two segments are missing
        slice_list = expected_segments * [seg, ]
        slice_list[0] = None
        slice_list[1] = None
        counter = 9
        mss.return_value = (counter, expected_segments, slice_list,
                            failure, projectable)
        res = reader._load_dataset(dataid, ds_info, file_handlers)
        self.assertTrue(slice_list[0] is empty_segment)
        self.assertTrue(slice_list[1] is empty_segment)

        # Disable padding
        res = reader._load_dataset(dataid, ds_info, file_handlers,
                                   pad_data=False)
        parent_load_dataset.assert_called_once_with(dataid, ds_info,
                                                    file_handlers)

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch('satpy.readers.yaml_reader._load_area_def')
    @patch('satpy.readers.yaml_reader._stack_area_defs')
    @patch('satpy.readers.yaml_reader.GEOSegmentYAMLReader._pad_earlier_segments_area')
    @patch('satpy.readers.yaml_reader.GEOSegmentYAMLReader._pad_later_segments_area')
    def test_load_area_def(self, pesa, plsa, sad, parent_load_area_def):
        """Test _load_area_def()."""
        from satpy.readers.yaml_reader import GEOSegmentYAMLReader
        reader = GEOSegmentYAMLReader()

        dataid = MagicMock()
        file_handlers = MagicMock()
        reader._load_area_def(dataid, file_handlers)
        pesa.assert_called_once()
        plsa.assert_called_once()
        sad.assert_called_once()
        parent_load_area_def.assert_not_called()
        # Disable padding
        reader._load_area_def(dataid, file_handlers, pad_data=False)
        parent_load_area_def.assert_called_once_with(dataid, file_handlers)

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch('satpy.readers.yaml_reader.AreaDefinition')
    def test_pad_later_segments_area(self, AreaDefinition):
        """Test _pad_later_segments_area()."""
        from satpy.readers.yaml_reader import GEOSegmentYAMLReader
        reader = GEOSegmentYAMLReader()

        expected_segments = 2
        segment = 1
        aex = [0, 1000, 200, 500]
        ashape = [200, 500]
        fh_1, _ = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, None)
        file_handlers = [fh_1]
        dataid = 'dataid'
        res = reader._pad_later_segments_area(file_handlers, dataid)
        self.assertEqual(len(res), 2)
        seg2_extent = (0, 1500, 200, 1000)
        expected_call = ('fill', 'fill', 'fill', 'some_crs', 500, 200,
                         seg2_extent)
        AreaDefinition.assert_called_once_with(*expected_call)

    @patch.object(yr.FileYAMLReader, "__init__", lambda x: None)
    @patch('satpy.readers.yaml_reader.AreaDefinition')
    def test_pad_earlier_segments_area(self, AreaDefinition):
        """Test _pad_earlier_segments_area()."""
        from satpy.readers.yaml_reader import GEOSegmentYAMLReader
        reader = GEOSegmentYAMLReader()

        expected_segments = 2
        segment = 2
        aex = [0, 1000, 200, 500]
        ashape = [200, 500]
        fh_2, seg2_area = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, None)

        file_handlers = [fh_2]
        dataid = 'dataid'
        area_defs = {2: seg2_area}
        res = reader._pad_earlier_segments_area(file_handlers, dataid, area_defs)
        self.assertEqual(len(res), 2)
        seg1_extent = (0, 500, 200, 0)
        expected_call = ('fill', 'fill', 'fill', 'some_crs', 500, 200,
                         seg1_extent)
        AreaDefinition.assert_called_once_with(*expected_call)

    def test_find_missing_segments(self):
        """Test _find_missing_segments()."""
        from satpy.readers.yaml_reader import _find_missing_segments as fms

        # Dataset with only one segment
        filename_info = {'segment': 1}
        fh_seg1 = MagicMock(filename_info=filename_info)
        projectable = 'projectable'
        get_dataset = MagicMock()
        get_dataset.return_value = projectable
        fh_seg1.get_dataset = get_dataset
        file_handlers = [fh_seg1]
        ds_info = {'file_type': []}
        dataid = 'dataid'
        res = fms(file_handlers, ds_info, dataid)
        counter, expected_segments, slice_list, failure, proj = res
        self.assertEqual(counter, 2)
        self.assertEqual(expected_segments, 1)
        self.assertTrue(projectable in slice_list)
        self.assertFalse(failure)
        self.assertTrue(proj is projectable)

        # Three expected segments, first and last missing
        filename_info = {'segment': 2}
        filetype_info = {'expected_segments': 3,
                         'file_type': 'foo'}
        fh_seg2 = MagicMock(filename_info=filename_info,
                            filetype_info=filetype_info)
        projectable = 'projectable'
        get_dataset = MagicMock()
        get_dataset.return_value = projectable
        fh_seg2.get_dataset = get_dataset
        file_handlers = [fh_seg2]
        ds_info = {'file_type': ['foo']}
        dataid = 'dataid'
        res = fms(file_handlers, ds_info, dataid)
        counter, expected_segments, slice_list, failure, proj = res
        self.assertEqual(counter, 3)
        self.assertEqual(expected_segments, 3)
        self.assertEqual(slice_list, [None, projectable, None])
        self.assertFalse(failure)
        self.assertTrue(proj is projectable)


@pytest.fixture
@patch.object(yr.GEOVariableSegmentYAMLReader, "__init__", lambda x: None)
def GVSYReader():
    """Get a fixture of the GEOVariableSegmentYAMLReader."""
    from satpy.readers.yaml_reader import GEOVariableSegmentYAMLReader
    reader = GEOVariableSegmentYAMLReader()
    reader.segment_infos = dict()
    reader.segment_heights = cache(reader._segment_heights)
    return reader


@pytest.fixture
def fake_geswh():
    """Get a fixture of the patched _get_empty_segment_with_height."""
    with patch('satpy.readers.yaml_reader._get_empty_segment_with_height') as geswh:
        yield geswh


@pytest.fixture
def fake_xr():
    """Get a fixture of the patched xarray."""
    with patch('satpy.readers.yaml_reader.xr') as xr:
        yield xr


@pytest.fixture
def fake_mss():
    """Get a fixture of the patched _find_missing_segments."""
    with patch('satpy.readers.yaml_reader._find_missing_segments') as mss:
        yield mss


@pytest.fixture
def fake_adef():
    """Get a fixture of the patched AreaDefinition."""
    with patch('satpy.readers.yaml_reader.AreaDefinition') as adef:
        yield adef


class TestGEOVariableSegmentYAMLReader:
    """Test GEOVariableSegmentYAMLReader."""

    def test_get_empty_segment(self, GVSYReader, fake_mss, fake_xr, fake_geswh):
        """Test execution of (overridden) get_empty_segment inside _load_dataset."""
        # Setup input, and output of mocked functions for first segment missing
        chk_pos_info = {
            '1km': {'start_position_row': 0,
                    'end_position_row': 0,
                    'segment_height': 0,
                    'grid_width': 11136},
            '2km': {'start_position_row': 140,
                    'end_position_row': None,
                    'segment_height': 278,
                    'grid_width': 5568}
        }
        expected_segments = 2
        segment = 2
        aex = [0, 1000, 200, 500]
        ashape = [278, 5568]
        fh_2, _ = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, chk_pos_info)

        GVSYReader.file_handlers = {'filetype1': [fh_2]}

        counter = 2
        seg = MagicMock(dims=['y', 'x'])
        slice_list = [None, seg]
        failure = False
        projectable = MagicMock()
        empty_segment = MagicMock()
        empty_segment.shape = [278, 5568]
        fake_xr.full_like.return_value = empty_segment
        dataid = MagicMock()
        ds_info = MagicMock()

        fake_mss.return_value = (counter, expected_segments, slice_list,
                                 failure, projectable)
        GVSYReader._load_dataset(dataid, ds_info, [fh_2])
        # the return of get_empty_segment
        fake_geswh.assert_called_once_with(empty_segment, 139, dim='y')

    def test_pad_earlier_segments_area(self, GVSYReader, fake_adef):
        """Test _pad_earlier_segments_area() for the variable segment case."""
        # setting to 0 or None values that shouldn't be relevant
        chk_pos_info = {
            '1km': {'start_position_row': 0,
                    'end_position_row': 0,
                    'segment_height': 0,
                    'grid_width': 11136},
            '2km': {'start_position_row': 140,
                    'end_position_row': None,
                    'segment_height': 278,
                    'grid_width': 5568}
        }
        expected_segments = 2
        segment = 2
        aex = [0, 1000, 200, 500]
        ashape = [278, 5568]
        fh_2, seg2_area = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, chk_pos_info)

        GVSYReader.file_handlers = {'filetype1': [fh_2]}
        dataid = 'dataid'
        area_defs = {2: seg2_area}
        res = GVSYReader._pad_earlier_segments_area([fh_2], dataid, area_defs)
        assert len(res) == 2

        # The later vertical segment (nr. 2) size is 278, which is exactly double the size
        # of the gap left by the missing first segment (139, as the second segment starts at line 140).
        # Therefore, the new vertical area extent for the first segment should be
        # half of the previous size (1000-500)/2=250.
        # The new area extent lower-left row is therefore 500-250=250
        seg1_extent = (0, 500, 200, 250)
        expected_call = ('fill', 'fill', 'fill', 'some_crs', 5568, 139,
                         seg1_extent)
        fake_adef.assert_called_once_with(*expected_call)

    def test_pad_later_segments_area(self, GVSYReader, fake_adef):
        """Test _pad_later_segments_area() in the variable padding case."""
        chk_pos_info = {
            '1km': {'start_position_row': None,
                    'end_position_row': 11136 - 278,
                    'segment_height': 556,
                    'grid_width': 11136},
            '2km': {'start_position_row': 0,
                    'end_position_row': 0,
                    'segment_height': 0,
                    'grid_width': 5568}}

        expected_segments = 2
        segment = 1
        aex = [0, 1000, 200, 500]
        ashape = [556, 11136]
        fh_1, _ = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, chk_pos_info)
        GVSYReader.file_handlers = {'filetype1': [fh_1]}
        dataid = 'dataid'
        res = GVSYReader._pad_later_segments_area([fh_1], dataid)
        assert len(res) == 2

        # The previous segment size is 556, which is exactly double the size of the gap left
        # by the missing last segment (278, as the second-to-last segment ends at line 11136 - 278 )
        # therefore, the new vertical area extent should be half of the previous size (1000-500)/2=250.
        # The new area extent lower-left row is therefore 1000+250=1250
        seg2_extent = (0, 1250, 200, 1000)
        expected_call = ('fill', 'fill', 'fill', 'some_crs', 11136, 278,
                         seg2_extent)
        fake_adef.assert_called_once_with(*expected_call)

    def test_pad_later_segments_area_for_multiple_segments_gap(self, GVSYReader, fake_adef):
        """Test _pad_later_segments_area() in the variable padding case for multiple gaps with multiple segments."""

        def side_effect_areadef(a, b, c, crs, width, height, aex):
            m = MagicMock()
            m.shape = [height, width]
            m.area_extent = aex
            m.crs = crs
            return m

        fake_adef.side_effect = side_effect_areadef

        chk_pos_info = {
            '1km': {'start_position_row': 11136 - 600 - 100 + 1,
                    'end_position_row': 11136 - 600,
                    'segment_height': 100,
                    'grid_width': 11136},
            '2km': {'start_position_row': 0,
                    'end_position_row': 0,
                    'segment_height': 0,
                    'grid_width': 5568}}
        expected_segments = 8
        segment = 1
        aex = [0, 1000, 200, 500]
        ashape = [100, 11136]
        fh_1, _ = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, chk_pos_info)
        chk_pos_info = {
            '1km': {'start_position_row': 11136 - 300 - 100 + 1,
                    'end_position_row': 11136 - 300,
                    'segment_height': 100,
                    'grid_width': 11136},
            '2km': {'start_position_row': 0,
                    'end_position_row': 0,
                    'segment_height': 0,
                    'grid_width': 5568}}
        segment = 4
        fh_4, _ = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, chk_pos_info)

        chk_pos_info = {
            '1km': {'start_position_row': 11136 - 100 + 1,
                    'end_position_row': None,
                    'segment_height': 100,
                    'grid_width': 11136},
            '2km': {'start_position_row': 0,
                    'end_position_row': 0,
                    'segment_height': 0,
                    'grid_width': 5568}}
        segment = 8
        fh_8, _ = _create_mocked_fh_and_areadef(aex, ashape, expected_segments, segment, chk_pos_info)

        GVSYReader.file_handlers = {'filetype1': [fh_1, fh_4, fh_8]}
        dataid = 'dataid'
        res = GVSYReader._pad_later_segments_area([fh_1, fh_4, fh_8], dataid)
        assert len(res) == 8

        # Regarding the segment sizes:
        # First group of missing segments:
        # The end position row of the gap is the start row of the last available segment-1:11136-300-100+1-1=10736
        # The start position row of the gap is the end row fo the first available segment+1: 11136-600+1=10837
        # hence the gap is 10736-10537+1=200 px high
        # The 200px have to be split between two missing segments, the most equal way to do it is with
        # sizes 100: 100+100=200
        # Second group:
        # The end position row of the gap is the start row of the last segment -1: 11136-100+1-1=11036
        # The start position row of the gap is the end row fo the first segment +1: 11136-300+1=10837
        # hence the gap is 11036-10837+1=200 px high
        # The 200px have to be split between three missing segments, the most equal way to do it is with
        # sizes 66 and 67: 66+67+67=200

        # Regarding the heights:
        # First group:
        # The first segment has 100px height and 500 area extent height.
        # The first padded segment has 100px height -> 500*100/100=500 area extent height ->1000+500=1500
        # The second padded segment has 100px height -> 500*100/100=500 area extent height ->1500+500=2000
        # Second group:
        # The first segment has 100px height and 500 area extent height.
        # The first padded segment has 66px height -> 500*66/100=330 area extent height ->1000+330=1330
        # The second padded segment has 67px height -> 500*67/100=335 area extent height ->1330+335=1665
        # The first padded segment has 67px height -> 500*67/100=335 area extent height ->1665+335=2000
        assert fake_adef.call_count == 5
        expected_call1 = ('fill', 'fill', 'fill', 'some_crs', 11136, 100,
                          (0, 1500.0, 200, 1000))
        expected_call2 = ('fill', 'fill', 'fill', 'some_crs', 11136, 100,
                          (0, 2000.0, 200, 1500))
        expected_call3 = ('fill', 'fill', 'fill', 'some_crs', 11136, 66,
                          (0, 1330.0, 200, 1000))
        expected_call4 = ('fill', 'fill', 'fill', 'some_crs', 11136, 67,
                          (0, 1665.0, 200, 1330.0))
        expected_call5 = ('fill', 'fill', 'fill', 'some_crs', 11136, 67,
                          (0, 2000.0, 200, 1665.0))

        fake_adef.side_effect = None
        fake_adef.assert_has_calls([call(*expected_call1),
                                    call(*expected_call2),
                                    call(*expected_call3),
                                    call(*expected_call4),
                                    call(*expected_call5)
                                    ])

    def test_get_empty_segment_with_height(self):
        """Test _get_empty_segment_with_height()."""
        from satpy.readers.yaml_reader import _get_empty_segment_with_height as geswh

        dim = 'y'

        # check expansion of empty segment
        empty_segment = xr.DataArray(np.ones((139, 5568)), dims=['y', 'x'])
        new_height = 140
        new_empty_segment = geswh(empty_segment, new_height, dim)
        assert new_empty_segment.shape == (140, 5568)

        # check reduction of empty segment
        empty_segment = xr.DataArray(np.ones((140, 5568)), dims=['y', 'x'])
        new_height = 139
        new_empty_segment = geswh(empty_segment, new_height, dim)
        assert new_empty_segment.shape == (139, 5568)

        # check that empty segment is not modified if it has the right height already
        empty_segment = xr.DataArray(np.ones((140, 5568)), dims=['y', 'x'])
        new_height = 140
        new_empty_segment = geswh(empty_segment, new_height, dim)
        assert new_empty_segment is empty_segment
