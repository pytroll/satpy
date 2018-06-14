#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author(s):
#
#   Panu Lahtinen <panu.lahtinen@fmi.fi
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


class TestDatasetDict(unittest.TestCase):
    """Test DatasetDict and its methods."""

    def setUp(self):
        """Create a test DatasetDict."""
        from satpy.dataset import DatasetID
        from satpy.readers import DatasetDict
        self.regular_dict = regular_dict = {
            DatasetID(name="test",
                      wavelength=(0, 0.5, 1),
                      resolution=1000): "1",
            DatasetID(name="testh",
                      wavelength=(0, 0.5, 1),
                      resolution=500): "1h",
            DatasetID(name="test2",
                      wavelength=(1, 1.5, 2),
                      resolution=1000): "2",
            DatasetID(name="test3",
                      wavelength=(1.2, 1.7, 2.2),
                      resolution=1000): "3",
            DatasetID(name="test4",
                      calibration="radiance",
                      polarization="V"): "4rad",
            DatasetID(name="test4",
                      calibration="reflectance",
                      polarization="H"): "4refl",
            DatasetID(name="test5",
                      modifiers=('mod1', 'mod2')): "5_2mod",
            DatasetID(name="test5",
                      modifiers=('mod2',)): "5_1mod",
            DatasetID(name='test6', level=100): '6_100',
            DatasetID(name='test6', level=200): '6_200',
        }
        self.test_dict = DatasetDict(regular_dict)

    def test_init_noargs(self):
        """Test DatasetDict init with no arguments."""
        from satpy.readers import DatasetDict
        d = DatasetDict()
        self.assertIsInstance(d, dict)

    def test_init_dict(self):
        """Test DatasetDict init with a regular dict argument."""
        from satpy.dataset import DatasetID
        from satpy.readers import DatasetDict
        regular_dict = {DatasetID(name="test", wavelength=(0, 0.5, 1)): "1", }
        d = DatasetDict(regular_dict)
        self.assertEqual(d, regular_dict)

    def test_getitem(self):
        """Test DatasetDict getitem with different arguments."""
        from satpy.dataset import DatasetID
        d = self.test_dict
        # access by name
        self.assertEqual(d["test"], "1")
        # access by exact wavelength
        self.assertEqual(d[1.5], "2")
        # access by near wavelength
        self.assertEqual(d[1.55], "2")
        # access by near wavelength of another dataset
        self.assertEqual(d[1.65], "3")
        # access by name with multiple levels
        self.assertEqual(d['test6'], '6_200')

        self.assertEqual(d[DatasetID(wavelength=1.5)], "2")
        self.assertEqual(d[DatasetID(wavelength=0.5, resolution=1000)], "1")
        self.assertEqual(d[DatasetID(wavelength=0.5, resolution=500)], "1h")
        self.assertEqual(d[DatasetID(name='test6', level=100)], '6_100')
        self.assertEqual(d[DatasetID(name='test6', level=200)], '6_200')

        # higher resolution is returned
        self.assertEqual(d[0.5], "1h")
        self.assertEqual(d['test4'], '4refl')
        self.assertEqual(d[DatasetID(name='test4', calibration='radiance')], '4rad')
        self.assertRaises(KeyError, d.getitem, '1h')

    def test_get_key(self):
        """Test 'get_key' special functions."""
        from satpy import DatasetID
        from satpy.readers import get_key
        d = self.test_dict
        res1 = get_key(DatasetID(name='test4'), d, calibration='radiance')
        res2 = get_key(DatasetID(name='test4'), d, calibration='radiance',
                       num_results=0)
        res3 = get_key(DatasetID(name='test4'), d, calibration='radiance',
                       num_results=3)
        self.assertEqual(len(res2), 1)
        self.assertEqual(len(res3), 1)
        res2 = res2[0]
        res3 = res3[0]
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

        res1 = get_key('test4', d, polarization='V')
        self.assertEqual(res1, DatasetID(name='test4', calibration='radiance',
                                         polarization='V'))

        res1 = get_key(0.5, d, resolution=500)
        self.assertEqual(res1, DatasetID(name='testh',
                                         wavelength=(0, 0.5, 1),
                                         resolution=500))

        res1 = get_key('test6', d, level=100)
        self.assertEqual(res1, DatasetID(name='test6',
                                         level=100))

        res1 = get_key('test5', d)
        res2 = get_key('test5', d, modifiers=('mod2',))
        res3 = get_key('test5', d, modifiers=('mod1', 'mod2',))
        self.assertEqual(res1, DatasetID(name='test5',
                                         modifiers=('mod2',)))
        self.assertEqual(res1, res2)
        self.assertNotEqual(res1, res3)

        # more than 1 result when default is to ask for 1 result
        self.assertRaises(KeyError, get_key, 'test4', d, best=False)

    def test_contains(self):
        """Test DatasetDict contains method."""
        from satpy.dataset import DatasetID
        d = self.test_dict
        self.assertIn('test', d)
        self.assertFalse(d.contains('test'))
        self.assertNotIn('test_bad', d)
        self.assertIn(0.5, d)
        self.assertFalse(d.contains(0.5))
        self.assertIn(1.5, d)
        self.assertIn(1.55, d)
        self.assertIn(1.65, d)
        self.assertIn(DatasetID(name='test4', calibration='radiance'), d)
        self.assertIn('test4', d)

    def test_keys(self):
        """Test keys method of DatasetDict."""
        from satpy import DatasetID
        d = self.test_dict
        self.assertEqual(len(d.keys()), len(self.regular_dict.keys()))
        self.assertTrue(all(isinstance(x, DatasetID) for x in d.keys()))
        name_keys = d.keys(names=True)
        self.assertListEqual(sorted(set(name_keys))[:4], [
            'test', 'test2', 'test3', 'test4'])
        wl_keys = tuple(d.keys(wavelengths=True))
        self.assertIn((0, 0.5, 1), wl_keys)
        self.assertIn((1, 1.5, 2), wl_keys)
        self.assertIn((1.2, 1.7, 2.2), wl_keys)
        self.assertIn(None, wl_keys)

    def test_setitem(self):
        """Test setitem method of DatasetDict."""
        d = self.test_dict
        d['new_ds'] = {'metadata': 'new_ds'}
        self.assertEqual(d['new_ds']['metadata'], 'new_ds')
        d[0.5] = {'calibration': 'radiance'}
        self.assertEqual(d[0.5]['resolution'], 500)
        self.assertEqual(d[0.5]['name'], 'testh')


class TestReaderLoader(unittest.TestCase):
    """Test the `load_readers` function.

    Assumes that the VIIRS SDR reader exists and works.
    """
    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler"""
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.tests.reader_tests.test_viirs_sdr import FakeHDF5FileHandler2
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIIRSSDRFileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler"""
        self.p.stop()

    def test_no_args(self):
        """Test no args provided.
        
        This should check the local directory which should have no files.
        """
        from satpy.readers import load_readers
        ri = load_readers()
        self.assertDictEqual(ri, {})

    def test_filenames_only(self):
        """Test with filenames specified"""
        from satpy.readers import load_readers
        ri = load_readers(filenames=[
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertListEqual(list(ri.keys()), ['viirs_sdr'])

    def test_filenames_and_reader(self):
        """Test with filenames and reader specified"""
        from satpy.readers import load_readers
        ri = load_readers(reader='viirs_sdr',
                filenames=[
                    'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertListEqual(list(ri.keys()), ['viirs_sdr'])

    def test_bad_reader_name_with_filenames(self):
        """Test bad reader name with filenames provided"""
        from satpy.readers import load_readers
        self.assertRaises(ValueError, load_readers, reader='i_dont_exist', filenames=[
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            ])

    def test_filenames_as_dict(self):
        """Test loading readers where filenames are organized by reader"""
        from satpy.readers import load_readers
        filenames = {
            'viirs_sdr': ['SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'],
        }
        ri = load_readers(filenames=filenames)
        self.assertListEqual(list(ri.keys()), ['viirs_sdr'])

    def test_filenames_as_dict_with_reader(self):
        """Test loading from a filenames dict with a single reader specified.

        This can happen in the deprecated Scene behavior of passing a reader
        and a base_dir.

        """
        from satpy.readers import load_readers
        filenames = {
            'viirs_sdr': ['SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'],
        }
        ri = load_readers(reader='viirs_sdr', filenames=filenames)
        self.assertListEqual(list(ri.keys()), ['viirs_sdr'])


class TestFindFilesAndReaders(unittest.TestCase):
    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler"""
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        from satpy.tests.reader_tests.test_viirs_sdr import FakeHDF5FileHandler2
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIIRSSDRFileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler"""
        self.p.stop()

    # def test_sensor(self):
    #     """Test with filenames and sensor specified"""
    #     from satpy.readers import load_readers
    #     ri = load_readers(sensor='viirs',
    #                       filenames=[
    #                           'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
    #                       ])
    #     self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
    #

    def test_reader_name(self):
        """Test with default base_dir and reader specified"""
        from satpy.readers import find_files_and_readers
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            ri = find_files_and_readers(reader='viirs_sdr')
            self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
            self.assertListEqual(ri['viirs_sdr'], [fn])
        finally:
            os.remove(fn)

    def test_reader_other_name(self):
        """Test with default base_dir and reader specified"""
        from satpy.readers import find_files_and_readers
        fn = 'S_NWC_CPP_npp_32505_20180204T1114116Z_20180204T1128227Z.nc'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            ri = find_files_and_readers(reader='nc_nwcsaf_pps')
            self.assertListEqual(list(ri.keys()), ['nc_nwcsaf_pps'])
            self.assertListEqual(ri['nc_nwcsaf_pps'], [fn])
        finally:
            os.remove(fn)

    def test_reader_name_matched_start_end_time(self):
        """Test with start and end time matching the filename"""
        from satpy.readers import find_files_and_readers
        from datetime import datetime, timedelta
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            ri = find_files_and_readers(reader='viirs_sdr',
                                        start_time=datetime(2012, 2, 25, 18, 0, 0),
                                        end_time=datetime(2012, 2, 25, 19, 0, 0),
                                        )
            self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
            self.assertListEqual(ri['viirs_sdr'], [fn])
        finally:
            os.remove(fn)

    def test_reader_name_matched_start_time(self):
        """Test with start matching the filename.

        Start time in the middle of the file time should still match the file.
        """
        from satpy.readers import find_files_and_readers
        from datetime import datetime, timedelta
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            ri = find_files_and_readers(reader='viirs_sdr',
                                        start_time=datetime(2012, 2, 25, 18, 1, 30),
                                        )
            self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
            self.assertListEqual(ri['viirs_sdr'], [fn])
        finally:
            os.remove(fn)

    def test_reader_name_matched_end_time(self):
        """Test with end matching the filename.

        End time in the middle of the file time should still match the file.

        """
        from satpy.readers import find_files_and_readers
        from datetime import datetime, timedelta
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            ri = find_files_and_readers(reader='viirs_sdr',
                                        end_time=datetime(2012, 2, 25, 18, 1, 30),
                                        )
            self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
            self.assertListEqual(ri['viirs_sdr'], [fn])
        finally:
            os.remove(fn)

    def test_reader_name_unmatched_start_end_time(self):
        """Test with start and end time matching the filename"""
        from satpy.readers import find_files_and_readers
        from datetime import datetime
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            self.assertRaises(ValueError, find_files_and_readers,
                              reader='viirs_sdr',
                              start_time=datetime(2012, 2, 26, 18, 0, 0),
                              end_time=datetime(2012, 2, 26, 19, 0, 0),
                              )
        finally:
            os.remove(fn)

    def test_no_parameters(self):
        """Test with no limiting parameters."""
        from satpy.readers import find_files_and_readers
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            ri = find_files_and_readers()
            self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
            self.assertListEqual(ri['viirs_sdr'], [fn])
        finally:
            os.remove(fn)

    def test_bad_sensor(self):
        """Test bad sensor doesn't find any files"""
        from satpy.readers import find_files_and_readers
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            self.assertRaises(ValueError, find_files_and_readers,
                              sensor='i_dont_exist')
        finally:
            os.remove(fn)

    def test_sensor(self):
        """Test that readers for the current sensor are loaded"""
        from satpy.readers import find_files_and_readers
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            # we can't easily know how many readers satpy has that support
            # 'viirs' so we just pass it and hope that this works
            ri = find_files_and_readers(sensor='viirs')
            self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
            self.assertListEqual(ri['viirs_sdr'], [fn])
        finally:
            os.remove(fn)

    def test_sensor_no_files(self):
        """Test that readers for the current sensor are loaded"""
        from satpy.readers import find_files_and_readers
        # we can't easily know how many readers satpy has that support
        # 'viirs' so we just pass it and hope that this works
        self.assertRaises(ValueError, find_files_and_readers,
                          sensor='viirs')


class TestYAMLFiles(unittest.TestCase):
    """Test and analyze the reader configuration files."""

    def test_filename_matches_reader_name(self):
        """Test that every reader filename matches the name in the YAML."""
        import yaml

        class IgnoreLoader(yaml.SafeLoader):
            def _ignore_all_tags(self, tag_suffix, node):
                return tag_suffix + ' ' + node.value
        IgnoreLoader.add_multi_constructor('', IgnoreLoader._ignore_all_tags)

        from satpy.config import glob_config
        from satpy.readers import read_reader_config
        for reader_config in glob_config('readers/*.yaml'):
            reader_fn = os.path.basename(reader_config)
            reader_fn_name = os.path.splitext(reader_fn)[0]
            reader_info = read_reader_config([reader_config],
                                             loader=IgnoreLoader)
            self.assertEqual(reader_fn_name, reader_info['name'],
                             "Reader YAML filename doesn't match reader "
                             "name in the YAML file.")

    def test_available_readers(self):
        """Test the 'available_readers' function."""
        from satpy import available_readers
        reader_names = available_readers()
        self.assertGreater(len(reader_names), 0)
        self.assertIsInstance(reader_names[0], str)
        self.assertIn('viirs_sdr', reader_names)  # needs h5py
        self.assertIn('abi_l1b', reader_names)  # needs netcdf4

        reader_infos = available_readers(as_dict=True)
        self.assertEqual(len(reader_names), len(reader_infos))
        self.assertIsInstance(reader_infos[0], dict)
        for reader_info in reader_infos:
            self.assertIn('name', reader_info)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestDatasetDict))
    mysuite.addTest(loader.loadTestsFromTestCase(TestReaderLoader))
    mysuite.addTest(loader.loadTestsFromTestCase(TestFindFilesAndReaders))
    mysuite.addTest(loader.loadTestsFromTestCase(TestYAMLFiles))

    return mysuite


if __name__ == "__main__":
    unittest.main()
