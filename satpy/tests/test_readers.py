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
    def test_init_noargs(self):
        from satpy.readers import DatasetDict
        d = DatasetDict()

    def test_init_dict(self):
        from satpy.dataset import DatasetID
        from satpy.readers import DatasetDict
        regular_dict = {DatasetID(name="test", wavelength=(0, 0.5, 1)): "1", }
        d = DatasetDict(regular_dict)
        self.assertEqual(d, regular_dict)

    def test_get_keys_by_datasetid(self):
        from satpy.readers import DatasetDict
        from satpy.dataset import DatasetID
        did_list = [DatasetID(
            name="test", wavelength=(0, 0.5, 1),
            resolution=1000), DatasetID(name="testh",
                                        wavelength=(0, 0.5, 1),
                                        resolution=500),
                    DatasetID(name="test2",
                              wavelength=(1, 1.5, 2),
                              resolution=1000)]
        val_list = ["1", "1h", "2"]
        d = DatasetDict(dict(zip(did_list, val_list)))
        self.assertIn(did_list[0],
                      d.get_keys_by_datasetid(DatasetID(wavelength=0.5)))
        self.assertIn(did_list[1],
                      d.get_keys_by_datasetid(DatasetID(wavelength=0.5)))
        self.assertIn(did_list[2],
                      d.get_keys_by_datasetid(DatasetID(wavelength=1.5)))
        self.assertIn(did_list[0],
                      d.get_keys_by_datasetid(DatasetID(resolution=1000)))
        self.assertIn(did_list[2],
                      d.get_keys_by_datasetid(DatasetID(resolution=1000)))

    def test_get_item(self):
        from satpy.dataset import DatasetID
        from satpy.readers import DatasetDict
        regular_dict = {
            DatasetID(name="test",
                      wavelength=(0, 0.5, 1),
                      resolution=1000): "1",
            DatasetID(name="testh",
                      wavelength=(0, 0.5, 1),
                      resolution=500): "1h",
            DatasetID(name="test2",
                      wavelength=(1, 1.5, 2),
                      resolution=1000): "2",
        }
        d = DatasetDict(regular_dict)

        self.assertEqual(d["test"], "1")
        self.assertEqual(d[1.5], "2")
        self.assertEqual(d[DatasetID(wavelength=1.5)], "2")
        self.assertEqual(d[DatasetID(wavelength=0.5, resolution=1000)], "1")
        self.assertEqual(d[DatasetID(wavelength=0.5, resolution=500)], "1h")


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
