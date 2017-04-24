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


class TestReaderFinder(unittest.TestCase):
    """Test the ReaderFinder class
    
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
        from satpy.readers import ReaderFinder
        rf = ReaderFinder()
        ri = rf()
        self.assertDictEqual(ri, {})

    def test_filenames_only(self):
        """Test with filenames specified"""
        from satpy.readers import ReaderFinder
        rf = ReaderFinder()
        ri = rf(filenames=[
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertListEqual(list(ri.keys()), ['viirs_sdr'])

    def test_filenames_and_reader(self):
        """Test with filenames and reader specified"""
        from satpy.readers import ReaderFinder
        rf = ReaderFinder()
        ri = rf(reader='viirs_sdr',
                filenames=[
                    'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertListEqual(list(ri.keys()), ['viirs_sdr'])

    def test_bad_reader_name_with_filenames(self):
        """Test bad reader name with filenames provided"""
        from satpy.readers import ReaderFinder
        rf = ReaderFinder()
        self.assertRaises(ValueError, rf, reader='i_dont_exist', filenames=[
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            ])

    def test_bad_sensor_with_filenames(self):
        """Test bad sensor with filenames provided"""
        from satpy.readers import ReaderFinder
        rf = ReaderFinder()
        self.assertRaises(ValueError, rf, sensor='i_dont_exist', filenames=[
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])

    def test_sensor(self):
        """Test with filenames and sensor specified"""
        from satpy.readers import ReaderFinder
        rf = ReaderFinder()
        ri = rf(sensor='viirs',
                filenames=[
                    'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
                ])
        self.assertListEqual(list(ri.keys()), ['viirs_sdr'])

    def test_reader_name_base_dir(self):
        """Test with default base_dir and reader specified"""
        from satpy.readers import ReaderFinder
        fn = 'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        # touch the file so it exists on disk
        open(fn, 'w')
        try:
            rf = ReaderFinder()
            ri = rf(reader='viirs_sdr')
            self.assertListEqual(list(ri.keys()), ['viirs_sdr'])
        finally:
            os.remove(fn)


class TestReaders(unittest.TestCase):
    '''Class for testing satpy.satin'''

    # def test_lonlat_to_geo_extent(self):
    #     '''Test conversion of longitudes and latitudes to area extent.'''
    #
    #     # MSG3 proj4 string from
    #     #  xrit.sat.load(..., only_metadata=True).proj4_params
    #     proj4_str = 'proj=geos lon_0=0.00 lat_0=0.00 ' \
    #         'a=6378169.00 b=6356583.80 h=35785831.00'
    #
    #     # MSG3 maximum extent
    #     max_extent=(-5567248.07, -5570248.48,
    #                  5570248.48, 5567248.07)
    #
    #     # Few area extents in longitudes/latitudes
    #     area_extents_ll = [[-68.328121068060341, # left longitude
    #                          18.363816196771392, # down latitude
    #                          74.770372053870972, # right longitude
    #                          75.66494585661934], # up latitude
    #                        # all corners outside Earth's disc
    #                        [1e30, 1e30, 1e30, 1e30]
    #                        ]
    #
    #     # And corresponding correct values in GEO projection
    #     geo_extents = [[-5010596.02, 1741593.72, 5570248.48, 5567248.07],
    #                    [-5567248.07, -5570248.48, 5570248.48, 5567248.07]]
    #
    #     for i in range(len(area_extents_ll)):
    #         res = satpy.satin.mipp_xrit.lonlat_to_geo_extent(area_extents_ll[i],
    #                                                         proj4_str,
    #                                                         max_extent=\
    #                                                             max_extent)
    #         for j in range(len(res)):
    #             self.assertAlmostEqual(res[j], geo_extents[i][j], 2)

    ## FIXME replace the following with tests on reader.select_files
    #
    # @mock.patch("glob.glob")
    # def test_find_sensors_readers_single_sensor_no_files(self, glob_mock,
    #                                                      **mock_objs):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     glob_mock.return_value = ["valid", "no_found_files", "not_valid"]
    #
    #     def fake_read_config(config_file):
    #         if config_file in ["valid", "no_found_files"]:
    #             return {"name": "fake_reader",
    #                     "sensor": ["foo"],
    #                     "config_files": config_file}
    #         else:
    #             raise ValueError("Fake ValueError")
    #
    #     def fake_get_filenames(reader_info):
    #         if reader_info["config_files"] == "valid":
    #             return ["file1", "file2"]
    #         return []
    #
    #     with mock.patch.multiple("satpy.readers.ReaderFinder",
    #                              read_reader_config=mock.DEFAULT,
    #                              get_filenames=mock.DEFAULT,
    #                              load_reader=mock.DEFAULT) as mock_objs:
    #         mock_objs["read_reader_config"].side_effect = fake_read_config
    #         mock_objs["get_filenames"].side_effect = fake_get_filenames
    #
    #         scn = Scene()
    #         finder = ReaderFinder(scn)
    #         finder._find_sensors_readers("foo", None)

    # def test_get_filenames_with_start_time_provided(self):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     from datetime import datetime
    #     scn = Scene()
    #     finder = ReaderFinder(scn)
    #     reader_info = {"file_patterns": ["foo"],
    #                    "start_time": datetime(2015, 6, 24, 0, 0)}
    #
    #     with mock.patch("satpy.readers.glob.iglob") as mock_iglob:
    #         mock_iglob.return_value = ["file1", "file2", "file3", "file4",
    #                                    "file5"]
    #         with mock.patch("satpy.readers.Parser") as mock_parser:
    #             mock_parser.return_value.parse.side_effect = [
    #                 {"start_time": datetime(2015, 6, 23, 23, 57),  # file1
    #                  "end_time": datetime(2015, 6, 23, 23, 59)},
    #                 {"start_time": datetime(2015, 6, 23, 23, 59),  # file2
    #                  "end_time": datetime(2015, 6, 24, 0, 1)},
    #                 {"start_time": datetime(2015, 6, 24, 0, 1),  # file3
    #                  "end_time": datetime(2015, 6, 24, 0, 3)},
    #                 {"start_time": datetime(2015, 6, 24, 0, 3),  # file4
    #                  "end_time": datetime(2015, 6, 24, 0, 5)},
    #                 {"start_time": datetime(2015, 6, 24, 0, 5),  # file5
    #                  "end_time": datetime(2015, 6, 24, 0, 7)},
    #             ]
    #             self.assertEqual(finder.get_filenames(reader_info), ["file2"])
    #

    # def test_get_filenames_with_start_time_and_end_time(self):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     from datetime import datetime
    #     scn = Scene()
    #     finder = ReaderFinder(scn)
    #     reader_info = {"file_patterns": ["foo"],
    #                    "start_time": datetime(2015, 6, 24, 0, 0),
    #                    "end_time": datetime(2015, 6, 24, 0, 6)}
    #     with mock.patch("satpy.readers.glob.iglob") as mock_iglob:
    #         mock_iglob.return_value = ["file1", "file2", "file3", "file4", "file5"]
    #         with mock.patch("satpy.readers.Parser") as mock_parser:
    #             mock_parser.return_value.parse.side_effect = [{"start_time": datetime(2015, 6, 23, 23, 57),  # file1
    #                                                            "end_time": datetime(2015, 6, 23, 23, 59)},
    #                                                           {"start_time": datetime(2015, 6, 23, 23, 59),  # file2
    #                                                            "end_time": datetime(2015, 6, 24, 0, 1)},
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 1),    # file3
    #                                                            "end_time": datetime(2015, 6, 24, 0, 3)},
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 3),    # file4
    #                                                            "end_time": datetime(2015, 6, 24, 0, 5)},
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 5),    # file5
    #                                                            "end_time": datetime(2015, 6, 24, 0, 7)},
    #                                                           ]
    #             self.assertEqual(finder.get_filenames(reader_info), ["file2", "file3", "file4", "file5"])
    #
    # def test_get_filenames_with_start_time_and_npp_style_end_time(self):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     from datetime import datetime
    #     scn = Scene()
    #     finder = ReaderFinder(scn)
    #     reader_info = {"file_patterns": ["foo"],
    #                    "start_time": datetime(2015, 6, 24, 0, 0),
    #                    "end_time": datetime(2015, 6, 24, 0, 6)}
    #     with mock.patch("satpy.readers.glob.iglob") as mock_iglob:
    #         mock_iglob.return_value = ["file1", "file2", "file3", "file4", "file5"]
    #         with mock.patch("satpy.readers.Parser") as mock_parser:
    #             mock_parser.return_value.parse.side_effect = [{"start_time": datetime(2015, 6, 23, 23, 57),  # file1
    #                                                            "end_time": datetime(1950, 1, 1, 23, 59)},
    #                                                           {"start_time": datetime(2015, 6, 23, 23, 59),  # file2
    #                                                            "end_time": datetime(1950, 1, 1, 0, 1)},
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 1),    # file3
    #                                                            "end_time": datetime(1950, 1, 1, 0, 3)},
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 3),    # file4
    #                                                            "end_time": datetime(1950, 1, 1, 0, 5)},
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 5),    # file5
    #                                                            "end_time": datetime(1950, 1, 1, 0, 7)},
    #                                                           ]
    #             self.assertEqual(finder.get_filenames(reader_info), ["file2", "file3", "file4", "file5"])
    #
    # def test_get_filenames_with_start_time(self):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     from datetime import datetime
    #     scn = Scene()
    #     finder = ReaderFinder(scn)
    #     reader_info = {"file_patterns": ["foo"],
    #                    "start_time": datetime(2015, 6, 24, 0, 0),
    #                    "end_time": datetime(2015, 6, 24, 0, 6)}
    #     with mock.patch("satpy.readers.glob.iglob") as mock_iglob:
    #         mock_iglob.return_value = ["file1", "file2", "file3", "file4", "file5"]
    #         with mock.patch("satpy.readers.Parser") as mock_parser:
    #             mock_parser.return_value.parse.side_effect = [{"start_time": datetime(2015, 6, 23, 23, 57)},  # file1
    #                                                           {"start_time": datetime(2015, 6, 23, 23, 59)},  # file2
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 1)},    # file3
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 3)},    # file4
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 5)},    # file5
    #                                                           ]
    #             self.assertEqual(finder.get_filenames(reader_info), ["file3", "file4", "file5"])
    #
    # def test_get_filenames_with_only_start_times_wrong(self):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     from datetime import datetime
    #     scn = Scene()
    #     finder = ReaderFinder(scn)
    #     reader_info = {"file_patterns": ["foo"],
    #                    "start_time": datetime(2015, 6, 24, 0, 0)}
    #     with mock.patch("satpy.readers.glob.iglob") as mock_iglob:
    #         mock_iglob.return_value = ["file1", "file2", "file3", "file4", "file5"]
    #         with mock.patch("satpy.readers.Parser") as mock_parser:
    #             mock_parser.return_value.parse.side_effect = [{"start_time": datetime(2015, 6, 23, 23, 57)},  # file1
    #                                                           {"start_time": datetime(2015, 6, 23, 23, 59)},  # file2
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 1)},    # file3
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 3)},    # file4
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 5)},    # file5
    #                                                           ]
    #             self.assertEqual(finder.get_filenames(reader_info), [])
    #
    # def test_get_filenames_with_only_start_times_right(self):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     from datetime import datetime
    #     scn = Scene()
    #     finder = ReaderFinder(scn)
    #     reader_info = {"file_patterns": ["foo"],
    #                    "start_time": datetime(2015, 6, 24, 0, 1)}
    #     with mock.patch("satpy.readers.glob.iglob") as mock_iglob:
    #         mock_iglob.return_value = ["file1", "file2", "file3", "file4", "file5"]
    #         with mock.patch("satpy.readers.Parser") as mock_parser:
    #             mock_parser.return_value.parse.side_effect = [{"start_time": datetime(2015, 6, 23, 23, 57)},  # file1
    #                                                           {"start_time": datetime(2015, 6, 23, 23, 59)},  # file2
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 1)},    # file3
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 3)},    # file4
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 5)},    # file5
    #                                                           ]
    #             self.assertEqual(finder.get_filenames(reader_info), ["file3"])
    #
    # def test_get_filenames_to_error(self):
    #     from satpy.scene import Scene
    #     from satpy.readers import ReaderFinder
    #     from datetime import datetime
    #     scn = Scene(start_time="bla")
    #     finder = ReaderFinder(scn)
    #     reader_info = {"file_patterns": ["foo"],
    #                    "start_time": None}
    #     with mock.patch("satpy.readers.glob.iglob") as mock_iglob:
    #         mock_iglob.return_value = ["file1", "file2", "file3", "file4", "file5"]
    #         with mock.patch("satpy.readers.Parser") as mock_parser:
    #             mock_parser.return_value.parse.side_effect = [{"start_time": datetime(2015, 6, 23, 23, 57)},  # file1
    #                                                           {"start_time": datetime(2015, 6, 23, 23, 59)},  # file2
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 1)},    # file3
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 3)},    # file4
    #                                                           {"start_time": datetime(2015, 6, 24, 0, 5)},    # file5
    #                                                           ]
    #             self.assertRaises(ValueError, finder.get_filenames, reader_info)


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestReaders))
    mysuite.addTest(loader.loadTestsFromTestCase(TestDatasetDict))
    mysuite.addTest(loader.loadTestsFromTestCase(TestReaderFinder))

    return mysuite


if __name__ == "__main__":
    unittest.main()
