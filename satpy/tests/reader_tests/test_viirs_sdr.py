#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.viirs_sdr module.
"""

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

import numpy as np
from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler"""
    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content"""
        start_time = filename_info['start_time']
        end_time = filename_info['end_time'].replace(year=start_time.year,
                                                     month=start_time.month,
                                                     day=start_time.day)

        prefix1 = 'Data_Products/{file_group}'.format(**filetype_info)
        prefix2 = '{prefix}/{file_group}_Aggr'.format(prefix=prefix1, **filetype_info)
        prefix3 = 'All_Data/{file_group}_All'.format(**filetype_info)
        begin_date = start_time.strftime('%Y%m%d')
        begin_time = start_time.strftime('%H%M%S.%fZ')
        ending_date = end_time.strftime('%Y%m%d')
        ending_time = end_time.strftime('%H%M%S.%fZ')
        if filename[:3] == 'SVI':
            geo_prefix = 'GIMGO'
        elif filename[:3] == 'SVM':
            geo_prefix = 'GMODO'
        else:
            geo_prefix = None
        file_content = {
            "{prefix2}/attr/AggregateBeginningDate": begin_date,
            "{prefix2}/attr/AggregateBeginningTime": begin_time,
            "{prefix2}/attr/AggregateEndingDate": ending_date,
            "{prefix2}/attr/AggregateEndingTime": ending_time,
            "{prefix2}/attr/G-Ring_Longitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "{prefix2}/attr/G-Ring_Latitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "{prefix2}/attr/AggregateBeginningOrbitNumber": "{0:d}".format(filename_info['orbit']),
            "{prefix2}/attr/AggregateEndingOrbitNumber": "{0:d}".format(filename_info['orbit']),
            "{prefix1}/attr/Instrument_Short_Name": "VIIRS",
            "/attr/Platform_Short_Name": "NPP",
        }
        if geo_prefix:
            file_content['/attr/N_GEO_Ref'] = geo_prefix + filename[5:]
        for k, v in list(file_content.items()):
            file_content[k.format(prefix1=prefix1, prefix2=prefix2)] = v

        if filename[:3] in ['SVM', 'SVI', 'SVD']:
            if filename[2:5] in ['M{:02d}'.format(x) for x in range(12)] + ['I01', 'I02', 'I03']:
                keys = ['Radiance', 'Reflectance']
            elif filename[2:5] in ['M{:02d}'.format(x) for x in range(12, 17)] + ['I04', 'I05']:
                keys = ['Radiance', 'BrightnessTemperature']
            else:
                # DNB
                keys = ['Radiance']

            for k in keys:
                k = prefix3 + "/" + k
                file_content[k] = DEFAULT_FILE_DATA.copy()
                file_content[k + "/shape"] = DEFAULT_FILE_SHAPE
                file_content[k + "Factors"] = DEFAULT_FILE_FACTORS.copy()
        elif filename[0] == 'G':
            if filename[:5] in ['GMODO', 'GIMGO']:
                lon_data = np.linspace(15, 55, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
                lat_data = np.linspace(55, 75, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
            else:
                lon_data = np.linspace(5, 45, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
                lat_data = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)

            for k in ["Latitude"]:
                k = prefix3 + "/" + k
                file_content[k] = lat_data
                file_content[k] = np.repeat([file_content[k]], DEFAULT_FILE_SHAPE[0], axis=0)
                file_content[k + "/shape"] = DEFAULT_FILE_SHAPE
            for k in ["Longitude"]:
                k = prefix3 + "/" + k
                file_content[k] = lon_data
                file_content[k] = np.repeat([file_content[k]], DEFAULT_FILE_SHAPE[0], axis=0)
                file_content[k + "/shape"] = DEFAULT_FILE_SHAPE

        return file_content


class TestVIIRSSDRReader(unittest.TestCase):
    """Test VIIRS SDR Reader"""
    yaml_file = "viirs_sdr.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler"""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIIRSSDRFileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler"""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_init_start_time_beyond(self):
        """Test basic init with start_time after the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        start_time=datetime(2012, 2, 26))
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 0)

    def test_init_end_time_beyond(self):
        """Test basic init with end_time before the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        end_time=datetime(2012, 2, 24))
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 0)

    def test_init_start_end_time(self):
        """Test basic init with end_time before the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        start_time=datetime(2012, 2, 24),
                        end_time=datetime(2012, 2, 26))
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_all_m_reflectances_no_geo(self):
        """Load all M band reflectances with no geo files provided"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVM01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM06_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM07_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM08_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM09_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM10_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM11_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['M01',
                     'M02',
                     'M03',
                     'M04',
                     'M05',
                     'M06',
                     'M07',
                     'M08',
                     'M09',
                     'M10',
                     'M11',
                     ])
        self.assertEqual(len(ds), 11)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'reflectance')
            self.assertEqual(d.info['units'], '%')
            self.assertNotIn('area', d.info)

    def test_load_all_m_reflectances_find_geo(self):
        """Load all M band reflectances with geo files not specified but existing"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVM01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM06_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM07_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM08_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM09_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM10_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM11_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        # make a fake geo file
        geo_fn = 'GMTCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5'
        open(geo_fn, 'w')

        try:
            r.create_filehandlers(loadables)
            ds = r.load(['M01',
                         'M02',
                         'M03',
                         'M04',
                         'M05',
                         'M06',
                         'M07',
                         'M08',
                         'M09',
                         'M10',
                         'M11',
                         ])
        finally:
            os.remove(geo_fn)

        self.assertEqual(len(ds), 11)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'reflectance')
            self.assertEqual(d.info['units'], '%')
            self.assertIn('area', d.info)
            self.assertIsNotNone(d.info['area'])

    def test_load_all_m_reflectances_provided_geo(self):
        """Load all M band reflectances with geo files provided"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVM01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM06_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM07_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM08_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM09_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM10_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM11_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GMTCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['M01',
                     'M02',
                     'M03',
                     'M04',
                     'M05',
                     'M06',
                     'M07',
                     'M08',
                     'M09',
                     'M10',
                     'M11',
                     ])
        self.assertEqual(len(ds), 11)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'reflectance')
            self.assertEqual(d.info['units'], '%')
            self.assertIn('area', d.info)
            self.assertIsNotNone(d.info['area'])
            self.assertEqual(d.info['area'].lons.min(), 5)
            self.assertEqual(d.info['area'].lats.min(), 45)

    def test_load_all_m_reflectances_use_nontc(self):
        """Load all M band reflectances but use non-TC geolocation"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs, use_tc=False)
        loadables = r.select_files_from_pathnames([
            'SVM01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM06_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM07_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM08_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM09_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM10_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM11_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GMTCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GMODO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['M01',
                     'M02',
                     'M03',
                     'M04',
                     'M05',
                     'M06',
                     'M07',
                     'M08',
                     'M09',
                     'M10',
                     'M11',
                     ])
        self.assertEqual(len(ds), 11)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'reflectance')
            self.assertEqual(d.info['units'], '%')
            self.assertIn('area', d.info)
            self.assertIsNotNone(d.info['area'])
            self.assertEqual(d.info['area'].lons.min(), 15)
            self.assertEqual(d.info['area'].lats.min(), 55)

    def test_load_all_m_reflectances_use_nontc2(self):
        """Load all M band reflectances but use non-TC geolocation (use_tc=None)"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs, use_tc=None)
        loadables = r.select_files_from_pathnames([
            'SVM01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM06_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM07_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM08_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM09_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM10_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM11_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GMODO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['M01',
                     'M02',
                     'M03',
                     'M04',
                     'M05',
                     'M06',
                     'M07',
                     'M08',
                     'M09',
                     'M10',
                     'M11',
                     ])
        self.assertEqual(len(ds), 11)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'reflectance')
            self.assertEqual(d.info['units'], '%')
            self.assertIn('area', d.info)
            self.assertIsNotNone(d.info['area'])
            self.assertEqual(d.info['area'].lons.min(), 15)
            self.assertEqual(d.info['area'].lats.min(), 55)

    def test_load_all_m_bts(self):
        """Load all M band brightness temperatures"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVM12_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM13_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM14_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM15_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM16_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GMTCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['M12',
                     'M13',
                     'M14',
                     'M15',
                     'M16',
                     ])
        self.assertEqual(len(ds), 5)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'brightness_temperature')
            self.assertEqual(d.info['units'], 'K')
            self.assertIn('area', d.info)
            self.assertIsNotNone(d.info['area'])

    def test_load_all_m_radiances(self):
        """Load all M band radiances"""
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVM01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM06_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM07_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM08_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM09_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM10_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM11_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM12_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM13_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM14_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM15_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVM16_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GMTCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load([
            DatasetID(name='M01', calibration='radiance'),
            DatasetID(name='M02', calibration='radiance'),
            DatasetID(name='M03', calibration='radiance'),
            DatasetID(name='M04', calibration='radiance'),
            DatasetID(name='M05', calibration='radiance'),
            DatasetID(name='M06', calibration='radiance'),
            DatasetID(name='M07', calibration='radiance'),
            DatasetID(name='M08', calibration='radiance'),
            DatasetID(name='M09', calibration='radiance'),
            DatasetID(name='M10', calibration='radiance'),
            DatasetID(name='M11', calibration='radiance'),
            DatasetID(name='M12', calibration='radiance'),
            DatasetID(name='M13', calibration='radiance'),
            DatasetID(name='M14', calibration='radiance'),
            DatasetID(name='M15', calibration='radiance'),
            DatasetID(name='M16', calibration='radiance'),
                     ])
        self.assertEqual(len(ds), 16)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'radiance')
            self.assertEqual(d.info['units'], 'W m-2 um-1 sr-1')
            self.assertIn('area', d.info)
            self.assertIsNotNone(d.info['area'])

    def test_load_dnb(self):
        """Load DNB dataset"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVDNB_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GDNBO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['DNB'])
        self.assertEqual(len(ds), 1)
        for d in ds.values():
            self.assertEqual(d.info['calibration'], 'radiance')
            self.assertEqual(d.info['units'], 'W m-2 sr-1')
            self.assertIn('area', d.info)
            self.assertIsNotNone(d.info['area'])

    def test_load_i_no_files(self):
        """Load I01 when only DNB files are provided"""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVDNB_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GDNBO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['I01'])
        self.assertEqual(len(ds), 0)


def suite():
    """The test suite for test_viirs_sdr.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIIRSSDRReader))

    return mysuite
