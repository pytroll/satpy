#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2019 Satpy developers
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
"""Module for testing the satpy.readers.viirs_sdr module."""

import os
import unittest
from unittest import mock
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

DATASET_KEYS = {'GDNBO': 'VIIRS-DNB-GEO',
                'SVDNB': 'VIIRS-DNB-SDR',
                'GITCO': 'VIIRS-IMG-GEO-TC',
                'GIMGO': 'VIIRS-IMG-GEO',
                'SVI01': 'VIIRS-I1-SDR',
                'SVI02': 'VIIRS-I2-SDR',
                'SVI03': 'VIIRS-I3-SDR',
                'SVI04': 'VIIRS-I4-SDR',
                'SVI05': 'VIIRS-I5-SDR',
                'GMTCO': 'VIIRS-MOD-GEO-TC',
                'GMODO': 'VIIRS-MOD-GEO',
                'SVM01': 'VIIRS-M1-SDR',
                'SVM02': 'VIIRS-M2-SDR',
                'SVM03': 'VIIRS-M3-SDR',
                'SVM04': 'VIIRS-M4-SDR',
                'SVM05': 'VIIRS-M5-SDR',
                'SVM06': 'VIIRS-M6-SDR',
                'SVM07': 'VIIRS-M7-SDR',
                'SVM08': 'VIIRS-M8-SDR',
                'SVM09': 'VIIRS-M9-SDR',
                'SVM10': 'VIIRS-M10-SDR',
                'SVM11': 'VIIRS-M11-SDR',
                'SVM12': 'VIIRS-M12-SDR',
                'SVM13': 'VIIRS-M13-SDR',
                'SVM14': 'VIIRS-M14-SDR',
                'SVM15': 'VIIRS-M15-SDR',
                'SVM16': 'VIIRS-M16-SDR',
                }


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def __init__(self, filename, filename_info, filetype_info, use_tc=None):
        """Create fake file handler."""
        super(FakeHDF5FileHandler2, self).__init__(filename, filename_info, filetype_info)
        self.datasets = filename_info['datasets'].split('-')
        self.use_tc = use_tc

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        start_time = filename_info['start_time']
        end_time = filename_info['end_time'].replace(year=start_time.year,
                                                     month=start_time.month,
                                                     day=start_time.day)
        final_content = {}
        for dataset in self.datasets:
            dataset_group = DATASET_KEYS[dataset]
            prefix1 = 'Data_Products/{dataset_group}'.format(dataset_group=dataset_group)
            prefix2 = '{prefix}/{dataset_group}_Aggr'.format(prefix=prefix1, dataset_group=dataset_group)
            prefix3 = 'All_Data/{dataset_group}_All'.format(dataset_group=dataset_group)
            prefix4 = '{prefix}/{dataset_group}_Gran_0'.format(prefix=prefix1, dataset_group=dataset_group)
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
                "{prefix2}/attr/AggregateNumberGranules": 1,
                "{prefix4}/attr/N_Number_Of_Scans": 48,
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
                file_content[k.format(prefix1=prefix1, prefix2=prefix2, prefix3=prefix3, prefix4=prefix4)] = v

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

        final_content.update(file_content)

        # convert to xarrays
        from xarray import DataArray
        import dask.array as da
        for key, val in final_content.items():
            if isinstance(val, np.ndarray):
                val = da.from_array(val, chunks=val.shape)
                if val.ndim > 1:
                    final_content[key] = DataArray(val, dims=('y', 'x'))
                else:
                    final_content[key] = DataArray(val)

        return final_content


class TestVIIRSSDRReader(unittest.TestCase):
    """Test VIIRS SDR Reader."""

    yaml_file = "viirs_sdr.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIIRSSDRFileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
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
                        filter_parameters={
                            'start_time': datetime(2012, 2, 26)
                        })
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 0)

    def test_init_end_time_beyond(self):
        """Test basic init with end_time before the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        filter_parameters={
                            'end_time': datetime(2012, 2, 24)
                        })
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 0)

    def test_init_start_end_time(self):
        """Test basic init with end_time before the provided files."""
        from satpy.readers import load_reader
        from datetime import datetime
        r = load_reader(self.reader_configs,
                        filter_parameters={
                            'start_time': datetime(2012, 2, 24),
                            'end_time': datetime(2012, 2, 26)
                        })
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    def test_load_all_m_reflectances_no_geo(self):
        """Load all M band reflectances with no geo files provided."""
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
            self.assertEqual(d.attrs['calibration'], 'reflectance')
            self.assertEqual(d.attrs['units'], '%')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertNotIn('area', d.attrs)

    def test_load_all_m_reflectances_find_geo(self):
        """Load all M band reflectances with geo files not specified but existing."""
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
            self.assertEqual(d.attrs['calibration'], 'reflectance')
            self.assertEqual(d.attrs['units'], '%')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_load_all_m_reflectances_provided_geo(self):
        """Load all M band reflectances with geo files provided."""
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
            self.assertEqual(d.attrs['calibration'], 'reflectance')
            self.assertEqual(d.attrs['units'], '%')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertEqual(d.attrs['area'].lons.min(), 5)
            self.assertEqual(d.attrs['area'].lats.min(), 45)
            self.assertEqual(d.attrs['area'].lons.attrs['rows_per_scan'], 16)
            self.assertEqual(d.attrs['area'].lats.attrs['rows_per_scan'], 16)

    def test_load_all_m_reflectances_use_nontc(self):
        """Load all M band reflectances but use non-TC geolocation."""
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
            self.assertEqual(d.attrs['calibration'], 'reflectance')
            self.assertEqual(d.attrs['units'], '%')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertEqual(d.attrs['area'].lons.min(), 15)
            self.assertEqual(d.attrs['area'].lats.min(), 55)
            self.assertEqual(d.attrs['area'].lons.attrs['rows_per_scan'], 16)
            self.assertEqual(d.attrs['area'].lats.attrs['rows_per_scan'], 16)

    def test_load_all_m_reflectances_use_nontc2(self):
        """Load all M band reflectances but use non-TC geolocation because TC isn't available."""
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
            self.assertEqual(d.attrs['calibration'], 'reflectance')
            self.assertEqual(d.attrs['units'], '%')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertEqual(d.attrs['area'].lons.min(), 15)
            self.assertEqual(d.attrs['area'].lats.min(), 55)
            self.assertEqual(d.attrs['area'].lons.attrs['rows_per_scan'], 16)
            self.assertEqual(d.attrs['area'].lats.attrs['rows_per_scan'], 16)

    def test_load_all_m_bts(self):
        """Load all M band brightness temperatures."""
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
            self.assertEqual(d.attrs['calibration'], 'brightness_temperature')
            self.assertEqual(d.attrs['units'], 'K')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_load_all_m_radiances(self):
        """Load all M band radiances."""
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
            DatasetID(name='M01', calibration='radiance', modifiers=None),
            DatasetID(name='M02', calibration='radiance', modifiers=None),
            DatasetID(name='M03', calibration='radiance', modifiers=None),
            DatasetID(name='M04', calibration='radiance', modifiers=None),
            DatasetID(name='M05', calibration='radiance', modifiers=None),
            DatasetID(name='M06', calibration='radiance', modifiers=None),
            DatasetID(name='M07', calibration='radiance', modifiers=None),
            DatasetID(name='M08', calibration='radiance', modifiers=None),
            DatasetID(name='M09', calibration='radiance', modifiers=None),
            DatasetID(name='M10', calibration='radiance', modifiers=None),
            DatasetID(name='M11', calibration='radiance', modifiers=None),
            DatasetID(name='M12', calibration='radiance', modifiers=None),
            DatasetID(name='M13', calibration='radiance', modifiers=None),
            DatasetID(name='M14', calibration='radiance', modifiers=None),
            DatasetID(name='M15', calibration='radiance', modifiers=None),
            DatasetID(name='M16', calibration='radiance', modifiers=None),
                     ])
        self.assertEqual(len(ds), 16)
        for d in ds.values():
            self.assertEqual(d.attrs['calibration'], 'radiance')
            self.assertEqual(d.attrs['units'], 'W m-2 um-1 sr-1')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_load_dnb(self):
        """Load DNB dataset."""
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
            self.assertEqual(d.attrs['calibration'], 'radiance')
            self.assertEqual(d.attrs['units'], 'W m-2 sr-1')
            self.assertEqual(d.attrs['rows_per_scan'], 16)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_load_i_no_files(self):
        """Load I01 when only DNB files are provided."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVDNB_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GDNBO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        self.assertNotIn('I01', [x.name for x in r.available_dataset_ids])
        ds = r.load(['I01'])
        self.assertEqual(len(ds), 0)

    def test_load_all_i_reflectances_provided_geo(self):
        """Load all I band reflectances with geo files provided."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVI02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVI03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GITCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['I01',
                     'I02',
                     'I03',
                     ])
        self.assertEqual(len(ds), 3)
        for d in ds.values():
            self.assertEqual(d.attrs['calibration'], 'reflectance')
            self.assertEqual(d.attrs['units'], '%')
            self.assertEqual(d.attrs['rows_per_scan'], 32)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])
            self.assertEqual(d.attrs['area'].lons.min(), 5)
            self.assertEqual(d.attrs['area'].lats.min(), 45)
            self.assertEqual(d.attrs['area'].lons.attrs['rows_per_scan'], 32)
            self.assertEqual(d.attrs['area'].lats.attrs['rows_per_scan'], 32)

    def test_load_all_i_bts(self):
        """Load all I band brightness temperatures."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVI04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVI05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GITCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['I04',
                     'I05',
                     ])
        self.assertEqual(len(ds), 2)
        for d in ds.values():
            self.assertEqual(d.attrs['calibration'], 'brightness_temperature')
            self.assertEqual(d.attrs['units'], 'K')
            self.assertEqual(d.attrs['rows_per_scan'], 32)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])

    def test_load_all_i_radiances(self):
        """Load all I band radiances."""
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVI02_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVI03_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVI04_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'SVI05_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
            'GITCO_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        ds = r.load([
            DatasetID(name='I01', calibration='radiance', modifiers=None),
            DatasetID(name='I02', calibration='radiance', modifiers=None),
            DatasetID(name='I03', calibration='radiance', modifiers=None),
            DatasetID(name='I04', calibration='radiance', modifiers=None),
            DatasetID(name='I05', calibration='radiance', modifiers=None),
        ])
        self.assertEqual(len(ds), 5)
        for d in ds.values():
            self.assertEqual(d.attrs['calibration'], 'radiance')
            self.assertEqual(d.attrs['units'], 'W m-2 um-1 sr-1')
            self.assertEqual(d.attrs['rows_per_scan'], 32)
            self.assertIn('area', d.attrs)
            self.assertIsNotNone(d.attrs['area'])


class FakeHDF5FileHandlerAggr(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def __init__(self, filename, filename_info, filetype_info, use_tc=None):
        """Create fake aggregated file handler."""
        super(FakeHDF5FileHandlerAggr, self).__init__(filename, filename_info, filetype_info)
        self.datasets = filename_info['datasets'].split('-')
        self.use_tc = use_tc

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        start_time = filename_info['start_time']
        end_time = filename_info['end_time'].replace(year=start_time.year,
                                                     month=start_time.month,
                                                     day=start_time.day)
        final_content = {}
        for dataset in self.datasets:
            dataset_group = DATASET_KEYS[dataset]
            prefix1 = 'Data_Products/{dataset_group}'.format(dataset_group=dataset_group)
            prefix2 = '{prefix}/{dataset_group}_Aggr'.format(prefix=prefix1, dataset_group=dataset_group)
            prefix3 = 'All_Data/{dataset_group}_All'.format(dataset_group=dataset_group)
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
                "{prefix3}/NumberOfScans": np.array([48, 48, 48, 48]),
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

            lats_lists = [
                np.array(
                    [
                        67.969505, 65.545685, 63.103046, 61.853905, 55.169273,
                        57.062447, 58.86063, 66.495514
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        72.74879, 70.2493, 67.84738, 66.49691, 58.77254,
                        60.465942, 62.11525, 71.08249
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        77.393425, 74.977875, 72.62976, 71.083435, 62.036346,
                        63.465122, 64.78075, 75.36842
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        81.67615, 79.49934, 77.278656, 75.369415, 64.72178,
                        65.78417, 66.66166, 79.00025
                    ],
                    dtype=np.float32)
            ]
            lons_lists = [
                np.array(
                    [
                        50.51393, 49.566296, 48.865967, 18.96082, -4.0238385,
                        -7.05221, -10.405702, 14.638646
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        53.52594, 51.685738, 50.439102, 14.629087, -10.247547,
                        -13.951393, -18.256989, 8.36572
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        59.386833, 55.770416, 53.38952, 8.353765, -18.062435,
                        -22.608992, -27.867302, -1.3537619
                    ],
                    dtype=np.float32),
                np.array(
                    [
                        72.50243, 64.17125, 59.15234, -1.3654504, -27.620953,
                        -33.091743, -39.28113, -17.749891
                    ],
                    dtype=np.float32)
            ]

            for granule in range(4):
                prefix_gran = '{prefix}/{dataset_group}_Gran_{idx}'.format(prefix=prefix1,
                                                                           dataset_group=dataset_group,
                                                                           idx=granule)
                file_content[prefix_gran + '/attr/G-Ring_Longitude'] = lons_lists[granule]
                file_content[prefix_gran + '/attr/G-Ring_Latitude'] = lats_lists[granule]
            if geo_prefix:
                file_content['/attr/N_GEO_Ref'] = geo_prefix + filename[5:]
            for k, v in list(file_content.items()):
                file_content[k.format(prefix1=prefix1, prefix2=prefix2, prefix3=prefix3)] = v

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

        final_content.update(file_content)

        # convert to xarrays
        from xarray import DataArray
        import dask.array as da
        for key, val in final_content.items():
            if isinstance(val, np.ndarray):
                val = da.from_array(val, chunks=val.shape)
                if val.ndim > 1:
                    final_content[key] = DataArray(val, dims=('y', 'x'))
                else:
                    final_content[key] = DataArray(val)

        return final_content


class TestAggrVIIRSSDRReader(unittest.TestCase):
    """Test VIIRS SDR Reader."""

    yaml_file = "viirs_sdr.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.config import config_search_paths
        from satpy.readers.viirs_sdr import VIIRSSDRFileHandler
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIIRSSDRFileHandler, '__bases__', (FakeHDF5FileHandlerAggr,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_bounding_box(self):
        """Test bounding box."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'SVI01_npp_d20120225_t1801245_e1802487_b01708_c20120226002130255476_noaa_ops.h5',
        ])
        r.create_filehandlers(loadables)
        # make sure we have some files
        expected_lons = [
            72.50243, 64.17125, 59.15234, 59.386833, 55.770416, 53.38952, 53.52594, 51.685738, 50.439102, 50.51393,
            49.566296, 48.865967, 18.96082, -4.0238385, -7.05221, -10.247547, -13.951393, -18.062435, -22.608992,
            -27.620953, -33.091743, -39.28113, -17.749891
        ]
        expected_lats = [
            81.67615, 79.49934, 77.278656, 77.393425, 74.977875, 72.62976, 72.74879, 70.2493, 67.84738, 67.969505,
            65.545685, 63.103046, 61.853905, 55.169273, 57.062447, 58.77254, 60.465942, 62.036346, 63.465122,
            64.72178, 65.78417, 66.66166, 79.00025
        ]
        lons, lats = r.file_handlers['generic_file'][0].get_bounding_box()
        np.testing.assert_allclose(lons, expected_lons)
        np.testing.assert_allclose(lats, expected_lats)
