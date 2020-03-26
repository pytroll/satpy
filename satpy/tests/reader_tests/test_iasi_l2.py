#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""Unit tests for IASI L2 reader"""

import os
import unittest

import numpy as np

SCAN_WIDTH = 120
NUM_LEVELS = 138
NUM_SCANLINES = 1
FNAME = "W_XX-EUMETSAT-kan,iasi,metopb+kan_C_EUMS_20170920103559_IASI_PW3_02_M01_20170920102217Z_20170920102912Z.hdf"
# Structure for the test data, to be written to HDF5 file
TEST_DATA = {
    # Not implemented in the reader
    'Amsu': {
        'FLG_AMSUBAD': {'data': np.zeros((NUM_SCANLINES, 30), dtype=np.uint8),
                        'attrs': {}}
    },
    # Not implemented in the reader
    'INFO': {
        'OmC': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                'attrs': {'long_name': "Cloud signal. Predicted average window channel 'Obs minus Calc",
                          'units': 'K'}},
        'mdist': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                  'attrs': {}}
    },
    'L1C': {
        'Latitude': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                     'attrs': {'units': 'degrees_north'}},
        'Longitude': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      'attrs': {'units': 'degrees_north'}},
        'SatAzimuth': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                       'attrs': {'units': 'degrees'}},
        'SatZenith': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      'attrs': {'units': 'degrees'}},
        'SensingTime_day': {'data': np.array([6472], dtype=np.uint16),
                            'attrs': {}},
        'SensingTime_msec': {'data': np.array([37337532], dtype=np.uint32),
                             'attrs': {}},
        'SunAzimuth': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                       'attrs': {'units': 'degrees'}},
        'SunZenith': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      'attrs': {'units': 'degrees'}},
    },
    # Not implemented in the reader
    'Maps': {
        'Height': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                   'attrs': {'units': 'm'}},
        'HeightStd': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                      'attrs': {'units': 'm'}},
    },
    # Not implemented in the reader
    'Mhs': {
        'FLG_MHSBAD': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.uint8),
                       'attrs': {}}
    },
    'PWLR': {
        'E': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH, 10), dtype=np.float32),
              'attrs': {'emissivity_wavenumbers': np.array([699.3, 826.4,
                                                            925.9, 1075.2,
                                                            1204.8, 1315.7,
                                                            1724.1, 2000.0,
                                                            2325.5, 2702.7],
                                                           dtype=np.float32)}},
        'O': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              'attrs': {'long_name': 'Ozone mixing ratio vertical profile',
                        'units': 'kg/kg'}},
        'OC': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {}},
        'P': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              'attrs': {'long_name': 'Atmospheric pressures at which the vertical profiles are given. '
                                     'Last value is the surface pressure',
                        'units': 'hpa'}},
        'QE': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {}},
        'QO': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {}},
        'QP': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {}},
        'QT': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {}},
        'QTs': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
                'attrs': {}},
        'QW': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {}},
        'T': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              'attrs': {'long_name': 'Temperature vertical profile', 'units': 'K'}},
        'Ts': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {'long_name': 'Surface skin temperature', 'units': 'K'}},
        'W': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH, NUM_LEVELS), dtype=np.float32),
              'attrs': {'long_name': 'Water vapour mixing ratio vertical profile', 'units': 'kg/kg'}},
        'WC': {'data': np.zeros((NUM_SCANLINES, SCAN_WIDTH), dtype=np.float32),
               'attrs': {'long_name': 'Water vapour total columnar amount', 'units': 'mm'}},
    }
}


def save_test_data(path):
    """Save the test to the indicated directory"""
    import h5py
    with h5py.File(os.path.join(path, FNAME), 'w') as fid:
        # Create groups
        for grp in TEST_DATA:
            fid.create_group(grp)
            # Write datasets
            for dset in TEST_DATA[grp]:
                fid[grp][dset] = TEST_DATA[grp][dset]['data']
                # Write dataset attributes
                for attr in TEST_DATA[grp][dset]['attrs']:
                    fid[grp][dset].attrs[attr] = \
                        TEST_DATA[grp][dset]['attrs'][attr]


class TestIasiL2(unittest.TestCase):
    """Test IASI L2 reader."""

    def setUp(self):
        """Create temporary data to test on."""
        import tempfile
        import datetime as dt
        from satpy.readers.iasi_l2 import IASIL2HDF5

        self.base_dir = tempfile.mkdtemp()
        save_test_data(self.base_dir)
        self.fname = os.path.join(self.base_dir, FNAME)
        self.fname_info = {'start_time': dt.datetime(2017, 9, 20, 10, 22, 17),
                           'end_time': dt.datetime(2017, 9, 20, 10, 29, 12),
                           'processing_time': dt.datetime(2017, 9, 20, 10, 35, 59),
                           'processing_location': 'kan',
                           'long_platform_id': 'metopb',
                           'instrument': 'iasi',
                           'platform_id': 'M01'}
        self.ftype_info = {'file_reader': IASIL2HDF5,
                           'file_patterns': ['{fname}.hdf'],
                           'file_type': 'iasi_l2_hdf5'}
        self.reader = IASIL2HDF5(self.fname, self.fname_info, self.ftype_info)

    def tearDown(self):
        """Remove the temporary directory created for a test"""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    def test_scene(self):
        """Test scene creation"""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FNAME)
        scn = Scene(reader='iasi_l2', filenames=[fname])
        self.assertTrue('start_time' in scn.attrs)
        self.assertTrue('end_time' in scn.attrs)
        self.assertTrue('sensor' in scn.attrs)
        self.assertTrue('iasi' in scn.attrs['sensor'])

    def test_scene_load_available_datasets(self):
        """Test that all datasets are available"""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FNAME)
        scn = Scene(reader='iasi_l2', filenames=[fname])
        scn.load(scn.available_dataset_names())

    def test_scene_load_pressure(self):
        """Test loading pressure data"""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FNAME)
        scn = Scene(reader='iasi_l2', filenames=[fname])
        scn.load(['pressure'])
        pres = scn['pressure'].compute()
        self.check_pressure(pres, scn.attrs)

    def test_scene_load_emissivity(self):
        """Test loading emissivity data"""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FNAME)
        scn = Scene(reader='iasi_l2', filenames=[fname])
        scn.load(['emissivity'])
        emis = scn['emissivity'].compute()
        self.check_emissivity(emis)

    def test_scene_load_sensing_times(self):
        """Test loading sensing times"""
        from satpy import Scene
        fname = os.path.join(self.base_dir, FNAME)
        scn = Scene(reader='iasi_l2', filenames=[fname])
        scn.load(['sensing_time'])
        times = scn['sensing_time'].compute()
        self.check_sensing_times(times)

    def test_init(self):
        """Test reader initialization"""
        self.assertEqual(self.reader.filename, self.fname)
        self.assertEqual(self.reader.finfo, self.fname_info)
        self.assertTrue(self.reader.lons is None)
        self.assertTrue(self.reader.lats is None)
        self.assertEqual(self.reader.mda['platform_name'], 'Metop-B')
        self.assertEqual(self.reader.mda['sensor'], 'iasi')

    def test_time_properties(self):
        """Test time properties"""
        import datetime as dt
        self.assertTrue(isinstance(self.reader.start_time, dt.datetime))
        self.assertTrue(isinstance(self.reader.end_time, dt.datetime))

    def test_get_dataset(self):
        """Test get_dataset() for different datasets"""
        from satpy import DatasetID
        info = {'eggs': 'spam'}
        key = DatasetID(name='pressure')
        data = self.reader.get_dataset(key, info).compute()
        self.check_pressure(data)
        self.assertTrue('eggs' in data.attrs)
        self.assertEqual(data.attrs['eggs'], 'spam')
        key = DatasetID(name='emissivity')
        data = self.reader.get_dataset(key, info).compute()
        self.check_emissivity(data)
        key = DatasetID(name='sensing_time')
        data = self.reader.get_dataset(key, info).compute()
        self.assertEqual(data.shape, (NUM_SCANLINES, SCAN_WIDTH))

    def check_pressure(self, pres, attrs=None):
        """Helper method for testing reading pressure dataset"""
        self.assertTrue(np.all(pres == 0.0))
        self.assertEqual(pres.x.size, SCAN_WIDTH)
        self.assertEqual(pres.y.size, NUM_SCANLINES)
        self.assertEqual(pres.level.size, NUM_LEVELS)
        if attrs:
            self.assertEqual(pres.attrs['start_time'], attrs['start_time'])
            self.assertEqual(pres.attrs['end_time'], attrs['end_time'])
        self.assertTrue('long_name' in pres.attrs)
        self.assertTrue('units' in pres.attrs)

    def check_emissivity(self, emis):
        """Helper method for testing reading emissivity dataset."""
        self.assertTrue(np.all(emis == 0.0))
        self.assertEqual(emis.x.size, SCAN_WIDTH)
        self.assertEqual(emis.y.size, NUM_SCANLINES)
        self.assertTrue('emissivity_wavenumbers' in emis.attrs)

    def check_sensing_times(self, times):
        """Helper method for testing reading sensing times"""
        # Times should be equal in blocks of four, but not beyond, so
        # there should be SCAN_WIDTH/4 different values
        for i in range(int(SCAN_WIDTH / 4)):
            self.assertEqual(np.unique(times[0, i*4:i*4+4]).size, 1)
        self.assertEqual(np.unique(times[0, :]).size, SCAN_WIDTH / 4)

    def test_read_dataset(self):
        """Test read_dataset() function"""
        import h5py
        from satpy.readers.iasi_l2 import read_dataset
        from satpy import DatasetID
        with h5py.File(self.fname, 'r') as fid:
            key = DatasetID(name='pressure')
            data = read_dataset(fid, key).compute()
            self.check_pressure(data)
            key = DatasetID(name='emissivity')
            data = read_dataset(fid, key).compute()
            self.check_emissivity(data)
            # This dataset doesn't have any attributes
            key = DatasetID(name='ozone_total_column')
            data = read_dataset(fid, key).compute()
            self.assertEqual(len(data.attrs), 0)

    def test_read_geo(self):
        """Test read_geo() function"""
        import h5py
        from satpy.readers.iasi_l2 import read_geo
        from satpy import DatasetID
        with h5py.File(self.fname, 'r') as fid:
            key = DatasetID(name='sensing_time')
            data = read_geo(fid, key).compute()
            self.assertEqual(data.shape, (NUM_SCANLINES, SCAN_WIDTH))
            key = DatasetID(name='latitude')
            data = read_geo(fid, key).compute()
            self.assertEqual(data.shape, (NUM_SCANLINES, SCAN_WIDTH))

    def test_form_datetimes(self):
        """Test _form_datetimes() function"""
        from satpy.readers.iasi_l2 import _form_datetimes
        days = TEST_DATA['L1C']['SensingTime_day']['data']
        msecs = TEST_DATA['L1C']['SensingTime_msec']['data']
        times = _form_datetimes(days, msecs)
        self.check_sensing_times(times)
