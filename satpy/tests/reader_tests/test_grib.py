#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module for testing the satpy.readers.grib module.
"""

import os
import sys
import numpy as np
import xarray as xr

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class FakeGRIB(object):
    """Fake GRIB file returned from xr.open_dataset."""
    def __init__(self, messages=None, proj_params=None):
        super(FakeGRIB, self).__init__()

        self.grib_file = None
        self.grib_file_nd = None

        vals = np.arange(25.).reshape((5, 5))
        latlon_vals = np.arange(25, 30)

        self.grib_file = xr.Dataset({'t': (['latitude', 'longitude'], vals),
                                    'u': (['latitude', 'longitude'], vals),
                                    'v': (['latitude', 'longitude'], vals)},
                                    coords={'latitude': (['latitude'], latlon_vals),
                                            'longitude': (['longitude'], latlon_vals),
                                            'time': np.datetime64('2018-07-22T00:00:00'),
                                            'valid_time': np.datetime64('2018-07-22T06:00:00')},
                                    attrs={'GRIB_edition': 2, 'GRIB_centre': 'kwbc',
                                           'Conventions': 'CF-1.7', 'GRIB_centreDescription': 'NCEP'})

        self.grib_file_irr = self.grib_file.copy(deep=True)

        for dsvars in self.grib_file.data_vars:
            dsvar = self.grib_file.data_vars[dsvars]
            dsvar.attrs['GRIB_shortName'] = dsvars
            dsvar.attrs['GRIB_typeOfLevel'] = 'maxWind'
            dsvar.attrs['jScansPositively'] = 0
            dsvar.attrs['GRIB_gridType'] = 'regular_ll'
            dsvar.attrs['GRIB_missingValue'] = 9999
            dsvar.attrs['units'] = 'K'

        for dsvars in self.grib_file_irr.data_vars:
            dsvar = self.grib_file_irr.data_vars[dsvars]
            dsvar.attrs['GRIB_shortName'] = dsvars
            dsvar.attrs['GRIB_typeOfLevel'] = 'maxWind'
            dsvar.attrs['jScansPositively'] = 0
            dsvar.attrs['GRIB_gridType'] = 'irregular_ll'
            dsvar.attrs['GRIB_missingValue'] = 9999
            dsvar.attrs['units'] = 'K'

        vals = np.arange(50.).reshape((2, 5, 5))
        lvl_vals = [1, 2]

        self.grib_file_nd = xr.Dataset({'t': (['potentialVorticity', 'latitude', 'longitude'], vals),
                                    'u': (['potentialVorticity', 'latitude', 'longitude'], vals),
                                    'v': (['potentialVorticity', 'latitude', 'longitude'], vals)},
                                    coords={'latitude': (['latitude'], latlon_vals),
                                            'longitude': (['longitude'], latlon_vals),
                                            'potentialVorticity': (['potentialVorticity'], lvl_vals),
                                            'time': np.datetime64('2018-07-22T00:00:00'),
                                            'valid_time': np.datetime64('2018-07-22T06:00:00')},
                                    attrs={'GRIB_edition': 2, 'GRIB_centre': 'kwbc',
                                           'Conventions': 'CF-1.7', 'GRIB_centreDescription': 'NCEP'})

        for dsvars in self.grib_file_nd.data_vars:
            dsvar = self.grib_file_nd.data_vars[dsvars]
            dsvar.attrs['GRIB_shortName'] = dsvars
            dsvar.attrs['GRIB_typeOfLevel'] = 'potentialVorticity'
            dsvar.attrs['jScansPositively'] = 0
            dsvar.attrs['GRIB_gridType'] = 'regular_ll'
            dsvar.attrs['GRIB_missingValue'] = 9999
            dsvar.attrs['units'] = 'K'


    def __enter__(self):
        return self
    def __exit__(self):
        pass
    def get_data(self):
        return self.grib_file
    def get_data_nd(self):
        return self.grib_file_nd
    def get_data_irr(self):
        return self.grib_file_irr


class TestGRIBReader(unittest.TestCase):
    """Test GRIB Reader"""
    yaml_file = "grib.yaml"

    def setUp(self):
        """Wrap pygrib to read fake data"""
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        try:
            import cfgrib
        except ImportError:
            cfgrib = None
        self.orig_cfgrib = cfgrib
        sys.modules['cfgrib'] = mock.MagicMock()

    def tearDown(self):
        """Re-enable cfgrib import."""
        sys.modules['cfgrib'] = self.orig_cfgrib

    @mock.patch('satpy.readers.grib.cfgrib')
    def test_init(self, cf):
        """Test basic init with no extra parameters."""
        cf.open_file.return_value = FakeGRIB()
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'gfs.t18z.sfluxgrbf106.grib2',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    #@mock.patch('satpy.readers.grib.cfgrib')
    @mock.patch('satpy.readers.grib.xr.open_dataset')
    def test_load_all(self, xar):
        """Test loading all test datasets"""
        xar.return_value = FakeGRIB().get_data()
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'gfs.t18z.sfluxgrbf106.grib2',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['t', 'u', 'v'])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')
            self.assertIsInstance(v, xr.DataArray)

    @mock.patch('satpy.readers.grib.xr.open_dataset')
    def test_load_all_by_level(self, xar):
        """Test loading all test datasets with a multidimensional grib file"""
        xar.return_value = FakeGRIB().get_data_nd()
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'gfs.t18z.sfluxgrbf106.grib2',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load([
            DatasetID(name='t', level=1),
            DatasetID(name='u', level=1),
            DatasetID(name='v', level=1),
            DatasetID(name='t', level=2),
            DatasetID(name='u', level=2),
            DatasetID(name='v', level=2),
        ])
        # using 3d dataset, so by default should load by level
        self.assertEqual(len(datasets), 6)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')
            self.assertIsInstance(v, xr.DataArray)

    @mock.patch('satpy.readers.grib.xr.open_dataset')
    def test_load_all_nd(self, xar):
        """Test loading all test datasets with a multidimensional grib file"""
        xar.return_value = FakeGRIB().get_data_nd()
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'gfs.t18z.sfluxgrbf106.grib2',
        ])
        r.create_filehandlers(loadables, fh_kwargs={'allow_nd': True})
        datasets = r.load([
            DatasetID(name='t', level=None),
            DatasetID(name='u', level=None),
            DatasetID(name='v', level=None),
        ])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')
            self.assertIsInstance(v, xr.DataArray)
            self.assertEqual(len(v.dims), 3)

#    @mock.patch('satpy.readers.grib.xr.open_dataset')
#    def test_load_all_lcc(self, xar):
#        """Test loading all test datasets with lcc projections"""
#        xar.return_value = FakeGRIB().get_data_irr()
#        from satpy.readers import load_reader
#        from satpy import DatasetID
#        r = load_reader(self.reader_configs)
#        loadables = r.select_files_from_pathnames([
#            'gfs.t18z.sfluxgrbf106.grib2',
#        ])
#        r.create_filehandlers(loadables)
#        datasets = r.load([
#            DatasetID(name='t', level=None),
#            DatasetID(name='u', level=None),
#            DatasetID(name='v', level=None)])
#        self.assertEqual(len(datasets), 3)
#        for v in datasets.values():
#            self.assertEqual(v.attrs['units'], 'K')
#            self.assertIsInstance(v, xr.DataArray)


def suite():
    """The test suite for test_grib."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGRIBReader))

    return mysuite
