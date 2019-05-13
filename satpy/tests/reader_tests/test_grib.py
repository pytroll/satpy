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


class FakeMessage(object):
    """Fake message returned by pygrib.open().message(x)."""

    def __init__(self, values, proj_params=None, latlons=None, **attrs):
        super(FakeMessage, self).__init__()
        self.attrs = attrs
        self.values = values
        if proj_params is None:
            proj_params = {'a': 6371229, 'b': 6371229, 'proj': 'cyl'}
        self.projparams = proj_params
        self._latlons = latlons

    def latlons(self):
        return self._latlons

    def __getitem__(self, item):
        return self.attrs[item]

    def valid_key(self, key):
        return True


class FakeGRIB(object):
    """Fake GRIB file returned by pygrib.open."""

    def __init__(self, messages=None, proj_params=None, latlons=None):
        super(FakeGRIB, self).__init__()
        if messages is not None:
            self._messages = messages
        else:
            self._messages = [
                FakeMessage(
                    values=np.arange(25.).reshape((5, 5)),
                    name='TEST',
                    shortName='t',
                    level=100,
                    pressureUnits='hPa',
                    cfName='air_temperature',
                    units='K',
                    dataDate=20180504,
                    dataTime=1200,
                    validityDate=20180504,
                    validityTime=1800,
                    distinctLongitudes=np.arange(5.),
                    distinctLatitudes=np.arange(5.),
                    missingValue=9999,
                    modelName='unknown',
                    minimum=100.,
                    maximum=200.,
                    typeOfLevel='isobaricInhPa',
                    jScansPositively=0,
                    proj_params=proj_params,
                    latlons=latlons,
                ),
                FakeMessage(
                    values=np.arange(25.).reshape((5, 5)),
                    name='TEST',
                    shortName='t',
                    level=200,
                    pressureUnits='hPa',
                    cfName='air_temperature',
                    units='K',
                    dataDate=20180504,
                    dataTime=1200,
                    validityDate=20180504,
                    validityTime=1800,
                    distinctLongitudes=np.arange(5.),
                    distinctLatitudes=np.arange(5.),
                    missingValue=9999,
                    modelName='unknown',
                    minimum=100.,
                    maximum=200.,
                    typeOfLevel='isobaricInhPa',
                    jScansPositively=0,
                    proj_params=proj_params,
                    latlons=latlons,
                ),
                FakeMessage(
                    values=np.arange(25.).reshape((5, 5)),
                    name='TEST',
                    shortName='t',
                    level=300,
                    pressureUnits='hPa',
                    cfName='air_temperature',
                    units='K',
                    dataDate=20180504,
                    dataTime=1200,
                    validityDate=20180504,
                    validityTime=1800,
                    distinctLongitudes=np.arange(5.),
                    distinctLatitudes=np.arange(5.),
                    missingValue=9999,
                    modelName='unknown',
                    minimum=100.,
                    maximum=200.,
                    typeOfLevel='isobaricInhPa',
                    jScansPositively=0,
                    proj_params=proj_params,
                    latlons=latlons,
                ),
            ]
        self.messages = len(self._messages)

    def message(self, msg_num):
        return self._messages[msg_num - 1]

    def seek(self, loc):
        return

    def __iter__(self):
        return iter(self._messages)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestGRIBReader(unittest.TestCase):
    """Test GRIB Reader"""
    yaml_file = "grib.yaml"

    def setUp(self):
        """Wrap pygrib to read fake data"""
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

        try:
            import pygrib
        except ImportError:
            pygrib = None
        self.orig_pygrib = pygrib
        sys.modules['pygrib'] = mock.MagicMock()

    def tearDown(self):
        """Re-enable pygrib import."""
        sys.modules['pygrib'] = self.orig_pygrib

    @mock.patch('satpy.readers.grib.pygrib')
    def test_init(self, pg):
        """Test basic init with no extra parameters."""
        pg.open.return_value = FakeGRIB()
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'gfs.t18z.sfluxgrbf106.grib2',
        ])
        self.assertTrue(len(loadables), 1)
        r.create_filehandlers(loadables)
        # make sure we have some files
        self.assertTrue(r.file_handlers)

    @mock.patch('satpy.readers.grib.pygrib')
    def test_load_all(self, pg):
        """Test loading all test datasets"""
        pg.open.return_value = FakeGRIB()
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'gfs.t18z.sfluxgrbf106.grib2',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load([
            DatasetID(name='t', level=100),
            DatasetID(name='t', level=200),
            DatasetID(name='t', level=300)])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')
            self.assertIsInstance(v, xr.DataArray)

    @mock.patch('satpy.readers.grib.pygrib')
    def test_load_all_lcc(self, pg):
        """Test loading all test datasets with lcc projections"""
        lons = np.array([
            [12.19, 0, 0, 0, 14.34208538],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [54.56534318, 0, 0, 0, 57.32843565]])
        lats = np.array([
            [-133.459, 0, 0, 0, -65.12555139],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [-152.8786225, 0, 0, 0, -49.41598659]])
        pg.open.return_value = FakeGRIB(
            proj_params={
                'a': 6371229, 'b': 6371229, 'proj': 'lcc',
                'lon_0': 265.0, 'lat_0': 25.0,
                'lat_1': 25.0, 'lat_2': 25.0},
            latlons=(lats, lons))
        from satpy.readers import load_reader
        from satpy import DatasetID
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'gfs.t18z.sfluxgrbf106.grib2',
        ])
        r.create_filehandlers(loadables)
        datasets = r.load([
            DatasetID(name='t', level=100),
            DatasetID(name='t', level=200),
            DatasetID(name='t', level=300)])
        self.assertEqual(len(datasets), 3)
        for v in datasets.values():
            self.assertEqual(v.attrs['units'], 'K')
            self.assertIsInstance(v, xr.DataArray)


def suite():
    """The test suite for test_grib."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestGRIBReader))

    return mysuite
