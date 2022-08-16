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
"""Module for testing the satpy.readers.grib module."""

import os
import sys
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from satpy.dataset import DataQuery

# Parameterized cases
TEST_ARGS = ('proj_params', 'lon_corners', 'lat_corners')
TEST_PARAMS = (
    (None, None, None),  # cyl default case
    (
        {
            'a': 6371229, 'b': 6371229, 'proj': 'lcc',
            'lon_0': 265.0, 'lat_0': 25.0,
            'lat_1': 25.0, 'lat_2': 25.0
        },
        [-133.459, -65.12555139, -152.8786225, -49.41598659],
        [12.19, 14.34208538, 54.56534318, 57.32843565]
    ),
)


def fake_gribdata():
    """Return some faked data for use as grib values."""
    return np.arange(25.).reshape((5, 5))


def _round_trip_projection_lonlat_check(area):
    """Check that X/Y coordinates can be transformed multiple times.

    Many GRIB files include non-standard projects that work for the
    initial transformation of X/Y coordinates to longitude/latitude,
    but may fail in the reverse transformation. For example, an eqc
    projection that goes from 0 longitude to 360 longitude. The X/Y
    coordinates may accurately go from the original X/Y metered space
    to the correct longitude/latitude, but transforming those coordinates
    back to X/Y space will produce the wrong result.

    """
    from pyproj import Proj
    p = Proj(area.crs)
    x, y = area.get_proj_vectors()
    lon, lat = p(x, y, inverse=True)
    x2, y2 = p(lon, lat)
    np.testing.assert_almost_equal(x, x2)
    np.testing.assert_almost_equal(y, y2)


class FakeMessage(object):
    """Fake message returned by pygrib.open().message(x)."""

    def __init__(self, values, proj_params=None, latlons=None, **attrs):
        """Init the message."""
        super(FakeMessage, self).__init__()
        self.attrs = attrs
        self.values = values
        if proj_params is None:
            proj_params = {'a': 6371229, 'b': 6371229, 'proj': 'cyl'}
        self.projparams = proj_params
        self._latlons = latlons

    def keys(self):
        """Get message keys."""
        return self.attrs.keys()

    def latlons(self):
        """Get coordinates."""
        return self._latlons

    def __getitem__(self, item):
        """Get item."""
        return self.attrs[item]

    def valid_key(self, key):
        """Validate key."""
        return True


class FakeGRIB(object):
    """Fake GRIB file returned by pygrib.open."""

    def __init__(self, messages=None, proj_params=None, latlons=None):
        """Init the grib file."""
        super(FakeGRIB, self).__init__()
        if messages is not None:
            self._messages = messages
        else:
            self._messages = [
                FakeMessage(
                    values=fake_gribdata(),
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
                    modelName='notknown',
                    minimum=100.,
                    maximum=200.,
                    typeOfLevel='isobaricInhPa',
                    jScansPositively=0,
                    proj_params=proj_params,
                    latlons=latlons,
                ),
                FakeMessage(
                    values=fake_gribdata(),
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
                    modelName='notknown',
                    minimum=100.,
                    maximum=200.,
                    typeOfLevel='isobaricInhPa',
                    jScansPositively=1,
                    proj_params=proj_params,
                    latlons=latlons,
                ),
                FakeMessage(
                    values=fake_gribdata(),
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
        """Get a message."""
        return self._messages[msg_num - 1]

    def seek(self, loc):
        """Seek."""
        return

    def __iter__(self):
        """Iterate."""
        return iter(self._messages)

    def __enter__(self):
        """Enter."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit."""


class TestGRIBReader:
    """Test GRIB Reader."""

    yaml_file = "grib.yaml"

    def setup_method(self):
        """Wrap pygrib to read fake data."""
        from satpy._config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))

        try:
            import pygrib
        except ImportError:
            pygrib = None
        self.orig_pygrib = pygrib
        sys.modules['pygrib'] = mock.MagicMock()

    def teardown_method(self):
        """Re-enable pygrib import."""
        sys.modules['pygrib'] = self.orig_pygrib

    def _get_test_datasets(self, dataids, fake_pygrib=None):
        from satpy.readers import load_reader
        if fake_pygrib is None:
            fake_pygrib = FakeGRIB()

        with mock.patch('satpy.readers.grib.pygrib') as pg:
            pg.open.return_value = fake_pygrib
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames([
                'gfs.t18z.sfluxgrbf106.grib2',
            ])
            r.create_filehandlers(loadables)
            datasets = r.load(dataids)
        return datasets

    @staticmethod
    def _get_fake_pygrib(proj_params, lon_corners, lat_corners):
        latlons = None
        if lon_corners is not None:
            lats = np.array([
                [lat_corners[0], 0, 0, 0, lat_corners[1]],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [lat_corners[2], 0, 0, 0, lat_corners[3]]])
            lons = np.array([
                [lon_corners[0], 0, 0, 0, lon_corners[1]],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [lon_corners[2], 0, 0, 0, lon_corners[3]]])
            latlons = (lats, lons)

        fake_pygrib = FakeGRIB(
            proj_params=proj_params,
            latlons=latlons)
        return fake_pygrib

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        with mock.patch('satpy.readers.grib.pygrib') as pg:
            pg.open.return_value = FakeGRIB()
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames([
                'gfs.t18z.sfluxgrbf106.grib2',
            ])
            assert len(loadables) == 1
            r.create_filehandlers(loadables)
            # make sure we have some files
            assert r.file_handlers

    def test_file_pattern(self):
        """Test matching of file patterns."""
        from satpy.readers import load_reader

        filenames = [
                "quinoa.grb",
                "tempeh.grb2",
                "tofu.grib2",
                "falafel.grib",
                "S_NWC_NWP_1900-01-01T00:00:00Z_999.grib"]

        r = load_reader(self.reader_configs)
        files = r.select_files_from_pathnames(filenames)
        assert len(files) == 4

    @pytest.mark.parametrize(TEST_ARGS, TEST_PARAMS)
    def test_load_all(self, proj_params, lon_corners, lat_corners):
        """Test loading all test datasets."""
        fake_pygrib = self._get_fake_pygrib(proj_params, lon_corners, lat_corners)
        dataids = [
            DataQuery(name='t', level=100, modifiers=tuple()),
            DataQuery(name='t', level=200, modifiers=tuple()),
            DataQuery(name='t', level=300, modifiers=tuple())
        ]
        datasets = self._get_test_datasets(dataids, fake_pygrib)

        assert len(datasets) == 3
        for v in datasets.values():
            assert v.attrs['units'] == 'K'
            assert isinstance(v, xr.DataArray)

    @pytest.mark.parametrize(TEST_ARGS, TEST_PARAMS)
    def test_area_def_crs(self, proj_params, lon_corners, lat_corners):
        """Check that the projection is accurate."""
        fake_pygrib = self._get_fake_pygrib(proj_params, lon_corners, lat_corners)
        dataids = [DataQuery(name='t', level=100, modifiers=tuple())]
        datasets = self._get_test_datasets(dataids, fake_pygrib)
        area = datasets['t'].attrs['area']
        if not hasattr(area, 'crs'):
            pytest.skip("Can't test with pyproj < 2.0")
        _round_trip_projection_lonlat_check(area)

    @pytest.mark.parametrize(TEST_ARGS, TEST_PARAMS)
    def test_missing_attributes(self, proj_params, lon_corners, lat_corners):
        """Check that the grib reader handles missing attributes in the grib file."""
        fake_pygrib = self._get_fake_pygrib(proj_params, lon_corners, lat_corners)

        # This has modelName
        query_contains = DataQuery(name='t', level=100, modifiers=tuple())
        # This does not have modelName
        query_not_contains = DataQuery(name='t', level=300, modifiers=tuple())
        dataset = self._get_test_datasets([query_contains, query_not_contains], fake_pygrib)
        assert dataset[query_contains].attrs['modelName'] == 'notknown'
        assert dataset[query_not_contains].attrs['modelName'] == 'unknown'

    @pytest.mark.parametrize(TEST_ARGS, TEST_PARAMS)
    def test_jscanspositively(self, proj_params, lon_corners, lat_corners):
        """Check that data is flipped if the jScansPositively is present."""
        fake_pygrib = self._get_fake_pygrib(proj_params, lon_corners, lat_corners)

        # This has no jScansPositively
        query_not_contains = DataQuery(name='t', level=100, modifiers=tuple())
        # This contains jScansPositively
        query_contains = DataQuery(name='t', level=200, modifiers=tuple())
        dataset = self._get_test_datasets([query_contains, query_not_contains], fake_pygrib)

        np.testing.assert_allclose(fake_gribdata(), dataset[query_not_contains].values)
        np.testing.assert_allclose(fake_gribdata(), dataset[query_contains].values[::-1])
