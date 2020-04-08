#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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

import sys
import numpy as np
import unittest
from unittest import mock
from datetime import datetime


class FakeMessage(object):
    """Fake message returned by pygrib.open().message(x)."""
    def __init__(self, values, proj_params=None, latlons=None, **attrs):
        super(FakeMessage, self).__init__()
        self.attrs = attrs
        self.values = values
        if proj_params is None:
            proj_params = {'a': 6378140.0, 'b': 6356755.0, 'lat_0': 0.0,
                           'lon_0': 0.0, 'proj': 'geos', 'h': 35785830.098}
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
                    name='Instantaneous rain rate',
                    shortName='irrate',
                    cfName='unknown',
                    units='kg m**-2 s**-1',
                    dataDate=20190603,
                    dataTime=1645,
                    missingValue=9999,
                    modelName='unknown',
                    centreDescription='Rome',
                    minimum=0.0,
                    maximum=0.01475,
                    Nx=3712,
                    Ny=3712,
                    NrInRadiusOfEarth=6.6107,
                    dx=3622,
                    dy=3610,
                    XpInGridLengths=1856.0,
                    YpInGridLengths=1856.0,
                    jScansPositively=0,
                    proj_params=proj_params,
                    latlons=latlons,
                )
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


class TestHSAFFileHandler(unittest.TestCase):
    """Test HSAF Reader"""

    def setUp(self):
        """Wrap pygrib to read fake data"""
        try:
            import pygrib
        except ImportError:
            pygrib = None
        self.orig_pygrib = pygrib
        sys.modules['pygrib'] = mock.MagicMock()

    def tearDown(self):
        """Re-enable pygrib import."""
        sys.modules['pygrib'] = self.orig_pygrib

    @mock.patch('satpy.readers.hsaf_grib.pygrib.open', return_value=FakeGRIB())
    def test_init(self, pg):
        """
        Test the init function, ensure that the correct dates and metadata
        are returned
        """
        pg.open.return_value = FakeGRIB()
        correct_dt = datetime(2019, 6, 3, 16, 45, 0)
        from satpy.readers.hsaf_grib import HSAFFileHandler
        fh = HSAFFileHandler('filename', mock.MagicMock(), mock.MagicMock())
        self.assertEqual(fh._analysis_time, correct_dt)
        self.assertEqual(fh.metadata['projparams']['lat_0'], 0.0)
        self.assertEqual(fh.metadata['shortName'], 'irrate')
        self.assertEqual(fh.metadata['nx'], 3712)

    @mock.patch('satpy.readers.hsaf_grib.pygrib.open', return_value=FakeGRIB())
    def test_get_area_def(self, pg):
        """
        Test the area definition setup, checks the size and extent
        """
        pg.open.return_value = FakeGRIB()
        from satpy.readers.hsaf_grib import HSAFFileHandler
        fh = HSAFFileHandler('filename', mock.MagicMock(), mock.MagicMock())
        area_def = HSAFFileHandler.get_area_def(fh, 'H03B')
        self.assertEqual(area_def.x_size, 3712)
        self.assertAlmostEqual(area_def.area_extent[0], -5569209.3026, places=3)
        self.assertAlmostEqual(area_def.area_extent[3], 5587721.9097, places=3)

    @mock.patch('satpy.readers.hsaf_grib.pygrib.open', return_value=FakeGRIB())
    def test_get_dataset(self, pg):
        """
        Test reading the actual datasets from a grib file
        """
        pg.open.return_value = FakeGRIB()
        from satpy.readers.hsaf_grib import HSAFFileHandler
        # Instantaneous precipitation
        fh = HSAFFileHandler('filename', mock.MagicMock(), mock.MagicMock())
        fh.filename = "H03B"
        ds_id = mock.Mock()
        ds_id.name = 'H03B'
        data = fh.get_dataset(ds_id, mock.Mock())
        np.testing.assert_array_equal(data.values, np.arange(25.).reshape((5, 5)))

        # Accumulated precipitation
        fh = HSAFFileHandler('filename', mock.MagicMock(), mock.MagicMock())
        fh.filename = "H05B"
        ds_id = mock.Mock()
        ds_id.name = 'H05B'
        data = fh.get_dataset(ds_id, mock.Mock())
        np.testing.assert_array_equal(data.values, np.arange(25.).reshape((5, 5)))
