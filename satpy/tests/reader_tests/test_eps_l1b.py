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
"""Test the eps l1b format."""

import os
from contextlib import suppress
from tempfile import mkstemp
from unittest import TestCase
from unittest import mock

import numpy as np
import xarray as xr
import satpy
from satpy import DatasetID
from satpy.readers import eps_l1b as eps
grh_dtype = np.dtype([("record_class", "|i1"),
                      ("INSTRUMENT_GROUP", "|i1"),
                      ("RECORD_SUBCLASS", "|i1"),
                      ("RECORD_SUBCLASS_VERSION", "|i1"),
                      ("RECORD_SIZE", ">u4"),
                      ("RECORD_START_TIME", "S6"),
                      ("RECORD_STOP_TIME", "S6")])


def create_sections(structure):
    """Create file sections."""
    sections = {}
    form = eps.XMLFormat(os.path.join(eps.CONFIG_PATH, "eps_avhrrl1b_6.5.xml"))
    for count, (rec_class, sub_class) in structure:
        try:
            the_dtype = form.dtype((rec_class, sub_class))
        except KeyError:
            continue
        item_size = the_dtype.itemsize + grh_dtype.itemsize
        the_dtype = np.dtype(grh_dtype.descr + the_dtype.descr)
        item = np.zeros(count, the_dtype)
        item['record_class'] = eps.record_class.index(rec_class)
        item['RECORD_SUBCLASS'] = sub_class
        item['RECORD_SIZE'] = item_size

        sections[(rec_class, sub_class)] = item
    return sections


class TestEPSL1B(TestCase):
    """Test the filehandler."""

    def setUp(self):
        """Set up the tests."""
        # ipr is not present in the xml format ?
        structure = [(1, ('mphr', 0)), (1, ('sphr', 0)), (11, ('ipr', 0)),
                     (1, ('geadr', 1)), (1, ('geadr', 2)), (1, ('geadr', 3)),
                     (1, ('geadr', 4)), (1, ('geadr', 5)), (1, ('geadr', 6)),
                     (1, ('geadr', 7)), (1, ('giadr', 1)), (1, ('giadr', 2)),
                     (1, ('veadr', 1)), (1080, ('mdr', 2))]

        sections = create_sections(structure)
        sections[('mphr', 0)]['TOTAL_MDR'] = b'TOTAL_MDR                     =   1080\n'
        sections[('mphr', 0)]['SPACECRAFT_ID'] = b'SPACECRAFT_ID                 = M03\n'
        sections[('mphr', 0)]['INSTRUMENT_ID'] = b'INSTRUMENT_ID                 = AVHR\n'
        sections[('sphr', 0)]['EARTH_VIEWS_PER_SCANLINE'] = b'EARTH_VIEWS_PER_SCANLINE      =  2048\n'

        _fd, fname = mkstemp()
        fd = open(_fd)

        self.filename = fname
        for _, arr in sections.items():
            arr.tofile(fd)
        fd.close()
        self.fh = eps.EPSAVHRRFile(self.filename, {'start_time': 'now',
                                                   'end_time': 'later'}, {})

    def test_read_all(self):
        """Test initialization."""
        self.fh._read_all()
        assert(self.fh.scanlines == 1080)
        assert(self.fh.pixels == 2048)

    def test_dataset(self):
        """Test getting a dataset."""
        did = DatasetID('1', calibration='reflectance')
        res = self.fh.get_dataset(did, {})
        assert(isinstance(res, xr.DataArray))
        assert(res.attrs['platform_name'] == 'Metop-C')
        assert(res.attrs['sensor'] == 'avhrr-3')
        assert(res.attrs['name'] == '1')
        assert(res.attrs['calibration'] == 'reflectance')

        did = DatasetID('4', calibration='brightness_temperature')
        res = self.fh.get_dataset(did, {})
        assert(isinstance(res, xr.DataArray))
        assert(res.attrs['platform_name'] == 'Metop-C')
        assert(res.attrs['sensor'] == 'avhrr-3')
        assert(res.attrs['name'] == '4')
        assert(res.attrs['calibration'] == 'brightness_temperature')

    def test_navigation(self):
        """Test the navigation."""
        did = DatasetID('longitude')
        res = self.fh.get_dataset(did, {})
        assert(isinstance(res, xr.DataArray))
        assert(res.attrs['platform_name'] == 'Metop-C')
        assert(res.attrs['sensor'] == 'avhrr-3')
        assert(res.attrs['name'] == 'longitude')

    def test_angles(self):
        """Test the navigation."""
        did = DatasetID('solar_zenith_angle')
        res = self.fh.get_dataset(did, {})
        assert(isinstance(res, xr.DataArray))
        assert(res.attrs['platform_name'] == 'Metop-C')
        assert(res.attrs['sensor'] == 'avhrr-3')
        assert(res.attrs['name'] == 'solar_zenith_angle')

    @mock.patch('satpy.readers.eps_l1b.EPSAVHRRFile.__getitem__')
    @mock.patch('satpy.readers.eps_l1b.EPSAVHRRFile.__init__')
    def test_get_full_angles_twice(self, mock__init__, mock__getitem__):
        """Test get full angles twice."""
        geotiemock = mock.Mock()
        metop20kmto1km = geotiemock.metop20kmto1km
        metop20kmto1km.side_effect = lambda x, y: (x.copy(), y.copy())

        def mock_getitem(key):
            data = {"ANGULAR_RELATIONS_FIRST": np.zeros((7, 4)),
                    "ANGULAR_RELATIONS": np.zeros((7, 103, 4)),
                    "ANGULAR_RELATIONS_LAST": np.zeros((7, 4)),
                    "NAV_SAMPLE_RATE": 20}
            return data[key]
        mock__init__.return_value = None
        mock__getitem__.side_effect = mock_getitem
        avhrr_reader = satpy.readers.eps_l1b.EPSAVHRRFile()
        avhrr_reader.sun_azi = None
        avhrr_reader.sat_azi = None
        avhrr_reader.sun_zen = None
        avhrr_reader.sat_zen = None
        avhrr_reader.scanlines = 7
        avhrr_reader.pixels = 2048

        with mock.patch.dict("sys.modules", geotiepoints=geotiemock):
            # Get dask arrays
            sun_azi, sun_zen, sat_azi, sat_zen = avhrr_reader.get_full_angles()
            # Convert to numpy array
            sun_zen_np1 = np.array(avhrr_reader.sun_zen)
            # Convert to numpy array again
            sun_zen_np2 = np.array(avhrr_reader.sun_zen)
            assert np.allclose(sun_zen_np1, sun_zen_np2)

    def tearDown(self):
        """Tear down the tests."""
        with suppress(OSError):
            os.remove(self.filename)
