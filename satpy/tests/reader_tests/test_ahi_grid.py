#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""The ahi_grid reader tests package."""

import unittest
from unittest import mock
import warnings
import numpy as np
import dask.array as da
from datetime import datetime
from pyresample.geometry import AreaDefinition
from satpy.readers.ahi_grid import AHIGriddedFileHandler


class TestAHIGriddedArea(unittest.TestCase):
    """Test the AHI gridded reader definition."""

    def setUp(self):
        self.FULLDISK_SIZES = {0.005: {'x_size': 24000,
                                       'y_size': 24000},
                               0.01: {'x_size': 12000,
                                      'y_size': 12000},
                               0.02: {'x_size': 6000,
                                      'y_size': 6000}}

        self.AHI_FULLDISK_EXTENT = [85., -60., 205., 60.]

    def make_fh(self, filetype):
        """Create a test file handler."""
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            fh = AHIGriddedFileHandler('somefile',
                                       {'area': 'fld'},
                                       filetype_info={'file_type': filetype})
            return fh

    def test_low_res(self):
        """Check size of the low resolution (2km) grid."""
        tmp_fh = self.make_fh('tir.01')
        self.assertEqual(self.FULLDISK_SIZES[0.02]['x_size'], tmp_fh.ncols)
        self.assertEqual(self.FULLDISK_SIZES[0.02]['y_size'], tmp_fh.nlines)

    def test_med_res(self):
        """Check size of the low resolution (1km) grid."""
        tmp_fh = self.make_fh('vis.02')
        self.assertEqual(self.FULLDISK_SIZES[0.01]['x_size'], tmp_fh.ncols)
        self.assertEqual(self.FULLDISK_SIZES[0.01]['y_size'], tmp_fh.nlines)

    def test_hi_res(self):
        """Check size of the low resolution (0.5km) grid."""
        tmp_fh = self.make_fh('ext.01')
        self.assertEqual(self.FULLDISK_SIZES[0.005]['x_size'], tmp_fh.ncols)
        self.assertEqual(self.FULLDISK_SIZES[0.005]['y_size'], tmp_fh.nlines)

    def test_area_def(self):
        """Check that a valid full disk area is produced."""

        good_area = AreaDefinition('gridded_himawari',
                                   'A gridded Himawari area',
                                   'longlat',
                                   'EPSG:4326',
                                   self.FULLDISK_SIZES[0.01]['x_size'],
                                   self.FULLDISK_SIZES[0.01]['y_size'],
                                   self.AHI_FULLDISK_EXTENT)

        tmp_fh = self.make_fh('vis.01')
        tmp_fh.get_area_def(None)
        self.assertEqual(tmp_fh.area.area_extent, good_area.area_extent)
        self.assertEqual(tmp_fh.area.area_id, good_area.area_id)
        self.assertEqual(tmp_fh.area.description, good_area.description)
        self.assertEqual(tmp_fh.area.area_extent, good_area.area_extent)
        self.assertEqual(tmp_fh.area.proj_str, good_area.proj_str)


class TestAHIGriddedFileHandler(unittest.TestCase):
    """Test case for the file reading."""

    def new_unzip(fname):
        """Fake unzipping."""
        if(fname[-3:] == 'bz2'):
            return fname[:-4]
        return fname

    @mock.patch('satpy.readers.ahi_grid.unzip_file',
                mock.MagicMock(side_effect=new_unzip))
    def setUp(self):
        """Create a test file handler."""
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            in_fname = 'test_file.bz2'
            fh = AHIGriddedFileHandler(in_fname,
                                       {'area': 'fld'},
                                       filetype_info={'file_type': 'tir.01'})

            # Check that the filename is altered for bz2 format files
            self.assertNotEqual(in_fname, fh.filename)
            self.fh = fh

    @mock.patch('satpy.readers.ahi_grid.os.path.exists')
    @mock.patch('satpy.readers.ahi_grid.np.loadtxt')
    def test_calibrate(self, os_exist, np_loadtxt):

        in_data = np.array([[100., 300., 500.],
                           [800., 1500., 2400.]])

        self.fh.calibrate()
