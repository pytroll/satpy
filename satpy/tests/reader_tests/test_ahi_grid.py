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
import numpy as np
import dask.array as da
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

    @staticmethod
    def make_fh(filetype, area='fld'):
        """Create a test file handler."""
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            fh = AHIGriddedFileHandler('somefile',
                                       {'area': area},
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

    def test_bad_area(self):
        """"Ensure an error is raised for an usupported area."""
        tmp_fh = self.make_fh('ext.01')
        tmp_fh.areaname = 'scanning'
        with self.assertRaises(NotImplementedError):
            tmp_fh.get_area_def(None)
        with self.assertRaises(NotImplementedError):
            self.make_fh('ext.01', area='scanning')


class TestAHIGriddedFileCalibration(unittest.TestCase):
    """Test case for the file calibration types."""

    def setUp(self):
        """Create a test file handler."""
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            in_fname = 'test_file'
            fh = AHIGriddedFileHandler(in_fname,
                                       {'area': 'fld'},
                                       filetype_info={'file_type': 'tir.01'})
            self.fh = fh

    @mock.patch('satpy.readers.ahi_grid.AHIGriddedFileHandler._get_luts')
    @mock.patch('satpy.readers.ahi_grid.os.path.exists')
    @mock.patch('satpy.readers.ahi_grid.np.loadtxt')
    def test_calibrate(self, np_loadtxt, os_exist, get_luts):
        """Test the calibration modes of AHI using the LUTs"""
        load_return = np.squeeze(np.dstack([np.arange(0, 2048, 1),
                                            np.arange(0, 120, 0.05859375)]))

        np_loadtxt.return_value = load_return
        get_luts.return_value = True

        in_data = np.array([[100., 300., 500.],
                           [800., 1500., 2040.]])

        refl_out = np.array([[5.859375, 17.578125, 29.296875],
                             [46.875, 87.890625, 119.53125]])

        os_exist.return_value = False
        # Check that the LUT download is called if we don't have the LUTS
        self.fh.calibrate(in_data, 'reflectance')
        get_luts.assert_called()

        os_exist.return_value = True
        # Ensure results equal if no calibration applied
        out_data = self.fh.calibrate(in_data, 'counts')
        np.testing.assert_equal(in_data, out_data)

        # Now ensure results equal if LUT calibration applied
        out_data = self.fh.calibrate(in_data, 'reflectance')
        np.testing.assert_allclose(refl_out, out_data)

        # Check that exception is raised if bad calibration is passed
        with self.assertRaises(NotImplementedError):
            self.fh.calibrate(in_data, 'lasers')

        # Check that exception is raised if no file is present
        np_loadtxt.side_effect = FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            self.fh.calibrate(in_data, 'reflectance')


class TestAHIGriddedFileHandler(unittest.TestCase):
    """Test case for the file reading."""

    @staticmethod
    def new_unzip(fname):
        """Fake unzipping."""
        if fname[-3:] == 'bz2':
            return fname[:-4]
        else:
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

        key = {'calibration': 'counts',
               'name': 'vis.01'}
        info = {'units': 'unitless',
                'standard_name': 'vis.01',
                'wavelength': 10.8,
                'resolution': 0.05}
        self.key = key
        self.info = info

    @mock.patch('satpy.readers.ahi_grid.np.memmap')
    def test_dataread(self, memmap):
        """Check that a dask array is returned from the read function."""
        test_arr = np.zeros((10, 10))
        memmap.return_value = test_arr
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            res = self.fh._read_data(mock.MagicMock())
            np.testing.assert_allclose(res, da.from_array(test_arr))

    @mock.patch('satpy.readers.ahi_grid.AHIGriddedFileHandler._read_data')
    def test_get_dataset(self, mocked_read):
        """Check that a good dataset is returned on request."""
        m = mock.mock_open()

        out_data = np.array([[100., 300., 500.],
                             [800., 1500., 2040.]])
        mocked_read.return_value = out_data
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            res = self.fh.get_dataset(self.key, self.info)
            mocked_read.assert_called()
            # Check output data is correct
            np.testing.assert_allclose(res.values, out_data)
            # Also check a couple of attributes
            self.assertEqual(res.attrs['name'], self.key['name'])
            self.assertEqual(res.attrs['wavelength'], self.info['wavelength'])


class TestAHIGriddedLUTs(unittest.TestCase):
    """Test case for the downloading and preparing LUTs."""
    def setUp(self):
        """Create a test file handler."""
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            in_fname = 'test_file'
            fh = AHIGriddedFileHandler(in_fname,
                                       {'area': 'fld'},
                                       filetype_info={'file_type': 'tir.01'})
            self.fh = fh

        key = {'calibration': 'counts',
               'name': 'vis.01'}
        info = {'units': 'unitless',
                'standard_name': 'vis.01',
                'wavelength': 10.8,
                'resolution': 0.05}
        self.key = key
        self.info = info

    @mock.patch('ftplib.FTP')
    def test_download_luts(self, mock_ftp):
        """Test that the FTP library is called for downloading LUTS."""
        m = mock.mock_open()
        with mock.patch('satpy.readers.ahi_grid.open', m, create=True):
            self.fh._download_luts('/test_file')
            mock_ftp.assert_called()

    @mock.patch('os.remove')
    def test_untar_luts(self, mock_remove):
        """Test that the untar library is used correctly."""
        m = mock.MagicMock()
        with mock.patch('tarfile.open', m, create=True):
            self.fh._untar_luts('/test_file', '/test_dir/')
            m.assert_called()
            mock_remove.assert_called()

    @mock.patch('os.path.join')
    @mock.patch('shutil.rmtree')
    @mock.patch('shutil.move')
    @mock.patch('satpy.readers.ahi_grid.AHIGriddedFileHandler._untar_luts')
    @mock.patch('satpy.readers.ahi_grid.AHIGriddedFileHandler._download_luts')
    @mock.patch('tempfile.gettempdir')
    @mock.patch('pathlib.Path')
    def test_get_luts(self, *mocks):
        """Check that the function to download LUTs operates successfully."""
        self.fh._get_luts()
        mocks[2].assert_called()
        mocks[3].assert_called()
