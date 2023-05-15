#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2018 Satpy developers
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
"""The HRIT msg reader tests package."""

import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import xarray as xr
from numpy import testing as npt

import satpy.tests.reader_tests.test_seviri_l1b_hrit_setup as setup
from satpy.readers.seviri_l1b_hrit import HRITMSGEpilogueFileHandler, HRITMSGFileHandler, HRITMSGPrologueFileHandler
from satpy.tests.reader_tests.test_seviri_base import ORBIT_POLYNOMIALS_INVALID
from satpy.tests.reader_tests.test_seviri_l1b_calibration import TestFileHandlerCalibrationBase
from satpy.tests.utils import assert_attrs_equal, make_dataid


class TestHRITMSGBase(unittest.TestCase):
    """Baseclass for SEVIRI HRIT reader tests."""

    def assert_attrs_equal(self, attrs, attrs_exp):
        """Assert equality of dataset attributes."""
        assert_attrs_equal(attrs, attrs_exp, tolerance=1e-4)


class TestHRITMSGFileHandlerHRV(TestHRITMSGBase):
    """Test the HRITFileHandler."""

    def setUp(self):
        """Set up the hrit file handler for testing HRV."""
        self.start_time = datetime(2016, 3, 3, 0, 0)
        self.nlines = 464
        self.reader = setup.get_fake_file_handler(
            start_time=self.start_time,
            nlines=self.nlines,
            ncols=5568,
        )
        self.reader.mda.update({
            'segment_sequence_number': 18,
            'planned_start_segment_number': 1
        })
        self.reader.fill_hrv = True

    @mock.patch('satpy.readers.hrit_base.np.memmap')
    def test_read_hrv_band(self, memmap):
        """Test reading the hrv band."""
        nbits = self.reader.mda['number_of_bits_per_pixel']
        memmap.return_value = np.random.randint(0, 256,
                                                size=int((464 * 5568 * nbits) / 8),
                                                dtype=np.uint8)
        res = self.reader.read_band('HRV', None)
        self.assertEqual(res.shape, (464, 5568))

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset(self, calibrate, parent_get_dataset):
        """Test getting the hrv dataset."""
        key = make_dataid(name='HRV', calibration='reflectance')
        info = setup.get_fake_dataset_info()

        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = xr.DataArray(data=np.zeros((464, 5568)), dims=('y', 'x'))
        res = self.reader.get_dataset(key, info)
        self.assertEqual(res.shape, (464, 11136))

        # Test method calls
        parent_get_dataset.assert_called_with(key, info)
        calibrate.assert_called_with(parent_get_dataset(), key['calibration'])

        self.assert_attrs_equal(res.attrs, setup.get_attrs_exp())
        np.testing.assert_equal(
            res['acq_time'],
            setup.get_acq_time_exp(self.start_time, self.nlines)
        )

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset_non_fill(self, calibrate, parent_get_dataset):
        """Test getting a non-filled hrv dataset."""
        key = make_dataid(name='HRV', calibration='reflectance')
        key.name = 'HRV'
        info = setup.get_fake_dataset_info()
        self.reader.fill_hrv = False
        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = xr.DataArray(data=np.zeros((464, 5568)), dims=('y', 'x'))
        res = self.reader.get_dataset(key, info)
        self.assertEqual(res.shape, (464, 5568))

        # Test method calls
        parent_get_dataset.assert_called_with(key, info)
        calibrate.assert_called_with(parent_get_dataset(), key['calibration'])

        self.assert_attrs_equal(res.attrs, setup.get_attrs_exp())
        np.testing.assert_equal(
            res['acq_time'],
            setup.get_acq_time_exp(self.start_time, self.nlines)
        )

    def test_get_area_def(self):
        """Test getting the area def."""
        from pyresample.utils import proj4_radius_parameters
        area = self.reader.get_area_def(make_dataid(name='HRV', resolution=1000))
        self.assertEqual(area.area_extent,
                         (-45561979844414.07, -3720765401003.719, 45602912357076.38, 77771774058.38356))
        proj_dict = area.proj_dict
        a, b = proj4_radius_parameters(proj_dict)
        self.assertEqual(a, 6378169.0)
        self.assertAlmostEqual(b, 6356583.8)
        self.assertEqual(proj_dict['h'], 35785831.0)
        self.assertEqual(proj_dict['lon_0'], 0.0)
        self.assertEqual(proj_dict['proj'], 'geos')
        self.assertEqual(proj_dict['units'], 'm')
        self.reader.fill_hrv = False
        area = self.reader.get_area_def(make_dataid(name='HRV', resolution=1000))
        npt.assert_allclose(area.defs[0].area_extent,
                            (-22017598561055.01, -2926674655354.9604, 23564847539690.22, 77771774058.38356))
        npt.assert_allclose(area.defs[1].area_extent,
                            (-30793529275853.656, -3720765401003.719, 14788916824891.568, -2926674655354.9604))

        self.assertEqual(area.defs[0].area_id, 'msg_seviri_fes_1km')
        self.assertEqual(area.defs[1].area_id, 'msg_seviri_fes_1km')


class TestHRITMSGFileHandler(TestHRITMSGBase):
    """Test the HRITFileHandler."""

    def setUp(self):
        """Set up the hrit file handler for testing."""
        self.start_time = datetime(2016, 3, 3, 0, 0)
        self.nlines = 464
        self.ncols = 3712
        self.projection_longitude = 9.5
        self.reader = setup.get_fake_file_handler(
            start_time=self.start_time,
            nlines=self.nlines,
            ncols=self.ncols,
            projection_longitude=self.projection_longitude
        )
        self.reader.mda.update({
            'segment_sequence_number': 18,
            'planned_start_segment_number': 1
        })

    def _get_fake_data(self):
        return xr.DataArray(
            data=np.zeros((self.nlines, self.ncols)),
            dims=('y', 'x')
        )

    def test_get_area_def(self):
        """Test getting the area def."""
        from pyresample.utils import proj4_radius_parameters
        area = self.reader.get_area_def(make_dataid(name='VIS006', resolution=3000))
        proj_dict = area.proj_dict
        a, b = proj4_radius_parameters(proj_dict)
        self.assertEqual(a, 6378169.0)
        self.assertAlmostEqual(b, 6356583.8)
        self.assertEqual(proj_dict['h'], 35785831.0)
        self.assertEqual(proj_dict['lon_0'], self.projection_longitude)
        self.assertEqual(proj_dict['proj'], 'geos')
        self.assertEqual(proj_dict['units'], 'm')
        self.assertEqual(area.area_extent,
                         (-77771774058.38356, -3720765401003.719,
                          30310525626438.438, 77771774058.38356))

        # Data shifted by 1.5km to N-W
        self.reader.mda['offset_corrected'] = False
        area = self.reader.get_area_def(make_dataid(name='VIS006', resolution=3000))
        self.assertEqual(area.area_extent,
                         (-77771772558.38356, -3720765402503.719,
                          30310525627938.438, 77771772558.38356))

        self.assertEqual(area.area_id, 'msg_seviri_rss_3km')

    @mock.patch('satpy.readers.hrit_base.np.memmap')
    def test_read_band(self, memmap):
        """Test reading a band."""
        nbits = self.reader.mda['number_of_bits_per_pixel']
        memmap.return_value = np.random.randint(0, 256,
                                                size=int((464 * 3712 * nbits) / 8),
                                                dtype=np.uint8)
        res = self.reader.read_band('VIS006', None)
        self.assertEqual(res.shape, (464, 3712))

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset(self, calibrate, parent_get_dataset):
        """Test getting the dataset."""
        data = self._get_fake_data()
        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = data

        key = make_dataid(name='VIS006', calibration='reflectance')
        info = setup.get_fake_dataset_info()
        res = self.reader.get_dataset(key, info)

        # Test method calls
        new_data = np.zeros_like(data.data).astype('float32')
        new_data[:, :] = np.nan
        expected = data.copy(data=new_data)

        expected['acq_time'] = (
            'y',
            setup.get_acq_time_exp(self.start_time, self.nlines)
        )
        xr.testing.assert_equal(res, expected)
        self.assert_attrs_equal(
            res.attrs,
            setup.get_attrs_exp(self.projection_longitude)
        )

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset_without_masking_bad_scan_lines(self, calibrate, parent_get_dataset):
        """Test getting the dataset."""
        data = self._get_fake_data()
        parent_get_dataset.return_value = mock.MagicMock()
        calibrate.return_value = data

        key = make_dataid(name='VIS006', calibration='reflectance')
        info = setup.get_fake_dataset_info()
        self.reader.mask_bad_quality_scan_lines = False
        res = self.reader.get_dataset(key, info)

        # Test method calls
        expected = data.copy()
        expected['acq_time'] = (
            'y',
            setup.get_acq_time_exp(self.start_time, self.nlines)
        )
        xr.testing.assert_equal(res, expected)
        self.assert_attrs_equal(
            res.attrs,
            setup.get_attrs_exp(self.projection_longitude)
        )

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITFileHandler.get_dataset')
    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.calibrate')
    def test_get_dataset_with_raw_metadata(self, calibrate, parent_get_dataset):
        """Test getting the dataset."""
        calibrate.return_value = self._get_fake_data()
        key = make_dataid(name='VIS006', calibration='reflectance')
        info = setup.get_fake_dataset_info()
        self.reader.include_raw_metadata = True
        res = self.reader.get_dataset(key, info)
        assert 'raw_metadata' in res.attrs

    def test_get_raw_mda(self):
        """Test provision of raw metadata."""
        self.reader.mda = {'segment': 1, 'loff': 123}
        self.reader.prologue_.reduce = lambda max_size: {'prologue': 1}
        self.reader.epilogue_.reduce = lambda max_size: {'epilogue': 1}
        expected = {'prologue': 1, 'epilogue': 1, 'segment': 1}
        self.assertDictEqual(self.reader._get_raw_mda(), expected)

        # Make sure _get_raw_mda() doesn't modify the original dictionary
        self.assertIn('loff', self.reader.mda)

    def test_satpos_no_valid_orbit_polynomial(self):
        """Test satellite position if there is no valid orbit polynomial."""
        reader = setup.get_fake_file_handler(
            start_time=self.start_time,
            nlines=self.nlines,
            ncols=self.ncols,
            projection_longitude=self.projection_longitude,
            orbit_polynomials=ORBIT_POLYNOMIALS_INVALID
        )
        self.assertNotIn(
            'satellite_actual_longitude',
            reader.mda['orbital_parameters']
        )


class TestHRITMSGPrologueFileHandler(unittest.TestCase):
    """Test the HRIT prologue file handler."""

    def setUp(self, *mocks):
        """Set up the test case."""
        fh = setup.get_fake_file_handler(
            start_time=datetime(2016, 3, 3, 0, 0),
            nlines=464,
            ncols=3712,
        )
        self.reader = fh.prologue_

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGPrologueFileHandler.read_prologue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def test_extra_kwargs(self, init, *mocks):
        """Test whether the prologue file handler accepts extra keyword arguments."""

        def init_patched(self, *args, **kwargs):
            self.mda = {}

        init.side_effect = init_patched

        HRITMSGPrologueFileHandler(filename='dummy_prologue_filename',
                                   filename_info={'service': ''},
                                   filetype_info=None,
                                   ext_calib_coefs={},
                                   mda_max_array_size=123,
                                   calib_mode='nominal')

    @mock.patch('satpy.readers.seviri_l1b_hrit.utils.reduce_mda')
    def test_reduce(self, reduce_mda):
        """Test metadata reduction."""
        reduce_mda.return_value = 'reduced'

        # Set buffer
        self.assertEqual(self.reader.reduce(123), 'reduced')

        # Read buffer
        self.assertEqual(self.reader.reduce(123), 'reduced')
        reduce_mda.assert_called_once()


class TestHRITMSGEpilogueFileHandler(unittest.TestCase):
    """Test the HRIT epilogue file handler."""

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGEpilogueFileHandler.read_epilogue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def setUp(self, init, *mocks):
        """Set up the test case."""

        def init_patched(self, *args, **kwargs):
            self.mda = {}

        init.side_effect = init_patched

        self.reader = HRITMSGEpilogueFileHandler(filename='dummy_epilogue_filename',
                                                 filename_info={'service': ''},
                                                 filetype_info=None,
                                                 calib_mode='nominal')

    @mock.patch('satpy.readers.seviri_l1b_hrit.HRITMSGEpilogueFileHandler.read_epilogue')
    @mock.patch('satpy.readers.hrit_base.HRITFileHandler.__init__', autospec=True)
    def test_extra_kwargs(self, init, *mocks):
        """Test whether the epilogue file handler accepts extra keyword arguments."""

        def init_patched(self, *args, **kwargs):
            self.mda = {}

        init.side_effect = init_patched

        HRITMSGEpilogueFileHandler(filename='dummy_epilogue_filename',
                                   filename_info={'service': ''},
                                   filetype_info=None,
                                   ext_calib_coefs={},
                                   mda_max_array_size=123,
                                   calib_mode='nominal')

    @mock.patch('satpy.readers.seviri_l1b_hrit.utils.reduce_mda')
    def test_reduce(self, reduce_mda):
        """Test metadata reduction."""
        reduce_mda.return_value = 'reduced'

        # Set buffer
        self.assertEqual(self.reader.reduce(123), 'reduced')
        reduce_mda.assert_called()

        # Read buffer
        reduce_mda.reset_mock()
        self.reader._reduced = 'red'
        self.assertEqual(self.reader.reduce(123), 'red')
        reduce_mda.assert_not_called()


class TestHRITMSGCalibration(TestFileHandlerCalibrationBase):
    """Unit tests for calibration."""

    @pytest.fixture(name='file_handler')
    def file_handler(self):
        """Create a mocked file handler."""
        prolog = {
            'RadiometricProcessing': {
                'Level15ImageCalibration': {
                    'CalSlope': self.gains_nominal,
                    'CalOffset': self.offsets_nominal,
                },
                'MPEFCalFeedback': {
                    'GSICSCalCoeff': self.gains_gsics,
                    'GSICSOffsetCount': self.offsets_gsics,
                }
            },
            'ImageDescription': {
                'Level15ImageProduction': {
                    'PlannedChanProcessing': self.radiance_types
                }
            }
        }
        epilog = {
            'ImageProductionStats': {
                'ActualScanningSummary': {
                    'ForwardScanStart': self.scan_time
                }
            }
        }
        mda = {
            'image_segment_line_quality': {
                'line_validity': np.array([3, 3]),
                'line_radiometric_quality': np.array([4, 4]),
                'line_geometric_quality': np.array([4, 4])
            },
        }

        with mock.patch(
            'satpy.readers.seviri_l1b_hrit.HRITMSGFileHandler.__init__',
            return_value=None
        ):
            fh = HRITMSGFileHandler()
            fh.platform_id = self.platform_id
            fh.mda = mda
            fh.prologue = prolog
            fh.epilogue = epilog
            return fh

    @pytest.mark.parametrize(
        ('channel', 'calibration', 'calib_mode', 'use_ext_coefs'),
        [
            # VIS channel, internal coefficients
            ('VIS006', 'counts', 'NOMINAL', False),
            ('VIS006', 'radiance', 'NOMINAL', False),
            ('VIS006', 'radiance', 'GSICS', False),
            ('VIS006', 'reflectance', 'NOMINAL', False),
            # VIS channel, external coefficients (mode should have no effect)
            ('VIS006', 'radiance', 'GSICS', True),
            ('VIS006', 'reflectance', 'NOMINAL', True),
            # IR channel, internal coefficients
            ('IR_108', 'counts', 'NOMINAL', False),
            ('IR_108', 'radiance', 'NOMINAL', False),
            ('IR_108', 'radiance', 'GSICS', False),
            ('IR_108', 'brightness_temperature', 'NOMINAL', False),
            ('IR_108', 'brightness_temperature', 'GSICS', False),
            # IR channel, external coefficients (mode should have no effect)
            ('IR_108', 'radiance', 'NOMINAL', True),
            ('IR_108', 'brightness_temperature', 'GSICS', True),
            # HRV channel, internal coefficiens
            ('HRV', 'counts', 'NOMINAL', False),
            ('HRV', 'radiance', 'NOMINAL', False),
            ('HRV', 'radiance', 'GSICS', False),
            ('HRV', 'reflectance', 'NOMINAL', False),
            # HRV channel, external coefficients (mode should have no effect)
            ('HRV', 'radiance', 'GSICS', True),
            ('HRV', 'reflectance', 'NOMINAL', True),
        ]
    )
    def test_calibrate(
            self, file_handler, counts, channel, calibration, calib_mode,
            use_ext_coefs
    ):
        """Test the calibration."""
        external_coefs = self.external_coefs if use_ext_coefs else {}
        expected = self._get_expected(
            channel=channel,
            calibration=calibration,
            calib_mode=calib_mode,
            use_ext_coefs=use_ext_coefs
        )

        fh = file_handler
        fh.mda['spectral_channel_id'] = self.spectral_channel_ids[channel]
        fh.channel_name = channel
        fh.calib_mode = calib_mode
        fh.ext_calib_coefs = external_coefs

        res = fh.calibrate(counts, calibration)
        xr.testing.assert_allclose(res, expected)

    def test_mask_bad_quality(self, file_handler):
        """Test the masking of bad quality scan lines."""
        channel = 'VIS006'
        expected = self._get_expected(
            channel=channel,
            calibration='radiance',
            calib_mode='NOMINAL',
            use_ext_coefs=False
        )

        fh = file_handler

        res = fh._mask_bad_quality(expected)
        new_data = np.zeros_like(expected.data).astype('float32')
        new_data[:, :] = np.nan
        expected = expected.copy(data=new_data)
        xr.testing.assert_equal(res, expected)
