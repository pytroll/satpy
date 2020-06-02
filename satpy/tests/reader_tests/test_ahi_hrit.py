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
"""The hrit ahi reader tests package."""

import numpy as np
import dask.array as da
from xarray import DataArray
import unittest
from unittest import mock


class TestHRITJMAFileHandler(unittest.TestCase):
    """Test the HRITJMAFileHandler."""

    @mock.patch('satpy.readers.hrit_jma.HRITFileHandler.__init__')
    def _get_reader(self, mocked_init, mda, filename_info=None):
        from satpy.readers.hrit_jma import HRITJMAFileHandler
        if not filename_info:
            filename_info = {}
        HRITJMAFileHandler.filename = 'filename'
        HRITJMAFileHandler.mda = mda
        return HRITJMAFileHandler('filename', filename_info, {})

    def _get_acq_time(self, nlines):
        """Get sample header entry for scanline acquisition times.

        Lines: 1, 21, 41, 61, ..., nlines
        Times: 1970-01-01 00:00 + (1, 21, 41, 61, ..., nlines) seconds

        So the interpolated times are expected to be 1970-01-01 +
        (1, 2, 3, 4, ..., nlines) seconds. Note that there will be some
        floating point inaccuracies, because timestamps are stored
        with only 6 decimals precision.
        """
        mjd_1970 = 40587.0
        lines_sparse = np.array(list(range(1, nlines, 20)) + [nlines])
        times_sparse = mjd_1970 + lines_sparse / 24 / 3600
        acq_time_s = ['LINE:={}\rTIME:={:.6f}\r'.format(l, t)
                      for l, t in zip(lines_sparse, times_sparse)]
        acq_time_b = ''.join(acq_time_s).encode()
        return acq_time_b

    def _get_mda(self, loff=5500.0, coff=5500.0, nlines=11000, ncols=11000,
                 segno=0, numseg=1, vis=True, platform='Himawari-8'):
        """Create metadata dict like HRITFileHandler would do it."""
        if vis:
            idf = b'$HALFTONE:=16\r_NAME:=VISIBLE\r_UNIT:=ALBEDO(%)\r' \
                  b'0:=-0.10\r1023:=100.00\r65535:=100.00\r'
        else:
            idf = b'$HALFTONE:=16\r_NAME:=INFRARED\r_UNIT:=KELVIN\r' \
                  b'0:=329.98\r1023:=130.02\r65535:=130.02\r'
        proj_h8 = b'GEOS(140.70)                    '
        proj_mtsat2 = b'GEOS(145.00)                    '
        proj_name = proj_h8 if platform == 'Himawari-8' else proj_mtsat2
        return {'image_segm_seq_no': segno,
                'total_no_image_segm': numseg,
                'projection_name': proj_name,
                'projection_parameters': {
                    'a': 6378169.00,
                    'b': 6356583.80,
                    'h': 35785831.00,
                },
                'cfac': 10233128,
                'lfac': 10233128,
                'coff': coff,
                'loff': loff,
                'number_of_columns': ncols,
                'number_of_lines': nlines,
                'image_data_function': idf,
                'image_observation_time': self._get_acq_time(nlines)}

    def test_init(self):
        """Test creating the file handler."""
        from satpy.readers.hrit_jma import UNKNOWN_AREA, HIMAWARI8

        # Test addition of extra metadata
        mda = self._get_mda()
        mda_expected = mda.copy()
        mda_expected.update(
            {'planned_end_segment_number': 1,
             'planned_start_segment_number': 1,
             'segment_sequence_number': 0,
             'unit': 'ALBEDO(%)'})
        mda_expected['projection_parameters']['SSP_longitude'] = 140.7
        reader = self._get_reader(mda=mda)
        self.assertEqual(reader.mda, mda_expected)

        # Check projection name
        self.assertEqual(reader.projection_name, 'GEOS(140.70)')

        # Check calibration table
        cal_expected = np.array([[0, -0.1],
                                 [1023,  100],
                                 [65535,  100]])
        self.assertTrue(np.all(reader.calibration_table == cal_expected))

        # Check if scanline timestamps are there (dedicated test below)
        self.assertIsInstance(reader.acq_time, np.ndarray)

        # Check platform
        self.assertEqual(reader.platform, HIMAWARI8)

        # Check is_segmented attribute
        expected = {0: False, 1: True, 8: True}
        for segno, is_segmented in expected.items():
            mda = self._get_mda(segno=segno)
            reader = self._get_reader(mda=mda)
            self.assertEqual(reader.is_segmented, is_segmented)

        # Check area IDs
        expected = [
            ({'area': 1}, 1),
            ({'area': 1234}, UNKNOWN_AREA),
            ({}, UNKNOWN_AREA)
        ]
        mda = self._get_mda()
        for filename_info, area_id in expected:
            reader = self._get_reader(mda=mda, filename_info=filename_info)
            self.assertEqual(reader.area_id, area_id)

    @mock.patch('satpy.readers.hrit_jma.HRITJMAFileHandler.__init__')
    def test_get_platform(self, mocked_init):
        """Test platform identification."""
        from satpy.readers.hrit_jma import HRITJMAFileHandler
        from satpy.readers.hrit_jma import PLATFORMS, UNKNOWN_PLATFORM

        mocked_init.return_value = None
        reader = HRITJMAFileHandler()

        for proj_name, platform in PLATFORMS.items():
            reader.projection_name = proj_name
            self.assertEqual(reader._get_platform(), platform)

        with mock.patch('logging.Logger.error') as mocked_log:
            reader.projection_name = 'invalid'
            self.assertEqual(reader._get_platform(), UNKNOWN_PLATFORM)
            mocked_log.assert_called()

    def test_get_area_def(self):
        """Test getting an AreaDefinition."""
        from satpy.readers.hrit_jma import (FULL_DISK, NORTH_HEMIS, SOUTH_HEMIS,
                                            AREA_NAMES)

        cases = [
            # Non-segmented, full disk
            {'loff': 1375.0, 'coff': 1375.0,
             'nlines': 2750, 'ncols': 2750,
             'segno': 0, 'numseg': 1,
             'area': FULL_DISK,
             'extent': (-5498000.088960204, -5498000.088960204,
                        5502000.089024927, 5502000.089024927)},
            # Non-segmented, northern hemisphere
            {'loff': 1325.0, 'coff': 1375.0,
             'nlines': 1375, 'ncols': 2750,
             'segno': 0, 'numseg': 1,
             'area': NORTH_HEMIS,
             'extent': (-5498000.088960204, -198000.00320373234,
                        5502000.089024927, 5302000.085788833)},
            # Non-segmented, southern hemisphere
            {'loff': 50, 'coff': 1375.0,
             'nlines': 1375, 'ncols': 2750,
             'segno': 0, 'numseg': 1,
             'area': SOUTH_HEMIS,
             'extent': (-5498000.088960204, -5298000.085724112,
                        5502000.089024927, 202000.0032684542)},
            # Segmented, segment #1
            {'loff': 1375.0, 'coff': 1375.0,
             'nlines': 275, 'ncols': 2750,
             'segno': 1, 'numseg': 10,
             'area': FULL_DISK,
             'extent': (-5498000.088960204, 4402000.071226413,
                        5502000.089024927, 5502000.089024927)},
            # Segmented, segment #7
            {'loff': 1375.0, 'coff': 1375.0,
             'nlines': 275, 'ncols': 2750,
             'segno': 7, 'numseg': 10,
             'area': FULL_DISK,
             'extent': (-5498000.088960204, -2198000.035564665,
                        5502000.089024927, -1098000.0177661523)},
        ]
        for case in cases:
            mda = self._get_mda(loff=case['loff'], coff=case['coff'],
                                nlines=case['nlines'], ncols=case['ncols'],
                                segno=case['segno'], numseg=case['numseg'])
            reader = self._get_reader(mda=mda,
                                      filename_info={'area': case['area']})
            area = reader.get_area_def('some_id')
            self.assertTupleEqual(area.area_extent, case['extent'])
            self.assertEqual(area.description, AREA_NAMES[case['area']]['long'])

    def test_calibrate(self):
        """Test calibration."""
        # Generate test data
        counts = np.linspace(0, 1200, 25).reshape(5, 5)
        counts[-1, -1] = 65535
        counts = DataArray(da.from_array(counts, chunks=5))
        refl = np.array(
            [[-0.1,            4.79247312,   9.68494624,  14.57741935,  19.46989247],
             [24.36236559,  29.25483871,  34.14731183,  39.03978495,  43.93225806],
             [48.82473118,  53.7172043,   58.60967742,  63.50215054,  68.39462366],
             [73.28709677,  78.17956989,  83.07204301,  87.96451613,  92.85698925],
             [97.74946237,  100.,         100.,         100.,         np.nan]]
        )
        bt = np.array(
            [[329.98,            320.20678397, 310.43356794, 300.66035191, 290.88713587],
             [281.11391984, 271.34070381, 261.56748778, 251.79427175, 242.02105572],
             [232.24783969, 222.47462366, 212.70140762, 202.92819159, 193.15497556],
             [183.38175953, 173.6085435,  163.83532747, 154.06211144, 144.28889541],
             [134.51567937, 130.02,       130.02,       130.02,       np.nan]]
        )

        # Choose an area near the subsatellite point to avoid masking
        # of space pixels
        mda = self._get_mda(nlines=5, ncols=5, loff=1375.0, coff=1375.0,
                            segno=0)
        reader = self._get_reader(mda=mda)

        # 1. Counts
        res = reader.calibrate(data=counts, calibration='counts')
        self.assertTrue(np.all(counts.values == res.values))

        # 2. Reflectance
        res = reader.calibrate(data=counts, calibration='reflectance')
        np.testing.assert_allclose(refl, res.values)  # also compares NaN

        # 3. Brightness temperature
        mda_bt = self._get_mda(nlines=5, ncols=5, loff=1375.0, coff=1375.0,
                               segno=0, vis=False)
        reader_bt = self._get_reader(mda=mda_bt)
        res = reader_bt.calibrate(data=counts,
                                  calibration='brightness_temperature')
        np.testing.assert_allclose(bt, res.values)  # also compares NaN

    def test_mask_space(self):
        """Test masking of space pixels."""
        mda = self._get_mda(loff=1375.0, coff=1375.0, nlines=275, ncols=1375,
                            segno=1, numseg=10)
        reader = self._get_reader(mda=mda)
        data = DataArray(da.ones((275, 1375), chunks=1024))
        masked = reader._mask_space(data)

        # First line of the segment should be space, in the middle of the
        # last line there should be some valid pixels
        np.testing.assert_allclose(masked.values[0, :], np.nan)
        self.assertTrue(np.all(masked.values[-1, 588:788] == 1))

    @mock.patch('satpy.readers.hrit_jma.HRITFileHandler.get_dataset')
    def test_get_dataset(self, base_get_dataset):
        """Test getting a dataset."""
        from satpy.readers.hrit_jma import HIMAWARI8

        mda = self._get_mda(loff=1375.0, coff=1375.0, nlines=275, ncols=1375,
                            segno=1, numseg=10)
        reader = self._get_reader(mda=mda)

        key = mock.MagicMock()
        key.calibration = 'reflectance'

        base_get_dataset.return_value = DataArray(da.ones((275, 1375),
                                                          chunks=1024),
                                                  dims=('y', 'x'))

        # Check attributes
        res = reader.get_dataset(key, {'units': '%', 'sensor': 'ahi'})
        self.assertEqual(res.attrs['units'], '%')
        self.assertEqual(res.attrs['sensor'], 'ahi')
        self.assertEqual(res.attrs['platform_name'], HIMAWARI8)
        self.assertEqual(res.attrs['satellite_longitude'], 140.7)
        self.assertEqual(res.attrs['satellite_latitude'], 0.)
        self.assertEqual(res.attrs['satellite_altitude'], 35785831.0)
        self.assertDictEqual(res.attrs['orbital_parameters'], {'projection_longitude': 140.7,
                                                               'projection_latitude': 0.,
                                                               'projection_altitude': 35785831.0})

        # Check if acquisition time is a coordinate
        self.assertIn('acq_time', res.coords)

        # Check called methods
        with mock.patch.object(reader, '_mask_space') as mask_space:
            with mock.patch.object(reader, 'calibrate') as calibrate:
                reader.get_dataset(key, {'units': '%', 'sensor': 'ahi'})
                mask_space.assert_called()
                calibrate.assert_called()

        with mock.patch('logging.Logger.error') as log_mock:
            reader.get_dataset(key, {'units': '%', 'sensor': 'jami'})
            log_mock.assert_called()

    def test_mjd2datetime64(self):
        """Test conversion from modified julian day to datetime64"""
        from satpy.readers.hrit_jma import mjd2datetime64
        self.assertEqual(mjd2datetime64(np.array([0])),
                         np.datetime64('1858-11-17', 'us'))
        self.assertEqual(mjd2datetime64(np.array([40587.5])),
                         np.datetime64('1970-01-01 12:00', 'us'))

    def test_get_acq_time(self):
        """Test computation of scanline acquisition times."""
        dt_line = np.arange(1, 11000+1).astype('timedelta64[s]')
        acq_time_exp = np.datetime64('1970-01-01', 'us') + dt_line

        for platform in ['Himawari-8', 'MTSAT-2']:
            # Results are not exactly identical because timestamps are stored in
            # the header with only 6 decimals precision (max diff here: 45 msec).
            mda = self._get_mda(platform=platform)
            reader = self._get_reader(mda=mda)
            np.testing.assert_allclose(reader.acq_time.astype(np.int64),
                                       acq_time_exp.astype(np.int64),
                                       atol=45000)
