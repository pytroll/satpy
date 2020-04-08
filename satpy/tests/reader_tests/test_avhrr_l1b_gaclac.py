#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009-2019 Satpy developers
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
"""Pygac interface."""

from datetime import datetime
from unittest import TestCase, main, TestLoader, TestSuite
import numpy as np
from unittest import mock

GAC_PATTERN = '{creation_site:3s}.{transfer_mode:4s}.{platform_id:2s}.D{start_time:%y%j.S%H%M}.E{end_time:%H%M}.B{orbit_number:05d}{end_orbit_last_digits:02d}.{station:2s}'  # noqa

GAC_POD_FILENAMES = ['NSS.GHRR.NA.D79184.S1150.E1337.B0008384.WI',
                     'NSS.GHRR.NA.D79184.S2350.E0137.B0008384.WI',
                     'NSS.GHRR.NA.D80021.S0927.E1121.B0295354.WI',
                     'NSS.GHRR.NA.D80021.S1120.E1301.B0295455.WI',
                     'NSS.GHRR.NA.D80021.S1256.E1450.B0295556.GC',
                     'NSS.GHRR.NE.D83208.S1219.E1404.B0171819.WI',
                     'NSS.GHRR.NG.D88002.S0614.E0807.B0670506.WI',
                     'NSS.GHRR.TN.D79183.S1258.E1444.B0369697.GC',
                     'NSS.GHRR.TN.D80003.S1147.E1332.B0630506.GC',
                     'NSS.GHRR.TN.D80003.S1328.E1513.B0630507.GC',
                     'NSS.GHRR.TN.D80003.S1509.E1654.B0630608.GC']

GAC_KLM_FILENAMES = ['NSS.GHRR.NK.D01235.S0252.E0446.B1703233.GC',
                     'NSS.GHRR.NL.D01288.S2315.E0104.B0549495.GC',
                     'NSS.GHRR.NM.D04111.S2305.E0050.B0947778.GC',
                     'NSS.GHRR.NN.D13011.S0559.E0741.B3939192.WI',
                     'NSS.GHRR.NP.D15361.S0121.E0315.B3547172.SV',
                     'NSS.GHRR.M1.D15362.S0031.E0129.B1699697.SV',
                     'NSS.GHRR.M2.D10178.S2359.E0142.B1914142.SV']

LAC_POD_FILENAMES = ['BRN.HRPT.ND.D95152.S1730.E1715.B2102323.UB',
                     'BRN.HRPT.ND.D95152.S1910.E1857.B2102424.UB',
                     'BRN.HRPT.NF.D85152.S1345.E1330.B0241414.UB',
                     'BRN.HRPT.NJ.D95152.S1233.E1217.B0216060.UB']

LAC_KLM_FILENAMES = ['BRN.HRPT.M1.D14152.S0958.E1012.B0883232.UB',
                     'BRN.HRPT.M1.D14152.S1943.E1958.B0883838.UB',
                     'BRN.HRPT.M2.D12153.S0912.E0922.B2914747.UB',
                     'BRN.HRPT.NN.D12153.S0138.E0152.B3622828.UB',
                     'BRN.HRPT.NN.D12153.S0139.E0153.B3622828.UB',
                     'BRN.HRPT.NN.D12153.S1309.E1324.B3623535.UB',
                     'BRN.HRPT.NP.D12153.S0003.E0016.B1707272.UB',
                     'BRN.HRPT.NP.D12153.S1134.E1148.B1707979.UB',
                     'BRN.HRPT.NP.D16184.S1256.E1311.B3813131.UB',
                     'BRN.HRPT.NP.D16184.S1438.E1451.B3813232.UB',
                     'BRN.HRPT.NP.D16184.S1439.E1451.B3813232.UB',
                     'BRN.HRPT.NP.D16185.S1245.E1259.B3814545.UB',
                     'BRN.HRPT.NP.D16185.S1427.E1440.B3814646.UB',
                     'NSS.FRAC.M2.D12153.S1729.E1910.B2915354.SV',
                     'NSS.LHRR.NP.D16306.S1803.E1814.B3985555.WI']


class TestGACLACFile(TestCase):
    """Test the GACLAC file handler."""

    def setUp(self):
        """Patch pygac imports."""
        self.pygac = mock.MagicMock()
        self.fhs = mock.MagicMock()
        modules = {
            'pygac': self.pygac,
            'pygac.gac_klm': self.pygac.gac_klm,
            'pygac.gac_pod': self.pygac.gac_pod,
            'pygac.lac_klm': self.pygac.lac_klm,
            'pygac.lac_pod': self.pygac.lac_pod,
            'pygac.utils': self.pygac.utils
        }

        self.module_patcher = mock.patch.dict('sys.modules', modules)
        self.module_patcher.start()

        # Import GACLACFile here to make it patchable. Otherwise self._get_fh
        # might import it first which would prevent a successful patch.
        from satpy.readers.avhrr_l1b_gaclac import GACLACFile
        self.GACLACFile = GACLACFile

    def tearDown(self):
        """Unpatch the pygac imports."""
        self.module_patcher.stop()

    def _get_fh(self, filename='NSS.GHRR.NG.D88002.S0614.E0807.B0670506.WI',
                **kwargs):
        """Create a file handler."""
        from trollsift import parse
        filename_info = parse(GAC_PATTERN, filename)
        return self.GACLACFile(filename, filename_info, {}, **kwargs)

    def test_init(self):
        """Test GACLACFile initialization."""
        from pygac.gac_klm import GACKLMReader
        from pygac.gac_pod import GACPODReader
        from pygac.lac_klm import LACKLMReader
        from pygac.lac_pod import LACPODReader

        for filenames, reader_cls in zip([GAC_POD_FILENAMES, GAC_KLM_FILENAMES, LAC_POD_FILENAMES, LAC_KLM_FILENAMES],
                                         [GACPODReader, GACKLMReader, LACPODReader, LACKLMReader]):
            for filename in filenames:
                fh = self._get_fh(filename)
                self.assertLess(fh.start_time, fh.end_time,
                                "Start time must precede end time.")
                self.assertIs(fh.reader_class, reader_cls,
                              'Wrong reader class assigned to {}'.format(filename))

    def test_get_dataset(self):
        """Test getting the dataset."""
        from pygac.gac_pod import GACPODReader
        from satpy.dataset import DatasetID

        fh = self._get_fh(strip_invalid_coords=False)

        lon_ones = np.ones((10, 10))
        lat_ones = np.ones((10, 10))
        ch_ones = np.ones((10, 10))
        acq_ones = np.ones((10, ), dtype='datetime64[us]')
        angle_ones = np.ones((10, 10))
        qualflags_ones = np.ones((10, 7))
        miss_lines = np.array([1, 2])

        # Channel
        key = DatasetID('1')
        info = {'name': '1', 'standard_name': 'reflectance'}

        GACPODReader.return_value.get_calibrated_channels.return_value.__getitem__.return_value = ch_ones
        GACPODReader.return_value.get_times.return_value = acq_ones
        GACPODReader.return_value.get_lonlat.return_value = lon_ones, lat_ones
        GACPODReader.return_value.get_qual_flags.return_value = qualflags_ones
        GACPODReader.return_value.get_miss_lines.return_value = miss_lines
        GACPODReader.return_value.get_midnight_scanline.return_value = 'midn_line'
        GACPODReader.return_value.mask = [0]

        GACPODReader.return_value.meta_data = {'missing_scanlines': miss_lines,
                                               'midnight_scanline': 'midn_line'}

        res = fh.get_dataset(key, info)
        np.testing.assert_allclose(res.data, ch_ones)
        np.testing.assert_array_equal(res.coords['acq_time'].data, acq_ones)
        self.assertTupleEqual(res.dims, ('y', 'x'))
        self.assertEqual(fh.start_time, datetime(1970, 1, 1, 0, 0, 0, 1))
        self.assertEqual(fh.end_time, datetime(1970, 1, 1, 0, 0, 0, 1))
        np.testing.assert_array_equal(res.attrs['missing_scanlines'], miss_lines)
        self.assertEqual(res.attrs['midnight_scanline'], 'midn_line')

        # Angles
        for item in ['solar_zenith_angle', 'sensor_zenith_angle',
                     'solar_azimuth_angle', 'sensor_azimuth_angle',
                     'sun_sensor_azimuth_difference_angle']:
            key = DatasetID(item)
            info = {'name': item}

            GACPODReader.return_value.get_angles.return_value = (angle_ones, ) * 5
            GACPODReader.return_value.get_tle_lines.return_value = 'tle1', 'tle2'
            res = fh.get_dataset(key, info)
            np.testing.assert_allclose(res.data, angle_ones)
            np.testing.assert_array_equal(res.coords['acq_time'].data, acq_ones)
            self.assertDictEqual(res.attrs['orbital_parameters'], {'tle': ('tle1', 'tle2')})

        # Longitude
        key = DatasetID('longitude')
        info = {'name': 'longitude', 'unit': 'degrees_east'}

        res = fh.get_dataset(key, info)
        np.testing.assert_allclose(res.data, lon_ones)
        self.assertEqual(res.attrs['unit'], 'degrees_east')
        np.testing.assert_array_equal(res.coords['acq_time'].data, acq_ones)

        # Latitude
        key = DatasetID('latitude')
        info = {'name': 'latitude', 'unit': 'degrees_north'}

        res = fh.get_dataset(key, info)
        np.testing.assert_allclose(res.data, lat_ones)
        self.assertEqual(res.attrs['unit'], 'degrees_north')
        np.testing.assert_array_equal(res.coords['acq_time'].data, acq_ones)

        # Quality flags
        key = DatasetID('qual_flags')
        info = {'name': 'qual_flags', 'long_name': 'My long name'}

        res = fh.get_dataset(key, info)
        np.testing.assert_allclose(res.data, qualflags_ones)
        self.assertTupleEqual(res.dims, ('y', 'num_flags'))
        np.testing.assert_array_equal(res.coords['acq_time'].data, acq_ones)
        self.assertEqual(res.attrs['long_name'], 'My long name')

        # Buffering
        GACPODReader.return_value.get_calibrated_channels.assert_called_once()
        GACPODReader.return_value.get_calibrated_channels.assert_called_once()
        GACPODReader.return_value.get_qual_flags.assert_called_once()

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile.slice')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_angle')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_channel')
    def test_get_dataset_extras(self, get_channel, get_angle, slc):
        """Test getting the dataset with extra options."""
        from satpy.dataset import DatasetID

        # Define test data
        lons = np.array([[1, 2],
                         [3, 4],
                         [5, 6],
                         [7, 8]])
        lats = lons.copy()
        angles = lons.copy()
        ch = np.array([[0.1, 0.2],
                       [0.3, 0.4],
                       [0.5, 0.6],
                       [0.7, 0.8]])
        acq = np.array([1, 2, 3, 4], dtype='datetime64[us]')

        # Mock reading
        reader = mock.MagicMock()
        reader.mask = [0]
        reader.get_lonlat.return_value = lons, lats
        reader.get_times.return_value = acq
        get_channel.return_value = ch
        get_angle.return_value = angles

        # Test slicing/stripping
        def slice_patched(data, times):
            if len(data.shape) == 2:
                return data[1:3, :], times[1:3]
            return data[1:3], times[1:3]

        slc.side_effect = slice_patched
        kwargs_list = [{'strip_invalid_coords': False,
                        'start_line': 123, 'end_line': 456},
                       {'strip_invalid_coords': True,
                        'start_line': None, 'end_line': None},
                       {'strip_invalid_coords': True,
                        'start_line': 123, 'end_line': 456}]
        for kwargs in kwargs_list:
            fh = self._get_fh(**kwargs)
            fh.reader = reader

            key = DatasetID('1')
            info = {'name': '1', 'standard_name': 'reflectance'}
            res = fh.get_dataset(key, info)
            np.testing.assert_array_equal(res.data, ch[1:3, :])
            np.testing.assert_array_equal(res.coords['acq_time'].data, acq[1:3])
            slc.assert_called_with(data=ch, times=acq)

        # Renaming of coordinates if interpolation is switched off
        fh = self._get_fh(interpolate_coords=False)
        fh.reader = reader

        key = DatasetID('latitude')
        info = {'name': 'latitude', 'unit': 'degrees_north'}
        res = fh.get_dataset(key, info)
        self.assertTupleEqual(res.dims, ('y', 'x_every_eighth'))

        key = DatasetID('solar_zenith_angle')
        info = {'name': 'solar_zenith_angle', 'unit': 'degrees'}
        res = fh.get_dataset(key, info)
        self.assertTupleEqual(res.dims, ('y', 'x_every_eighth'))

    def test_get_angle(self):
        """Test getting the angle."""
        reader = mock.MagicMock()
        reader.get_angles.return_value = 1, 2, 3, 4, 5
        fh = self._get_fh()
        fh.reader = reader

        # Test angle readout
        res = fh._get_angle('sensor_zenith_angle')
        self.assertEqual(res, 2)
        self.assertDictEqual(fh.angles, {'sensor_zenith_angle': 2,
                                         'sensor_azimuth_angle': 1,
                                         'solar_zenith_angle': 4,
                                         'solar_azimuth_angle': 3,
                                         'sun_sensor_azimuth_difference_angle': 5})

        # Test buffering
        fh._get_angle('sensor_azimuth_angle')
        reader.get_angles.assert_called_once()

    def test_strip_invalid_lat(self):
        """Test stripping invalid coordinates."""
        import pygac.utils

        reader = mock.MagicMock()
        reader.get_lonlat.return_value = None, None
        fh = self._get_fh()
        fh.reader = reader

        # Test stripping
        pygac.utils.strip_invalid_lat.return_value = 1, 2
        start, end = fh._strip_invalid_lat()
        self.assertTupleEqual((start, end), (1, 2))

        # Test buffering
        fh._strip_invalid_lat()
        pygac.utils.strip_invalid_lat.assert_called_once()

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._slice')
    def test_slice(self, _slice):
        """Test slicing."""
        def slice_patched(data):
            if len(data.shape) == 2:
                return data[1:3, :], 'midn_line', np.array([1., 2., 3.])
            return data[1:3], 'foo', np.array([0, 0, 0])

        _slice.side_effect = slice_patched
        data = np.zeros((4, 2))
        times = np.array([1, 2, 3, 4], dtype='datetime64[us]')

        fh = self._get_fh()
        data_slc, times_slc = fh.slice(data, times)
        np.testing.assert_array_equal(data_slc, data[1:3])
        np.testing.assert_array_equal(times_slc, times[1:3])
        self.assertEqual(fh.start_time, datetime(1970, 1, 1, 0, 0, 0, 2))
        self.assertEqual(fh.end_time, datetime(1970, 1, 1, 0, 0, 0, 3))
        self.assertEqual(fh.midnight_scanline, 'midn_line')
        np.testing.assert_array_equal(fh.missing_scanlines, np.array([1, 2, 3]))
        self.assertEqual(fh.missing_scanlines.dtype, int)

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_qual_flags')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._strip_invalid_lat')
    def test__slice(self, strip_invalid_lat, get_qual_flags):
        """Test slicing."""
        import pygac.utils
        pygac.utils.check_user_scanlines.return_value = 1, 2
        pygac.utils.slice_channel.return_value = 'sliced', 'miss_lines', 'midn_line'
        strip_invalid_lat.return_value = 3, 4
        get_qual_flags.return_value = 'qual_flags'

        data = np.zeros((2, 2))

        # a) Only start/end line given
        fh = self._get_fh(start_line=5, end_line=6, strip_invalid_coords=False)
        data_slc, midn_line, miss_lines = fh._slice(data)
        self.assertEqual(data_slc, 'sliced')
        self.assertEqual(midn_line, 'midn_line')
        self.assertEqual(miss_lines, 'miss_lines')
        pygac.utils.check_user_scanlines.assert_called_with(
            start_line=5, end_line=6,
            first_valid_lat=None, last_valid_lat=None, along_track=2)
        pygac.utils.slice_channel.assert_called_with(
            data, start_line=1, end_line=2,
            first_valid_lat=None, last_valid_lat=None,
            midnight_scanline=None, miss_lines=None, qual_flags='qual_flags')

        # b) Only strip_invalid_coords=True
        fh = self._get_fh(strip_invalid_coords=True)
        fh._slice(data)
        pygac.utils.check_user_scanlines.assert_called_with(
            start_line=0, end_line=0,
            first_valid_lat=3, last_valid_lat=4, along_track=2)

        # c) Both
        fh = self._get_fh(start_line=5, end_line=6, strip_invalid_coords=True)
        fh._slice(data)
        pygac.utils.check_user_scanlines.assert_called_with(
            start_line=5, end_line=6,
            first_valid_lat=3, last_valid_lat=4, along_track=2)
