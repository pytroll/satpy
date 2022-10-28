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
from unittest import TestCase, mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

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


@mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile.__init__', return_value=None)
def _get_fh_mocked(init_mock, **attrs):
    """Create a mocked file handler with the given attributes."""
    from satpy.readers.avhrr_l1b_gaclac import GACLACFile

    fh = GACLACFile()
    for name, value in attrs.items():
        setattr(fh, name, value)
    return fh


def _get_reader_mocked(along_track=3):
    """Create a mocked reader."""
    reader = mock.MagicMock(spacecraft_name='spacecraft_name',
                            meta_data={'foo': 'bar'})
    reader.mask = [0, 0]
    reader.get_times.return_value = np.arange(along_track)
    reader.get_tle_lines.return_value = 'tle'
    return reader


class PygacPatcher(TestCase):
    """Patch pygac."""

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
            'pygac.utils': self.pygac.utils,
            'pygac.calibration': self.pygac.calibration,
        }

        self.module_patcher = mock.patch.dict('sys.modules', modules)
        self.module_patcher.start()

    def tearDown(self):
        """Unpatch the pygac imports."""
        self.module_patcher.stop()


class GACLACFilePatcher(PygacPatcher):
    """Patch pygac."""

    def setUp(self):
        """Patch GACLACFile."""
        super().setUp()

        # Import GACLACFile here to make it patchable. Otherwise self._get_fh
        # might import it first which would prevent a successful patch.
        from satpy.readers.avhrr_l1b_gaclac import GACLACFile
        self.GACLACFile = GACLACFile


class TestGACLACFile(GACLACFilePatcher):
    """Test the GACLAC file handler."""

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

        kwargs = {'start_line': 1,
                  'end_line': 2,
                  'strip_invalid_coords': True,
                  'interpolate_coords': True,
                  'adjust_clock_drift': True,
                  'tle_dir': 'tle_dir',
                  'tle_name': 'tle_name',
                  'tle_thresh': 123,
                  'calibration': 'calibration'}
        for filenames, reader_cls in zip([GAC_POD_FILENAMES, GAC_KLM_FILENAMES, LAC_POD_FILENAMES, LAC_KLM_FILENAMES],
                                         [GACPODReader, GACKLMReader, LACPODReader, LACKLMReader]):
            for filename in filenames:
                fh = self._get_fh(filename, **kwargs)
                self.assertLess(fh.start_time, fh.end_time,
                                "Start time must precede end time.")
                self.assertIs(fh.reader_class, reader_cls,
                              'Wrong reader class assigned to {}'.format(filename))

    def test_read_raw_data(self):
        """Test raw data reading."""
        fh = _get_fh_mocked(reader=None,
                            interpolate_coords='interpolate_coords',
                            creation_site='creation_site',
                            reader_kwargs={'foo': 'bar'},
                            filename='myfile')
        reader = mock.MagicMock(mask=[0])
        reader_cls = mock.MagicMock(return_value=reader)
        fh.reader_class = reader_cls
        fh.read_raw_data()
        reader_cls.assert_called_with(interpolate_coords='interpolate_coords',
                                      creation_site='creation_site',
                                      foo='bar')
        reader.read.assert_called_with('myfile')

        # Test exception if all data is masked
        reader.mask = [1]
        fh.reader = None
        with self.assertRaises(ValueError):
            fh.read_raw_data()

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._update_attrs')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile.slice')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_channel')
    def test_get_dataset_slice(self, get_channel, slc, *mocks):
        """Get a slice of a dataset."""
        from satpy.tests.utils import make_dataid

        # Test slicing/stripping
        def slice_patched(data, times):
            if len(data.shape) == 2:
                return data[1:3, :], times[1:3]
            return data[1:3], times[1:3]

        ch = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12],
                       [13, 14, 15]])
        acq = np.array([0, 1, 2, 3, 4])
        slc.side_effect = slice_patched
        get_channel.return_value = ch
        kwargs_list = [{'strip_invalid_coords': False,
                        'start_line': 123, 'end_line': 456},
                       {'strip_invalid_coords': True,
                        'start_line': None, 'end_line': None},
                       {'strip_invalid_coords': True,
                        'start_line': 123, 'end_line': 456}]
        for kwargs in kwargs_list:
            fh = _get_fh_mocked(
                reader=_get_reader_mocked(along_track=len(acq)),
                chn_dict={'1': 0},
                **kwargs
            )

            key = make_dataid(name='1', calibration='reflectance')
            info = {'name': '1', 'standard_name': 'reflectance'}
            res = fh.get_dataset(key, info)
            np.testing.assert_array_equal(res.data, ch[1:3, :])
            np.testing.assert_array_equal(res.coords['acq_time'].data, acq[1:3])
            np.testing.assert_array_equal(slc.call_args_list[-1][1]['times'], acq)
            np.testing.assert_array_equal(slc.call_args_list[-1][1]['data'], ch)

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._update_attrs')
    def test_get_dataset_latlon(self, *mocks):
        """Test getting the latitudes and longitudes."""
        from satpy.tests.utils import make_dataid

        lons = np.ones((3, 3))
        lats = 2 * lons
        reader = _get_reader_mocked()
        reader.get_lonlat.return_value = lons, lats
        fh = _get_fh_mocked(
            reader=reader,
            start_line=None,
            end_line=None,
            strip_invalid_coords=False,
            interpolate_coords=True
        )

        # With interpolation of coordinates
        for name, exp_data in zip(['longitude', 'latitude'], [lons, lats]):
            key = make_dataid(name=name)
            info = {'name': name, 'standard_name': 'my_standard_name'}
            res = fh.get_dataset(key=key, info=info)
            exp = xr.DataArray(exp_data,
                               name=res.name,
                               dims=('y', 'x'),
                               coords={'acq_time': ('y', [0, 1, 2])})
            xr.testing.assert_equal(res, exp)

        # Without interpolation of coordinates
        fh.interpolate_coords = False
        for name, _exp_data in zip(['longitude', 'latitude'], [lons, lats]):
            key = make_dataid(name=name)
            info = {'name': name, 'standard_name': 'my_standard_name'}
            res = fh.get_dataset(key=key, info=info)
            self.assertTupleEqual(res.dims, ('y', 'x_every_eighth'))

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._update_attrs')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_angle')
    def test_get_dataset_angles(self, get_angle, *mocks):
        """Test getting the angles."""
        from satpy.readers.avhrr_l1b_gaclac import ANGLES
        from satpy.tests.utils import make_dataid

        ones = np.ones((3, 3))
        get_angle.return_value = ones
        reader = _get_reader_mocked()
        fh = _get_fh_mocked(
            reader=reader,
            start_line=None,
            end_line=None,
            strip_invalid_coords=False,
            interpolate_coords=True
        )

        # With interpolation of coordinates
        for angle in ANGLES:
            key = make_dataid(name=angle)
            info = {'name': angle, 'standard_name': 'my_standard_name'}
            res = fh.get_dataset(key=key, info=info)
            exp = xr.DataArray(ones,
                               name=res.name,
                               dims=('y', 'x'),
                               coords={'acq_time': ('y', [0, 1, 2])})
            xr.testing.assert_equal(res, exp)

        # Without interpolation of coordinates
        fh.interpolate_coords = False
        for angle in ANGLES:
            key = make_dataid(name=angle)
            info = {'name': angle, 'standard_name': 'my_standard_name'}
            res = fh.get_dataset(key=key, info=info)
            self.assertTupleEqual(res.dims, ('y', 'x_every_eighth'))

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._update_attrs')
    def test_get_dataset_qual_flags(self, *mocks):
        """Test getting the qualitiy flags."""
        from satpy.tests.utils import make_dataid

        qual_flags = np.ones((3, 7))
        reader = _get_reader_mocked()
        reader.get_qual_flags.return_value = qual_flags
        fh = _get_fh_mocked(
            reader=reader,
            start_line=None,
            end_line=None,
            strip_invalid_coords=False,
            interpolate_coords=True
        )

        key = make_dataid(name='qual_flags')
        info = {'name': 'qual_flags'}
        res = fh.get_dataset(key=key, info=info)
        exp = xr.DataArray(qual_flags,
                           name=res.name,
                           dims=('y', 'num_flags'),
                           coords={'acq_time': ('y', [0, 1, 2]),
                                   'num_flags': ['Scan line number',
                                                 'Fatal error flag',
                                                 'Insufficient data for calibration',
                                                 'Insufficient data for calibration',
                                                 'Solar contamination of blackbody in channels 3',
                                                 'Solar contamination of blackbody in channels 4',
                                                 'Solar contamination of blackbody in channels 5']})
        xr.testing.assert_equal(res, exp)

    def test_get_channel(self):
        """Test getting the channels."""
        from satpy.tests.utils import make_dataid

        counts = np.moveaxis(np.array([[[1, 2, 3],
                                        [4, 5, 6]]]), 0, 2)
        calib_channels = 2 * counts
        reader = _get_reader_mocked()
        reader.get_counts.return_value = counts
        reader.get_calibrated_channels.return_value = calib_channels
        fh = _get_fh_mocked(reader=reader, counts=None, calib_channels=None,
                            chn_dict={'1': 0})

        key = make_dataid(name='1', calibration='counts')
        # Counts
        res = fh._get_channel(key=key)
        np.testing.assert_array_equal(res, [[1, 2, 3],
                                            [4, 5, 6]])
        np.testing.assert_array_equal(fh.counts, counts)

        # Reflectance and Brightness Temperature
        for calib in ['reflectance', 'brightness_temperature']:
            key = make_dataid(name='1', calibration=calib)
            res = fh._get_channel(key=key)
            np.testing.assert_array_equal(res, [[2, 4, 6],
                                                [8, 10, 12]])
            np.testing.assert_array_equal(fh.calib_channels, calib_channels)

        # Invalid
        with pytest.raises(ValueError):
            key = make_dataid(name='7', calibration='coffee')

        # Buffering
        reader.get_counts.reset_mock()
        key = make_dataid(name='1', calibration='counts')
        fh._get_channel(key=key)
        reader.get_counts.assert_not_called()

        reader.get_calibrated_channels.reset_mock()
        for calib in ['reflectance', 'brightness_temperature']:
            key = make_dataid(name='1', calibration=calib)
            fh._get_channel(key)
            reader.get_calibrated_channels.assert_not_called()

    def test_get_angle(self):
        """Test getting the angle."""
        from satpy.tests.utils import make_dataid

        reader = mock.MagicMock()
        reader.get_angles.return_value = 1, 2, 3, 4, 5
        fh = _get_fh_mocked(reader=reader, angles=None)

        # Test angle readout
        key = make_dataid(name='sensor_zenith_angle')
        res = fh._get_angle(key)
        self.assertEqual(res, 2)
        self.assertDictEqual(fh.angles, {'sensor_zenith_angle': 2,
                                         'sensor_azimuth_angle': 1,
                                         'solar_zenith_angle': 4,
                                         'solar_azimuth_angle': 3,
                                         'sun_sensor_azimuth_difference_angle': 5})

        # Test buffering
        key = make_dataid(name='sensor_azimuth_angle')
        fh._get_angle(key)
        reader.get_angles.assert_called_once()

    def test_strip_invalid_lat(self):
        """Test stripping invalid coordinates."""
        import pygac.utils

        reader = mock.MagicMock()
        reader.get_lonlat.return_value = None, None
        fh = _get_fh_mocked(reader=reader, first_valid_lat=None)

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

        def _slice_patched(data):
            return data[1:3]
        _slice.side_effect = _slice_patched

        data = np.zeros((4, 2))
        times = np.array([1, 2, 3, 4], dtype='datetime64[us]')

        fh = _get_fh_mocked(start_line=1, end_line=3, strip_invalid_coords=False)
        data_slc, times_slc = fh.slice(data, times)
        np.testing.assert_array_equal(data_slc, data[1:3])
        np.testing.assert_array_equal(times_slc, times[1:3])
        self.assertEqual(fh.start_time, datetime(1970, 1, 1, 0, 0, 0, 2))
        self.assertEqual(fh.end_time, datetime(1970, 1, 1, 0, 0, 0, 3))

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_qual_flags')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._strip_invalid_lat')
    def test__slice(self, strip_invalid_lat, get_qual_flags):
        """Test slicing."""
        import pygac.utils
        pygac.utils.check_user_scanlines.return_value = 1, 2
        pygac.utils.slice_channel.return_value = 'sliced'
        strip_invalid_lat.return_value = 3, 4
        get_qual_flags.return_value = 'qual_flags'

        data = np.zeros((2, 2))

        # a) Only start/end line given
        fh = _get_fh_mocked(start_line=5, end_line=6, strip_invalid_coords=False)
        data_slc = fh._slice(data)
        self.assertEqual(data_slc, 'sliced')
        pygac.utils.check_user_scanlines.assert_called_with(
            start_line=5, end_line=6,
            first_valid_lat=None, last_valid_lat=None, along_track=2)
        pygac.utils.slice_channel.assert_called_with(
            data, start_line=1, end_line=2,
            first_valid_lat=None, last_valid_lat=None)

        # b) Only strip_invalid_coords=True
        fh = _get_fh_mocked(start_line=None, end_line=None, strip_invalid_coords=True)
        fh._slice(data)
        pygac.utils.check_user_scanlines.assert_called_with(
            start_line=0, end_line=0,
            first_valid_lat=3, last_valid_lat=4, along_track=2)

        # c) Both
        fh = _get_fh_mocked(start_line=5, end_line=6, strip_invalid_coords=True)
        fh._slice(data)
        pygac.utils.check_user_scanlines.assert_called_with(
            start_line=5, end_line=6,
            first_valid_lat=3, last_valid_lat=4, along_track=2)

        # Test slicing with older pygac versions
        pygac.utils.slice_channel.return_value = ('sliced', 'foo', 'bar')
        data_slc = fh._slice(data)
        self.assertEqual(data_slc, 'sliced')


class TestGetDataset(GACLACFilePatcher):
    """Test the get_dataset method."""

    def setUp(self):
        """Set up the instance."""
        self.exp = xr.DataArray(da.ones((3, 3)),
                                name='1',
                                dims=('y', 'x'),
                                coords={'acq_time': ('y', [0, 1, 2])},
                                attrs={'name': '1',
                                       'platform_name': 'spacecraft_name',
                                       'orbit_number': 123,
                                       'sensor': 'sensor',
                                       'foo': 'bar',
                                       'standard_name': 'my_standard_name'})
        self.exp.coords['acq_time'].attrs['long_name'] = 'Mean scanline acquisition time'
        super().setUp()

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile.__init__', return_value=None)
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile.read_raw_data')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_channel', return_value=np.ones((3, 3)))
    def test_get_dataset_channels(self, get_channel, *mocks):
        """Test getting the channel datasets."""
        pygac_reader = _get_reader_mocked()
        fh = self._create_file_handler(pygac_reader)

        # Test calibration to reflectance as well as attributes.
        key, res = self._get_dataset(fh)
        exp = self._create_expected(res.name)
        exp.attrs['orbital_parameters'] = {'tle': 'tle'}

        xr.testing.assert_identical(res, exp)
        get_channel.assert_called_with(key)

        self._check_get_channel_calls(fh, get_channel)

    @staticmethod
    def _get_dataset(fh):
        from satpy.tests.utils import make_dataid

        key = make_dataid(name='1', calibration='reflectance')
        info = {'name': '1', 'standard_name': 'my_standard_name'}
        res = fh.get_dataset(key=key, info=info)
        return key, res

    @staticmethod
    def _create_file_handler(reader):
        """Mock reader and file handler."""
        fh = _get_fh_mocked(
            reader=reader,
            chn_dict={'1': 0, '5': 0},
            start_line=None,
            end_line=None,
            strip_invalid_coords=False,
            filename_info={'orbit_number': 123},
            sensor='sensor',
        )
        return fh

    @staticmethod
    def _create_expected(name):
        exp = xr.DataArray(da.ones((3, 3)),
                           name=name,
                           dims=('y', 'x'),
                           coords={'acq_time': ('y', [0, 1, 2])},
                           attrs={'name': '1',
                                  'platform_name': 'spacecraft_name',
                                  'orbit_number': 123,
                                  'sensor': 'sensor',
                                  'foo': 'bar',
                                  'standard_name': 'my_standard_name'})
        exp.coords['acq_time'].attrs['long_name'] = 'Mean scanline acquisition time'
        return exp

    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile.__init__', return_value=None)
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile.read_raw_data')
    @mock.patch('satpy.readers.avhrr_l1b_gaclac.GACLACFile._get_channel', return_value=np.ones((3, 3)))
    def test_get_dataset_no_tle(self, get_channel, *mocks):
        """Test getting the channel datasets when no TLEs are present."""
        pygac_reader = _get_reader_mocked()
        pygac_reader.get_tle_lines = mock.MagicMock()
        pygac_reader.get_tle_lines.side_effect = RuntimeError()

        fh = self._create_file_handler(pygac_reader)

        # Test calibration to reflectance as well as attributes.
        key, res = self._get_dataset(fh)
        exp = self._create_expected(res.name)
        xr.testing.assert_identical(res, exp)
        get_channel.assert_called_with(key)

        self._check_get_channel_calls(fh, get_channel)

    @staticmethod
    def _check_get_channel_calls(fh, get_channel):
        """Check _get_channel() calls."""
        from satpy.tests.utils import make_dataid

        for key in [make_dataid(name='1', calibration='counts'),
                    make_dataid(name='5', calibration='brightness_temperature')]:
            fh.get_dataset(key=key, info={'name': 1})
            get_channel.assert_called_with(key)
