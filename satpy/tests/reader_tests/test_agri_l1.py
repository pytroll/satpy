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
"""The agri_l1 reader tests package."""

import os
from unittest import mock

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler

ALL_BAND_NAMES = ["C01", "C02", "C03", "C04", "C05", "C06", "C07",
                  "C08", "C09", "C10", "C11", "C12", "C13", "C14"]

CHANNELS_BY_RESOLUTION = {500: ["C02"],
                          1000: ["C01", "C02", "C03"],
                          2000: ["C01", "C02", "C03", "C04", "C05", "C06", "C07"],
                          4000: ALL_BAND_NAMES,
                          'GEO': 'solar_azimuth_angle'
                          }

RESOLUTION_LIST = [500, 1000, 2000, 4000]

AREA_EXTENTS_BY_RESOLUTION = {'FY4A': {
    500: (-5495271.006002, -5496021.008869, -5493270.998357, -5495521.006957),
    1000: (-5494521.070252, -5496021.076004, -5490521.054912, -5495021.072169),
    2000: (-5493021.198696, -5496021.210274, -5485021.167823, -5494021.202556),
    4000: (-5490021.187119, -5496021.210274, -5474021.125371, -5492021.194837)
},
    'FY4B': {
        500: (-5495271.006002, -5496021.008869, -5493270.998357, -5495521.006957),
        1000: (-5494521.070252, -5496021.076004, -5490521.054912, -5495021.072169),
        2000: (-5493021.198696, -5496021.210274, -5485021.167823, -5494021.202556),
        4000: (-5490021.187119, -5496021.210274, -5474021.125371, -5492021.194837)
    }}


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def make_test_data(self, cwl, ch, prefix, dims, file_type):
        """Make test data."""
        if prefix == 'CAL':
            data = xr.DataArray(
                da.from_array((np.arange(10.) + 1.) / 10., [dims[0] * dims[1]]),
                attrs={
                    'Slope': np.array(1.), 'Intercept': np.array(0.),
                    'FillValue': np.array(-65535.0),
                    'units': 'NUL',
                    'center_wavelength': '{}um'.format(cwl).encode('utf-8'),
                    'band_names': 'band{}(band number is range from 1 to 14)'
                    .format(ch).encode('utf-8'),
                    'long_name': 'Calibration table of {}um Channel'.format(cwl).encode('utf-8'),
                    'valid_range': np.array([0, 1.5]),
                },
                dims='_const')

        elif prefix == 'NOM':
            data = xr.DataArray(
                da.from_array(np.arange(10, dtype=np.uint16).reshape((2, 5)) + 1,
                              [dim for dim in dims]),
                attrs={
                    'Slope': np.array(1.), 'Intercept': np.array(0.),
                    'FillValue': np.array(65535),
                    'units': 'DN',
                    'center_wavelength': '{}um'.format(cwl).encode('utf-8'),
                    'band_names': 'band{}(band number is range from 1 to 14)'
                    .format(ch).encode('utf-8'),
                    'long_name': 'Calibration table of {}um Channel'.format(cwl).encode('utf-8'),
                    'valid_range': np.array([0, 4095]),
                },
                dims=('_RegLength', '_RegWidth'))

        elif prefix == 'GEO':
            data = xr.DataArray(
                da.from_array(np.arange(0., 360., 36., dtype=np.float32).reshape((2, 5)),
                              [dim for dim in dims]),
                attrs={
                    'Slope': np.array(1.), 'Intercept': np.array(0.),
                    'FillValue': np.array(65535.),
                    'units': 'NUL',
                    'band_names': 'NUL',
                    'valid_range': np.array([0., 360.]),
                },
                dims=('_RegLength', '_RegWidth'))

        elif prefix == 'COEF':
            if file_type == '500':
                data = self._create_coeff_array(1)

            elif file_type == '1000':
                data = self._create_coeff_array(3)

            elif file_type == '2000':
                data = self._create_coeff_array(7)

            elif file_type == '4000':
                data = self._create_coeff_array(14)

        return data

    def _create_coeff_array(self, nb_channels):
        data = xr.DataArray(
            da.from_array((np.arange(nb_channels * 2).reshape((nb_channels, 2)) + 1.) /
                          np.array([1E4, 1E2]), [nb_channels, 2]),
            attrs={
                'Slope': 1., 'Intercept': 0.,
                'FillValue': 0,
                'units': 'NUL',
                'band_names': 'NUL',
                'long_name': b'Calibration coefficient (SCALE and OFFSET)',
                'valid_range': [-500, 500],
            },
            dims=('_num_channel', '_coefs'))
        return data

    def _create_channel_data(self, chs, cwls, file_type):
        dim_0 = 2
        dim_1 = 5
        data = {}
        for index, _cwl in enumerate(cwls):
            data['CALChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'CAL',
                                                                           [dim_0, dim_1], file_type)
            data['Calibration/CALChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'CAL',
                                                                                       [dim_0, dim_1], file_type)
            data['NOMChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'NOM',
                                                                           [dim_0, dim_1], file_type)
            data['Data/NOMChannel' + '%02d' % chs[index]] = self.make_test_data(cwls[index], chs[index], 'NOM',
                                                                                [dim_0, dim_1], file_type)
            data['CALIBRATION_COEF(SCALE+OFFSET)'] = self.make_test_data(cwls[index], chs[index], 'COEF',
                                                                         [dim_0, dim_1], file_type)
            data['Calibration/CALIBRATION_COEF(SCALE+OFFSET)'] = self.make_test_data(cwls[index], chs[index], 'COEF',
                                                                                     [dim_0, dim_1], file_type)
        return data

    def _get_500m_data(self, file_type):
        chs = [2]
        cwls = [0.65]
        data = self._create_channel_data(chs, cwls, file_type)

        return data

    def _get_1km_data(self, file_type):
        chs = np.linspace(1, 3, 3)
        cwls = [0.47, 0.65, 0.83]
        data = self._create_channel_data(chs, cwls, file_type)

        return data

    def _get_2km_data(self, file_type):
        chs = np.linspace(1, 7, 7)
        cwls = [0.47, 0.65, 0.83, 1.37, 1.61, 2.22, 3.72]
        data = self._create_channel_data(chs, cwls, file_type)

        return data

    def _get_4km_data(self, file_type):
        chs = np.linspace(1, 14, 14)
        cwls = [0.47, 0.65, 0.83, 1.37, 1.61, 2.22, 3.72, 3.72, 6.25, 7.10, 8.50, 10.8, 12, 13.5]
        data = self._create_channel_data(chs, cwls, file_type)

        return data

    def _get_geo_data(self, file_type):
        dim_0 = 2
        dim_1 = 5
        data = {'NOMSunAzimuth': self.make_test_data('NUL', 'NUL', 'GEO',
                                                     [dim_0, dim_1], file_type),
                'Navigation/NOMSunAzimuth': self.make_test_data('NUL', 'NUL', 'GEO',
                                                                [dim_0, dim_1], file_type)}
        return data

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        global_attrs = {
            '/attr/NOMCenterLat': np.array(0.0),
            '/attr/NOMCenterLon': np.array(104.7),
            '/attr/NOMSatHeight': np.array(42164140.0),
            '/attr/dEA': np.array(6378.14),
            '/attr/dObRecFlat': np.array(298.257223563),
            '/attr/OBIType': 'REGC',
            '/attr/RegLength': np.array(2.0),
            '/attr/RegWidth': np.array(5.0),
            '/attr/Begin Line Number': np.array(0),
            '/attr/End Line Number': np.array(1),
            '/attr/Observing Beginning Date': '2019-06-03', '/attr/Observing Beginning Time': '00:30:01.807',
            '/attr/Observing Ending Date': '2019-06-03', '/attr/Observing Ending Time': '00:34:07.572',
            '/attr/Satellite Name': 'FY4A', '/attr/Sensor Identification Code': 'AGRI', '/attr/Sensor Name': 'AGRI',
        }

        data = {}
        if self.filetype_info['file_type'] == 'agri_l1_0500m':
            data = self._get_500m_data('500')
        elif self.filetype_info['file_type'] == 'agri_l1_1000m':
            data = self._get_1km_data('1000')
        elif self.filetype_info['file_type'] == 'agri_l1_2000m':
            data = self._get_2km_data('2000')
            global_attrs['/attr/Observing Beginning Time'] = '00:30:01'
            global_attrs['/attr/Observing Ending Time'] = '00:34:07'
        elif self.filetype_info['file_type'] == 'agri_l1_4000m':
            data = self._get_4km_data('4000')
        elif self.filetype_info['file_type'] == 'agri_l1_4000m_geo':
            data = self._get_geo_data('4000')

        test_content = {}
        test_content.update(global_attrs)
        test_content.update(data)

        return test_content


def _create_filenames_from_resolutions(satname, *resolutions):
    """Create filenames from the given resolutions."""
    if 'GEO' in resolutions:
        return [f"{satname}-_AGRI--_N_REGC_1047E_L1-_GEO-_MULT_NOM_20190603003000_20190603003416_4000M_V0001.HDF"]
    pattern = (f"{satname}-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20190603003000_20190603003416_"
               "{resolution:04d}M_V0001.HDF")
    return [pattern.format(resolution=resolution) for resolution in resolutions]


class Test_HDF_AGRI_L1_cal:
    """Test VIRR L1B Reader."""

    yaml_file = "agri_fy4a_l1.yaml"

    def setup_method(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy._config import config_search_paths
        from satpy.readers.agri_l1 import HDF_AGRI_L1
        from satpy.readers.fy4_base import FY4Base
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.fy4 = mock.patch.object(FY4Base, '__bases__', (FakeHDF5FileHandler2,))
        self.p = mock.patch.object(HDF_AGRI_L1.__class__, (self.fy4,))
        self.fake_handler = self.fy4.start()
        self.p.is_local = True
        self.satname = 'FY4A'

        self.expected = {
            1: np.array([[2.01, 2.02, 2.03, 2.04, 2.05], [2.06, 2.07, 2.08, 2.09, 2.1]]),
            2: np.array([[4.03, 4.06, 4.09, 4.12, 4.15], [4.18, 4.21, 4.24, 4.27, 4.3]]),
            3: np.array([[6.05, 6.1, 6.15, 6.2, 6.25], [6.3, 6.35, 6.4, 6.45, 6.5]]),
            4: np.array([[8.07, 8.14, 8.21, 8.28, 8.35], [8.42, 8.49, 8.56, 8.63, 8.7]]),
            5: np.array([[10.09, 10.18, 10.27, 10.36, 10.45], [10.54, 10.63, 10.72, 10.81, 10.9]]),
            6: np.array([[12.11, 12.22, 12.33, 12.44, 12.55], [12.66, 12.77, 12.88, 12.99, 13.1]]),
            7: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]]),
            8: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]]),
            9: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]]),
            10: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]]),
            11: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]]),
            12: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]]),
            13: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]]),
            14: np.array([[0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1., np.nan]])
        }

    def teardown_method(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_times_correct(self):
        """Test that the reader handles the two possible time formats correctly."""
        reader = self._create_reader_for_resolutions(1000)
        np.testing.assert_almost_equal(reader.start_time.microsecond, 807000)
        reader = self._create_reader_for_resolutions(2000)
        np.testing.assert_almost_equal(reader.start_time.microsecond, 0)

    def test_fy4a_channels_are_loaded_with_right_resolution(self):
        """Test all channels are loaded with the right resolution."""
        reader = self._create_reader_for_resolutions(*RESOLUTION_LIST)

        available_datasets = reader.available_dataset_ids

        for resolution_to_test in RESOLUTION_LIST:
            self._check_keys_for_dsq(available_datasets, resolution_to_test)

    def test_agri_all_bands_have_right_units(self):
        """Test all bands have the right units."""
        reader = self._create_reader_for_resolutions(*RESOLUTION_LIST)

        band_names = ALL_BAND_NAMES
        res = reader.load(band_names)
        assert len(res) == 14

        for band_name in band_names:
            assert res[band_name].shape == (2, 5)
            self._check_units(band_name, res)

    def test_agri_orbital_parameters_are_correct(self):
        """Test orbital parameters are set correctly."""
        reader = self._create_reader_for_resolutions(*RESOLUTION_LIST)

        band_names = ALL_BAND_NAMES
        res = reader.load(band_names)

        # check whether the data type of orbital_parameters is float
        orbital_parameters = res[band_names[0]].attrs['orbital_parameters']
        for attr in orbital_parameters:
            assert isinstance(orbital_parameters[attr], float)
        assert orbital_parameters['satellite_nominal_latitude'] == 0.
        assert orbital_parameters['satellite_nominal_longitude'] == 104.7
        assert orbital_parameters['satellite_nominal_altitude'] == 42164140.0

    @staticmethod
    def _check_keys_for_dsq(available_datasets, resolution_to_test):
        from satpy.dataset.data_dict import get_key
        from satpy.tests.utils import make_dsq

        band_names = CHANNELS_BY_RESOLUTION[resolution_to_test]
        for band_name in band_names:
            ds_q = make_dsq(name=band_name, resolution=resolution_to_test)
            res = get_key(ds_q, available_datasets, num_results=0, best=False)
            if band_name < 'C07':
                assert len(res) == 2
            else:
                assert len(res) == 3

    def test_agri_counts_calibration(self):
        """Test loading data at counts calibration."""
        from satpy.tests.utils import make_dsq
        reader = self._create_reader_for_resolutions(*RESOLUTION_LIST)

        ds_ids = []
        band_names = CHANNELS_BY_RESOLUTION[4000]
        for band_name in band_names:
            ds_ids.append(make_dsq(name=band_name, calibration='counts'))
        res = reader.load(ds_ids)
        assert len(res) == 14

        for band_name in band_names:
            assert res[band_name].shape == (2, 5)
            assert res[band_name].attrs['calibration'] == "counts"
            assert res[band_name].dtype == np.uint16
            assert res[band_name].attrs['units'] == "1"

    @pytest.mark.parametrize("satname", ['FY4A', 'FY4B'])
    def test_agri_geo(self, satname):
        """Test loading data for angles."""
        from satpy.tests.utils import make_dsq
        self.satname = satname
        reader = self._create_reader_for_resolutions('GEO')
        band_name = 'solar_azimuth_angle'
        ds_ids = [make_dsq(name=band_name)]
        res = reader.load(ds_ids)
        assert len(res) == 1

        np.testing.assert_almost_equal(np.nanmin(res[band_name]), 0.)
        np.testing.assert_almost_equal(np.nanmax(res[band_name]), 324.)

        assert res[band_name].shape == (2, 5)
        assert res[band_name].dtype == np.float32

    def _create_reader_for_resolutions(self, *resolutions):
        from satpy.readers import load_reader
        filenames = _create_filenames_from_resolutions(self.satname, *resolutions)
        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        assert len(filenames) == len(files)
        reader.create_filehandlers(files)
        # Make sure we have some files
        assert reader.file_handlers
        return reader

    @pytest.mark.parametrize("resolution_to_test", RESOLUTION_LIST)
    @pytest.mark.parametrize("satname", ['FY4A', 'FY4B'])
    def test_agri_for_one_resolution(self, resolution_to_test, satname):
        """Test loading data when only one resolution is available."""
        self.satname = satname
        reader = self._create_reader_for_resolutions(resolution_to_test)

        available_datasets = reader.available_dataset_ids
        band_names = CHANNELS_BY_RESOLUTION[resolution_to_test]
        self._assert_which_channels_are_loaded(available_datasets, band_names, resolution_to_test)
        res = reader.load(band_names)
        assert len(res) == len(band_names)
        self._check_calibration_and_units(band_names, res)
        for band_name in band_names:
            np.testing.assert_allclose(res[band_name].attrs['area'].area_extent,
                                       AREA_EXTENTS_BY_RESOLUTION[satname][resolution_to_test])

    def _check_calibration_and_units(self, band_names, result):
        for index, band_name in enumerate(band_names):
            assert result[band_name].attrs['sensor'].islower()
            assert result[band_name].shape == (2, 5)
            np.testing.assert_allclose(result[band_name].values, self.expected[index + 1], equal_nan=True)
            self._check_units(band_name, result)

    @staticmethod
    def _check_units(band_name, result):
        if band_name < 'C07':
            assert result[band_name].attrs['calibration'] == "reflectance"
        else:
            assert result[band_name].attrs['calibration'] == 'brightness_temperature'
        if band_name < 'C07':
            assert result[band_name].attrs['units'] == "%"
        else:
            assert result[band_name].attrs['units'] == "K"

    @staticmethod
    def _assert_which_channels_are_loaded(available_datasets, band_names, resolution_to_test):
        from satpy.dataset.data_dict import get_key
        from satpy.tests.utils import make_dsq

        other_resolutions = RESOLUTION_LIST.copy()
        other_resolutions.remove(resolution_to_test)
        for band_name in band_names:
            for resolution in other_resolutions:
                ds_q = make_dsq(name=band_name, resolution=resolution)
                with pytest.raises(KeyError):
                    _ = get_key(ds_q, available_datasets, num_results=0, best=False)

            ds_q = make_dsq(name=band_name, resolution=resolution_to_test)
            res = get_key(ds_q, available_datasets, num_results=0, best=False)
            if band_name < 'C07':
                assert len(res) == 2
            else:
                assert len(res) == 3
