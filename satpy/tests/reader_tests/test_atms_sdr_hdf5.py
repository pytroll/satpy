#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022-2023 Pytroll developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Module for testing the ATMS SDR HDF5 reader."""

import os
from datetime import datetime
from unittest import mock

import numpy as np
import pytest

from satpy._config import config_search_paths
from satpy.readers import load_reader
from satpy.readers.atms_sdr_hdf5 import ATMS_CHANNEL_NAMES
from satpy.readers.viirs_atms_sdr_base import DATASET_KEYS
from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (1, 96)
# Mimicking one scan line of data
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)
DEFAULT_FILE_FACTORS = np.array([2.0, 1.0], dtype=np.float32)


class FakeHDF5_ATMS_SDR_FileHandler(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    _num_test_granules = 1
    _num_scans_per_gran = [12]
    _num_of_bands = 22

    def __init__(self, filename, filename_info, filetype_info, include_factors=True):
        """Create fake file handler."""
        self.include_factors = include_factors
        super().__init__(filename, filename_info, filetype_info)

    @staticmethod
    def _add_basic_metadata_to_file_content(file_content, filename_info, num_grans):
        start_time = filename_info['start_time']
        end_time = filename_info['end_time'].replace(year=start_time.year,
                                                     month=start_time.month,
                                                     day=start_time.day)
        begin_date = start_time.strftime('%Y%m%d')
        begin_time = start_time.strftime('%H%M%S.%fZ')
        ending_date = end_time.strftime('%Y%m%d')
        ending_time = end_time.strftime('%H%M%S.%fZ')
        new_file_content = {
            "{prefix2}/attr/AggregateNumberGranules": num_grans,
            "{prefix2}/attr/AggregateBeginningDate": begin_date,
            "{prefix2}/attr/AggregateBeginningTime": begin_time,
            "{prefix2}/attr/AggregateEndingDate": ending_date,
            "{prefix2}/attr/AggregateEndingTime": ending_time,
            "{prefix2}/attr/G-Ring_Longitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "{prefix2}/attr/G-Ring_Latitude": np.array([0.0, 0.1, 0.2, 0.3]),
            "{prefix2}/attr/AggregateBeginningOrbitNumber": "{0:d}".format(filename_info['orbit']),
            "{prefix2}/attr/AggregateEndingOrbitNumber": "{0:d}".format(filename_info['orbit']),
            "{prefix1}/attr/Instrument_Short_Name": "ATMS",
            "/attr/Platform_Short_Name": "J01",
        }
        file_content.update(new_file_content)

    def _add_granule_specific_info_to_file_content(self, file_content, dataset_group,
                                                   num_granules, num_scans_per_granule, gran_group_prefix):
        lons_lists = self._get_per_granule_lons()
        lats_lists = self._get_per_granule_lats()
        file_content["{prefix3}/NumberOfScans"] = np.array([1] * num_granules)
        for granule_idx in range(num_granules):
            prefix_gran = '{prefix}/{dataset_group}_Gran_{idx}'.format(prefix=gran_group_prefix,
                                                                       dataset_group=dataset_group,
                                                                       idx=granule_idx)
            num_scans = num_scans_per_granule[granule_idx]
            file_content[prefix_gran + '/attr/N_Number_Of_Scans'] = num_scans
            file_content[prefix_gran + '/attr/G-Ring_Longitude'] = lons_lists[granule_idx]
            file_content[prefix_gran + '/attr/G-Ring_Latitude'] = lats_lists[granule_idx]

    @staticmethod
    def _get_per_granule_lons():
        return [
            np.array(
                [
                    50.51393, 49.566296, 48.865967, 18.96082, -4.0238385,
                    -7.05221, -10.405702, 14.638646
                ],
                dtype=np.float32),
            np.array(
                [
                    53.52594, 51.685738, 50.439102, 14.629087, -10.247547,
                    -13.951393, -18.256989, 8.36572
                ],
                dtype=np.float32),
            np.array(
                [
                    59.386833, 55.770416, 53.38952, 8.353765, -18.062435,
                    -22.608992, -27.867302, -1.3537619
                ],
                dtype=np.float32),
            np.array(
                [
                    72.50243, 64.17125, 59.15234, -1.3654504, -27.620953,
                    -33.091743, -39.28113, -17.749891
                ],
                dtype=np.float32)
        ]

    @staticmethod
    def _get_per_granule_lats():
        return [
            np.array(
                [
                    67.969505, 65.545685, 63.103046, 61.853905, 55.169273,
                    57.062447, 58.86063, 66.495514
                ],
                dtype=np.float32),
            np.array(
                [
                    72.74879, 70.2493, 67.84738, 66.49691, 58.77254,
                    60.465942, 62.11525, 71.08249
                ],
                dtype=np.float32),
            np.array(
                [
                    77.393425, 74.977875, 72.62976, 71.083435, 62.036346,
                    63.465122, 64.78075, 75.36842
                ],
                dtype=np.float32),
            np.array(
                [
                    81.67615, 79.49934, 77.278656, 75.369415, 64.72178,
                    65.78417, 66.66166, 79.00025
                ],
                dtype=np.float32),
        ]

    def _add_data_info_to_file_content(self, file_content, filename, data_var_prefix, num_grans):
        # ATMS SDR files always produce data with 12 scans per granule even if there are less? FIXME!
        total_rows = DEFAULT_FILE_SHAPE[0] * 12 * num_grans
        new_shape = (total_rows, DEFAULT_FILE_SHAPE[1], self._num_of_bands)
        key = 'BrightnessTemperature'
        key = data_var_prefix + "/" + key
        file_content[key] = np.repeat(DEFAULT_FILE_DATA.copy(), 12 * num_grans, axis=0)
        file_content[key] = np.repeat(file_content[key][:, :, np.newaxis], self._num_of_bands, axis=2)
        file_content[key + "/shape"] = new_shape
        if self.include_factors:
            file_content[key + "Factors"] = np.repeat(
                DEFAULT_FILE_FACTORS.copy()[None, :], num_grans, axis=0).ravel()

    @staticmethod
    def _add_geolocation_info_to_file_content(file_content, filename, data_var_prefix, num_grans):
        # ATMS SDR files always produce data with 12 scans per granule even if there are less? FIXME!
        total_rows = DEFAULT_FILE_SHAPE[0] * 12 * num_grans
        new_shape = (total_rows, DEFAULT_FILE_SHAPE[1])

        lon_data = np.linspace(15, 55, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)
        lat_data = np.linspace(55, 75, DEFAULT_FILE_SHAPE[1]).astype(DEFAULT_FILE_DTYPE)

        for k in ["Latitude"]:
            k = data_var_prefix + "/" + k
            file_content[k] = lat_data
            file_content[k] = np.repeat([file_content[k]], total_rows, axis=0)
            file_content[k + "/shape"] = new_shape
        for k in ["Longitude"]:
            k = data_var_prefix + "/" + k
            file_content[k] = lon_data
            file_content[k] = np.repeat([file_content[k]], total_rows, axis=0)
            file_content[k + "/shape"] = new_shape

        angles = ['SolarZenithAngle',
                  'SolarAzimuthAngle',
                  'SatelliteZenithAngle',
                  'SatelliteAzimuthAngle']
        for k in angles:
            k = data_var_prefix + "/" + k
            file_content[k] = lon_data  # close enough to SZA
            file_content[k] = np.repeat([file_content[k]], total_rows, axis=0)
            file_content[k + "/shape"] = new_shape

    @staticmethod
    def _add_geo_ref(file_content, filename):
        geo_prefix = 'GATMO'
        file_content['/attr/N_GEO_Ref'] = geo_prefix + filename[5:]

    @staticmethod
    def _convert_numpy_content_to_dataarray(final_content):
        import dask.array as da
        from xarray import DataArray
        for key, val in final_content.items():
            if isinstance(val, np.ndarray):
                val = da.from_array(val, chunks=val.shape)
                if val.ndim > 2:
                    final_content[key] = DataArray(val, dims=('y', 'x', 'z'))
                elif val.ndim > 1:
                    final_content[key] = DataArray(val, dims=('y', 'x'))
                else:
                    final_content[key] = DataArray(val)

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        final_content = {}
        for dataset in self.datasets:
            dataset_group = DATASET_KEYS[dataset]
            prefix1 = 'Data_Products/{dataset_group}'.format(dataset_group=dataset_group)
            prefix2 = '{prefix}/{dataset_group}_Aggr'.format(prefix=prefix1, dataset_group=dataset_group)
            prefix3 = 'All_Data/{dataset_group}_All'.format(dataset_group=dataset_group)

            file_content = {}
            self._add_basic_metadata_to_file_content(file_content, filename_info, self._num_test_granules)
            self._add_granule_specific_info_to_file_content(file_content, dataset_group,
                                                            self._num_test_granules, self._num_scans_per_gran,
                                                            prefix1)
            self._add_geo_ref(file_content, filename)

            for k, v in list(file_content.items()):
                file_content[k.format(prefix1=prefix1, prefix2=prefix2, prefix3=prefix3)] = v

            if filename[:5] in ['SATMS', 'TATMS']:
                self._add_data_info_to_file_content(file_content, filename, prefix3,
                                                    self._num_test_granules)
            elif filename[0] == 'G':
                self._add_geolocation_info_to_file_content(file_content, filename, prefix3,
                                                           self._num_test_granules)
            final_content.update(file_content)
        self._convert_numpy_content_to_dataarray(final_content)
        return final_content


class TestATMS_SDR_Reader:
    """Test ATMS SDR Reader."""

    yaml_file = "atms_sdr_hdf5.yaml"

    def _assert_bt_properties(self, data_arr, num_scans=1, with_area=True):
        assert np.issubdtype(data_arr.dtype, np.float32)

        assert data_arr.attrs['calibration'] == 'brightness_temperature'
        assert data_arr.attrs['units'] == 'K'
        assert data_arr.attrs['rows_per_scan'] == num_scans
        if with_area:
            assert 'area' in data_arr.attrs
            assert data_arr.attrs['area'] is not None
            assert data_arr.attrs['area'].shape == data_arr.shape
        else:
            assert 'area' not in data_arr.attrs

    def setup_method(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.readers.viirs_atms_sdr_base import JPSS_SDR_FileHandler

        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(JPSS_SDR_FileHandler, '__bases__', (FakeHDF5_ATMS_SDR_FileHandler,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def teardown_method(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_init(self):
        """Test basic init with no extra parameters."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            '/path/to/atms/sdr/data/SATMS_j01_d20221220_t0910240_e0921356_b26361_c20221220100456348770_cspp_dev.h5',
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    def test_init_start_end_time(self):
        """Test basic init with start and end times around the start/end times of the provided file."""
        r = load_reader(self.reader_configs,
                        filter_parameters={
                            'start_time': datetime(2022, 12, 19),
                            'end_time': datetime(2022, 12, 21)
                        })
        loadables = r.select_files_from_pathnames([
            'SATMS_j01_d20221220_t0910240_e0921356_b26361_c20221220100456348770_cspp_dev.h5',
        ])
        assert len(loadables) == 1
        r.create_filehandlers(loadables)
        # make sure we have some files
        assert r.file_handlers

    @pytest.mark.parametrize("files, expected",
                             [(['SATMS_j01_d20221220_t0910240_e0921356_b26361_c20221220100456348770_cspp_dev.h5',
                                'GATMO_j01_d20221220_t0910240_e0921356_b26361_c20221220100456680030_cspp_dev.h5'],
                               True),
                              (['SATMS_j01_d20221220_t0910240_e0921356_b26361_c20221220100456348770_cspp_dev.h5', ],
                               False)]
                             )
    def test_load_all_bands(self, files, expected):
        """Load brightness temperatures for all 22 ATMS channels, with/without geolocation."""
        from satpy.readers import load_reader
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames(files)
        r.create_filehandlers(loadables)
        ds = r.load(ATMS_CHANNEL_NAMES)
        assert len(ds) == 22
        for d in ds.values():
            self._assert_bt_properties(d, with_area=expected)
