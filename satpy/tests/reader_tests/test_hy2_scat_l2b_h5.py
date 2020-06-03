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
"""Module for testing the satpy.readers.hy2_scat_l2b_h5 module.
"""

import os
import numpy as np
import xarray as xr
import dask.array as da
from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler

import unittest
from unittest import mock

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_LAT_DATA = np.linspace(45, 65, DEFAULT_FILE_SHAPE[1]).astype(np.float32)
DEFAULT_LAT_DATA = np.repeat([DEFAULT_LAT_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_LON_DATA = np.linspace(-10, 10, DEFAULT_FILE_SHAPE[1]).astype(np.float32)
DEFAULT_LON_DATA = np.repeat([DEFAULT_LON_DATA], DEFAULT_FILE_SHAPE[0], axis=0)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler"""

    def _get_geo_data(self, num_rows, num_cols):
        geo = {
            'wvc_lon':
                xr.DataArray(
                    da.ones((num_rows, num_cols), chunks=1024,
                            dtype=np.float32),
                    attrs={
                        'fill_value': 1.7e+38,
                        'scale_factor': 1.,
                        'add_offset': 0.,
                        'units': 'degree',
                        'valid range': [0, 359.99],
                    },
                    dims=('y', 'x')),
            'wvc_lat':
                xr.DataArray(
                    da.ones((num_rows, num_cols), chunks=1024,
                            dtype=np.float32),
                    attrs={
                        'fill_value': 1.7e+38,
                        'scale_factor': 1.,
                        'add_offset': 0.,
                        'units': 'degree',
                        'valid range': [-90.0, 90.0],
                    },
                    dims=('y', 'x')),
        }
        return geo

    def _get_selection_data(self, num_rows, num_cols):
        selection = {
            'wvc_selection':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int8),
                attrs={
                    'fill_value': 0,
                    'scale_factor': 1.,
                    'add_offset': 0.,
                    'units': 'count',
                    'valid range': [1, 8],
                },
                dims=('y', 'x')),
            'wind_speed_selection':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int16),
                attrs={
                    'fill_value': -32767,
                    'scale_factor': 0.1,
                    'add_offset': 0.,
                    'units': 'deg',
                    'valid range': [0, 3599],
                },
                dims=('y', 'x')),
            'wind_dir_selection':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int16),
                attrs={
                    'fill_value': -32767,
                    'scale_factor': 0.01,
                    'add_offset': 0.,
                    'units': 'm/s',
                    'valid range': [0, 5000],
                },
                dims=('y', 'x')),
            'model_dir':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int16),
                attrs={
                    'fill_value': -32767,
                    'scale_factor': 0.01,
                    'add_offset': 0.,
                    'units': 'm/s',
                    'valid range': [0, 5000],
                },
                dims=('y', 'x')),
            'model_speed':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int16),
                attrs={
                    'fill_value': -32767,
                    'scale_factor': 0.1,
                    'add_offset': 0.,
                    'units': 'deg',
                    'valid range': [0, 3599],
                },
                dims=('y', 'x')),
            'num_ambigs':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int8),
                attrs={
                    'fill_value': 0,
                    'scale_factor': 1.,
                    'add_offset': 0.,
                    'units': 'count',
                    'valid range': [1, 8],
                },
                dims=('y', 'x')),
            'num_in_aft':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int8),
                attrs={
                    'fill_value': 0,
                    'scale_factor': 1.,
                    'add_offset': 0.,
                    'units': 'count',
                    'valid range': [1, 127],
                },
                dims=('y', 'x')),
            'num_in_fore':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int8),
                attrs={
                    'fill_value': 0,
                    'scale_factor': 1.,
                    'add_offset': 0.,
                    'units': 'count',
                    'valid range': [1, 127],
                },
                dims=('y', 'x')),
            'num_out_aft':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int8),
                attrs={
                    'fill_value': 0,
                    'scale_factor': 1.,
                    'add_offset': 0.,
                    'units': 'count',
                    'valid range': [1, 127],
                },
                dims=('y', 'x')),
            'num_out_fore':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.int8),
                attrs={
                    'fill_value': 0,
                    'scale_factor': 1.,
                    'add_offset': 0.,
                    'units': 'count',
                    'valid range': [1, 127],
                },
                dims=('y', 'x')),
            'wvc_quality_flag':
            xr.DataArray(
                da.ones((num_rows, num_cols), chunks=1024,
                        dtype=np.uint16),
                attrs={
                    'fill_value': 2.14748e+09,
                    'scale_factor': 1.,
                    'add_offset': 0.,
                    'units': 'na',
                    'valid range': [1, 2.14748e+09],
                },
                dims=('y', 'x')),
        }
        return selection

    def _get_all_ambiguities_data(self, num_rows, num_cols, num_amb):
        all_amb = {
            'max_likelihood_est':
                xr.DataArray(
                    da.ones((num_rows, num_cols, num_amb), chunks=1024,
                            dtype=np.int16),
                    attrs={
                        'fill_value': -32767,
                        'scale_factor': 1.,
                        'add_offset': 0.,
                        'units': 'na',
                        'valid range': [0, 32767],
                    },
                    dims=('y', 'x', 'selection')),
            'wind_dir':
                xr.DataArray(
                    da.ones((num_rows, num_cols, num_amb), chunks=1024,
                            dtype=np.int16),
                    attrs={
                        'fill_value': -32767,
                        'scale_factor': 0.1,
                        'add_offset': 0.,
                        'units': 'deg',
                        'valid range': [0, 3599],
                    },
                    dims=('y', 'x', 'selection')),
            'wind_speed':
                xr.DataArray(
                    da.ones((num_rows, num_cols, num_amb), chunks=1024,
                            dtype=np.int16),
                    attrs={
                        'fill_value': -32767,
                        'scale_factor': 0.01,
                        'add_offset': 0.,
                        'units': 'm/s',
                        'valid range': [0, 5000],
                    },
                    dims=('y', 'x', 'selection')),
        }
        return all_amb

    def _get_wvc_row_time(self, num_rows):
        data = ["20200326T01:11:07.639",
                "20200326T01:11:11.443",
                "20200326T01:11:15.246",
                "20200326T01:11:19.049",
                "20200326T01:11:22.856",
                "20200326T01:11:26.660",
                "20200326T01:11:30.464",
                "20200326T01:11:34.268",
                "20200326T01:11:38.074",
                "20200326T01:11:41.887"]
        wvc_row_time = {
            'wvc_row_time':
                xr.DataArray(data,
                             attrs={
                                 'fill_value': "",
                             },
                             dims=('y',)),
        }
        return wvc_row_time

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        num_rows = 300
        num_cols = 10
        num_amb = 8
        global_attrs = {
            '/attr/Equator_Crossing_Longitude': '246.408397',
            '/attr/Equator_Crossing_Time': '20200326T01:37:15.875',
            '/attr/HDF_Version_Id': 'HDF5-1.8.16',
            '/attr/Input_L2A_Filename': 'H2B_OPER_SCA_L2A_OR_20200326T010839_20200326T025757_07076_dps_250_20.h5',
            '/attr/Instrument_ShorName': 'HSCAT-B',
            '/attr/L2A_Inputdata_Version': '10',
            '/attr/L2B_Actual_WVC_Rows': np.int32(num_rows),
            '/attr/L2B_Algorithm_Descriptor': ('Wind retrieval processing uses the multiple solution scheme (MSS) for '
                                               'wind inversion with the NSCAT-4 GMF,and a circular median filter '
                                               'method (CMF) for ambiguity removal. The ECMWF/NCEP forescate data are '
                                               'used as background winds in the CMF'),
            '/attr/L2B_Data_Version': '10',
            '/attr/L2B_Expected_WVC_Rows': np.int32(num_rows),
            '/attr/L2B_Number_WVC_cells': np.int32(num_cols),
            '/attr/L2B_Processing_Type': 'OPER',
            '/attr/L2B_Processor_Name': 'hy2_sca_l2b_pro',
            '/attr/L2B_Processor_Version': '01.00',
            '/attr/Long_Name': 'HY-2B/SCAT Level 2B Ocean Wind Vectors in 25.0 km Swath Grid',
            '/attr/Orbit_Inclination': np.float32(99.3401),
            '/attr/Orbit_Number': '07076',
            '/attr/Output_L2B_Filename': 'H2B_OPER_SCA_L2B_OR_20200326T011107_20200326T025540_07076_dps_250_20_owv.h5',
            '/attr/Platform_LongName': 'Haiyang 2B Ocean Observing Satellite',
            '/attr/Platform_ShortName': 'HY-2B',
            '/attr/Platform_Type': 'spacecraft',
            '/attr/Producer_Agency': 'Ministry of Natural Resources of the People\'s Republic of China',
            '/attr/Producer_Institution': 'NSOAS',
            '/attr/Production_Date_Time': '20200326T06:23:10',
            '/attr/Range_Beginning_Time': '20200326T01:11:07',
            '/attr/Range_Ending_Time': '20200326T02:55:40',
            '/attr/Rev_Orbit_Period': '14 days',
            '/attr/Short_Name': 'HY-2B SCAT-L2B-25km',
            '/attr/Sigma0_Granularity': 'whole pulse',
            '/attr/WVC_Size': '25000m*25000m',
        }

        test_content = {}
        test_content.update(global_attrs)
        data = {}
        data = self._get_geo_data(num_rows, num_cols)
        test_content.update(data)
        data = self._get_selection_data(num_rows, num_cols)
        test_content.update(data)
        data = self._get_all_ambiguities_data(num_rows, num_cols, num_amb)
        test_content.update(data)

        data = self._get_wvc_row_time(num_rows)
        test_content.update(data)

        return test_content


class TestHY2SCATL2BH5Reader(unittest.TestCase):
    """Test HY2 Scatterometer L2B H5 Reader."""
    yaml_file = "hy2_scat_l2b_h5.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.readers.hy2_scat_l2b_h5 import HY2SCATL2BH5FileHandler
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(HY2SCATL2BH5FileHandler, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def test_load_geo(self):
        """Test loading data."""
        from satpy.readers import load_reader
        filenames = [
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,HY2B+SM_C_EUMP_20200326------_07077_o_250_l2b.h5', ]

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)

        res = reader.load(['wvc_lon', 'wvc_lat'])
        self.assertEqual(2, len(res))

    def test_load_data_selection(self):
        """Test loading data."""
        from satpy.readers import load_reader
        filenames = [
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,HY2B+SM_C_EUMP_20200326------_07077_o_250_l2b.h5', ]

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)
        res = reader.load(['wind_speed_selection',
                           'wind_dir_selection',
                           'wvc_selection'])
        self.assertEqual(3, len(res))

    def test_load_data_all_ambiguities(self):
        """Test loading data."""
        from satpy.readers import load_reader
        filenames = [
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,HY2B+SM_C_EUMP_20200326------_07077_o_250_l2b.h5', ]

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)
        res = reader.load(['wind_speed',
                           'wind_dir',
                           'max_likelihood_est',
                           'model_dir',
                           'model_speed',
                           'num_ambigs',
                           'num_in_aft',
                           'num_in_fore',
                           'num_out_aft',
                           'num_out_fore',
                           'wvc_quality_flag'])
        self.assertEqual(11, len(res))

    def test_load_data_row_times(self):
        """Test loading data."""
        from satpy.readers import load_reader
        filenames = [
            'W_XX-EUMETSAT-Darmstadt,SURFACE+SATELLITE,HY2B+SM_C_EUMP_20200326------_07077_o_250_l2b.h5', ]

        reader = load_reader(self.reader_configs)
        files = reader.select_files_from_pathnames(filenames)
        self.assertTrue(1, len(files))
        reader.create_filehandlers(files)
        # Make sure we have some files
        self.assertTrue(reader.file_handlers)
        res = reader.load(['wvc_row_time'])
        self.assertEqual(1, len(res))
