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
"""Tests for the SEVIRI L1b HDF4 from ICARE reader."""
import os
import unittest
from unittest import mock

import dask.array as da
import numpy as np

from satpy.readers import load_reader
from satpy.tests.reader_tests.test_hdf4_utils import FakeHDF4FileHandler

DEFAULT_FILE_DTYPE = np.uint16
DEFAULT_FILE_SHAPE = (10, 300)
DEFAULT_FILE_DATA = np.arange(DEFAULT_FILE_SHAPE[0] * DEFAULT_FILE_SHAPE[1],
                              dtype=DEFAULT_FILE_DTYPE).reshape(DEFAULT_FILE_SHAPE)


class FakeHDF4FileHandler2(FakeHDF4FileHandler):
    """Swap in HDF4 file handler."""

    def get_test_content(self, filename, filename_info, filename_type):
        """Mimic reader input file content."""
        file_content = {}
        file_content['/attr/Nadir_Pixel_Size'] = 3000.
        file_content['/attr/Beginning_Acquisition_Date'] = "2004-12-29T12:15:00Z"
        file_content['/attr/End_Acquisition_Date'] = "2004-12-29T12:27:44Z"
        file_content['/attr/Geolocation'] = ('1.3642337E7', '1856.0', '1.3642337E7', '1856.0')
        file_content['/attr/Altitude'] = '42164.0'
        file_content['/attr/Geographic_Projection'] = 'geos'
        file_content['/attr/Projection_Longitude'] = '0.0'
        file_content['/attr/Sub_Satellite_Longitude'] = '3.4'
        file_content['/attr/Sensors'] = 'MSG1/SEVIRI'
        file_content['/attr/Zone'] = 'G'
        file_content['/attr/_FillValue'] = 1
        file_content['/attr/scale_factor'] = 1.
        file_content['/attr/add_offset'] = 0.

        # test one IR and one VIS channel
        file_content['Normalized_Radiance'] = DEFAULT_FILE_DATA
        file_content['Normalized_Radiance/attr/_FillValue'] = 1
        file_content['Normalized_Radiance/attr/scale_factor'] = 1.
        file_content['Normalized_Radiance/attr/add_offset'] = 0.
        file_content['Normalized_Radiance/shape'] = DEFAULT_FILE_SHAPE

        file_content['Brightness_Temperature'] = DEFAULT_FILE_DATA
        file_content['Brightness_Temperature/attr/_FillValue'] = 1
        file_content['Brightness_Temperature/attr/scale_factor'] = 1.
        file_content['Brightness_Temperature/attr/add_offset'] = 0.
        file_content['Brightness_Temperature/shape'] = DEFAULT_FILE_SHAPE

        # convert to xarrays
        from xarray import DataArray
        for key, val in file_content.items():
            if isinstance(val, np.ndarray):
                attrs = {}
                for a in ['_FillValue', 'scale_factor', 'add_offset']:
                    if key + '/attr/' + a in file_content:
                        attrs[a] = file_content[key + '/attr/' + a]
                file_content[key] = DataArray(da.from_array(val), dims=('x', 'y'), attrs=attrs)
        return file_content


class TestSEVIRIICAREReader(unittest.TestCase):
    """Test SEVIRI L1b HDF4 from ICARE Reader."""

    yaml_file = 'seviri_l1b_icare.yaml'

    def setUp(self):
        """Wrap HDF4 file handler with own fake file handler."""
        from satpy._config import config_search_paths
        from satpy.readers.seviri_l1b_icare import SEVIRI_ICARE
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        self.p = mock.patch.object(SEVIRI_ICARE, '__bases__', (FakeHDF4FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF4 file handler."""
        self.p.stop()

    def compare_areas(self, v):
        """Compare produced AreaDefinition with expected."""
        test_area = {'area_id': 'geosmsg',
                     'width': 10,
                     'height': 300,
                     'area_extent': (-5567248.2834071,
                                     -5570248.6866857,
                                     -5537244.2506213,
                                     -4670127.7031114)}
        self.assertEqual(v.attrs['area'].area_id, test_area['area_id'])
        self.assertEqual(v.attrs['area'].width, test_area['width'])
        self.assertEqual(v.attrs['area'].height, test_area['height'])
        np.testing.assert_almost_equal(v.attrs['area'].area_extent,
                                       test_area['area_extent'])

    def test_init(self):
        """Test basic init with no extra parameters."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GEO_L1B-MSG1_2004-12-29T12-15-00_G_VIS08_V1-04.hdf',
            'GEO_L1B-MSG1_2004-12-29T12-15-00_G_IR108_V1-04.hdf'
        ])
        self.assertEqual(len(loadables), 2)
        r.create_filehandlers(loadables)
        self.assertTrue(r.file_handlers)

    def test_load_dataset_vis(self):
        """Test loading all datasets from a full swath file."""
        from datetime import datetime
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GEO_L1B-MSG1_2004-12-29T12-15-00_G_VIS08_V1-04.hdf'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['VIS008'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            dt = datetime(2004, 12, 29, 12, 27, 44)
            self.assertEqual(v.attrs['end_time'], dt)
            self.assertEqual(v.attrs['calibration'], 'reflectance')

    def test_load_dataset_ir(self):
        """Test loading all datasets from a full swath file."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GEO_L1B-MSG1_2004-12-29T12-15-00_G_IR108_V1-04.hdf'
        ])
        r.create_filehandlers(loadables)
        datasets = r.load(['IR_108'])
        self.assertEqual(len(datasets), 1)
        for v in datasets.values():
            self.assertEqual(v.attrs['calibration'], 'brightness_temperature')

    def test_area_def_lores(self):
        """Test loading all datasets from an area of interest file."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GEO_L1B-MSG1_2004-12-29T12-15-00_G_VIS08_V1-04.hdf'
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['VIS008'])
        self.compare_areas(ds['VIS008'])
        self.assertEqual(ds['VIS008'].attrs['area'].proj_id, 'msg_lowres')

    def test_area_def_hires(self):
        """Test loading all datasets from an area of interest file."""
        r = load_reader(self.reader_configs)
        loadables = r.select_files_from_pathnames([
            'GEO_L1B-MSG1_2004-12-29T12-15-00_G_HRV_V1-04.hdf',
        ])
        r.create_filehandlers(loadables)
        ds = r.load(['HRV'])
        self.compare_areas(ds['HRV'])
        self.assertEqual(ds['HRV'].attrs['area'].proj_id, 'msg_hires')

    def test_sensor_names(self):
        """Check satellite name conversion is correct, including error case."""
        file_data = FakeHDF4FileHandler2.get_test_content(mock.MagicMock(),
                                                          mock.MagicMock(),
                                                          mock.MagicMock(),
                                                          mock.MagicMock())
        sensor_list = {'Meteosat-08': 'MSG1/SEVIRI',
                       'Meteosat-09': 'MSG2/SEVIRI',
                       'Meteosat-10': 'MSG3/SEVIRI',
                       'Meteosat-11': 'MSG4/SEVIRI'}
        with mock.patch('satpy.tests.reader_tests.test_seviri_l1b_icare.'
                        'FakeHDF4FileHandler2.get_test_content') as patched_func:
            def _run_target():
                patched_func.return_value = file_data
                return self.p.target(mock.MagicMock(),
                                     mock.MagicMock(),
                                     mock.MagicMock()).sensor_name

            for sat in sensor_list:
                file_data['/attr/Sensors'] = sensor_list[sat]
                plat, sens = _run_target()
                self.assertEqual(plat, sat)

            with self.assertRaises(NameError):
                file_data['/attr/Sensors'] = 'BADSAT/NOSENSE'
                plat, sens = _run_target()

    def test_bad_bandname(self):
        """Check reader raises an error if a band bandname is passed."""
        with self.assertRaises(NameError):
            self.p.target(mock.MagicMock(),
                          mock.MagicMock(),
                          mock.MagicMock())._get_dsname({'name': 'badband'})

    def test_nocompute(self):
        """Test that dask does not compute anything in the reader itself."""
        import dask

        from satpy.tests.utils import CustomScheduler
        with dask.config.set(scheduler=CustomScheduler(max_computes=0)):
            r = load_reader(self.reader_configs)
            loadables = r.select_files_from_pathnames([
                'GEO_L1B-MSG1_2004-12-29T12-15-00_G_VIS08_V1-04.hdf'
            ])
            r.create_filehandlers(loadables)
            r.load(['VIS008'])
