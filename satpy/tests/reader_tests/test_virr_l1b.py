#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Satpy developers
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
"""Test for readers/virr_l1b.py.
"""
from satpy.tests.reader_tests.test_hdf5_utils import FakeHDF5FileHandler
import sys
import numpy as np
import dask.array as da
import xarray as xr
import os

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class FakeHDF5FileHandler2(FakeHDF5FileHandler):
    """Swap-in HDF5 File Handler."""

    def make_test_data(self, dims):
        return xr.DataArray(da.from_array(np.ones([dim for dim in dims], dtype=np.float32) * 10, [dim for dim in dims]))

    def _make_file(self, platform_id, geolocation_prefix, l1b_prefix, ECWN, Emissive_units):
        dim_0 = 19
        dim_1 = 20
        test_file = {
            # Satellite data.
            '/attr/Day Or Night Flag': 'D', '/attr/Observing Beginning Date': '2018-12-25',
            '/attr/Observing Beginning Time': '21:41:47.090', '/attr/Observing Ending Date': '2018-12-25',
            '/attr/Observing Ending Time': '21:47:28.254', '/attr/Satellite Name': platform_id,
            '/attr/Sensor Identification Code': 'VIRR',
            # Emissive data.
            l1b_prefix + 'EV_Emissive': self.make_test_data([3, dim_0, dim_1]),
            l1b_prefix + 'EV_Emissive/attr/valid_range': [0, 50000],
            l1b_prefix + 'Emissive_Radiance_Scales': self.make_test_data([dim_0, dim_1]),
            l1b_prefix + 'EV_Emissive/attr/units': Emissive_units,
            l1b_prefix + 'Emissive_Radiance_Offsets': self.make_test_data([dim_0, dim_1]),
            '/attr/' + ECWN: [2610.31, 917.6268, 836.2546],
            # Reflectance data.
            l1b_prefix + 'EV_RefSB': self.make_test_data([7, dim_0, dim_1]),
            l1b_prefix + 'EV_RefSB/attr/valid_range': [0, 32767], l1b_prefix + 'EV_RefSB/attr/units': 'none',
            '/attr/RefSB_Cal_Coefficients': np.ones(14, dtype=np.float32) * 2
        }
        for attribute in ['Latitude', 'Longitude', geolocation_prefix + 'SolarZenith',
                          geolocation_prefix + 'SensorZenith', geolocation_prefix + 'SolarAzimuth',
                          geolocation_prefix + 'SensorAzimuth']:
            test_file[attribute] = self.make_test_data([dim_0, dim_1])
            test_file[attribute + '/attr/Intercept'] = 0.
            test_file[attribute + '/attr/units'] = 'degrees'
            if 'Solar' in attribute or 'Sensor' in attribute:
                test_file[attribute + '/attr/Slope'] = .01
                if 'Azimuth' in attribute:
                    test_file[attribute + '/attr/valid_range'] = [0, 18000]
                else:
                    test_file[attribute + '/attr/valid_range'] = [-18000, 18000]
            else:
                test_file[attribute + '/attr/Slope'] = 1.
                if 'Longitude' == attribute:
                    test_file[attribute + '/attr/valid_range'] = [-180., 180.]
                else:
                    test_file[attribute + '/attr/valid_range'] = [-90., 90.]
        return test_file

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content."""
        if filename_info['platform_id'] == 'FY3B':
            return self._make_file('FY3B', '', '', 'Emmisive_Centroid_Wave_Number', 'milliWstts/m^2/cm^(-1)/steradian')
        return self._make_file(filename_info['platform_id'], 'Geolocation/', 'Data/',
                               'Emissive_Centroid_Wave_Number', 'none')


class TestVIRRL1BReader(unittest.TestCase):
    """Test VIRR L1B Reader."""
    yaml_file = "virr_l1b.yaml"

    def setUp(self):
        """Wrap HDF5 file handler with our own fake handler."""
        from satpy.readers.virr_l1b import VIRR_L1B
        from satpy.config import config_search_paths
        self.reader_configs = config_search_paths(os.path.join('readers', self.yaml_file))
        # http://stackoverflow.com/questions/12219967/how-to-mock-a-base-class-with-python-mock-library
        self.p = mock.patch.object(VIRR_L1B, '__bases__', (FakeHDF5FileHandler2,))
        self.fake_handler = self.p.start()
        self.p.is_local = True

    def tearDown(self):
        """Stop wrapping the HDF5 file handler."""
        self.p.stop()

    def _band_helper(self, attributes, units, calibration, standard_name,
                     file_type, band_index_size, resolution):
        self.assertEqual(units, attributes['units'])
        self.assertEqual(calibration, attributes['calibration'])
        self.assertEqual(standard_name, attributes['standard_name'])
        self.assertEqual(file_type, attributes['file_type'])
        self.assertTrue(attributes['band_index'] in range(band_index_size))
        self.assertEqual(resolution, attributes['resolution'])
        self.assertEqual(('longitude', 'latitude'), attributes['coordinates'])

    def _fy3_helper(self, platform_name, reader, Emissive_units):
        import datetime
        band_values = {'R1': 22.0, 'R2': 22.0, 'R3': 22.0, 'R4': 22.0, 'R5': 22.0, 'R6': 22.0, 'R7': 22.0,
                       'E1': 496.542155, 'E2': 297.444511, 'E3': 288.956557, 'solar_zenith_angle': .1,
                       'satellite_zenith_angle': .1, 'solar_azimuth_angle': .1, 'satellite_azimuth_angle': .1,
                       'longitude': 10}
        datasets = reader.load([band for band in band_values])
        for dataset in datasets:
            # Object returned by get_dataset.
            ds = datasets[dataset.name]
            attributes = ds.attrs
            self.assertTrue(isinstance(ds.data, da.Array))
            self.assertEqual('VIRR', attributes['sensor'])
            self.assertEqual(platform_name, attributes['platform_name'])
            self.assertEqual(datetime.datetime(2018, 12, 25, 21, 41, 47, 90000), attributes['start_time'])
            self.assertEqual(datetime.datetime(2018, 12, 25, 21, 47, 28, 254000), attributes['end_time'])
            self.assertEqual((19, 20), datasets[dataset.name].shape)
            self.assertEqual(('y', 'x'), datasets[dataset.name].dims)
            if 'R' in dataset.name:
                self._band_helper(attributes, '%', 'reflectance',
                                  'toa_bidirectional_reflectance', 'virr_l1b',
                                  7, 1000)
            elif 'E' in dataset.name:
                self._band_helper(attributes, Emissive_units, 'brightness_temperature',
                                  'toa_brightness_temperature', 'virr_l1b', 3, 1000)
            elif dataset.name in ['longitude', 'latitude']:
                self.assertEqual('degrees', attributes['units'])
                self.assertTrue(attributes['standard_name'] in ['longitude', 'latitude'])
                self.assertEqual(['virr_l1b', 'virr_geoxx'], attributes['file_type'])
                self.assertEqual(1000, attributes['resolution'])
            else:
                self.assertEqual('degrees', attributes['units'])
                self.assertTrue(
                    attributes['standard_name'] in ['solar_zenith_angle', 'sensor_zenith_angle', 'solar_azimuth_angle',
                                                    'sensor_azimuth_angle'])
                self.assertEqual(['virr_geoxx', 'virr_l1b'], attributes['file_type'])
                self.assertEqual(('longitude', 'latitude'), attributes['coordinates'])
            self.assertEqual(band_values[dataset.name],
                             round(float(np.array(ds[ds.shape[0] // 2][ds.shape[1] // 2])), 6))

    def test_fy3b_file(self):
        from satpy.readers import load_reader
        FY3B_reader = load_reader(self.reader_configs)
        FY3B_file = FY3B_reader.select_files_from_pathnames(['tf2018359214943.FY3B-L_VIRRX_L1B.HDF'])
        self.assertTrue(1, len(FY3B_file))
        FY3B_reader.create_filehandlers(FY3B_file)
        # Make sure we have some files
        self.assertTrue(FY3B_reader.file_handlers)
        self._fy3_helper('FY3B', FY3B_reader, 'milliWstts/m^2/cm^(-1)/steradian')

    def test_fy3c_file(self):
        from satpy.readers import load_reader
        FY3C_reader = load_reader(self.reader_configs)
        FY3C_files = FY3C_reader.select_files_from_pathnames(['tf2018359143912.FY3C-L_VIRRX_GEOXX.HDF',
                                                              'tf2018359143912.FY3C-L_VIRRX_L1B.HDF'])
        self.assertTrue(2, len(FY3C_files))
        FY3C_reader.create_filehandlers(FY3C_files)
        # Make sure we have some files
        self.assertTrue(FY3C_reader.file_handlers)
        self._fy3_helper('FY3C', FY3C_reader, '1')


def suite():
    """The test suite for test_virr_l1b."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestVIRRL1BReader))
    return mysuite
