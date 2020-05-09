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
"""The ami_l1b reader tests package."""

import numpy as np
import xarray as xr
import dask.array as da

import unittest
from unittest import mock


class FakeDataset(object):
    """Mimic xarray Dataset object."""

    def __init__(self, info, attrs):
        """Initialize test data."""
        for var_name, var_data in list(info.items()):
            if isinstance(var_data, np.ndarray):
                info[var_name] = xr.DataArray(var_data)
        self.info = info
        self.attrs = attrs

    def __getitem__(self, key):
        """Mimic getitem method."""
        return self.info[key]

    def __contains__(self, key):
        """Mimic contains method."""
        return key in self.info

    def rename(self, *args, **kwargs):
        """Mimic rename method."""
        return self

    def close(self):
        """Act like close method."""
        return


class TestAMIL1bNetCDFBase(unittest.TestCase):
    """Common setup for NC_ABI_L1B tests."""

    @mock.patch('satpy.readers.ami_l1b.xr')
    def setUp(self, xr_, counts=None):
        """Create a fake dataset using the given counts data."""
        from satpy.readers.ami_l1b import AMIL1bNetCDF

        if counts is None:
            rad_data = (np.arange(10.).reshape((2, 5)) + 1.) * 50.
            rad_data = (rad_data + 1.) / 0.5
            rad_data = rad_data.astype(np.int16)
            counts = xr.DataArray(
                da.from_array(rad_data, chunks='auto'),
                dims=('y', 'x'),
                attrs={
                    'channel_name': "VI006",
                    'detector_side': 2,
                    'number_of_total_pixels': 484000000,
                    'number_of_error_pixels': 113892451,
                    'max_pixel_value': 32768,
                    'min_pixel_value': 6,
                    'average_pixel_value': 8228.98770845248,
                    'stddev_pixel_value': 13621.130386551,
                    'number_of_total_bits_per_pixel': 16,
                    'number_of_data_quality_flag_bits_per_pixel': 2,
                    'number_of_valid_bits_per_pixel': 12,
                    'data_quality_flag_meaning':
                        "0:good_pixel, 1:conditionally_usable_pixel, 2:out_of_scan_area_pixel, 3:error_pixel",
                    'ground_sample_distance_ew': 1.4e-05,
                    'ground_sample_distance_ns': 1.4e-05,
                }
            )
        sc_position = xr.DataArray(0., attrs={
            'sc_position_center_pixel': [-26113466.1974016, 33100139.1630508, 3943.75470244799],
        })
        xr_.open_dataset.return_value = FakeDataset(
            {
                'image_pixel_values': counts,
                'sc_position': sc_position,
            },
            {
                "satellite_name": "GK-2A",
                "observation_start_time": 623084431.957882,
                "observation_end_time": 623084975.606133,
                "projection_type": "GEOS",
                "sub_longitude": 2.23751210105673,
                "cfac": 81701355.6133574,
                "lfac": -81701355.6133574,
                "coff": 11000.5,
                "loff": 11000.5,
                "nominal_satellite_height": 42164000.,
                "earth_equatorial_radius": 6378137.,
                "earth_polar_radius": 6356752.3,
                "number_of_columns": 22000,
                "number_of_lines": 22000,
                "observation_mode": "FD",
                "channel_spatial_resolution": "0.5",
                "Radiance_to_Albedo_c": 1,
                "DN_to_Radiance_Gain": -0.0144806550815701,
                "DN_to_Radiance_Offset": 118.050903320312,
                "Teff_to_Tbb_c0": -0.141418528203155,
                "Teff_to_Tbb_c1": 1.00052232906885,
                "Teff_to_Tbb_c2": -0.00000036287276076109,
                "light_speed": 2.9979245800E+08,
                "Boltzmann_constant_k": 1.3806488000E-23,
                "Plank_constant_h": 6.6260695700E-34,
            }
        )

        self.reader = AMIL1bNetCDF('filename',
                                   {'platform_shortname': 'gk2a'},
                                   {'filetype': 'info'})


class TestAMIL1bNetCDF(TestAMIL1bNetCDFBase):
    """Test the AMI L1b reader."""

    def _check_orbital_parameters(self, orb_params):
        """Check that orbital parameters match expected values."""
        exp_params = {
            'projection_altitude': 35785863.0,
            'projection_latitude': 0.0,
            'projection_longitude': 128.2,
            'satellite_actual_altitude': 35782654.56070405,
            'satellite_actual_latitude': 0.005364927,
            'satellite_actual_longitude': 128.2707,
        }
        for key, val in exp_params.items():
            self.assertAlmostEqual(val, orb_params[key], places=3)

    def test_filename_grouping(self):
        """Test that filenames are grouped properly."""
        from satpy.readers import group_files
        filenames = [
            'gk2a_ami_le1b_ir087_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_ir096_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_ir105_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_ir112_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_ir123_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_ir133_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_nr013_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_nr016_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_sw038_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_vi004_fd010ge_201909300300.nc',
            'gk2a_ami_le1b_vi005_fd010ge_201909300300.nc',
            'gk2a_ami_le1b_vi006_fd005ge_201909300300.nc',
            'gk2a_ami_le1b_vi008_fd010ge_201909300300.nc',
            'gk2a_ami_le1b_wv063_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_wv069_fd020ge_201909300300.nc',
            'gk2a_ami_le1b_wv073_fd020ge_201909300300.nc']
        groups = group_files(filenames, reader='ami_l1b')
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['ami_l1b']), 16)

    def test_basic_attributes(self):
        """Test getting basic file attributes."""
        from datetime import datetime
        self.assertEqual(self.reader.start_time,
                         datetime(2019, 9, 30, 3, 0, 31, 957882))
        self.assertEqual(self.reader.end_time,
                         datetime(2019, 9, 30, 3, 9, 35, 606133))

    def test_get_dataset(self):
        """Test gettting radiance data."""
        from satpy import DatasetID
        key = DatasetID(name='VI006', calibration='radiance')
        res = self.reader.get_dataset(key, {
            'file_key': 'image_pixel_values',
            'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
            'units': 'W m-2 um-1 sr-1',
        })
        exp = {'calibration': 'radiance',
               'modifiers': (),
               'platform_name': 'GEO-KOMPSAT-2A',
               'sensor': 'ami',
               'units': 'W m-2 um-1 sr-1'}
        for key, val in exp.items():
            self.assertEqual(val, res.attrs[key])
        self._check_orbital_parameters(res.attrs['orbital_parameters'])

    def test_bad_calibration(self):
        """Test that asking for a bad calibration fails."""
        from satpy import DatasetID
        self.assertRaises(ValueError, self.reader.get_dataset,
                          DatasetID(name='VI006', calibration='_bad_'),
                          {'file_key': 'image_pixel_values',
                           'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
                           'units': 'W m-2 um-1 sr-1',
                           })

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def(self, adef):
        """Test the area generation."""
        self.reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        exp = {'a': 6378137.0, 'b': 6356752.3, 'h': 35785863.0,
               'lon_0': 128.2, 'proj': 'geos', 'units': 'm'}
        for key, val in exp.items():
            self.assertIn(key, call_args[3])
            self.assertAlmostEqual(val, call_args[3][key])
        self.assertEqual(call_args[4], self.reader.nc.attrs['number_of_columns'])
        self.assertEqual(call_args[5], self.reader.nc.attrs['number_of_lines'])
        np.testing.assert_allclose(call_args[6],
                                   [-5511022.902, -5511022.902, 5511022.902, 5511022.902])

    def test_get_dataset_vis(self):
        """Test get visible calibrated data."""
        from satpy import DatasetID
        key = DatasetID(name='VI006', calibration='reflectance')
        res = self.reader.get_dataset(key, {
            'file_key': 'image_pixel_values',
            'standard_name': 'toa_bidirectional_reflectance',
            'units': '%',
        })
        exp = {'calibration': 'reflectance',
               'modifiers': (),
               'platform_name': 'GEO-KOMPSAT-2A',
               'sensor': 'ami',
               'units': '%'}
        for key, val in exp.items():
            self.assertEqual(val, res.attrs[key])
        self._check_orbital_parameters(res.attrs['orbital_parameters'])

    def test_get_dataset_counts(self):
        """Test get counts data."""
        from satpy import DatasetID
        key = DatasetID(name='VI006', calibration='counts')
        res = self.reader.get_dataset(key, {
            'file_key': 'image_pixel_values',
            'standard_name': 'counts',
            'units': '1',
        })
        exp = {'calibration': 'counts',
               'modifiers': (),
               'platform_name': 'GEO-KOMPSAT-2A',
               'sensor': 'ami',
               'units': '1'}
        for key, val in exp.items():
            self.assertEqual(val, res.attrs[key])
        self._check_orbital_parameters(res.attrs['orbital_parameters'])


class TestAMIL1bNetCDFIRCal(TestAMIL1bNetCDFBase):
    """Test IR specific things about the AMI reader."""

    def setUp(self):
        """Create test data for IR calibration tests."""
        count_data = (np.arange(10).reshape((2, 5))) + 7000
        count_data = count_data.astype(np.uint16)
        count = xr.DataArray(
            da.from_array(count_data, chunks='auto'),
            dims=('y', 'x'),
            attrs={
                'channel_name': "IR087",
                'detector_side': 2,
                'number_of_total_pixels': 484000000,
                'number_of_error_pixels': 113892451,
                'max_pixel_value': 32768,
                'min_pixel_value': 6,
                'average_pixel_value': 8228.98770845248,
                'stddev_pixel_value': 13621.130386551,
                'number_of_total_bits_per_pixel': 16,
                'number_of_data_quality_flag_bits_per_pixel': 2,
                'number_of_valid_bits_per_pixel': 13,
                'data_quality_flag_meaning':
                    "0:good_pixel, 1:conditionally_usable_pixel, 2:out_of_scan_area_pixel, 3:error_pixel",
                'ground_sample_distance_ew': 1.4e-05,
                'ground_sample_distance_ns': 1.4e-05,
            }
        )
        super(TestAMIL1bNetCDFIRCal, self).setUp(counts=count)

    def test_ir_calibrate(self):
        """Test IR calibration."""
        from satpy import DatasetID
        from satpy.readers.ami_l1b import rad2temp
        ds_id = DatasetID(name='IR087', wavelength=[8.415, 8.59, 8.765],
                          calibration='brightness_temperature')
        ds_info = {
            'file_key': 'image_pixel_values',
            'wavelength': [8.415, 8.59, 8.765],
            'standard_name': 'toa_brightness_temperature',
            'units': 'K',
        }
        with mock.patch('satpy.readers.ami_l1b.rad2temp', wraps=rad2temp) as r2t_mock:
            res = self.reader.get_dataset(ds_id, ds_info)
            r2t_mock.assert_called_once()
        expected = np.array([[238.34385135, 238.31443527, 238.28500087, 238.25554813, 238.22607701],
                             [238.1965875, 238.16707956, 238.13755317, 238.10800829, 238.07844489]])
        np.testing.assert_allclose(res.data.compute(), expected, equal_nan=True)
        # make sure the attributes from the file are in the data array
        self.assertEqual(res.attrs['standard_name'], 'toa_brightness_temperature')

        # test builtin coefficients
        self.reader.calib_mode = 'FILE'
        with mock.patch('satpy.readers.ami_l1b.rad2temp', wraps=rad2temp) as r2t_mock:
            res = self.reader.get_dataset(ds_id, ds_info)
            r2t_mock.assert_not_called()
        # file coefficients are pretty close, give some wiggle room
        np.testing.assert_allclose(res.data.compute(), expected, equal_nan=True, atol=0.04)
        # make sure the attributes from the file are in the data array
        self.assertEqual(res.attrs['standard_name'], 'toa_brightness_temperature')
