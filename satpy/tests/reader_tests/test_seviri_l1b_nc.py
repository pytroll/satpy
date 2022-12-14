#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2020 Satpy developers
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

from datetime import datetime
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from satpy.readers.seviri_l1b_nc import NCSEVIRIFileHandler
from satpy.tests.reader_tests.test_seviri_base import ORBIT_POLYNOMIALS
from satpy.tests.reader_tests.test_seviri_l1b_calibration import TestFileHandlerCalibrationBase
from satpy.tests.utils import assert_attrs_equal, make_dataid

channel_keys_dict = {'VIS006': 'ch1', 'IR_108': 'ch9'}


def to_cds_time(time):
    """Convert datetime to (days, msecs) since 1958-01-01."""
    if isinstance(time, datetime):
        time = np.datetime64(time)
    t0 = np.datetime64('1958-01-01 00:00')
    delta = time - t0
    days = (delta / np.timedelta64(1, 'D')).astype(int)
    msecs = delta / np.timedelta64(1, 'ms') - days * 24 * 3600 * 1E3
    return days, msecs


class TestNCSEVIRIFileHandler(TestFileHandlerCalibrationBase):
    """Unit tests for SEVIRI netCDF reader."""

    def _get_fake_dataset(self, counts, h5netcdf):
        """Create a fake dataset.

        Args:
            counts (xr.DataArray):
                Array with data.
            h5netcdf (boolean):
                If True an array attribute will be created which is common
                for the h5netcdf backend in xarray for scalar values.

        """
        acq_time_day = np.repeat([1, 1], 11).reshape(2, 11)
        acq_time_msec = np.repeat([1000, 2000], 11).reshape(2, 11)
        line_validity = np.repeat([3, 3], 11).reshape(2, 11)
        line_geom_radio_quality = np.repeat([4, 4], 11).reshape(2, 11)
        orbit_poly_start_day, orbit_poly_start_msec = to_cds_time(
            np.array([datetime(2019, 12, 31, 18),
                      datetime(2019, 12, 31, 22)],
                     dtype='datetime64')
        )
        orbit_poly_end_day, orbit_poly_end_msec = to_cds_time(
            np.array([datetime(2019, 12, 31, 22),
                      datetime(2020, 1, 1, 2)],
                     dtype='datetime64')
        )
        counts = counts.rename({
            'y': 'num_rows_vis_ir',
            'x': 'num_columns_vis_ir'
        })
        scan_time_days, scan_time_msecs = to_cds_time(self.scan_time)
        ds = xr.Dataset(
            {
                'ch1': counts.copy(),
                'ch9': counts.copy(),
                'HRV': (('num_rows_hrv', 'num_columns_hrv'), [[1, 2, 3],
                                                              [4, 5, 6],
                                                              [7, 8, 9]]),
                'planned_chan_processing': self.radiance_types,
                'channel_data_visir_data_l10_line_mean_acquisition_time_day': (
                    ('num_rows_vis_ir', 'channels_vis_ir_dim'),
                    acq_time_day
                ),
                'channel_data_visir_data_l10_line_mean_acquisition_msec': (
                    ('num_rows_vis_ir', 'channels_vis_ir_dim'),
                    acq_time_msec
                ),
                'channel_data_visir_data_line_validity': (
                    ('num_rows_vis_ir', 'channels_vis_ir_dim'),
                    line_validity
                ),
                'channel_data_visir_data_line_geometric_quality': (
                    ('num_rows_vis_ir', 'channels_vis_ir_dim'),
                    line_geom_radio_quality
                ),
                'channel_data_visir_data_line_radiometric_quality': (
                    ('num_rows_vis_ir', 'channels_vis_ir_dim'),
                    line_geom_radio_quality
                ),
                'orbit_polynomial_x': (
                    ('orbit_polynomial_dim_row',
                     'orbit_polynomial_dim_col'),
                    ORBIT_POLYNOMIALS['X'][0:2]
                ),
                'orbit_polynomial_y': (
                    ('orbit_polynomial_dim_row',
                     'orbit_polynomial_dim_col'),
                    ORBIT_POLYNOMIALS['Y'][0:2]
                ),
                'orbit_polynomial_z': (
                    ('orbit_polynomial_dim_row',
                     'orbit_polynomial_dim_col'),
                    ORBIT_POLYNOMIALS['Z'][0:2]
                ),
                'orbit_polynomial_start_time_day': (
                    'orbit_polynomial_dim_row',
                    orbit_poly_start_day
                ),
                'orbit_polynomial_start_time_msec': (
                    'orbit_polynomial_dim_row',
                    orbit_poly_start_msec
                ),
                'orbit_polynomial_end_time_day': (
                    'orbit_polynomial_dim_row',
                    orbit_poly_end_day
                ),
                'orbit_polynomial_end_time_msec': (
                    'orbit_polynomial_dim_row',
                    orbit_poly_end_msec
                ),
            },
            attrs={
                'equatorial_radius': 6378.169,
                'north_polar_radius': 6356.5838,
                'south_polar_radius': 6356.5838,
                'longitude_of_SSP': 0.0,
                'nominal_longitude': -3.5,
                'satellite_id': self.platform_id,
                'true_repeat_cycle_start_day': scan_time_days,
                'true_repeat_cycle_start_mi_sec': scan_time_msecs,
                'planned_repeat_cycle_end_day': scan_time_days,
                'planned_repeat_cycle_end_mi_sec': scan_time_msecs,
                'north_most_line': 3712,
                'east_most_pixel': 1,
                'west_most_pixel': 3712,
                'south_most_line': 1,
                'vis_ir_grid_origin': 0,
                'vis_ir_column_dir_grid_step': 3.0004032,
                'vis_ir_line_dir_grid_step': 3.0004032,
                'type_of_earth_model': '0x02',
            }
        )

        if h5netcdf:
            nattrs = {'equatorial_radius': np.array([6378.169]),
                      'north_polar_radius': np.array([6356.5838]),
                      'south_polar_radius': np.array([6356.5838]),
                      'longitude_of_SSP': np.array([0.0]),
                      'vis_ir_column_dir_grid_step': np.array([3.0004032]),
                      'vis_ir_line_dir_grid_step': np.array([3.0004032])
                      }

            ds.attrs.update(nattrs)

        ds['ch1'].attrs.update({
            'scale_factor': self.gains_nominal[0],
            'add_offset': self.offsets_nominal[0]
        })
        # IR_108 is dataset with key ch9
        ds['ch9'].attrs.update({
            'scale_factor': self.gains_nominal[8],
            'add_offset': self.offsets_nominal[8],
        })

        # Add some attributes so that the reader can strip them
        strip_attrs = {
            'comment': None,
            'long_name': None,
            'valid_min': None,
            'valid_max': None
        }
        for name in ['ch1', 'ch9']:
            ds[name].attrs.update(strip_attrs)

        return ds

    @pytest.fixture
    def h5netcdf(self):
        """Fixture for xr backend choice."""
        return False

    @pytest.fixture(name='file_handler')
    def file_handler(self, counts, h5netcdf):
        """Create a mocked file handler."""
        with mock.patch(
            'satpy.readers.seviri_l1b_nc.open_dataset',
            return_value=self._get_fake_dataset(counts=counts, h5netcdf=h5netcdf)
        ):
            return NCSEVIRIFileHandler(
                'filename',
                {'platform_shortname': 'MSG3',
                 'start_time': self.scan_time,
                 'service': 'MSG'},
                {'filetype': 'info'}
            )

    @pytest.mark.parametrize(
        ('channel', 'calibration', 'use_ext_coefs'),
        [
            # VIS channel, internal coefficients
            ('VIS006', 'counts', False),
            ('VIS006', 'radiance', False),
            ('VIS006', 'reflectance', False),
            # VIS channel, external coefficients
            ('VIS006', 'radiance', True),
            ('VIS006', 'reflectance', True),
            # IR channel, internal coefficients
            ('IR_108', 'counts', False),
            ('IR_108', 'radiance', False),
            ('IR_108', 'brightness_temperature', False),
            # IR channel, external coefficients
            ('IR_108', 'radiance', True),
            ('IR_108', 'brightness_temperature', True),
            # FUTURE: Enable once HRV reading has been fixed.
            # # HRV channel, internal coefficiens
            # ('HRV', 'counts', False),
            # ('HRV', 'radiance', False),
            # ('HRV', 'reflectance', False),
            # # HRV channel, external coefficients (mode should have no effect)
            # ('HRV', 'radiance', True),
            # ('HRV', 'reflectance', True),
        ]
    )
    def test_calibrate(
            self, file_handler, channel, calibration, use_ext_coefs
    ):
        """Test the calibration."""
        external_coefs = self.external_coefs if use_ext_coefs else {}
        expected = self._get_expected(
            channel=channel,
            calibration=calibration,
            calib_mode='NOMINAL',
            use_ext_coefs=use_ext_coefs
        )
        fh = file_handler
        fh.ext_calib_coefs = external_coefs
        dataset_id = make_dataid(name=channel, calibration=calibration)

        key = channel_keys_dict[channel]

        res = fh.calibrate(fh.nc[key], dataset_id)
        xr.testing.assert_allclose(res, expected)

    def test_mask_bad_quality(self, file_handler):
        """Test masking of bad quality scan lines."""
        channel = 'VIS006'
        key = channel_keys_dict[channel]
        dataset_info = {
            'nc_key': key,
            'units': 'units',
            'wavelength': 'wavelength',
            'standard_name': 'standard_name'
        }
        expected = self._get_expected(
            channel=channel,
            calibration='radiance',
            calib_mode='NOMINAL',
            use_ext_coefs=False
        )

        fh = file_handler

        res = fh._mask_bad_quality(fh.nc[key], dataset_info)
        new_data = np.zeros_like(expected.data).astype('float32')
        new_data[:, :] = np.nan
        expected = expected.copy(data=new_data)
        xr.testing.assert_allclose(res, expected)

    @pytest.mark.parametrize(
        ('channel', 'calibration', 'mask_bad_quality_scan_lines'),
        [
            ('VIS006', 'reflectance', True),
            ('VIS006', 'reflectance', False),
            ('IR_108', 'brightness_temperature', True)
         ]
    )
    def test_get_dataset(self, file_handler, channel, calibration, mask_bad_quality_scan_lines):
        """Test getting the dataset."""
        dataset_id = make_dataid(name=channel, calibration=calibration)
        key = channel_keys_dict[channel]
        dataset_info = {
            'nc_key': key,
            'units': 'units',
            'wavelength': 'wavelength',
            'standard_name': 'standard_name'
        }

        file_handler.mask_bad_quality_scan_lines = mask_bad_quality_scan_lines
        res = file_handler.get_dataset(dataset_id, dataset_info)

        # Test scanline acquisition times
        expected = self._get_expected(
            channel=channel,
            calibration=calibration,
            calib_mode='NOMINAL',
            use_ext_coefs=False
        )
        expected.attrs = {
            'orbital_parameters': {
                'satellite_actual_longitude': -3.541742131915741,
                'satellite_actual_latitude': -0.5203765167594427,
                'satellite_actual_altitude': 35783419.16135868,
                'satellite_nominal_longitude': -3.5,
                'satellite_nominal_latitude': 0.0,
                'projection_longitude': 0.0,
                'projection_latitude': 0.0,
                'projection_altitude': 35785831.0
            },
            'georef_offset_corrected': True,
            'platform_name': 'Meteosat-11',
            'sensor': 'seviri',
            'units': 'units',
            'wavelength': 'wavelength',
            'standard_name': 'standard_name'
        }
        expected['acq_time'] = ('y', [np.datetime64('1958-01-02 00:00:01'),
                                      np.datetime64('1958-01-02 00:00:02')])
        expected = expected[::-1]  # reader flips data upside down
        if mask_bad_quality_scan_lines:
            expected = file_handler._mask_bad_quality(expected, dataset_info)

        xr.testing.assert_allclose(res, expected)

        for key in ['sun_earth_distance_correction_applied',
                    'sun_earth_distance_correction_factor']:
            res.attrs.pop(key, None)
        assert_attrs_equal(res.attrs, expected.attrs, tolerance=1e-4)

    def test_satpos_no_valid_orbit_polynomial(self, file_handler):
        """Test satellite position if there is no valid orbit polynomial."""
        dataset_id = make_dataid(name='VIS006', calibration='counts')
        dataset_info = {
            'name': 'VIS006',
            'nc_key': 'ch1',
            'units': 'units',
            'wavelength': 'wavelength',
            'standard_name': 'standard_name'
        }
        file_handler.nc['orbit_polynomial_start_time_day'] = 0
        file_handler.nc['orbit_polynomial_end_time_day'] = 0
        res = file_handler.get_dataset(dataset_id, dataset_info)
        assert 'satellite_actual_longitude' not in res.attrs[
            'orbital_parameters']

    @pytest.mark.parametrize('h5netcdf', [True])
    def test_h5netcdf_pecularity(self, file_handler, h5netcdf):
        """Test conversion of attributes when xarray is used with h5netcdf backend."""
        fh = file_handler
        assert isinstance(fh.mda['projection_parameters']['a'], float)
