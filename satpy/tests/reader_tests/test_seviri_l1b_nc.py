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
from satpy.tests.reader_tests.test_seviri_l1b_calibration import (
    TestFileHandlerCalibrationBase
)
from satpy.tests.reader_tests.test_seviri_base import ORBIT_POLYNOMIALS
from satpy.tests.utils import make_dataid


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

    def _get_fake_dataset(self, counts):
        """Create a fake dataset."""
        acq_time_day = np.repeat([1, 1], 11).reshape(2, 11)
        acq_time_msec = np.repeat([1000, 2000], 11).reshape(2, 11)
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
                'VIS006': counts.copy(),
                'IR_108': counts.copy(),
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
                'vis_ir_column_dir_grid_step': 3.0004032,
                'vis_ir_line_dir_grid_step': 3.0004032,
                'type_of_earth_model': '0x02',
            }
        )
        ds['VIS006'].attrs.update({
            'scale_factor': self.gains_nominal[0],
            'add_offset': self.offsets_nominal[0]
        })
        ds['IR_108'].attrs.update({
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
        for name in ['VIS006', 'IR_108']:
            ds[name].attrs.update(strip_attrs)

        return ds

    @pytest.fixture(name='file_handler')
    def file_handler(self, counts):
        """Create a mocked file handler."""
        with mock.patch(
            'satpy.readers.seviri_l1b_nc.xr.open_dataset',
            return_value=self._get_fake_dataset(counts)
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
        file_handler.nc = file_handler.nc.rename({
            'num_rows_vis_ir': 'y',
            'num_columns_vis_ir': 'x'
        })
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

        res = fh.calibrate(fh.nc[channel], dataset_id)
        xr.testing.assert_allclose(res, expected)

    @pytest.mark.parametrize(
        ('channel', 'calibration'),
        [
            ('VIS006', 'reflectance'),
            ('IR_108', 'brightness_temperature')
         ]
    )
    def test_get_dataset(self, file_handler, channel, calibration):
        """Test getting the dataset."""
        dataset_id = make_dataid(name=channel, calibration=calibration)
        dataset_info = {
            'nc_key': channel,
            'units': 'units',
            'wavelength': 'wavelength',
            'standard_name': 'standard_name'
        }
        res = file_handler.get_dataset(dataset_id, dataset_info)

        # Test scanline acquisition times
        xr.testing.assert_allclose(
            res['acq_time'].drop_vars('acq_time'),
            xr.DataArray(
                [datetime(1958, 1, 2, 0, 0, 2),
                 datetime(1958, 1, 2, 0, 0, 1)],
                dims='y'
            )
        )

        # Test attributes
        orb_params = res.attrs['orbital_parameters']
        assert orb_params['projection_longitude'] == 0.0
        assert orb_params['projection_latitude'] == 0.0
        assert orb_params['projection_altitude'] == 35785831.0
        assert orb_params['satellite_nominal_longitude'] == -3.5
        assert orb_params['satellite_nominal_latitude'] == 0.0
        assert orb_params['satellite_actual_longitude'] == pytest.approx(
            -3.541742131915741
        )
        assert orb_params['satellite_actual_latitude'] == pytest.approx(
            -0.5203765167594427
        )
        assert orb_params['satellite_actual_altitude'] == pytest.approx(
            35783419.16135868
        )
        assert res.attrs['georef_offset_corrected'] is True
        strip_attrs = ["comment", "long_name", "nc_key", "scale_factor",
                       "add_offset", "valid_min", "valid_max"]
        assert not any([k in res.attrs.keys() for k in strip_attrs])
