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
"""Test fixed bugs."""


from unittest.mock import patch

import dask.array as da
import numpy as np
from xarray import DataArray, Dataset

from satpy.tests.utils import make_dataid

abi_file_list = ['/data/OR_ABI-L1b-RadF-M3C01_G16_s20180722030423_e20180722041189_c20180722041235-118900_0.nc',
                 '/data/OR_ABI-L1b-RadF-M3C02_G16_s20180722030423_e20180722041190_c20180722041228-120000_0.nc',
                 '/data/OR_ABI-L1b-RadF-M3C03_G16_s20180722030423_e20180722041190_c20180722041237-119000_0.nc',
                 '/data/OR_ABI-L1b-RadF-M3C04_G16_s20180722030423_e20180722041189_c20180722041221.nc',
                 '/data/OR_ABI-L1b-RadF-M3C05_G16_s20180722030423_e20180722041190_c20180722041237-119101_0.nc',
                 '/data/OR_ABI-L1b-RadF-M3C06_G16_s20180722030423_e20180722041195_c20180722041227.nc',
                 '/data/OR_ABI-L1b-RadF-M3C07_G16_s20180722030423_e20180722041201_c20180722041238.nc',
                 '/data/OR_ABI-L1b-RadF-M3C08_G16_s20180722030423_e20180722041190_c20180722041238.nc',
                 '/data/OR_ABI-L1b-RadF-M3C09_G16_s20180722030423_e20180722041195_c20180722041256.nc',
                 '/data/OR_ABI-L1b-RadF-M3C10_G16_s20180722030423_e20180722041201_c20180722041250.nc',
                 '/data/OR_ABI-L1b-RadF-M3C11_G16_s20180722030423_e20180722041189_c20180722041254.nc',
                 '/data/OR_ABI-L1b-RadF-M3C12_G16_s20180722030423_e20180722041195_c20180722041256.nc',
                 '/data/OR_ABI-L1b-RadF-M3C13_G16_s20180722030423_e20180722041201_c20180722041259.nc',
                 '/data/OR_ABI-L1b-RadF-M3C14_G16_s20180722030423_e20180722041190_c20180722041258.nc',
                 '/data/OR_ABI-L1b-RadF-M3C15_G16_s20180722030423_e20180722041195_c20180722041259.nc',
                 '/data/OR_ABI-L1b-RadF-M3C16_G16_s20180722030423_e20180722041202_c20180722041259.nc']


def generate_fake_abi_xr_dataset(filename, chunks=None, **kwargs):
    """Create a fake xarray dataset for abi data.

    This is an incomplete copy of existing file structures.
    """
    dataset = Dataset(attrs={
        'time_coverage_start': '2018-03-13T20:30:42.3Z',
        'time_coverage_end': '2018-03-13T20:41:18.9Z',
    })

    projection = DataArray(
        [-214748364],
        attrs={
            'long_name': 'GOES-R ABI fixed grid projection',
            'grid_mapping_name': 'geostationary',
            'perspective_point_height': 35786023.0,
            'semi_major_axis': 6378137.0,
            'semi_minor_axis': 6356752.31414,
            'inverse_flattening': 298.2572221,
            'latitude_of_projection_origin': 0.0,
            'longitude_of_projection_origin': -75.0,
            'sweep_angle_axis': 'x'
        })
    dataset['goes_imager_projection'] = projection

    if 'C01' in filename or 'C03' in filename or 'C05' in filename:
        stop = 10847
        step = 2
        scale = 2.8e-05
        offset = 0.151858
    elif 'C02' in filename:
        stop = 21693
        step = 4
        scale = 1.4e-05
        offset = 0.151865
    else:
        stop = 5424
        step = 1
        scale = 5.6e-05
        offset = 0.151844

    y = DataArray(
        da.arange(0, stop, step),
        attrs={
            'scale_factor': -scale,
            'add_offset': offset,
            'units': 'rad',
            'axis': 'Y',
            'long_name': 'GOES fixed grid projection y-coordinate',
            'standard_name': 'projection_y_coordinate'
            },
        dims=['y'])

    dataset['y'] = y

    x = DataArray(
        da.arange(0, stop, step),
        attrs={
            'scale_factor': scale,
            'add_offset': -offset,
            'units': 'rad',
            'axis': 'X',
            'long_name': 'GOES fixed grid projection x-coordinate',
            'standard_name': 'projection_x_coordinate'
        },
        dims=['x'])

    dataset['x'] = x

    rad = DataArray(
        da.random.randint(0, 1025, size=[len(y), len(x)], dtype=np.int16, chunks=chunks),
        attrs={
            '_FillValue': np.array(1023),
            'long_name': 'ABI L1b Radiances',
            'standard_name': 'toa_outgoing_radiance_per_unit_wavelength',
            '_Unsigned': 'true',
            'sensor_band_bit_depth': 10,
            'valid_range': np.array([0, 1022], dtype=np.int16),
            'scale_factor': 0.8121064,
            'add_offset': -25.936647,
            'units': 'W m-2 sr-1 um-1',
            'resolution': 'y: 0.000028 rad x: 0.000028 rad',
            'grid_mapping': 'goes_imager_projection',
            'cell_methods': 't: point area: point'
        },
        dims=['y', 'x']
    )

    dataset['Rad'] = rad

    sublat = DataArray(0.0, attrs={
        'long_name': 'nominal satellite subpoint latitude (platform latitude)',
        'standard_name': 'latitude',
        '_FillValue': -999.0,
        'units': 'degrees_north'})
    dataset['nominal_satellite_subpoint_lat'] = sublat

    sublon = DataArray(-75.0, attrs={
        'long_name': 'nominal satellite subpoint longitude (platform longitude)',
        'standard_name': 'longitude',
        '_FillValue': -999.0,
        'units': 'degrees_east'})

    dataset['nominal_satellite_subpoint_lon'] = sublon

    satheight = DataArray(35786.023, attrs={
        'long_name': 'nominal satellite height above GRS 80 ellipsoid (platform altitude)',
        'standard_name': 'height_above_reference_ellipsoid',
        '_FillValue': -999.0,
        'units': 'km'})

    dataset['nominal_satellite_height'] = satheight

    yaw_flip_flag = DataArray(0, attrs={
        'long_name': 'Flag indicating the spacecraft is operating in yaw flip configuration',
        '_Unsigned': 'true',
        '_FillValue': np.array(-1),
        'valid_range': np.array([0, 1], dtype=np.int8),
        'units': '1',
        'flag_values': '0 1',
        'flag_meanings': 'false true'})

    dataset['yaw_flip_flag'] = yaw_flip_flag

    return dataset


@patch('xarray.open_dataset')
def test_1258(fake_open_dataset):
    """Save true_color from abi with radiance doesn't need two resamplings."""
    from satpy import Scene
    fake_open_dataset.side_effect = generate_fake_abi_xr_dataset

    scene = Scene(abi_file_list, reader='abi_l1b')
    scene.load(['true_color_nocorr', 'C04'], calibration='radiance')
    resampled_scene = scene.resample(scene.coarsest_area(), resampler='native')
    assert len(resampled_scene.keys()) == 2


@patch('xarray.open_dataset')
def test_1088(fake_open_dataset):
    """Check that copied arrays gets resampled."""
    from satpy import Scene
    fake_open_dataset.side_effect = generate_fake_abi_xr_dataset

    scene = Scene(abi_file_list, reader='abi_l1b')
    scene.load(['C04'], calibration='radiance')

    my_id = make_dataid(name='my_name', wavelength=(10, 11, 12))
    scene[my_id] = scene['C04'].copy()
    resampled = scene.resample('eurol')
    assert resampled[my_id].shape == (2048, 2560)


@patch('xarray.open_dataset')
def test_no_enums(fake_open_dataset):
    """Check that no enums are inserted in the resulting attrs."""
    from enum import Enum

    from satpy import Scene
    fake_open_dataset.side_effect = generate_fake_abi_xr_dataset

    scene = Scene(abi_file_list, reader='abi_l1b')
    scene.load(['C04'], calibration='radiance')
    for value in scene['C04'].attrs.values():
        assert not isinstance(value, Enum)
