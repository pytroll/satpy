#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Satpy developers
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
"""The scmi_abi_l1b reader tests package."""

import unittest
from unittest import mock
import numpy as np
import xarray as xr


class FakeDataset(object):
    def __init__(self, info, attrs, dims=None):
        for var_name, var_data in list(info.items()):
            if isinstance(var_data, np.ndarray):
                info[var_name] = xr.DataArray(var_data)
        self.info = info
        self.attrs = attrs
        self.dims = dims or {}

    def __getitem__(self, key):
        return self.info.get(key, self.dims.get(key))

    def __contains__(self, key):
        return key in self.info or key in self.dims

    def rename(self, *args, **kwargs):
        return self

    def close(self):
        return


class TestSCMIFileHandler(unittest.TestCase):
    """Test the SCMIFileHandler reader."""

    @mock.patch('satpy.readers.scmi.xr')
    def setUp(self, xr_):
        """Setup for test."""
        from satpy.readers.scmi import SCMIFileHandler
        rad_data = (np.arange(10.).reshape((2, 5)) + 1.)
        rad_data = (rad_data + 1.) / 0.5
        rad_data = rad_data.astype(np.int16)
        self.expected_rad = rad_data.astype(np.float64) * 0.5 + -1.
        self.expected_rad[-1, -2] = np.nan
        time = xr.DataArray(0.)
        rad = xr.DataArray(
            rad_data,
            dims=('y', 'x'),
            attrs={
                'scale_factor': 0.5,
                'add_offset': -1.,
                '_FillValue': 20,
                'standard_name': 'toa_bidirectional_reflectance',
            },
            coords={
                'time': time,
            }
        )
        xr_.open_dataset.return_value = FakeDataset(
            {
                'Sectorized_CMI': rad,
                "nominal_satellite_subpoint_lat": np.array(0.0),
                "nominal_satellite_subpoint_lon": np.array(-89.5),
                "nominal_satellite_height": np.array(35786.02),
            },
            {
                'start_date_time': "2017210120000",
                'satellite_id': 'GOES-16',
                'satellite_longitude': -90.,
                'satellite_latitude': 0.,
                'satellite_altitude': 35785831.,
            },
            {'y': 2, 'x': 5},
        )

        self.reader = SCMIFileHandler('filename',
                                      {'platform_shortname': 'G16'},
                                      {'filetype': 'info'})

    def test_basic_attributes(self):
        """Test getting basic file attributes."""
        from datetime import datetime
        from satpy import DatasetID
        self.assertEqual(self.reader.start_time,
                         datetime(2017, 7, 29, 12, 0, 0, 0))
        self.assertEqual(self.reader.end_time,
                         datetime(2017, 7, 29, 12, 0, 0, 0))
        self.assertEqual(self.reader.get_shape(DatasetID(name='C05'), {}),
                         (2, 5))

    def test_data_load(self):
        """Test data loading."""
        from satpy import DatasetID
        res = self.reader.get_dataset(
            DatasetID(name='C05', calibration='reflectance'), {})

        np.testing.assert_allclose(res.data, self.expected_rad, equal_nan=True)
        self.assertNotIn('scale_factor', res.attrs)
        self.assertNotIn('_FillValue', res.attrs)
        self.assertEqual(res.attrs['standard_name'],
                         'toa_bidirectional_reflectance')


class TestSCMIFileHandlerArea(unittest.TestCase):
    """Test the SCMIFileHandler's area creation."""

    @mock.patch('satpy.readers.scmi.xr')
    def create_reader(self, proj_name, proj_attrs, xr_):
        """Create a fake reader."""
        from satpy.readers.scmi import SCMIFileHandler
        proj = xr.DataArray([], attrs=proj_attrs)
        x__ = xr.DataArray(
            [0, 1],
            attrs={'scale_factor': 2., 'add_offset': -1., 'units': 'meters'},
        )
        y__ = xr.DataArray(
            [0, 1],
            attrs={'scale_factor': -2., 'add_offset': 1., 'units': 'meters'},
        )
        xr_.open_dataset.return_value = FakeDataset({
            'goes_imager_projection': proj,
            'x': x__,
            'y': y__,
            'Sectorized_CMI': np.ones((2, 2))},
            {
                'satellite_id': 'GOES-16',
                'grid_mapping': proj_name,
            },
            {
                'y': y__.size,
                'x': x__.size,
            }
        )

        return SCMIFileHandler('filename',
                               {'platform_shortname': 'G16'},
                               {'filetype': 'info'})

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_geos(self, adef):
        """Test the area generation for geos projection."""
        reader = self.create_reader(
            'goes_imager_projection',
            {
                'semi_major_axis': 1.,
                'semi_minor_axis': 1.,
                'perspective_point_height': 1.,
                'longitude_of_projection_origin': -90.,
                'sweep_angle_axis': u'x',
                'grid_mapping_name': 'geostationary',
            }
        )
        reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {
            'a': 1.0, 'b': 1.0, 'h': 1.0, 'lon_0': -90.0, 'lat_0': 0.0,
            'proj': 'geos', 'sweep': 'x', 'units': 'm'})
        self.assertEqual(call_args[4], reader.ncols)
        self.assertEqual(call_args[5], reader.nlines)
        np.testing.assert_allclose(call_args[6], (-2., -2., 2, 2.))

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_lcc(self, adef):
        """Test the area generation for lcc projection."""
        reader = self.create_reader(
            'goes_imager_projection',
            {
                'semi_major_axis': 1.,
                'semi_minor_axis': 1.,
                'longitude_of_central_meridian': -90.,
                'standard_parallel': 25.,
                'latitude_of_projection_origin': 25.,
                'grid_mapping_name': 'lambert_conformal_conic',
            }
        )
        reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {
            'a': 1.0, 'b': 1.0, 'lon_0': -90.0, 'lat_0': 25.0, 'lat_1': 25.0,
            'proj': 'lcc', 'units': 'm'})
        self.assertEqual(call_args[4], reader.ncols)
        self.assertEqual(call_args[5], reader.nlines)
        np.testing.assert_allclose(call_args[6], (-2., -2., 2, 2.))

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_stere(self, adef):
        """Test the area generation for stere projection."""
        reader = self.create_reader(
            'goes_imager_projection',
            {
                'semi_major_axis': 1.,
                'semi_minor_axis': 1.,
                'straight_vertical_longitude_from_pole': -90.,
                'standard_parallel': 60.,
                'latitude_of_projection_origin': 90.,
                'grid_mapping_name': 'polar_stereographic',
            }
        )
        reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {
            'a': 1.0, 'b': 1.0, 'lon_0': -90.0, 'lat_0': 90.0, 'lat_ts': 60.0,
            'proj': 'stere', 'units': 'm'})
        self.assertEqual(call_args[4], reader.ncols)
        self.assertEqual(call_args[5], reader.nlines)
        np.testing.assert_allclose(call_args[6], (-2., -2., 2, 2.))

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_merc(self, adef):
        """Test the area generation for merc projection."""
        reader = self.create_reader(
            'goes_imager_projection',
            {
                'semi_major_axis': 1.,
                'semi_minor_axis': 1.,
                'longitude_of_projection_origin': -90.,
                'standard_parallel': 0.,
                'grid_mapping_name': 'mercator',
            }
        )
        reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {
            'a': 1.0, 'b': 1.0, 'lon_0': -90.0, 'lat_0': 0.0, 'lat_ts': 0.0,
            'proj': 'merc', 'units': 'm'})
        self.assertEqual(call_args[4], reader.ncols)
        self.assertEqual(call_args[5], reader.nlines)
        np.testing.assert_allclose(call_args[6], (-2., -2., 2, 2.))

    @mock.patch('satpy.readers.abi_base.geometry.AreaDefinition')
    def test_get_area_def_bad(self, adef):
        """Test the area generation for bad projection."""
        reader = self.create_reader(
            'goes_imager_projection',
            {
                'semi_major_axis': 1.,
                'semi_minor_axis': 1.,
                'longitude_of_projection_origin': -90.,
                'standard_parallel': 0.,
                'grid_mapping_name': 'fake',
            }
        )
        self.assertRaises(ValueError, reader.get_area_def, None)
