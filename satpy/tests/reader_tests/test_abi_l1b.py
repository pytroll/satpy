#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Satpy developers
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
"""The abi_l1b reader tests package.
"""

import sys
import numpy as np
import xarray as xr

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class FakeDataset(object):
    def __init__(self, info, attrs):
        for var_name, var_data in list(info.items()):
            if isinstance(var_data, np.ndarray):
                info[var_name] = xr.DataArray(var_data)
        self.info = info
        self.attrs = attrs

    def __getitem__(self, key):
        return self.info[key]

    def rename(self, *args, **kwargs):
        return self

    def close(self):
        return


class Test_NC_ABI_L1B_Base(unittest.TestCase):
    """Common setup for NC_ABI_L1B tests"""

    @mock.patch('satpy.readers.abi_l1b.xr')
    def setUp(self, xr_, rad=None):
        """Create a fake dataset using the given radiance data"""
        from satpy.readers.abi_l1b import NC_ABI_L1B

        x_image = xr.DataArray(0.)
        y_image = xr.DataArray(0.)
        time = xr.DataArray(0.)
        if rad is None:
            rad_data = (np.arange(10.).reshape((2, 5)) + 1.) * 50.
            rad_data = (rad_data + 1.) / 0.5
            rad_data = rad_data.astype(np.int16)
            rad = xr.DataArray(
                rad_data,
                dims=('y', 'x'),
                attrs={
                    'scale_factor': 0.5,
                    'add_offset': -1.,
                    '_FillValue': 1002,
                    'units': 'W m-2 um-1 sr-1'
                }
            )
        rad['time'] = time
        rad['x_image'] = x_image
        rad['y_image'] = y_image
        x__ = xr.DataArray(
            range(5),
            attrs={'scale_factor': 2., 'add_offset': -1.},
        )
        y__ = xr.DataArray(
            range(2),
            attrs={'scale_factor': -2., 'add_offset': 1.},
        )
        proj = xr.DataArray(
            [],
            attrs={
                'semi_major_axis': 1.,
                'semi_minor_axis': 1.,
                'perspective_point_height': 1.,
                'longitude_of_projection_origin': -90.,
                'latitude_of_projection_origin': 0.,
                'sweep_angle_axis': u'x'
            }
        )
        yaw_flip = xr.DataArray([1])
        xr_.open_dataset.return_value = FakeDataset({
            'Rad': rad,
            'band_id': np.array(8),
            'x': x__,
            'y': y__,
            'x_image': x_image,
            'y_image': y_image,
            'goes_imager_projection': proj,
            'yaw_flip_flag': yaw_flip,
            "planck_fk1": np.array(13432.1),
            "planck_fk2": np.array(1497.61),
            "planck_bc1": np.array(0.09102),
            "planck_bc2": np.array(0.99971),
            "esun": np.array(2017),
            "nominal_satellite_subpoint_lat": np.array(0.0),
            "nominal_satellite_subpoint_lon": np.array(-89.5),
            "nominal_satellite_height": np.array(35786.02),
            "earth_sun_distance_anomaly_in_AU": np.array(0.99)},
            {
                "time_coverage_start": "2017-09-20T17:30:40.8Z",
                "time_coverage_end": "2017-09-20T17:41:17.5Z",
            })

        self.reader = NC_ABI_L1B('filename',
                                 {'platform_shortname': 'G16', 'observation_type': 'Rad',
                                  'scene_abbr': 'C', 'scan_mode': 'M3'},
                                 {'filetype': 'info'})


class Test_NC_ABI_L1B(Test_NC_ABI_L1B_Base):
    """Test the NC_ABI_L1B reader."""

    def test_basic_attributes(self):
        """Test getting basic file attributes."""
        from datetime import datetime
        from satpy import DatasetID
        self.assertEqual(self.reader.start_time,
                         datetime(2017, 9, 20, 17, 30, 40, 800000))
        self.assertEqual(self.reader.end_time,
                         datetime(2017, 9, 20, 17, 41, 17, 500000))
        self.assertEqual(self.reader.get_shape(DatasetID(name='C05'), {}),
                         (2, 5))

    def test_get_dataset(self):
        from satpy import DatasetID
        key = DatasetID(name='Rad', calibration='radiance')
        res = self.reader.get_dataset(key, {'info': 'info'})
        exp = {'calibration': 'radiance',
               'instrument_ID': None,
               'modifiers': (),
               'name': 'Rad',
               'observation_type': 'Rad',
               'orbital_parameters': {'projection_altitude': 1.0,
                                      'projection_latitude': 0.0,
                                      'projection_longitude': -90.0,
                                      'satellite_nominal_altitude': 35786.02,
                                      'satellite_nominal_latitude': 0.0,
                                      'satellite_nominal_longitude': -89.5,
                                      'yaw_flip': True},
               'orbital_slot': None,
               'platform_name': 'GOES-16',
               'platform_shortname': 'G16',
               'production_site': None,
               'satellite_altitude': 35786.02,
               'satellite_latitude': 0.0,
               'satellite_longitude': -89.5,
               'scan_mode': 'M3',
               'scene_abbr': 'C',
               'scene_id': None,
               'sensor': 'abi',
               'timeline_ID': None,
               'units': 'W m-2 um-1 sr-1'}
        self.assertDictEqual(res.attrs, exp)

    def test_bad_calibration(self):
        """Test that asking for a bad calibration fails."""
        from satpy import DatasetID
        self.assertRaises(ValueError, self.reader.get_dataset,
                          DatasetID(name='C05', calibration='_bad_'), {})

    @mock.patch('satpy.readers.abi_l1b.geometry.AreaDefinition')
    def test_get_area_def(self, adef):
        """Test the area generation."""
        self.reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {'a': 1.0, 'b': 1.0, 'h': 1.0, 'lon_0': -90.0, 'proj': 'geos',
                                            'sweep': 'x', 'units': 'm'})
        self.assertEqual(call_args[4], self.reader.ncols)
        self.assertEqual(call_args[5], self.reader.nlines)
        np.testing.assert_allclose(call_args[6], (-2, -2, 8, 2))


class Test_NC_ABI_L1B_ir_cal(Test_NC_ABI_L1B_Base):
    def setUp(self):
        """Setup for test."""
        rad_data = (np.arange(10.).reshape((2, 5)) + 1.) * 50.
        rad_data = (rad_data + 1.) / 0.5
        rad_data = rad_data.astype(np.int16)
        rad = xr.DataArray(
            rad_data,
            dims=('y', 'x'),
            attrs={
                'scale_factor': 0.5,
                'add_offset': -1.,
                '_FillValue': 1002,
            }
        )
        super(Test_NC_ABI_L1B_ir_cal, self).setUp(rad=rad)

    def test_ir_calibrate(self):
        """Test IR calibration."""
        from satpy import DatasetID
        res = self.reader.get_dataset(
            DatasetID(name='C05', calibration='brightness_temperature'), {})

        expected = np.array([[267.55572248, 305.15576503, 332.37383249, 354.73895301, 374.19710115],
                             [391.68679226, 407.74064808, 422.69329105, 436.77021913, np.nan]])
        self.assertTrue(np.allclose(res.data, expected, equal_nan=True))
        # make sure the attributes from the file are in the data array
        self.assertNotIn('scale_factor', res.attrs)
        self.assertNotIn('_FillValue', res.attrs)
        self.assertEqual(res.attrs['standard_name'],
                         'toa_brightness_temperature')
        self.assertEqual(res.attrs['long_name'], 'Brightness Temperature')


class Test_NC_ABI_L1B_vis_cal(Test_NC_ABI_L1B_Base):
    def setUp(self):
        """Setup for test."""
        rad_data = (np.arange(10.).reshape((2, 5)) + 1.)
        rad_data = (rad_data + 1.) / 0.5
        rad_data = rad_data.astype(np.int16)
        rad = xr.DataArray(
            rad_data,
            dims=('y', 'x'),
            attrs={
                'scale_factor': 0.5,
                'add_offset': -1.,
                '_FillValue': 20,
            }
        )
        super(Test_NC_ABI_L1B_vis_cal, self).setUp(rad=rad)

    def test_vis_calibrate(self):
        """Test VIS calibration."""
        from satpy import DatasetID
        res = self.reader.get_dataset(
            DatasetID(name='C05', calibration='reflectance'), {})

        expected = np.array([[0.15265617, 0.30531234, 0.45796851, 0.61062468, 0.76328085],
                             [0.91593702, 1.06859319, 1.22124936, np.nan, 1.52656171]])
        self.assertTrue(np.allclose(res.data, expected, equal_nan=True))
        self.assertNotIn('scale_factor', res.attrs)
        self.assertNotIn('_FillValue', res.attrs)
        self.assertEqual(res.attrs['standard_name'],
                         'toa_bidirectional_reflectance')
        self.assertEqual(res.attrs['long_name'],
                         'Bidirectional Reflectance')


def suite():
    """The test suite for test_scene."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_NC_ABI_L1B))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_NC_ABI_L1B_ir_cal))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_NC_ABI_L1B_vis_cal))
    return mysuite
