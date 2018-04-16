# -*- coding: utf-8 -*-

# Copyright (c) 2017 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

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
"""The abi_l1b reader tests package.
"""

import sys
import numpy as np

from satpy.readers.abi_l1b import NC_ABI_L1B
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
    def __init__(self, info):
        for var_name, var_data in list(info.items()):
            if isinstance(var_data, np.ndarray):
                info[var_name] = xr.DataArray(var_data)
        self.info = info

    def __getitem__(self, key):
        return self.info[key]

    def rename(self, *args, **kwargs):
        return self

    def close(self):
        return


class Test_NC_ABI_L1B_ir_cal(unittest.TestCase):
    """Test the NC_ABI_L1B reader."""
    @mock.patch('satpy.readers.abi_l1b.xr')
    def setUp(self, xr):
        """Setup for test."""
        xr.open_dataset.return_value = FakeDataset({
            'band_id': np.array(8),
            'Rad': np.arange(10.).reshape((2, 5)),
            "planck_fk1": np.array(13432.1),
            "planck_fk2": np.array(1497.61),
            "planck_bc1": np.array(0.09102),
            "planck_bc2": np.array(0.99971),
            "esun": np.array(2017),
            "earth_sun_distance_anomaly_in_AU": np.array(0.99)})

        self.reader = NC_ABI_L1B('filename',
                                 {'platform_shortname': 'G16'},
                                 {'filetype': 'info'})

    def test_ir_calibrate(self):
        """Test IR calibration."""
        data = xr.DataArray((np.arange(10.).reshape((2, 5)) + 1) * 50)

        res = self.reader._ir_calibrate(data)

        expected = np.array([[267.55572248, 305.15576503, 332.37383249,
                                 354.73895301, 374.19710115],
                                [391.68679226, 407.74064808, 422.69329105,
                                 436.77021913, 450.13141732]])
        self.assertTrue(np.allclose(res.data, expected))


class Test_NC_ABI_L1B_vis_cal(unittest.TestCase):
    """Test the NC_ABI_L1B reader."""

    @mock.patch('satpy.readers.abi_l1b.xr')
    def setUp(self, xr):
        """Setup for test."""
        xr.open_dataset.return_value = FakeDataset({
            'band_id': np.array(5),
            'Rad': np.arange(10.).reshape((2, 5)),
            "planck_fk1": np.array(13432.1),
            "planck_fk2": np.array(1497.61),
            "planck_bc1": np.array(0.09102),
            "planck_bc2": np.array(0.99971),
            "esun": np.array(2017),
            "earth_sun_distance_anomaly_in_AU": np.array(0.99)})

        self.reader = NC_ABI_L1B('filename',
                                 {'platform_shortname': 'G16'},
                                 {'filetype': 'info'})

    def test_vis_calibrate(self):
        """Test VIS calibration."""
        data = xr.DataArray((np.arange(10.).reshape((2, 5)) + 1) * 100)

        res = self.reader._vis_calibrate(data)

        expected = np.array([[0.15265617, 0.30531234, 0.45796851,
                                 0.61062468, 0.76328085],
                                [0.91593702, 1.06859319, 1.22124936,
                                 1.37390553, 1.52656171]])
        self.assertTrue(np.allclose(res.data, expected))


class Test_NC_ABI_L1B_area(unittest.TestCase):
    """Test the NC_ABI_L1B reader."""
    @mock.patch('satpy.readers.abi_l1b.xr')
    def setUp(self, xr_):
        """Setup for test."""
        import xarray as xr
        proj = xr.DataArray(
            [],
            attrs={
                'semi_major_axis': 1.,
                'semi_minor_axis': 1.,
                'perspective_point_height': 1.,
                'longitude_of_projection_origin': -90.,
                'sweep_angle_axis': u'x'
            }
        )
        x__ = xr.DataArray(
            [-1., 1.],
            attrs={'scale_factor': 1., 'add_offset': 0.},
        )
        y__ = xr.DataArray(
            [-1., 1.],
            attrs={'scale_factor': 1., 'add_offset': 0.},
        )
        xr_.open_dataset.return_value = FakeDataset({
            'goes_imager_projection': proj,
            'x': x__,
            'y': y__,
            'Rad': np.ones((10, 10))})

        self.reader = NC_ABI_L1B('filename',
                                 {'platform_shortname': 'G16'},
                                 {'filetype': 'info'})

    @mock.patch('satpy.readers.abi_l1b.geometry.AreaDefinition')
    def test_get_area_def(self, adef):
        """Test the area generation."""
        self.reader.get_area_def(None)

        self.assertEqual(adef.call_count, 1)
        call_args = tuple(adef.call_args)[0]
        self.assertDictEqual(call_args[3], {'a': 1.0, 'b': 1.0, 'h': 1.0,
                                      'lon_0': -90.0, 'proj': 'geos',
                                      'sweep': 'x', 'units': 'm'})
        self.assertEqual(call_args[4], self.reader.ncols)
        self.assertEqual(call_args[5], self.reader.nlines)
        np.testing.assert_allclose(call_args[6], (-1.1111111111111112,
                                                  1.1111111111111112,
                                                  1.1111111111111112,
                                                  -1.1111111111111112))


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(Test_NC_ABI_L1B_ir_cal))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_NC_ABI_L1B_vis_cal))
    mysuite.addTest(loader.loadTestsFromTestCase(Test_NC_ABI_L1B_area))
    return mysuite
