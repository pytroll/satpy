#!/usr/bin/python
# Copyright (c) 2016.
#

# Author(s):
#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""
"""

import unittest

try:
    from unittest import mock
except ImportError:
    import mock


class TestCache(unittest.TestCase):
    """
    Test the caching functionnality
    """

    def test_kd_cache(self):
        """Test the cache in kd resampler.
        """
        import satpy.resample
        satpy.resample.CACHE_SIZE = 3

        with mock.patch('satpy.resample.get_neighbour_info') as get_neighbour_info:
            get_neighbour_info.return_value = [9, 9, 9, 9]
            with mock.patch('satpy.resample.get_sample_from_neighbour_info'):
                with mock.patch('satpy.resample.KDTreeResampler.get_hash') as get_hash:
                    get_hash.side_effect = ['a', 'b', 'a', 'c', 'd']
                    in_area = mock.MagicMock()
                    out_area = mock.MagicMock()
                    resampler = satpy.resample.KDTreeResampler(in_area, out_area)
                    resampler.resample('hej')
                    get_neighbour_info.assert_called_with(in_area, out_area, 10000,
                                                          segments=None, epsilon=0, neighbours=1, nprocs=1,
                                                          reduce_data=True)
                    self.assertEqual(list(resampler.caches.keys()), ['a'])

                    in_area = mock.MagicMock()
                    out_area = mock.MagicMock()
                    resampler = satpy.resample.KDTreeResampler(in_area, out_area)
                    resampler.resample('hej')
                    get_neighbour_info.assert_called_with(in_area, out_area, 10000,
                                                          segments=None, epsilon=0, neighbours=1, nprocs=1,
                                                          reduce_data=True)
                    self.assertEqual(list(resampler.caches.keys()), ['a', 'b'])

                    in_area = mock.MagicMock()
                    out_area = mock.MagicMock()
                    resampler = satpy.resample.KDTreeResampler(in_area, out_area)
                    resampler.resample('hej')
                    self.assertEqual(list(resampler.caches.keys()), ['b', 'a'])
                    self.assertNotEqual(get_neighbour_info.call_args,
                                        mock.call(in_area, out_area, 10000,
                                                  segments=None, epsilon=0, neighbours=1,
                                                  nprocs=1, reduce_data=True))

                    resampler = satpy.resample.KDTreeResampler(in_area, out_area)
                    resampler.resample('hej')
                    self.assertEqual(list(resampler.caches.keys()), ['b', 'a', 'c'])

                    resampler = satpy.resample.KDTreeResampler(in_area, out_area)
                    resampler.resample('hej')
                    self.assertEqual(list(resampler.caches.keys()), ['a', 'c', 'd'])


class TestNativeResampler(unittest.TestCase):
    def test_expand_reduce(self):
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        d_arr = da.zeros((6, 20), chunks=4)
        new_arr = NativeResampler.expand_reduce(d_arr, {0: 2., 1: 2.})
        self.assertEqual(new_arr.shape, (12, 40))
        new_arr = NativeResampler.expand_reduce(d_arr, {0: .5, 1: .5})
        self.assertEqual(new_arr.shape, (3, 10))
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 1. / 3, 1: 1.})
        new_arr = NativeResampler.expand_reduce(d_arr, {0: 1., 1: 1.})
        self.assertEqual(new_arr.shape, (6, 20))
        self.assertIs(new_arr, d_arr)
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 0.333323423, 1: 1.})
        self.assertRaises(ValueError, NativeResampler.expand_reduce,
                          d_arr, {0: 1.333323423, 1: 1.})

        n_arr = np.zeros((6, 20))
        new_arr = NativeResampler.expand_reduce(n_arr, {0: 2., 1: 1.0})
        self.assertTrue(np.all(new_arr.compute()[::2, :] == n_arr))

    def test_expand_dims(self):
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        ds1 = DataArray(da.zeros((100, 50), chunks=85), dims=('y', 'x'),
                        coords={'y': da.arange(100, chunks=85),
                                'x': da.arange(50, chunks=85)})
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        target = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size=100,
            y_size=200,
            area_extent=(-1000., -1500., 1000., 1500.),
        )
        # source geo def doesn't actually matter
        resampler = NativeResampler(None, target)
        new_arr = resampler.resample(ds1)
        self.assertEqual(new_arr.shape, (200, 100))
        new_arr2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_arr == new_arr2))

    def test_expand_without_dims(self):
        from satpy.resample import NativeResampler
        import numpy as np
        import dask.array as da
        from xarray import DataArray
        from pyresample.geometry import AreaDefinition
        from pyresample.utils import proj4_str_to_dict
        ds1 = DataArray(da.zeros((100, 50), chunks=85))
        proj_dict = proj4_str_to_dict('+proj=lcc +datum=WGS84 +ellps=WGS84 '
                                      '+lon_0=-95. +lat_0=25 +lat_1=25 '
                                      '+units=m +no_defs')
        target = AreaDefinition(
            'test',
            'test',
            'test',
            proj_dict,
            x_size=100,
            y_size=200,
            area_extent=(-1000., -1500., 1000., 1500.),
        )
        # source geo def doesn't actually matter
        resampler = NativeResampler(None, target)
        new_arr = resampler.resample(ds1)
        self.assertEqual(new_arr.shape, (200, 100))
        new_arr2 = resampler.resample(ds1.compute())
        self.assertTrue(np.all(new_arr == new_arr2))


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestNativeResampler))
    # FIXME: Fix these tests
    # mysuite.addTest(loader.loadTestsFromTestCase(TestCache))

    return mysuite