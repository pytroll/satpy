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
from satpy.resample import KDTreeResampler
import tempfile
import shutil
import os

try:
    from unittest import mock
except ImportError:
    import mock


class TestKDTreeResampler(unittest.TestCase):
    """Test the kd-tree resampler."""

    def test_kd_resampling(self):
        """Test the kd resampler."""
        source_area = mock.MagicMock()
        target_area = mock.MagicMock()

        with mock.patch('satpy.resample.XArrayResamplerNN'):
            resampler = KDTreeResampler(source_area, target_area)
            resampler.precompute()
            resampler.resampler.get_neighbour_info.assert_called_with()

            try:
                the_dir = tempfile.mkdtemp()
                with mock.patch('satpy.resample.KDTreeResampler._create_cache_filename') as create_filename:
                    resampler = KDTreeResampler(source_area, target_area)
                    create_filename.return_value = os.path.join(the_dir, 'test_cache.npz')
                    with mock.patch('satpy.resample.np.load') as load:
                        with mock.patch('satpy.resample.np.savez') as savez:
                            load.side_effect = IOError()
                            resampler.precompute(cache_dir=the_dir)
                            # assert saving
                            self.assertEqual(len(savez.mock_calls), 1)
                            nbcalls = len(resampler.resampler.get_neighbour_info.mock_calls)
                            # test reusing the resampler
                            load.side_effect = None

                            class FakeNPZ(dict):
                                def close(self):
                                    pass

                            load.return_value = FakeNPZ(valid_input_index=1,
                                                        valid_output_index=2,
                                                        index_array=3,
                                                        distance_array=4)
                            self.assertEqual(len(savez.mock_calls), 1)
                            resampler.precompute(cache_dir=the_dir)
                            self.assertEqual(len(load.mock_calls), 1)
                            self.assertEqual(len(resampler.resampler.get_neighbour_info.mock_calls), nbcalls)

                            # test loading saved resampler
                            resampler = KDTreeResampler(source_area, target_area)
                            resampler.precompute(cache_dir=the_dir)
                            self.assertEqual(len(load.mock_calls), 2)
                            self.assertEqual(len(resampler.resampler.get_neighbour_info.mock_calls), nbcalls)
            finally:
                shutil.rmtree(the_dir)

            data = mock.MagicMock()
            data.name = 'hej'
            data.data = [1, 2, 3]
            fill_value = 8
            resampler.compute(data, fill_value=fill_value)
            resampler.resampler.get_sample_from_neighbour_info.assert_called_with(data, fill_value)

            data.attrs = {'_FillValue': 8}
            resampler.compute(data)
            resampler.resampler.get_sample_from_neighbour_info.assert_called_with(data, fill_value)


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
    mysuite.addTest(loader.loadTestsFromTestCase(TestKDTreeResampler))

    return mysuite


if __name__ == '__main__':
    unittest.main()
