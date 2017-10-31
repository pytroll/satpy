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


def suite():
    """The test suite for test_scene.
    """
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestCache))

    return mysuite