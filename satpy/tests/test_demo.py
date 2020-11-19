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
"""Tests for the satpy.demo module."""

import os
import unittest
from unittest import mock


class _GlobHelper(object):
    """Create side effect function for mocking gcsfs glob method."""

    def __init__(self, num_results):
        """Initialize side_effect function for mocking gcsfs glob method.

        Args:
            num_results (int or list): Number of results for each glob call
                to return. If a list then number of results per call. The
                last number is used for any additional calls.

        """
        self.current_call = 0
        if not isinstance(num_results, (list, tuple)):
            num_results = [num_results]
        self.num_results = num_results

    def __call__(self, pattern):
        """Mimic glob by being used as the side effect function."""
        try:
            num_results = self.num_results[self.current_call]
        except IndexError:
            num_results = self.num_results[-1]
        self.current_call += 1
        return [pattern + '.{:03d}'.format(idx) for idx in range(num_results)]


class TestDemo(unittest.TestCase):
    """Test demo data download functions."""

    def setUp(self):
        """Create temporary directory to save files to."""
        import tempfile
        self.base_dir = tempfile.mkdtemp()
        self.prev_dir = os.getcwd()
        os.chdir(self.base_dir)

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        os.chdir(self.prev_dir)
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    @mock.patch('satpy.demo._google_cloud_platform.gcsfs')
    def test_get_us_midlatitude_cyclone_abi(self, gcsfs_mod):
        """Test data download function."""
        from satpy.demo import get_us_midlatitude_cyclone_abi
        gcsfs_mod.GCSFileSystem = mock.MagicMock()
        gcsfs_inst = mock.MagicMock()
        gcsfs_mod.GCSFileSystem.return_value = gcsfs_inst
        gcsfs_inst.glob.return_value = ['a.nc', 'b.nc']
        # expected 16 files, got 2
        self.assertRaises(AssertionError, get_us_midlatitude_cyclone_abi)
        # unknown access method
        self.assertRaises(NotImplementedError, get_us_midlatitude_cyclone_abi, method='unknown')

        gcsfs_inst.glob.return_value = ['a.nc'] * 16
        filenames = get_us_midlatitude_cyclone_abi()
        expected = os.path.join('.', 'abi_l1b', '20190314_us_midlatitude_cyclone', 'a.nc')
        for fn in filenames:
            self.assertEqual(expected, fn)

    @mock.patch('satpy.demo._google_cloud_platform.gcsfs')
    def test_get_hurricane_florence_abi(self, gcsfs_mod):
        """Test data download function."""
        from satpy.demo import get_hurricane_florence_abi
        gcsfs_mod.GCSFileSystem = mock.MagicMock()
        gcsfs_inst = mock.MagicMock()
        gcsfs_mod.GCSFileSystem.return_value = gcsfs_inst
        # only return 5 results total
        gcsfs_inst.glob.side_effect = _GlobHelper([5, 0])
        # expected 16 files * 10 frames, got 16 * 5
        self.assertRaises(AssertionError, get_hurricane_florence_abi)
        self.assertRaises(NotImplementedError, get_hurricane_florence_abi, method='unknown')

        gcsfs_inst.glob.side_effect = _GlobHelper([int(240 / 16), 0, 0, 0] * 16)
        filenames = get_hurricane_florence_abi()
        self.assertEqual(10 * 16, len(filenames))

        gcsfs_inst.glob.side_effect = _GlobHelper([int(240 / 16), 0, 0, 0] * 16)
        filenames = get_hurricane_florence_abi(channels=[2, 3, 4])
        self.assertEqual(10 * 3, len(filenames))

        gcsfs_inst.glob.side_effect = _GlobHelper([int(240 / 16), 0, 0, 0] * 16)
        filenames = get_hurricane_florence_abi(channels=[2, 3, 4], num_frames=5)
        self.assertEqual(5 * 3, len(filenames))

        gcsfs_inst.glob.side_effect = _GlobHelper([int(240 / 16), 0, 0, 0] * 16)
        filenames = get_hurricane_florence_abi(num_frames=5)
        self.assertEqual(5 * 16, len(filenames))


class TestGCPUtils(unittest.TestCase):
    """Test Google Cloud Platform utilities."""

    @mock.patch('satpy.demo._google_cloud_platform.urlopen')
    def test_is_gcp_instance(self, uo):
        """Test is_google_cloud_instance."""
        from satpy.demo._google_cloud_platform import is_google_cloud_instance, URLError
        uo.side_effect = URLError("Test Environment")
        self.assertFalse(is_google_cloud_instance())

    @mock.patch('satpy.demo._google_cloud_platform.gcsfs')
    def test_get_bucket_files(self, gcsfs_mod):
        """Test get_bucket_files basic cases."""
        from satpy.demo._google_cloud_platform import get_bucket_files
        gcsfs_mod.GCSFileSystem = mock.MagicMock()
        gcsfs_inst = mock.MagicMock()
        gcsfs_mod.GCSFileSystem.return_value = gcsfs_inst
        gcsfs_inst.glob.return_value = ['a.nc', 'b.nc']
        filenames = get_bucket_files('*.nc', '.')
        expected = [os.path.join('.', 'a.nc'), os.path.join('.', 'b.nc')]
        self.assertEqual(expected, filenames)

        gcsfs_inst.glob.side_effect = _GlobHelper(10)
        filenames = get_bucket_files(['*.nc', '*.txt'], '.', pattern_slice=slice(2, 5))
        self.assertEqual(len(filenames), 3 * 2)
        gcsfs_inst.glob.side_effect = None  # reset mock side effect

        gcsfs_inst.glob.return_value = ['a.nc', 'b.nc']
        self.assertRaises(OSError, get_bucket_files, '*.nc', 'does_not_exist')

        open('a.nc', 'w').close()  # touch the file
        gcsfs_inst.get.reset_mock()
        gcsfs_inst.glob.return_value = ['a.nc']
        filenames = get_bucket_files('*.nc', '.')
        self.assertEqual([os.path.join('.', 'a.nc')], filenames)
        gcsfs_inst.get.assert_not_called()

        # force redownload
        gcsfs_inst.get.reset_mock()
        gcsfs_inst.glob.return_value = ['a.nc']
        filenames = get_bucket_files('*.nc', '.', force=True)
        self.assertEqual([os.path.join('.', 'a.nc')], filenames)
        gcsfs_inst.get.assert_called_once()

        # if we don't get any results then we expect an exception
        gcsfs_inst.get.reset_mock()
        gcsfs_inst.glob.return_value = []
        self.assertRaises(OSError, get_bucket_files, '*.nc', '.')

    @mock.patch('satpy.demo._google_cloud_platform.gcsfs', None)
    def test_no_gcsfs(self):
        """Test that 'gcsfs' is required."""
        from satpy.demo._google_cloud_platform import get_bucket_files
        self.assertRaises(RuntimeError, get_bucket_files, '*.nc', '.')
