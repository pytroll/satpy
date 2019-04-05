#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 SatPy developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Tests for the satpy.demo module."""

import os
import sys

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest
try:
    from unittest import mock
except ImportError:
    import mock


class TestDemo(unittest.TestCase):
    """Test demo data download functions."""

    @mock.patch('satpy.demo.google_cloud_platform.gcsfs')
    def test_get_us_midlatitude_cyclone_abi(self, gcsfs_mod):
        """Test data download function."""
        from satpy.demo import get_us_midlatitude_cyclone_abi
        gcsfs_mod.GCSFileSystem = mock.MagicMock()
        gcsfs_inst = mock.MagicMock()
        gcsfs_mod.GCSFileSystem.return_value = gcsfs_inst
        gcsfs_inst.glob.return_value = ['a.nc', 'b.nc']
        self.assertRaises(AssertionError, get_us_midlatitude_cyclone_abi)
        self.assertRaises(NotImplementedError, get_us_midlatitude_cyclone_abi, method='unknown')

        gcsfs_inst.glob.return_value = ['a.nc'] * 16
        filenames = get_us_midlatitude_cyclone_abi()
        expected = os.path.join('.', 'abi_l1b', '20190314_us_midlatitude_cyclone', 'a.nc')
        for fn in filenames:
            self.assertEqual(expected, fn)


class TestGCPUtils(unittest.TestCase):
    """Test Google Cloud Platform utilities."""

    @mock.patch('satpy.demo.google_cloud_platform.urlopen')
    def test_is_gcp_instance(self, uo):
        """Test is_google_cloud_instance."""
        from satpy.demo.google_cloud_platform import is_google_cloud_instance, URLError
        uo.side_effect = URLError("Test Environment")
        self.assertFalse(is_google_cloud_instance())

    @mock.patch('satpy.demo.google_cloud_platform.gcsfs.GCSFileSystem')
    def test_get_bucket_files(self, gcsfs_cls):
        """Test get_bucket_files basic cases."""
        from satpy.demo.google_cloud_platform import get_bucket_files
        gcsfs_inst = mock.MagicMock()
        gcsfs_cls.return_value = gcsfs_inst
        gcsfs_inst.glob.return_value = ['a.nc', 'b.nc']
        filenames = get_bucket_files('*.nc', '.')
        expected = [os.path.join('.', 'a.nc'), os.path.join('.', 'b.nc')]
        self.assertEqual(expected, filenames)

        gcsfs_inst.glob.return_value = ['a.nc', 'b.nc']
        self.assertRaises(OSError, get_bucket_files, '*.nc', 'does_not_exist')

        # touch the file
        open('a.nc', 'w').close()
        gcsfs_inst.get.reset_mock()
        gcsfs_inst.glob.return_value = ['a.nc']
        filenames = get_bucket_files('*.nc', '.')
        self.assertEqual(['./a.nc'], filenames)
        gcsfs_inst.get.assert_not_called()

        # force redownload
        gcsfs_inst.get.reset_mock()
        gcsfs_inst.glob.return_value = ['a.nc']
        filenames = get_bucket_files('*.nc', '.', force=True)
        self.assertEqual(['./a.nc'], filenames)
        gcsfs_inst.get.assert_called_once()

        # if we don't get any results then we expect an exception
        gcsfs_inst.get.reset_mock()
        gcsfs_inst.glob.return_value = []
        self.assertRaises(OSError, get_bucket_files, '*.nc', '.')

    @mock.patch('satpy.demo.google_cloud_platform.gcsfs', None)
    def test_no_gcsfs(self):
        """Test that 'gcsfs' is required."""
        from satpy.demo.google_cloud_platform import get_bucket_files
        self.assertRaises(RuntimeError, get_bucket_files, '*.nc', '.')


def suite():
    """The test suite for test_demo."""
    loader = unittest.TestLoader()
    mysuite = unittest.TestSuite()
    mysuite.addTest(loader.loadTestsFromTestCase(TestDemo))
    mysuite.addTest(loader.loadTestsFromTestCase(TestGCPUtils))
    return mysuite


if __name__ == "__main__":
    unittest.main()
