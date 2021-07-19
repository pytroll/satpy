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
import sys
import unittest
from contextlib import contextmanager
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


class TestAHIDemoDownload:
    """Test the AHI demo data download."""

    @mock.patch.dict(sys.modules, {'s3fs': mock.MagicMock()})
    def test_ahi_full_download(self):
        """Test that the himawari download works as expected."""
        from satpy.demo import download_typhoon_surigae_ahi
        from tempfile import gettempdir
        files = download_typhoon_surigae_ahi(base_dir=gettempdir())
        assert len(files) == 160

    @mock.patch.dict(sys.modules, {'s3fs': mock.MagicMock()})
    def test_ahi_partial_download(self):
        """Test that the himawari download works as expected."""
        from satpy.demo import download_typhoon_surigae_ahi
        from tempfile import gettempdir
        files = download_typhoon_surigae_ahi(base_dir=gettempdir(), segments=[4, 9], channels=[1, 2, 3])
        assert len(files) == 6


class TestSEVIRIHRITDemoDownload(unittest.TestCase):
    """Test case for downloading an hrit tarball."""

    def setUp(self):
        """Set up the test case."""
        self.files = ['hrit',
                      'hrit/H-000-MSG4__-MSG4________-_________-EPI______-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000001___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000002___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000003___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000004___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000005___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000009___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000010___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000011___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000012___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000013___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000014___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000015___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000016___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000017___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000018___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000019___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000020___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000021___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000022___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000023___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-HRV______-000024___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_016___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_016___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_016___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_039___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_039___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_039___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_087___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_087___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_087___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_097___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_097___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_097___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_108___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_108___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_108___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_120___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_120___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_120___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_134___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_134___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-IR_134___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-_________-PRO______-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-VIS006___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-VIS006___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-VIS006___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-VIS008___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-VIS008___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-VIS008___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-WV_062___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-WV_062___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-WV_062___-000008___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-WV_073___-000006___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-WV_073___-000007___-202003090800-__',
                      'hrit/H-000-MSG4__-MSG4________-WV_073___-000008___-202003090800-__']

    def test_download_filestream(self):
        """Test that downloading a file works."""
        from satpy.demo import download_filestream
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='w', delete=False) as fd:
            fd.write('lots of data')
            filename = fd.name
        try:
            url = "file:///" + filename
            assert download_filestream(url).read() == b"lots of data"
        finally:
            from contextlib import suppress
            with suppress(PermissionError):
                os.remove(filename)

    def test_unpack_tarball_stream(self):
        """Test unpacking a tarball stream."""
        from satpy.demo import unpack_tarball_stream

        from tempfile import TemporaryDirectory
        import glob

        with make_fake_tarball(self.files) as tmp_filename:
            with TemporaryDirectory() as tmp_dirname_output:
                with open(tmp_filename, 'rb') as fd:
                    unpack_tarball_stream(fd, tmp_dirname_output)
                    os.chdir(tmp_dirname_output)
                    assert set(glob.glob(os.path.join("hrit", "*"))) == set(self.files[1:])

    def test_unpack_tarball_stream_returns_filenames(self):
        """Test unpacking a tarball returns the filenames."""
        from satpy.demo import unpack_tarball_stream
        from tempfile import TemporaryDirectory

        with make_fake_tarball(self.files) as tmp_filename:
            with TemporaryDirectory() as tmp_dirname_output:
                with open(tmp_filename, 'rb') as fd:
                    res_files = unpack_tarball_stream(fd, tmp_dirname_output)
                    assert res_files == self.files


@contextmanager
def make_fake_tarball(files):
    """Make a fake tarball."""
    from tempfile import TemporaryDirectory, NamedTemporaryFile
    import tarfile
    from pathlib import Path
    with TemporaryDirectory() as tmp_dirname_input:
        with NamedTemporaryFile(mode='wb') as tfd:
            tmp_filename = tfd.name
            tfd.close()
            with tarfile.open(tmp_filename, mode="w:gz") as tf:
                os.makedirs(os.path.join(tmp_dirname_input, "hrit"))
                tf.add(os.path.join(tmp_dirname_input, "hrit"), arcname="hrit")
                for filename in files[1:]:
                    Path(os.path.join(tmp_dirname_input, filename)).touch()
                    tf.add(os.path.join(tmp_dirname_input, filename), arcname=filename)
            yield tmp_filename
