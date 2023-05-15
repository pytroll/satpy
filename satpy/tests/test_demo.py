#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019-2021 Satpy developers
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
from __future__ import annotations

import contextlib
import io
import os
import sys
import tarfile
import unittest
from collections import defaultdict
from unittest import mock

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - tmpdir
# - monkeypatch


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
        from satpy.demo._google_cloud_platform import URLError, is_google_cloud_instance
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
        from tempfile import gettempdir

        from satpy.demo import download_typhoon_surigae_ahi
        files = download_typhoon_surigae_ahi(base_dir=gettempdir())
        assert len(files) == 160

    @mock.patch.dict(sys.modules, {'s3fs': mock.MagicMock()})
    def test_ahi_partial_download(self):
        """Test that the himawari download works as expected."""
        from tempfile import gettempdir

        from satpy.demo import download_typhoon_surigae_ahi
        files = download_typhoon_surigae_ahi(base_dir=gettempdir(), segments=[4, 9], channels=[1, 2, 3])
        assert len(files) == 6


def _create_and_populate_dummy_tarfile(fn):
    """Populate a dummy tarfile with dummy files."""
    fn.parent.mkdir(exist_ok=True, parents=True)
    with tarfile.open(fn, mode="x:gz") as tf:
        for i in range(3):
            with open(f"fci-rc{i:d}", "w"):
                pass
            tf.addfile(tf.gettarinfo(name=f"fci-rc{i:d}"))


def test_fci_download(tmp_path, monkeypatch):
    """Test download of FCI test data."""
    from satpy.demo import download_fci_test_data
    monkeypatch.chdir(tmp_path)

    def fake_download_url(url, nm):
        """Create a dummy tarfile.

        Create a dummy tarfile.

        Intended as a drop-in replacement for demo.utils.download_url.
        """
        _create_and_populate_dummy_tarfile(nm)

    with mock.patch("satpy.demo.fci.utils.download_url", new=fake_download_url):
        files = download_fci_test_data(tmp_path)
    assert len(files) == 3
    assert files == ["fci-rc0", "fci-rc1", "fci-rc2"]
    for f in files:
        assert os.path.exists(f)


class _FakeRequest:
    """Fake object to act like a requests return value when downloading a file."""

    requests_log: list[str] = []

    def __init__(self, url, stream=None):
        self._filename = os.path.basename(url)
        self.headers = {}
        self.requests_log.append(url)
        del stream  # just mimicking requests 'get'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def raise_for_status(self):
        return

    def _get_fake_bytesio(self):
        filelike_obj = io.BytesIO()
        filelike_obj.write(self._filename.encode("ascii"))
        filelike_obj.seek(0)
        return filelike_obj

    def iter_content(self, chunk_size):
        """Return generator of 'chunk_size' at a time."""
        bytes_io = self._get_fake_bytesio()
        x = bytes_io.read(chunk_size)
        while x:
            yield x
            x = bytes_io.read(chunk_size)


@mock.patch('satpy.demo.utils.requests')
class TestVIIRSSDRDemoDownload:
    """Test VIIRS SDR downloading."""

    ALL_BAND_PREFIXES = ("SVI01", "SVI02", "SVI03", "SVI04", "SVI05",
                         "SVM01", "SVM02", "SVM03", "SVM04", "SVM05", "SVM06", "SVM07", "SVM08", "SVM09", "SVM10",
                         "SVM11", "SVM12", "SVM13", "SVM14", "SVM15", "SVM16",
                         "SVDNB")
    ALL_GEO_PREFIXES = ("GITCO", "GMTCO", "GDNBO")

    def test_download(self, _requests, tmpdir):
        """Test downloading VIIRS SDR data."""
        from satpy.demo import get_viirs_sdr_20170128_1229
        _requests.get.side_effect = _FakeRequest
        with mock_filesystem():
            files = get_viirs_sdr_20170128_1229(base_dir=str(tmpdir))
            assert len(files) == 10 * (16 + 5 + 1 + 3)  # 10 granules * (5 I bands + 16 M bands + 1 DNB + 3 geolocation)
            self._assert_bands_in_filenames_and_contents(self.ALL_BAND_PREFIXES + self.ALL_GEO_PREFIXES, files, 10)

    def test_do_not_download_the_files_twice(self, _requests, tmpdir):
        """Test re-downloading VIIRS SDR data."""
        from satpy.demo import get_viirs_sdr_20170128_1229
        get_mock = mock.MagicMock()
        _requests.get.return_value.__enter__ = get_mock
        with mock_filesystem():
            files = get_viirs_sdr_20170128_1229(base_dir=str(tmpdir))
            new_files = get_viirs_sdr_20170128_1229(base_dir=str(tmpdir))

        total_num_files = 10 * (16 + 5 + 1 + 3)  # 10 granules * (5 I bands + 16 M bands + 1 DNB + 3 geolocation)
        assert len(new_files) == total_num_files
        assert get_mock.call_count == total_num_files
        assert new_files == files

    def test_download_channels_num_granules_im(self, _requests, tmpdir):
        """Test downloading VIIRS SDR I/M data with select granules."""
        from satpy.demo import get_viirs_sdr_20170128_1229
        _requests.get.side_effect = _FakeRequest
        with mock_filesystem():
            files = get_viirs_sdr_20170128_1229(base_dir=str(tmpdir),
                                                channels=("I01", "M01"))
            assert len(files) == 10 * (1 + 1 + 2)  # 10 granules * (1 I band + 1 M band + 2 geolocation)
            self._assert_bands_in_filenames_and_contents(("SVI01", "SVM01", "GITCO", "GMTCO"), files, 10)

    def test_download_channels_num_granules_im_twice(self, _requests, tmpdir):
        """Test re-downloading VIIRS SDR I/M data with select granules."""
        from satpy.demo import get_viirs_sdr_20170128_1229
        get_mock = mock.MagicMock()
        _requests.get.return_value.__enter__ = get_mock
        with mock_filesystem():
            files = get_viirs_sdr_20170128_1229(base_dir=str(tmpdir),
                                                channels=("I01", "M01"))
            num_first_batch = 10 * (1 + 1 + 2)  # 10 granules * (1 I band + 1 M band + 2 geolocation)
            assert len(files) == num_first_batch

            files = get_viirs_sdr_20170128_1229(base_dir=str(tmpdir),
                                                channels=("I01", "M01"),
                                                granules=(2, 3))
            assert len(files) == 2 * (1 + 1 + 2)  # 2 granules * (1 I band + 1 M band + 2 geolocation)
            assert get_mock.call_count == num_first_batch

    def test_download_channels_num_granules_dnb(self, _requests, tmpdir):
        """Test downloading and re-downloading VIIRS SDR DNB data with select granules."""
        from satpy.demo import get_viirs_sdr_20170128_1229
        _requests.get.side_effect = _FakeRequest
        with mock_filesystem():
            files = get_viirs_sdr_20170128_1229(base_dir=str(tmpdir),
                                                channels=("DNB",),
                                                granules=(5, 6, 7, 8, 9))
            assert len(files) == 5 * (1 + 1)  # 5 granules * (1 DNB + 1 geolocation)
            self._assert_bands_in_filenames_and_contents(("SVDNB", "GDNBO"), files, 5)

    def _assert_bands_in_filenames_and_contents(self, band_prefixes, filenames, num_files_per_band):
        self._assert_bands_in_filenames(band_prefixes, filenames, num_files_per_band)
        self._assert_file_contents(filenames)

    @staticmethod
    def _assert_bands_in_filenames(band_prefixes, filenames, num_files_per_band):
        for band_name in band_prefixes:
            files_for_band = [x for x in filenames if band_name in x]
            assert files_for_band
            assert len(set(files_for_band)) == num_files_per_band

    @staticmethod
    def _assert_file_contents(filenames):
        for fn in filenames:
            with open(fn, "rb") as fake_hdf5_file:
                assert fake_hdf5_file.read().decode("ascii") == os.path.basename(fn)


@contextlib.contextmanager
def mock_filesystem():
    """Create a mock filesystem, patching `open` and `os.path.isfile`."""
    class FakeFile:
        """Fake file based on BytesIO."""

        def __init__(self):
            self.io = io.BytesIO()

        def __enter__(self):
            return self.io

        def __exit__(self, *args, **kwargs):
            self.io.seek(0)

    fake_fs = defaultdict(FakeFile)
    mo = mock.mock_open()

    def fun(filename, *args, **kwargs):
        return fake_fs[filename]

    mo.side_effect = fun
    with mock.patch("builtins.open", mo):
        with mock.patch("os.path.isfile") as isfile:
            isfile.side_effect = (lambda target: target in fake_fs)
            yield


def test_fs():
    """Test the mock filesystem."""
    with mock_filesystem():
        with open("somefile", "w") as fd:
            fd.write(b"bla")
        with open("someotherfile", "w") as fd:
            fd.write(b"bli")
        with open("somefile", "r") as fd:
            assert fd.read() == b"bla"
        with open("someotherfile", "r") as fd:
            assert fd.read() == b"bli"
        assert os.path.isfile("somefile")
        assert not os.path.isfile("missingfile")


class TestSEVIRIHRITDemoDownload(unittest.TestCase):
    """Test case for downloading an hrit tarball."""

    def setUp(self):
        """Set up the test case."""
        from satpy.demo.seviri_hrit import generate_subset_of_filenames
        self.subdir = os.path.join(".", "seviri_hrit", "20180228_1500")
        self.files = generate_subset_of_filenames(base_dir=self.subdir)

        self.patcher = mock.patch('satpy.demo.utils.requests.get', autospec=True)
        self.get_mock = self.patcher.start()

        _FakeRequest.requests_log = []

    def tearDown(self):
        """Tear down the test case."""
        self.patcher.stop()

    def test_download_gets_files_with_contents(self):
        """Test downloading SEVIRI HRIT data with content."""
        from satpy.demo import download_seviri_hrit_20180228_1500
        self.get_mock.side_effect = _FakeRequest
        with mock_filesystem():
            files = download_seviri_hrit_20180228_1500()
            assert len(files) == 114
            assert set(files) == set(self.files)
            for the_file in files:
                with open(the_file, mode="r") as fd:
                    assert fd.read().decode("utf8") == os.path.basename(the_file)

    def test_download_from_zenodo(self):
        """Test downloading SEVIRI HRIT data from zenodo."""
        from satpy.demo import download_seviri_hrit_20180228_1500
        self.get_mock.side_effect = _FakeRequest
        with mock_filesystem():
            download_seviri_hrit_20180228_1500()
            assert _FakeRequest.requests_log[0].startswith("https://zenodo.org")

    def test_download_a_subset_of_files(self):
        """Test downloading a subset of files."""
        from satpy.demo import download_seviri_hrit_20180228_1500
        with mock_filesystem():
            files = download_seviri_hrit_20180228_1500(subset={"HRV": [1, 2, 3], "IR_108": [1, 2], "EPI": None})
            assert set(files) == set(os.path.join(self.subdir, filename) for filename in [
                'H-000-MSG4__-MSG4________-_________-EPI______-201802281500-__',
                'H-000-MSG4__-MSG4________-HRV______-000001___-201802281500-__',
                'H-000-MSG4__-MSG4________-HRV______-000002___-201802281500-__',
                'H-000-MSG4__-MSG4________-HRV______-000003___-201802281500-__',
                'H-000-MSG4__-MSG4________-IR_108___-000001___-201802281500-__',
                'H-000-MSG4__-MSG4________-IR_108___-000002___-201802281500-__',
            ])

    def test_do_not_download_same_file_twice(self):
        """Test that files are not downloaded twice."""
        from satpy.demo import download_seviri_hrit_20180228_1500
        get_mock = mock.MagicMock()
        self.get_mock.return_value.__enter__ = get_mock
        with mock_filesystem():
            files = download_seviri_hrit_20180228_1500(subset={"HRV": [1, 2, 3], "IR_108": [1, 2], "EPI": None})
            new_files = download_seviri_hrit_20180228_1500(subset={"HRV": [1, 2, 3], "IR_108": [1, 2], "EPI": None})
            assert set(files) == set(new_files)
            assert get_mock.call_count == 6

    def test_download_to_output_directory(self):
        """Test downloading to an output directory."""
        from tempfile import gettempdir

        from satpy.demo import download_seviri_hrit_20180228_1500
        with mock_filesystem():
            base_dir = gettempdir()
            files = download_seviri_hrit_20180228_1500(base_dir=base_dir)
            assert files[0].startswith(base_dir)
