#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Test for ancillary data downloading."""

from unittest import mock

import pytest
import yaml

from satpy.aux_download import DataDownloadMixin
from satpy.modifiers import ModifierBase

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmpdir

pooch = pytest.importorskip("pooch")

README_URL = "https://raw.githubusercontent.com/pytroll/satpy/main/README.rst"


class UnfriendlyModifier(ModifierBase, DataDownloadMixin):
    """Fake modifier that raises an exception in __init__."""

    def __init__(self, name, prerequisites=None, optional_prerequisites=None, **kwargs):
        """Raise an exception if we weren't provided any prerequisites."""
        if not prerequisites or len(prerequisites) != 1:
            raise ValueError("Unexpected number of prereqs")
        super().__init__(name, prerequisites, optional_prerequisites, **kwargs)
        self.register_data_files({'url': kwargs['url'],
                                  'filename': kwargs['filename'],
                                  'known_hash': kwargs['known_hash']})


def _setup_custom_composite_config(base_dir):
    from satpy.composites import StaticImageCompositor
    from satpy.modifiers.atmosphere import ReflectanceCorrector
    composite_config = base_dir.mkdir("composites").join("visir.yaml")
    with open(composite_config, 'w') as comp_file:
        yaml.dump({
            "sensor_name": "visir",
            "modifiers": {
                "test_modifier": {
                    "modifier": ReflectanceCorrector,
                    "url": README_URL,
                    "known_hash": None,
                },
                "unfriendly_modifier": {
                    "modifier": UnfriendlyModifier,
                    "url": README_URL,
                    "filename": "unfriendly.rst",
                    "known_hash": None,
                }
            },
            "composites": {
                "test_static": {
                    "compositor": StaticImageCompositor,
                    "url": README_URL,
                    "known_hash": None,
                },
            },
        }, comp_file)


def _setup_custom_reader_config(base_dir):
    reader_config = base_dir.mkdir("readers").join("fake.yaml")
    with open(reader_config, 'wt') as comp_file:
        # abstract base classes can't be converted so we do raw string
        comp_file.write("""
reader:
  name: "fake"
  reader: !!python/name:satpy.readers.yaml_reader.FileYAMLReader
  data_files:
    - url: {}
      known_hash: null
    - url: {}
      filename: "README2.rst"
      known_hash: null
file_types: {{}}
""".format(README_URL, README_URL))


def _setup_custom_writer_config(base_dir):
    writer_config = base_dir.mkdir("writers").join("fake.yaml")
    with open(writer_config, 'wt') as comp_file:
        # abstract base classes can't be converted so we do raw string
        comp_file.write("""
writer:
  name: "fake"
  writer: !!python/name:satpy.writers.Writer
  data_files:
    - url: {}
      known_hash: null
    - url: {}
      filename: "README2.rst"
      known_hash: null
""".format(README_URL, README_URL))


def _assert_reader_files_downloaded(readers, found_files):
    r_cond1 = 'readers/README.rst' in found_files
    r_cond2 = 'readers/README2.rst' in found_files
    if readers is not None and not readers:
        r_cond1 = not r_cond1
        r_cond2 = not r_cond2
    assert r_cond1
    assert r_cond2


def _assert_writer_files_downloaded(writers, found_files):
    w_cond1 = 'writers/README.rst' in found_files
    w_cond2 = 'writers/README2.rst' in found_files
    if writers is not None and not writers:
        w_cond1 = not w_cond1
        w_cond2 = not w_cond2
    assert w_cond1
    assert w_cond2


def _assert_comp_files_downloaded(comp_sensors, found_files):
    comp_cond = 'composites/README.rst' in found_files
    if comp_sensors is not None and not comp_sensors:
        comp_cond = not comp_cond
    assert comp_cond


def _assert_mod_files_downloaded(comp_sensors, found_files):
    mod_cond = 'modifiers/README.rst' in found_files
    unfriendly_cond = 'modifiers/unfriendly.rst' in found_files
    if comp_sensors is not None and not comp_sensors:
        mod_cond = not mod_cond
    assert mod_cond
    assert not unfriendly_cond


class TestDataDownload:
    """Test basic data downloading functionality."""

    @pytest.fixture(autouse=True)
    def _setup_custom_configs(self, tmpdir):
        _setup_custom_composite_config(tmpdir)
        _setup_custom_reader_config(tmpdir)
        _setup_custom_writer_config(tmpdir)
        self.tmpdir = tmpdir

    @pytest.mark.parametrize('comp_sensors', [[], None, ['visir']])
    @pytest.mark.parametrize('writers', [[], None, ['fake']])
    @pytest.mark.parametrize('readers', [[], None, ['fake']])
    def test_find_registerable(self, readers, writers, comp_sensors):
        """Test that find_registerable finds some things."""
        import satpy
        from satpy.aux_download import find_registerable_files
        with satpy.config.set(config_path=[self.tmpdir]), \
             mock.patch('satpy.aux_download._FILE_REGISTRY', {}):
            found_files = find_registerable_files(
                readers=readers, writers=writers,
                composite_sensors=comp_sensors,
            )

            _assert_reader_files_downloaded(readers, found_files)
            _assert_writer_files_downloaded(writers, found_files)
            _assert_comp_files_downloaded(comp_sensors, found_files)
            _assert_mod_files_downloaded(comp_sensors, found_files)

    def test_limited_find_registerable(self):
        """Test that find_registerable doesn't find anything when limited."""
        import satpy
        from satpy.aux_download import find_registerable_files
        file_registry = {}
        with satpy.config.set(config_path=[self.tmpdir]), \
             mock.patch('satpy.aux_download._FILE_REGISTRY', file_registry):
            found_files = find_registerable_files(
                readers=[], writers=[], composite_sensors=[],
            )
            assert not found_files

    def test_retrieve(self):
        """Test retrieving a single file."""
        import satpy
        from satpy.aux_download import find_registerable_files, retrieve
        file_registry = {}
        with satpy.config.set(config_path=[self.tmpdir], data_dir=str(self.tmpdir)), \
             mock.patch('satpy.aux_download._FILE_REGISTRY', file_registry):
            comp_file = 'composites/README.rst'
            found_files = find_registerable_files()
            assert comp_file in found_files
            assert not self.tmpdir.join(comp_file).exists()
            retrieve(comp_file)
            assert self.tmpdir.join(comp_file).exists()

    def test_offline_retrieve(self):
        """Test retrieving a single file when offline."""
        import satpy
        from satpy.aux_download import find_registerable_files, retrieve
        file_registry = {}
        with satpy.config.set(config_path=[self.tmpdir], data_dir=str(self.tmpdir), download_aux=True), \
             mock.patch('satpy.aux_download._FILE_REGISTRY', file_registry):
            comp_file = 'composites/README.rst'
            found_files = find_registerable_files()
            assert comp_file in found_files

            # the file doesn't exist, we can't download it
            assert not self.tmpdir.join(comp_file).exists()
            with satpy.config.set(download_aux=False):
                pytest.raises(RuntimeError, retrieve, comp_file)

            # allow downloading and get it
            retrieve(comp_file)
            assert self.tmpdir.join(comp_file).exists()

            # turn off downloading and make sure we get local file
            with satpy.config.set(download_aux=False):
                local_file = retrieve(comp_file)
                assert local_file

    def test_offline_retrieve_all(self):
        """Test registering and retrieving all files fails when offline."""
        import satpy
        from satpy.aux_download import retrieve_all
        with satpy.config.set(config_path=[self.tmpdir], data_dir=str(self.tmpdir), download_aux=False):
            pytest.raises(RuntimeError, retrieve_all)

    def test_retrieve_all(self):
        """Test registering and retrieving all files."""
        import satpy
        from satpy.aux_download import retrieve_all
        file_registry = {}
        file_urls = {}
        with satpy.config.set(config_path=[self.tmpdir], data_dir=str(self.tmpdir)), \
             mock.patch('satpy.aux_download._FILE_REGISTRY', file_registry), \
             mock.patch('satpy.aux_download._FILE_URLS', file_urls), \
             mock.patch('satpy.aux_download.find_registerable_files'):
            comp_file = 'composites/README.rst'
            file_registry[comp_file] = None
            file_urls[comp_file] = README_URL
            assert not self.tmpdir.join(comp_file).exists()
            retrieve_all()
            assert self.tmpdir.join(comp_file).exists()

    def test_no_downloads_in_tests(self):
        """Test that tests aren't allowed to download stuff."""
        import satpy
        from satpy.aux_download import register_file, retrieve

        file_registry = {}
        with satpy.config.set(config_path=[self.tmpdir], data_dir=str(self.tmpdir),
                              download_aux=True), \
             mock.patch('satpy.aux_download._FILE_REGISTRY', file_registry):
            cache_key = 'myfile.rst'
            register_file(README_URL, cache_key)
            assert not self.tmpdir.join(cache_key).exists()
            pytest.raises(RuntimeError, retrieve, cache_key)
            # touch the file so it gets created
            open(self.tmpdir.join(cache_key), 'w').close()
            # offline downloading should still be allowed
            with satpy.config.set(download_aux=False):
                retrieve(cache_key)

    def test_download_script(self):
        """Test basic functionality of the download script."""
        import satpy
        from satpy.aux_download import retrieve_all_cmd
        file_registry = {}
        file_urls = {}
        with satpy.config.set(config_path=[self.tmpdir]), \
             mock.patch('satpy.aux_download._FILE_REGISTRY', file_registry), \
             mock.patch('satpy.aux_download._FILE_URLS', file_urls), \
             mock.patch('satpy.aux_download.find_registerable_files'):
            comp_file = 'composites/README.rst'
            file_registry[comp_file] = None
            file_urls[comp_file] = README_URL
            assert not self.tmpdir.join(comp_file).exists()
            retrieve_all_cmd(argv=["--data-dir", str(self.tmpdir)])
            assert self.tmpdir.join(comp_file).exists()
