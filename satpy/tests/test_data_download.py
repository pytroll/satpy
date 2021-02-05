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

from satpy.readers.yaml_reader import FileYAMLReader
from satpy.writers import Writer
from satpy.data_download import DataDownloadMixin

import pytest
import yaml

pooch = pytest.importorskip("pooch")

README_URL = "https://raw.githubusercontent.com/pytroll/satpy/master/README.rst"


def _setup_custom_composite_config(base_dir):
    from satpy.composites import StaticImageCompositor
    composite_config = base_dir.mkdir("composites").join("visir.yaml")
    with open(composite_config, 'w') as comp_file:
        yaml.dump({
            "sensor_name": "visir",
            "composites": {
                "test_static": {
                    "compositor": StaticImageCompositor,
                    "url": README_URL,
                    "known_hash": None,
                },
            },
        }, comp_file)


class FakeMixedReader(FileYAMLReader, DataDownloadMixin):
    """Fake reader that uses the data download mixin."""

    def __init__(self, *args, **kwargs):
        """Initialize reader and data downloading."""
        super().__init__(*args, **kwargs)
        self.data_files = []
        self.register_data_files()

    @property
    def start_time(self):
        """Start time of the reader."""
        return None

    @property
    def end_time(self):
        """End time of the reader."""
        return None

    def filter_selected_filenames(self, filenames):
        """Filter provided filenames by parameters in reader configuration."""
        return filenames

    def load(self, dataset_keys):
        """Load some data."""
        return {}


def _setup_custom_reader_config(base_dir):
    reader_config = base_dir.mkdir("readers").join("fake.yaml")
    with open(reader_config, 'wt') as comp_file:
        # abstract base classes can't be converted so we do raw string
        comp_file.write("""
reader:
  name: "fake"
  reader: !!python/name:satpy.tests.test_data_download.FakeMixedReader
  data_files:
    - url: {}
      known_hash: null
    - url: {}
      filename: "README2.rst"
      known_hash: null
file_types: {{}}
""".format(README_URL, README_URL))


class FakeMixedWriter(Writer, DataDownloadMixin):
    """Fake reader that uses the data download mixin."""

    def __init__(self, *args, **kwargs):
        """Initialize writer and data downloading."""
        super().__init__(*args, **kwargs)
        self.data_files = []
        self.register_data_files()


def _setup_custom_writer_config(base_dir):
    writer_config = base_dir.mkdir("writers").join("fake.yaml")
    with open(writer_config, 'wt') as comp_file:
        # abstract base classes can't be converted so we do raw string
        comp_file.write("""
writer:
  name: "fake"
  writer: !!python/name:satpy.tests.test_data_download.FakeMixedWriter
  data_files:
    - url: {}
      known_hash: null
    - url: {}
      filename: "README2.rst"
      known_hash: null
""".format(README_URL, README_URL))


def _setup_custom_configs(base_dir):
    _setup_custom_composite_config(base_dir)
    _setup_custom_reader_config(base_dir)
    _setup_custom_writer_config(base_dir)


class TestDataDownload:
    """Test basic data downloading functionality."""

    def test_find_registerable(self, tmpdir):
        """Test that find_registerable finds some things."""
        import satpy
        from satpy.data_download import find_registerable_files
        _setup_custom_configs(tmpdir)
        file_registry = {}
        with satpy.config.set(config_path=[tmpdir]), \
             mock.patch('satpy.data_download._FILE_REGISTRY', file_registry):
            found_files = find_registerable_files()
            assert 'composites/StaticImageCompositor/README.rst' in found_files
            assert 'readers/FakeMixedReader/README.rst' in found_files
            assert 'readers/FakeMixedReader/README2.rst' in found_files
            assert 'writers/FakeMixedWriter/README.rst' in found_files
            assert 'writers/FakeMixedWriter/README2.rst' in found_files

    def test_retrieve(self, tmpdir):
        """Test retrieving a single file."""
        import satpy
        from satpy.data_download import find_registerable_files, retrieve
        _setup_custom_configs(tmpdir)
        file_registry = {}
        with satpy.config.set(config_path=[tmpdir], data_dir=str(tmpdir)), \
             mock.patch('satpy.data_download._FILE_REGISTRY', file_registry):
            comp_file = 'composites/StaticImageCompositor/README.rst'
            found_files = find_registerable_files()
            assert comp_file in found_files
            assert not tmpdir.join(comp_file).exists()
            retrieve(comp_file)
            assert tmpdir.join(comp_file).exists()

    def test_retrieve_all(self, tmpdir):
        """Test registering and retrieving all files."""
        import satpy
        from satpy.data_download import retrieve_all
        _setup_custom_configs(tmpdir)
        file_registry = {}
        file_urls = {}
        with satpy.config.set(config_path=[tmpdir], data_dir=str(tmpdir)), \
             mock.patch('satpy.data_download._FILE_REGISTRY', file_registry), \
             mock.patch('satpy.data_download._FILE_URLS', file_urls), \
             mock.patch('satpy.data_download.find_registerable_files'):
            comp_file = 'composites/StaticImageCompositor/README.rst'
            file_registry[comp_file] = None
            file_urls[comp_file] = README_URL
            assert not tmpdir.join(comp_file).exists()
            retrieve_all()
            assert tmpdir.join(comp_file).exists()
