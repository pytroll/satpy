# Copyright (c) 2025 Satpy developers
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
"""Tests for the writer config module."""
from __future__ import annotations

import os


class TestYAMLFiles:
    """Test and analyze the writer configuration files."""

    def test_filename_matches_writer_name(self):
        """Test that every writer filename matches the name in the YAML."""
        import yaml

        class IgnoreLoader(yaml.SafeLoader):

            def _ignore_all_tags(self, tag_suffix, node):
                return tag_suffix + " " + node.value
        IgnoreLoader.add_multi_constructor("", IgnoreLoader._ignore_all_tags)

        from satpy._config import glob_config
        from satpy.writers.core.config import read_writer_config
        for writer_config in glob_config("writers/*.yaml"):
            writer_fn = os.path.basename(writer_config)
            writer_fn_name = os.path.splitext(writer_fn)[0]
            writer_info = read_writer_config([writer_config],
                                             loader=IgnoreLoader)
            assert writer_fn_name == writer_info["name"]

    def test_available_writers(self):
        """Test the 'available_writers' function."""
        from satpy import available_writers
        writer_names = available_writers()
        assert len(writer_names) > 0
        assert isinstance(writer_names[0], str)
        assert "geotiff" in writer_names

        writer_infos = available_writers(as_dict=True)
        assert len(writer_names) == len(writer_infos)
        assert isinstance(writer_infos[0], dict)
        for writer_info in writer_infos:
            assert "name" in writer_info
