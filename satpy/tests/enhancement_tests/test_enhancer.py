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
"""Test enhancer class."""
from __future__ import annotations

import pytest

from satpy.enhancements.enhancer import Enhancer


class TestEnhancer:
    """Test basic `Enhancer` functionality with builtin configs."""

    def test_basic_init_no_args(self):
        """Test Enhancer init with no arguments passed."""
        e = Enhancer()
        assert e.enhancement_tree is not None

    def test_basic_init_no_enh(self):
        """Test Enhancer init requesting no enhancements."""
        e = Enhancer(enhancement_config_file=False)
        assert e.enhancement_tree is None

    def test_basic_init_provided_enh(self):
        """Test Enhancer init with string enhancement configs."""
        e = Enhancer(enhancement_config_file=["""enhancements:
  enh1:
    standard_name: toa_bidirectional_reflectance
    operations:
    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs: {stretch: linear}
"""])
        assert e.enhancement_tree is not None

    def test_init_nonexistent_enh_file(self):
        """Test Enhancer init with a nonexistent enhancement configuration file."""
        with pytest.raises(ValueError, match="YAML file doesn't exist or string is not YAML dict:.*"):
            Enhancer(enhancement_config_file="is_not_a_valid_filename_?.yaml")

    def test_print_tree(self, capsys):
        """Test enhancement decision tree printing."""
        enh = Enhancer()
        enh.enhancement_tree.print_tree()
        stdout = capsys.readouterr().out
        lines = stdout.splitlines()
        assert lines[0].startswith("name=<wildcard>")
        # make sure lines are indented
        assert lines[1].startswith("  reader=")
        assert lines[2].startswith("    platform_name=")
