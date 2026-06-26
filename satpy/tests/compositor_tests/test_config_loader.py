#!/usr/bin/env python
# Copyright (c) 2018-2025 Satpy developers
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

"""Tests for compositor config handling."""

import pytest

import satpy
from satpy.composites.config_loader import load_compositor_configs_for_sensors


def test_bad_sensor_yaml_configs(tmp_path):
    """Test composite YAML file with no sensor isn't loaded.

    But the bad YAML also shouldn't crash composite configuration loading.

    """
    comp_dir = tmp_path / "composites"
    comp_dir.mkdir()
    comp_yaml = comp_dir / "fake_sensor.yaml"
    with satpy.config.set(config_path=[tmp_path]):
        _create_fake_composite_config(comp_yaml)

        # no sensor_name in YAML, quietly ignored
        comps, _ = load_compositor_configs_for_sensors(["fake_sensor"])
        assert "fake_sensor" in comps
        assert "fake_composite" not in comps["fake_sensor"]


def _create_fake_composite_config(yaml_filename: str):
    import yaml

    from satpy.composites.aux_data import StaticImageCompositor

    with open(yaml_filename, "w") as comp_file:
        yaml.dump({
            "composites": {
                "fake_composite": {
                    "compositor": StaticImageCompositor,
                    "url": "http://example.com/image.png",
                },
            },
        },
            comp_file,
        )



# 8< v1.0
import satpy._instruments as inst_utils  # noqa


class TestUserConfigWithDeprecatedFilename:
    """Test finding user config with deprecated filename."""

    @pytest.fixture
    def user_comp_dir(self, tmp_path):
        """Get directory with user composites."""
        return tmp_path / "etc" / "composites"

    @pytest.fixture(autouse=True)
    def user_config_file(self, user_comp_dir):
        """Write user config with old instrument in the filename."""
        user_comp_dir.mkdir(parents=True)
        depr_file = user_comp_dir / "old-name.yaml"
        _create_fake_composite_config(depr_file)

    def test_finding_user_config(self, user_comp_dir, monkeypatch):
        """Test finding user config with old instrument in the filename."""
        monkeypatch.setitem(inst_utils.RENAMED_COMP_INSTRUMENTS, "New Name", "old-name")
        with satpy.config.set(config_path=[str(user_comp_dir.parent)]):
            with pytest.warns(DeprecationWarning, match="has been renamed"):
                comps, _ = load_compositor_configs_for_sensors(["New Name"])
            assert "New Name" in comps
# >8 v1.0
