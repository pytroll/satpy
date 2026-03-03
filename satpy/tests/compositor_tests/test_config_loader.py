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

import satpy


def test_bad_sensor_yaml_configs(tmp_path):
    """Test composite YAML file with no sensor isn't loaded.

    But the bad YAML also shouldn't crash composite configuration loading.

    """
    from satpy.composites.config_loader import load_compositor_configs_for_sensors

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


def test_composite_warning_is_emitted_on_compositor_use(tmp_path):
    """Test that composite 'warnings' fires when the compositor is fetched, not when configs are loaded."""
    import pytest

    from satpy.composites.config_loader import load_compositor_configs_for_sensors
    from satpy.dataset import DataQuery
    from satpy.dependency_tree import DependencyTree

    comp_dir = tmp_path / "composites"
    comp_dir.mkdir()
    _create_fake_composite_config_with_warnings(comp_dir / "fake_sensor.yaml")

    with satpy.config.set(config_path=[tmp_path]):
        comps, _ = load_compositor_configs_for_sensors(["fake_sensor"])   # no warning here

    tree = DependencyTree({}, comps, {})
    with pytest.warns(DeprecationWarning, match="Use new_composite instead"):
        tree.get_compositor(DataQuery(name="old_composite"))


def _create_fake_composite_config_with_warnings(yaml_filename):
    import yaml

    from satpy.composites.aux_data import StaticImageCompositor

    with open(yaml_filename, "w") as comp_file:
        yaml.dump({
            "sensor_name": "fake_sensor",
            "composites": {
                "old_composite": {
                    "compositor": StaticImageCompositor,
                    "url": "http://example.com/image.png",
                    "warnings": {"DeprecationWarning": "Use new_composite instead"},
                },
            },
        }, comp_file)


def test_composite_tags_stored_on_compositor(tmp_path):
    """Test that a composite with 'tags' in its YAML has those tags stored in its attrs."""
    from satpy.composites.config_loader import load_compositor_configs_for_sensors

    comp_dir = tmp_path / "composites"
    comp_dir.mkdir()
    comp_yaml = comp_dir / "fake_sensor.yaml"
    _create_fake_composite_config_with_tags(comp_yaml, tags=["wmo"])

    with satpy.config.set(config_path=[tmp_path]):
        comps, _ = load_compositor_configs_for_sensors(["fake_sensor"])

    compositor = next(iter(comps["fake_sensor"].values()))
    assert compositor.attrs.get("tags") == ["wmo"]


def _create_fake_composite_config_with_tags(yaml_filename, tags):
    import yaml

    from satpy.composites.aux_data import StaticImageCompositor

    with open(yaml_filename, "w") as comp_file:
        yaml.dump({
            "sensor_name": "fake_sensor",
            "composites": {
                "tagged_composite": {
                    "compositor": StaticImageCompositor,
                    "url": "http://example.com/image.png",
                    "tags": tags,
                },
            },
        }, comp_file)
