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
"""Test objects and functions in the satpy.config module."""
from __future__ import annotations

import contextlib
import os
import sys
import unittest
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Callable, Iterator
from unittest import mock

import pytest

import satpy
from satpy import DatasetDict
from satpy._config import cached_entry_points
from satpy.composites.config_loader import load_compositor_configs_for_sensors

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path


class TestBuiltinAreas(unittest.TestCase):
    """Test that the builtin areas are all valid."""

    def test_areas_pyproj(self):
        """Test all areas have valid projections with pyproj."""
        import numpy as np
        import pyproj
        import xarray as xr
        from pyresample import parse_area_file
        from pyresample.geometry import SwathDefinition

        from satpy.resample import get_area_file

        lons = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lats = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lons = xr.DataArray(lons)
        lats = xr.DataArray(lats)
        swath_def = SwathDefinition(lons, lats)
        all_areas = parse_area_file(get_area_file())
        for area_obj in all_areas:
            if hasattr(area_obj, 'freeze'):
                try:
                    area_obj = area_obj.freeze(lonslats=swath_def)
                except RuntimeError:
                    # we didn't provide enough info to freeze, hard to guess
                    # in a generic test so just skip this area
                    continue
            proj_dict = area_obj.proj_dict
            _ = pyproj.Proj(proj_dict)

    def test_areas_rasterio(self):
        """Test all areas have valid projections with rasterio."""
        try:
            from rasterio.crs import CRS
        except ImportError:
            return unittest.skip("Missing rasterio dependency")
        if not hasattr(CRS, 'from_dict'):
            return unittest.skip("RasterIO 1.0+ required")

        import numpy as np
        import xarray as xr
        from pyresample import parse_area_file
        from pyresample.geometry import SwathDefinition

        from satpy.resample import get_area_file

        lons = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lats = np.array([[0, 0.1, 0.2], [0.05, 0.15, 0.25]])
        lons = xr.DataArray(lons)
        lats = xr.DataArray(lats)
        swath_def = SwathDefinition(lons, lats)
        all_areas = parse_area_file(get_area_file())
        for area_obj in all_areas:
            if hasattr(area_obj, 'freeze'):
                try:
                    area_obj = area_obj.freeze(lonslats=swath_def)
                except RuntimeError:
                    # we didn't provide enough info to freeze, hard to guess
                    # in a generic test so just skip this area
                    continue
            proj_dict = area_obj.proj_dict
            if proj_dict.get('proj') in ('ob_tran', 'nsper') and \
                    'wktext' not in proj_dict:
                # FIXME: rasterio doesn't understand ob_tran unless +wktext
                # See: https://github.com/pyproj4/pyproj/issues/357
                # pyproj 2.0+ seems to drop wktext from PROJ dict
                continue
            _ = CRS.from_dict(proj_dict)


@contextlib.contextmanager
def fake_plugin_etc_path(
        tmp_path: Path,
        entry_point_names: dict[str, list[str]],
) -> Iterator[Path]:
    """Create a fake satpy plugin entry point.

    This mocks the necessary methods to trick Satpy into thinking a plugin
    package is installed and has made a satpy plugin available.

    """
    etc_path, entry_points, module_paths = _get_entry_points_and_etc_paths(tmp_path, entry_point_names)
    fake_iter_entry_points = _create_fake_iter_entry_points(entry_points)
    fake_importlib_files = _create_fake_importlib_files(module_paths)
    with mock.patch('satpy._config.entry_points', fake_iter_entry_points), \
            mock.patch('satpy._config.impr_files', fake_importlib_files):
        yield etc_path


def _get_entry_points_and_etc_paths(
        tmp_path: Path,
        entry_point_names: dict[str, list[str]]
) -> tuple[Path, dict[str, list[EntryPoint]], dict[str, Path]]:
    module_path = tmp_path / "satpy_plugin"
    etc_path = module_path / "etc"
    etc_path.mkdir(parents=True, exist_ok=True)
    entry_points: dict[str, list[EntryPoint]] = {}
    entry_point_module_paths: dict[str, Path] = {}
    for ep_group, entry_point_values in entry_point_names.items():
        entry_points[ep_group] = []
        for entry_point_value in entry_point_values:
            parts = [part.strip() for part in entry_point_value.split("=")]
            ep_name = parts[0]
            ep_value = parts[1]
            ep_module = ep_value.split(":")[0].strip()
            ep = EntryPoint(name=ep_name, group=ep_group, value=ep_value)
            entry_points[ep_group].append(ep)
            entry_point_module_paths[ep_module] = module_path
    return etc_path, entry_points, entry_point_module_paths


def _create_fake_iter_entry_points(entry_points: dict[str, list[EntryPoint]]) -> Callable[[], dict[str, EntryPoint]]:
    def _fake_iter_entry_points() -> dict:
        return entry_points
    return _fake_iter_entry_points


def _create_fake_importlib_files(module_paths: dict[str, Path]) -> Callable[[str], Path]:
    def _fake_importlib_files(module_name: str) -> Path:
        return module_paths[module_name]
    return _fake_importlib_files


@pytest.fixture
def fake_composite_plugin_etc_path(tmp_path: Path) -> Iterator[Path]:
    """Create a fake plugin entry point with a fake compositor YAML configuration file."""
    yield from _create_yamlbased_plugin(
        tmp_path,
        "composites",
        "fake_sensor.yaml",
        _write_fake_composite_yaml,
    )


def _write_fake_composite_yaml(yaml_filename: str) -> None:
    with open(yaml_filename, "w") as comps_file:
        comps_file.write("""
    sensor_name: visir/fake_sensor

    composites:
        fake_composite:
            compositor: !!python/name:satpy.composites.GenericCompositor
            prerequisites:
            - 3.9
            - 10.8
            - 12.0
            standard_name: fake composite

    """)


@pytest.fixture
def fake_reader_plugin_etc_path(tmp_path: Path) -> Iterator[Path]:
    """Create a fake plugin entry point with a fake reader YAML configuration file."""
    yield from _create_yamlbased_plugin(
        tmp_path,
        "readers",
        "fake_reader.yaml",
        _write_fake_reader_yaml,
    )


def _write_fake_reader_yaml(yaml_filename: str) -> None:
    reader_name = os.path.splitext(os.path.basename(yaml_filename))[0]
    with open(yaml_filename, "w") as comps_file:
        comps_file.write(f"""
reader:
    name: {reader_name}
    sensors: [fake_sensor]
    reader: !!python/name:satpy.readers.yaml_reader.FileYAMLReader
datasets: {{}}
""")


@pytest.fixture
def fake_writer_plugin_etc_path(tmp_path: Path) -> Iterator[Path]:
    """Create a fake plugin entry point with a fake writer YAML configuration file."""
    yield from _create_yamlbased_plugin(
        tmp_path,
        "writers",
        "fake_writer.yaml",
        _write_fake_writer_yaml,
    )


def _write_fake_writer_yaml(yaml_filename: str) -> None:
    writer_name = os.path.splitext(os.path.basename(yaml_filename))[0]
    with open(yaml_filename, "w") as comps_file:
        comps_file.write(f"""
writer:
    name: {writer_name}
    writer: !!python/name:satpy.writers.Writer
""")


@pytest.fixture
def fake_enh_plugin_etc_path(tmp_path: Path) -> Iterator[Path]:
    """Create a fake plugin entry point with a fake enhancement YAML configure files.

    This creates a ``fake_sensor.yaml`` and ``generic.yaml`` enhancement configuration.

    """
    yield from _create_yamlbased_plugin(
        tmp_path,
        "enhancements",
        "fake_sensor.yaml",
        _write_fake_enh_yamls,
    )


def _write_fake_enh_yamls(yaml_filename: str) -> None:
    with open(yaml_filename, "w") as comps_file:
        comps_file.write("""
enhancements:
    some_custom_plugin_enh:
        name: fake_name
        operations:
        - name: stretch
          method: !!python/name:satpy.enhancements.stretch
          kwargs:
            stretch: crude
            min_stretch: -100.0
            max_stretch: 0.0
""")

    generic_filename = os.path.join(os.path.dirname(yaml_filename), "generic.yaml")
    with open(generic_filename, "w") as comps_file:
        comps_file.write("""
enhancements:
    default:
        operations:
        - name: stretch
          method: !!python/name:satpy.enhancements.stretch
          kwargs:
            stretch: crude
            min_stretch: -1.0
            max_stretch: 1.0
""")


def _create_yamlbased_plugin(
        tmp_path: Path,
        component_type: str,
        yaml_name: str,
        yaml_func: Callable[[str], None]
) -> Iterator[Path]:
    entry_point_dict = {f"satpy.{component_type}": [f"example_{component_type} = satpy_plugin"]}
    with fake_plugin_etc_path(tmp_path, entry_point_dict) as plugin_etc_path:
        comps_dir = os.path.join(plugin_etc_path, component_type)
        os.makedirs(comps_dir, exist_ok=True)
        comps_filename = os.path.join(comps_dir, yaml_name)
        yaml_func(comps_filename)
        yield plugin_etc_path


class TestPluginsConfigs:
    """Test that plugins are working."""

    def setup_method(self):
        """Set up the test."""
        cached_entry_points.cache_clear()

    def test_get_plugin_configs(self, fake_composite_plugin_etc_path):
        """Check that the plugin configs are looked for."""
        from satpy._config import get_entry_points_config_dirs

        with satpy.config.set(config_path=[]):
            dirs = get_entry_points_config_dirs('satpy.composites')
            assert dirs == [str(fake_composite_plugin_etc_path)]

    def test_load_entry_point_composite(self, fake_composite_plugin_etc_path):
        """Test that composites can be loaded from plugin entry points."""
        with satpy.config.set(config_path=[]):
            compositors, _ = load_compositor_configs_for_sensors(["fake_sensor"])
            assert "fake_sensor" in compositors
            comp_dict = DatasetDict(compositors["fake_sensor"])
            assert "fake_composite" in comp_dict
            comp_obj = comp_dict["fake_composite"]
            assert comp_obj.attrs["name"] == "fake_composite"
            assert comp_obj.attrs["prerequisites"] == [3.9, 10.8, 12.0]

    @pytest.mark.parametrize("specified_reader", [None, "fake_reader"])
    def test_plugin_reader_configs(self, fake_reader_plugin_etc_path, specified_reader):
        """Test that readers can be loaded from plugin entry points."""
        from satpy.readers import configs_for_reader
        reader_yaml_path = fake_reader_plugin_etc_path / "readers" / "fake_reader.yaml"
        self._get_and_check_reader_writer_configs(specified_reader, configs_for_reader, reader_yaml_path)

    def test_plugin_reader_available_readers(self, fake_reader_plugin_etc_path):
        """Test that readers can be loaded from plugin entry points."""
        from satpy.readers import available_readers
        self._check_available_component(available_readers, "fake_reader")

    @pytest.mark.parametrize("specified_writer", [None, "fake_writer"])
    def test_plugin_writer_configs(self, fake_writer_plugin_etc_path, specified_writer):
        """Test that writers can be loaded from plugin entry points."""
        from satpy.writers import configs_for_writer
        writer_yaml_path = fake_writer_plugin_etc_path / "writers" / "fake_writer.yaml"
        self._get_and_check_reader_writer_configs(specified_writer, configs_for_writer, writer_yaml_path)

    def test_plugin_writer_available_writers(self, fake_writer_plugin_etc_path):
        """Test that readers can be loaded from plugin entry points."""
        from satpy.writers import available_writers
        self._check_available_component(available_writers, "fake_writer")

    @staticmethod
    def _get_and_check_reader_writer_configs(specified_component, configs_func, exp_yaml):
        with satpy.config.set(config_path=[]):
            configs = list(configs_func(specified_component))
        assert any(str(exp_yaml) in config_list for config_list in configs)

    @staticmethod
    def _check_available_component(available_func, exp_component):
        with satpy.config.set(config_path=[]):
            available_components = available_func()
        assert exp_component in available_components

    @pytest.mark.parametrize(
        ("sensor_name", "exp_result"),
        [
            ("fake_sensor", 1.0),  # uses the sensor specific entry
            ("fake_sensor2", 0.5),  # uses the generic.yaml default
        ]
    )
    def test_plugin_enhancements_generic_sensor(self, fake_enh_plugin_etc_path, sensor_name, exp_result):
        """Test that enhancements from a plugin are available."""
        import dask.array as da
        import numpy as np
        import xarray as xr
        from trollimage.xrimage import XRImage

        from satpy.writers import Enhancer

        data_arr = xr.DataArray(
            da.zeros((10, 10), dtype=np.float32),
            dims=("y", "x"),
            attrs={
                "sensor": {sensor_name},
                "name": "fake_name",
            })
        img = XRImage(data_arr)

        enh = Enhancer()
        enh.add_sensor_enhancements(data_arr.attrs["sensor"])
        enh.apply(img, **img.data.attrs)

        res_data = img.data.values
        np.testing.assert_allclose(res_data, exp_result)


class TestConfigObject:
    """Test basic functionality of the central config object."""

    def test_custom_config_file(self):
        """Test adding a custom configuration file using SATPY_CONFIG."""
        import tempfile
        from importlib import reload

        import yaml

        import satpy
        my_config_dict = {
            'cache_dir': "/path/to/cache",
        }
        try:
            with tempfile.NamedTemporaryFile(mode='w+t', suffix='.yaml', delete=False) as tfile:
                yaml.dump(my_config_dict, tfile)
                tfile.close()
                with mock.patch.dict('os.environ', {'SATPY_CONFIG': tfile.name}):
                    reload(satpy._config)
                    reload(satpy)
                    assert satpy.config.get('cache_dir') == '/path/to/cache'
        finally:
            os.remove(tfile.name)

    def test_deprecated_env_vars(self):
        """Test that deprecated variables are mapped to new config."""
        from importlib import reload

        import satpy
        old_vars = {
            'PPP_CONFIG_DIR': '/my/ppp/config/dir',
            'SATPY_ANCPATH': '/my/ancpath',
        }

        with mock.patch.dict('os.environ', old_vars):
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('data_dir') == '/my/ancpath'
            assert satpy.config.get('config_path') == ['/my/ppp/config/dir']

    def test_config_path_multiple(self):
        """Test that multiple config paths are accepted."""
        from importlib import reload

        import satpy
        exp_paths, env_paths = _os_specific_multipaths()
        old_vars = {
            'SATPY_CONFIG_PATH': env_paths,
        }

        with mock.patch.dict('os.environ', old_vars):
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('config_path') == exp_paths

    def test_config_path_multiple_load(self):
        """Test that config paths from subprocesses load properly.

        Satpy modifies the config path environment variable when it is imported.
        If Satpy is imported again from a subprocess then it should be able to parse this
        modified variable.
        """
        from importlib import reload

        import satpy
        exp_paths, env_paths = _os_specific_multipaths()
        old_vars = {
            'SATPY_CONFIG_PATH': env_paths,
        }

        with mock.patch.dict('os.environ', old_vars):
            # these reloads will update env variable "SATPY_CONFIG_PATH"
            reload(satpy._config)
            reload(satpy)

            # load the updated env variable and parse it again.
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('config_path') == exp_paths

    def test_bad_str_config_path(self):
        """Test that a str config path isn't allowed."""
        from importlib import reload

        import satpy
        old_vars = {
            'SATPY_CONFIG_PATH': '/my/configs1',
        }

        # single path from env var still works
        with mock.patch.dict('os.environ', old_vars):
            reload(satpy._config)
            reload(satpy)
            assert satpy.config.get('config_path') == ['/my/configs1']

        # strings are not allowed, lists are
        with satpy.config.set(config_path='/single/string/paths/are/bad'):
            pytest.raises(ValueError, satpy._config.get_config_path_safe)

    def test_tmp_dir_is_writable(self):
        """Check that the default temporary directory is writable."""
        import satpy
        assert _is_writable(satpy.config["tmp_dir"])


def test_is_writable():
    """Test writable directory check."""
    assert _is_writable(os.getcwd())
    assert not _is_writable("/foo/bar")


def _is_writable(directory):
    import tempfile
    try:
        with tempfile.TemporaryFile(dir=directory):
            return True
    except OSError:
        return False


def _os_specific_multipaths():
    exp_paths = ['/my/configs1', '/my/configs2', '/my/configs3']
    if sys.platform.startswith("win"):
        exp_paths = ["C:" + p for p in exp_paths]
    path_str = os.pathsep.join(exp_paths)
    return exp_paths, path_str
