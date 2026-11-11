"""Shared preparation and utilities for testing.

This module is executed automatically by pytest.

"""
import os

import pytest

import satpy

TEST_ETC_DIR = os.path.join(os.path.dirname(__file__), "etc")


@pytest.fixture(autouse=True)
def _reset_satpy_config(tmpdir):
    """Set satpy config to logical defaults for tests."""
    test_config = {
        "cache_dir": str(tmpdir / "cache"),
        "data_dir": str(tmpdir / "data"),
        "config_path": [],
        "cache_lonlats": False,
        "cache_sensor_angles": False,
    }
    with satpy.config.set(test_config):
        yield


@pytest.fixture(autouse=True)
def _clear_function_caches():
    """Clear out global function-level caches that may cause conflicts between tests."""
    from satpy.composites.config_loader import load_compositor_configs_for_sensor
    load_compositor_configs_for_sensor.cache_clear()


@pytest.fixture
def include_test_etc():
    """Tell Satpy to use the config 'etc' directory from the tests directory."""
    with satpy.config.set(config_path=[TEST_ETC_DIR]):
        yield TEST_ETC_DIR


@pytest.fixture(autouse=True, scope="session")
def _forbid_pyspectral_downloads():
    from pyspectral.testing import forbid_pyspectral_downloads

    with forbid_pyspectral_downloads():
        yield
