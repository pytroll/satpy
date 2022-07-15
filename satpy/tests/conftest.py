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
"""Shared preparation and utilities for testing.

This module is executed automatically by pytest.

"""

import pytest

import satpy


@pytest.fixture(autouse=True)
def reset_satpy_config(tmpdir):
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
def clear_function_caches():
    """Clear out global function-level caches that may cause conflicts between tests."""
    from satpy.composites.config_loader import load_compositor_configs_for_sensor
    load_compositor_configs_for_sensor.cache_clear()
