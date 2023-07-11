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
import os
import pathlib
import shutil

import pytest
import requests

import satpy

TEST_ETC_DIR = os.path.join(os.path.dirname(__file__), 'etc')


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


@pytest.fixture
def include_test_etc():
    """Tell Satpy to use the config 'etc' directory from the tests directory."""
    with satpy.config.set(config_path=[TEST_ETC_DIR]):
        yield TEST_ETC_DIR


_url_sample_file = (
        "https://go.dwd-nextcloud.de/index.php/s/z87KfL72b9dM5xm/download/"
        "IASI_SND_02_M01_20190605002352Z_20190605020856Z_N_O_20190605011702Z.nat")


@pytest.fixture(scope="session")
def sample_file(tmp_path_factory):
    """Obtain sample file."""
    fn = pathlib.Path("/media/nas/x21308/IASI/IASI_SND_02_M01_20190605002352Z_20190605020856Z_N_O_20190605011702Z.nat")
    if fn.exists():
        return fn
    fn = tmp_path_factory.mktemp("data") / "IASI_SND_02_M01_20190605002352Z_20190605020856Z_N_O_20190605011702Z.nat"
    data = requests.get(_url_sample_file, stream=True)
    with fn.open(mode="wb") as fp:
        shutil.copyfileobj(data.raw, fp)
    return fn
