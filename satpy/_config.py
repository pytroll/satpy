#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2019 Satpy developers
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
"""Satpy Configuration directory and file handling."""
from __future__ import annotations

import ast
import glob
import logging
import os
import sys
import tempfile
from collections import OrderedDict
from importlib.metadata import entry_points
from pathlib import Path

try:
    from importlib.resources import files as impr_files  # type: ignore
except ImportError:
    # Python 3.8
    def impr_files(module_name: str) -> Path:
        """Get path to module as a backport for Python 3.8."""
        from importlib.resources import path as impr_path

        with impr_path(module_name, "__init__.py") as pkg_init_path:
            return pkg_init_path.parent

import appdirs
from donfig import Config

from satpy._compat import cache

LOG = logging.getLogger(__name__)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
# FIXME: Use package_resources?
PACKAGE_CONFIG_PATH = os.path.join(BASE_PATH, 'etc')

_satpy_dirs = appdirs.AppDirs(appname='satpy', appauthor='pytroll')
_CONFIG_DEFAULTS = {
    'tmp_dir': tempfile.gettempdir(),
    'cache_dir': _satpy_dirs.user_cache_dir,
    'cache_lonlats': False,
    'cache_sensor_angles': False,
    'config_path': [],
    'data_dir': _satpy_dirs.user_data_dir,
    'demo_data_dir': '.',
    'download_aux': True,
    'sensor_angles_position_preference': 'actual',
}

# Satpy main configuration object
# See https://donfig.readthedocs.io/en/latest/configuration.html
# for more information.
#
# Configuration values will be loaded from files at:
# 1. The builtin package satpy.yaml (not present currently)
# 2. $SATPY_ROOT_CONFIG (default: /etc/satpy/satpy.yaml)
# 3. <python-env-prefix>/etc/satpy/satpy.yaml
# 4. ~/.config/satpy/satpy.yaml
# 5. ~/.satpy/satpy.yaml
# 6. $SATPY_CONFIG_PATH/satpy.yaml if present (colon separated)
_CONFIG_PATHS = [
    os.path.join(PACKAGE_CONFIG_PATH, 'satpy.yaml'),
    os.getenv('SATPY_ROOT_CONFIG', os.path.join('/etc', 'satpy', 'satpy.yaml')),
    os.path.join(sys.prefix, 'etc', 'satpy', 'satpy.yaml'),
    os.path.join(_satpy_dirs.user_config_dir, 'satpy.yaml'),
    os.path.join(os.path.expanduser('~'), '.satpy', 'satpy.yaml'),
]
# The above files can also be directories. If directories all files
# with `.yaml`., `.yml`, or `.json` extensions will be used.

_ppp_config_dir = os.getenv('PPP_CONFIG_DIR', None)
_satpy_config_path = os.getenv('SATPY_CONFIG_PATH', None)

if _ppp_config_dir is not None and _satpy_config_path is None:
    LOG.warning("'PPP_CONFIG_DIR' is deprecated. Please use 'SATPY_CONFIG_PATH' instead.")
    _satpy_config_path = _ppp_config_dir

if _satpy_config_path is not None:
    if _satpy_config_path.startswith("["):
        # 'SATPY_CONFIG_PATH' is set by previous satpy config as a reprsentation of a 'list'
        # need to use 'ast.literal_eval' to parse the string back to a list
        _satpy_config_path_list = ast.literal_eval(_satpy_config_path)
    else:
        # colon-separated are ordered by custom -> builtins
        # i.e. last-applied/highest priority to first-applied/lowest priority
        _satpy_config_path_list = _satpy_config_path.split(os.pathsep)

    os.environ['SATPY_CONFIG_PATH'] = repr(_satpy_config_path_list)
    for config_dir in _satpy_config_path_list:
        _CONFIG_PATHS.append(os.path.join(config_dir, 'satpy.yaml'))

_ancpath = os.getenv('SATPY_ANCPATH', None)
_data_dir = os.getenv('SATPY_DATA_DIR', None)
if _ancpath is not None and _data_dir is None:
    LOG.warning("'SATPY_ANCPATH' is deprecated. Please use 'SATPY_DATA_DIR' instead.")
    os.environ['SATPY_DATA_DIR'] = _ancpath

config = Config("satpy", defaults=[_CONFIG_DEFAULTS], paths=_CONFIG_PATHS)


def get_config_path_safe():
    """Get 'config_path' and check for proper 'list' type."""
    config_path = config.get('config_path')
    if not isinstance(config_path, list):
        raise ValueError("Satpy config option 'config_path' must be a "
                         "list, not '{}'".format(type(config_path)))
    return config_path


def get_entry_points_config_dirs(name, include_config_path=True):
    """Get the config directories for all entry points of given name."""
    dirs = []
    for entry_point in cached_entry_points().get(name, []):
        module = _entry_point_module(entry_point)
        new_dir = str(impr_files(module) / "etc")
        if not dirs or dirs[-1] != new_dir:
            dirs.append(new_dir)
    if include_config_path:
        dirs.extend(config.get('config_path')[::-1])
    return dirs


@cache
def cached_entry_points():
    """Return entry_points.

    This is a dummy proxy to allow caching.
    """
    return entry_points()


def _entry_point_module(entry_point):
    try:
        return entry_point.module
    except AttributeError:
        # Python 3.8
        return entry_point.value.split(":")[0].strip()


def config_search_paths(filename, search_dirs=None, **kwargs):
    """Get series of configuration base paths where Satpy configs are located."""
    if search_dirs is None:
        search_dirs = get_config_path_safe()[::-1]

    paths = [filename, os.path.basename(filename)]
    paths += [os.path.join(search_dir, filename) for search_dir in search_dirs]
    paths += [os.path.join(PACKAGE_CONFIG_PATH, filename)]
    paths = [os.path.abspath(path) for path in paths]

    if kwargs.get("check_exists", True):
        paths = [x for x in paths if os.path.isfile(x)]

    paths = list(OrderedDict.fromkeys(paths))
    # flip the order of the list so builtins are loaded first
    return paths[::-1]


def glob_config(pattern, search_dirs=None):
    """Return glob results for all possible configuration locations.

    Note: This method does not check the configuration "base" directory if the pattern includes a subdirectory.
          This is done for performance since this is usually used to find *all* configs for a certain component.
    """
    patterns = config_search_paths(pattern, search_dirs=search_dirs,
                                   check_exists=False)
    for pattern_fn in patterns:
        for path in glob.iglob(pattern_fn):
            yield path


def get_config_path(filename):
    """Get the path to the highest priority version of a config file."""
    paths = config_search_paths(filename)
    for path in paths[::-1]:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find file in configuration path: "
                            "'{}'".format(filename))
