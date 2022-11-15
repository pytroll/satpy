#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020-2021 Satpy developers
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
"""Classes for loading compositor and modifier configuration files."""
from __future__ import annotations

import logging
import os
import warnings
from functools import lru_cache, update_wrapper
from typing import Callable, Iterable

import yaml
from yaml import UnsafeLoader

import satpy
from satpy import DataID, DataQuery
from satpy._config import config_search_paths, get_entry_points_config_dirs, glob_config
from satpy.dataset.dataid import minimal_default_keys_config
from satpy.utils import recursive_dict_update

logger = logging.getLogger(__name__)


def _convert_dep_info_to_data_query(dep_info):
    key_item = dep_info.copy()
    key_item.pop('prerequisites', None)
    key_item.pop('optional_prerequisites', None)
    if 'modifiers' in key_item:
        key_item['modifiers'] = tuple(key_item['modifiers'])
    key = DataQuery.from_dict(key_item)
    return key


class _CompositeConfigHelper:
    """Helper class for parsing composite configurations.

    The provided `loaded_compositors` dictionary is updated inplace.

    """

    def __init__(self, loaded_compositors, sensor_id_keys):
        self.loaded_compositors = loaded_compositors
        self.sensor_id_keys = sensor_id_keys

    def _create_comp_from_info(self, composite_info, loader):
        key = DataID(self.sensor_id_keys, **composite_info)
        comp = loader(_satpy_id=key, **composite_info)
        return key, comp

    def _handle_inline_comp_dep(self, dep_info, dep_num, parent_name):
        # Create an unique temporary name for the composite
        sub_comp_name = '_' + parent_name + '_dep_{}'.format(dep_num)
        dep_info['name'] = sub_comp_name
        self._load_config_composite(dep_info)

    @staticmethod
    def _get_compositor_loader_from_config(composite_name, composite_info):
        try:
            loader = composite_info.pop('compositor')
        except KeyError:
            raise ValueError("'compositor' key missing or empty for '{}'. Option keys = {}".format(
                composite_name, str(composite_info.keys())))
        return loader

    def _process_composite_deps(self, composite_info):
        dep_num = -1
        for prereq_type in ['prerequisites', 'optional_prerequisites']:
            prereqs = []
            for dep_info in composite_info.get(prereq_type, []):
                dep_num += 1
                if not isinstance(dep_info, dict):
                    prereqs.append(dep_info)
                    continue
                elif 'compositor' in dep_info:
                    self._handle_inline_comp_dep(
                        dep_info, dep_num, composite_info['name'])
                prereq_key = _convert_dep_info_to_data_query(dep_info)
                prereqs.append(prereq_key)
            composite_info[prereq_type] = prereqs

    def _load_config_composite(self, composite_info):
        composite_name = composite_info['name']
        loader = self._get_compositor_loader_from_config(composite_name, composite_info)
        self._process_composite_deps(composite_info)
        key, comp = self._create_comp_from_info(composite_info, loader)
        self.loaded_compositors[key] = comp

    def _load_config_composites(self, configured_composites):
        for composite_name, composite_info in configured_composites.items():
            composite_info['name'] = composite_name
            self._load_config_composite(composite_info)

    def parse_config(self, configured_composites, composite_configs):
        """Parse composite configuration dictionary."""
        try:
            self._load_config_composites(configured_composites)
        except (ValueError, KeyError):
            raise RuntimeError("Failed to load composites from configs "
                               "'{}'".format(composite_configs))


class _ModifierConfigHelper:
    """Helper class for parsing modifier configurations.

    The provided `loaded_modifiers` dictionary is updated inplace.

    """

    def __init__(self, loaded_modifiers, sensor_id_keys):
        self.loaded_modifiers = loaded_modifiers
        self.sensor_id_keys = sensor_id_keys

    @staticmethod
    def _get_modifier_loader_from_config(modifier_name, modifier_info):
        try:
            loader = modifier_info.pop('modifier', None)
            if loader is None:
                loader = modifier_info.pop('compositor')
                warnings.warn("Modifier '{}' uses deprecated 'compositor' "
                              "key to point to Python class, replace "
                              "with 'modifier'.".format(modifier_name))
        except KeyError:
            raise ValueError("'modifier' key missing or empty for '{}'. Option keys = {}".format(
                modifier_name, str(modifier_info.keys())))
        return loader

    def _process_modifier_deps(self, modifier_info):
        for prereq_type in ['prerequisites', 'optional_prerequisites']:
            prereqs = []
            for dep_info in modifier_info.get(prereq_type, []):
                if not isinstance(dep_info, dict):
                    prereqs.append(dep_info)
                    continue
                prereq_key = _convert_dep_info_to_data_query(dep_info)
                prereqs.append(prereq_key)
            modifier_info[prereq_type] = prereqs

    def _load_config_modifier(self, modifier_info):
        modifier_name = modifier_info['name']
        loader = self._get_modifier_loader_from_config(modifier_name, modifier_info)
        self._process_modifier_deps(modifier_info)
        self.loaded_modifiers[modifier_name] = (loader, modifier_info)

    def _load_config_modifiers(self, configured_modifiers):
        for modifier_name, modifier_info in configured_modifiers.items():
            modifier_info['name'] = modifier_name
            self._load_config_modifier(modifier_info)

    def parse_config(self, configured_modifiers, composite_configs):
        """Parse modifier configuration dictionary."""
        try:
            self._load_config_modifiers(configured_modifiers)
        except (ValueError, KeyError):
            raise RuntimeError("Failed to load modifiers from configs "
                               "'{}'".format(composite_configs))


def _load_config(composite_configs):
    if not isinstance(composite_configs, (list, tuple)):
        composite_configs = [composite_configs]

    conf = {}
    for composite_config in composite_configs:
        with open(composite_config, 'r', encoding='utf-8') as conf_file:
            conf = recursive_dict_update(conf, yaml.load(conf_file, Loader=UnsafeLoader))
    try:
        sensor_name = conf['sensor_name']
    except KeyError:
        logger.debug('No "sensor_name" tag found in %s, skipping.',
                     composite_configs)
        return {}, {}, {}

    sensor_compositors = {}
    sensor_modifiers = {}

    dep_id_keys = None
    sensor_deps = sensor_name.split('/')[:-1]
    if sensor_deps:
        # get dependent
        for sensor_dep in sensor_deps:
            dep_comps, dep_mods, dep_id_keys = load_compositor_configs_for_sensor(sensor_dep)
        # the last parent should include all of its parents so only add the last one
        sensor_compositors.update(dep_comps)
        sensor_modifiers.update(dep_mods)

    id_keys = _get_sensor_id_keys(conf, dep_id_keys)
    mod_config_helper = _ModifierConfigHelper(sensor_modifiers, id_keys)
    configured_modifiers = conf.get('modifiers', {})
    mod_config_helper.parse_config(configured_modifiers, composite_configs)

    comp_config_helper = _CompositeConfigHelper(sensor_compositors, id_keys)
    configured_composites = conf.get('composites', {})
    comp_config_helper.parse_config(configured_composites, composite_configs)
    return sensor_compositors, sensor_modifiers, id_keys


def _get_sensor_id_keys(conf, parent_id_keys):
    try:
        id_keys = conf['composite_identification_keys']
    except KeyError:
        id_keys = parent_id_keys
        if not id_keys:
            id_keys = minimal_default_keys_config
    return id_keys


def _lru_cache_with_config_path(func: Callable):
    """Use lru_cache but include satpy's current config_path."""
    @lru_cache()
    def _call_without_config_path_wrapper(sensor_name, _):
        return func(sensor_name)

    def _add_config_path_wrapper(sensor_name: str):
        config_path = satpy.config.get("config_path")
        # make sure config_path is hashable, but keep original order since it matters
        config_path = tuple(config_path)
        return _call_without_config_path_wrapper(sensor_name, config_path)

    wrapper = update_wrapper(_add_config_path_wrapper, func)
    wrapper = _update_cached_wrapper(wrapper, _call_without_config_path_wrapper)
    return wrapper


def _update_cached_wrapper(wrapper, cached_func):
    for meth_name in ("cache_clear", "cache_parameters", "cache_info"):
        if hasattr(cached_func, meth_name):
            setattr(wrapper, meth_name, getattr(cached_func, meth_name))
    return wrapper


@_lru_cache_with_config_path
def load_compositor_configs_for_sensor(sensor_name: str) -> tuple[dict[str, dict], dict[str, dict], dict]:
    """Load compositor, modifier, and DataID key information from configuration files for the specified sensor.

    Args:
        sensor_name: Sensor name that has matching ``sensor_name.yaml``
            config files.

    Returns:
        (comps, mods, data_id_keys): Where `comps` is a dictionary:

                composite ID -> compositor object

            And `mods` is a dictionary:

                modifier name -> (modifier class, modifiers options)

            Add `data_id_keys` is a dictionary:

                DataID key -> key properties

    """
    config_filename = sensor_name + ".yaml"
    logger.debug("Looking for composites config file %s", config_filename)
    paths = get_entry_points_config_dirs('satpy.composites')
    composite_configs = config_search_paths(
        os.path.join("composites", config_filename),
        search_dirs=paths, check_exists=True)
    if not composite_configs:
        logger.debug("No composite config found called %s",
                     config_filename)
        return {}, {}, minimal_default_keys_config
    return _load_config(composite_configs)


def load_compositor_configs_for_sensors(sensor_names: Iterable[str]) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load compositor and modifier configuration files for the specified sensors.

    Args:
        sensor_names (list of strings): Sensor names that have matching
            ``sensor_name.yaml`` config files.

    Returns:
        (comps, mods): Where `comps` is a dictionary:

                sensor_name -> composite ID -> compositor object

            And `mods` is a dictionary:

                sensor_name -> modifier name -> (modifier class,
                modifiers options)

    """
    comps = {}
    mods = {}
    for sensor_name in sensor_names:
        sensor_comps, sensor_mods = load_compositor_configs_for_sensor(sensor_name)[:2]
        comps[sensor_name] = sensor_comps
        mods[sensor_name] = sensor_mods
    return comps, mods


def all_composite_sensors():
    """Get all sensor names from available composite configs."""
    paths = get_entry_points_config_dirs('satpy.composites')
    composite_configs = glob_config(
        os.path.join("composites", "*.yaml"),
        search_dirs=paths)
    yaml_names = set([os.path.splitext(os.path.basename(fn))[0]
                      for fn in composite_configs])
    non_sensor_yamls = ('visir',)
    sensor_names = [x for x in yaml_names if x not in non_sensor_yamls]
    return sensor_names
