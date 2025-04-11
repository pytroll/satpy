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
"""Helpers to apply enhancements."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from yaml import UnsafeLoader

from satpy._config import config_search_paths, get_entry_points_config_dirs
from satpy.decision_tree import DecisionTree
from satpy.utils import get_logger, recursive_dict_update

LOG = get_logger(__name__)


class EnhancementDecisionTree(DecisionTree):
    """The enhancement decision tree."""

    def __init__(self, *decision_dicts, **kwargs):
        """Init the decision tree."""
        match_keys = kwargs.pop("match_keys",
                                ("name",
                                 "reader",
                                 "platform_name",
                                 "sensor",
                                 "standard_name",
                                 "units",
                                 ))
        self.prefix = kwargs.pop("config_section", "enhancements")
        multival_keys = kwargs.pop("multival_keys", ["sensor"])
        super(EnhancementDecisionTree, self).__init__(
            decision_dicts, match_keys, multival_keys)

    def add_config_to_tree(self, *decision_dict: str | Path | dict) -> None:
        """Add configuration to tree."""
        conf: dict = {}
        for config_file in decision_dict:
            config_dict = self._get_config_dict_from_user(config_file)
            recursive_dict_update(conf, config_dict)
        self._build_tree(conf)

    def _get_config_dict_from_user(self, config_file: str | Path | dict) -> dict:
        if isinstance(config_file, (str, Path)) and os.path.isfile(config_file):
            config_dict = self._get_yaml_enhancement_dict(config_file)
        elif isinstance(config_file, dict):
            config_dict = config_file
        elif isinstance(config_file, str):
            LOG.debug("Loading enhancement config string")
            config_dict = yaml.load(config_file, Loader=UnsafeLoader)
            if not isinstance(config_dict, dict):
                raise ValueError(
                    "YAML file doesn't exist or string is not YAML dict: {}".format(config_file))
        else:
            raise ValueError(f"Unexpected type for enhancement configuration: {type(config_file)}")
        return config_dict

    def _get_yaml_enhancement_dict(self, config_file: str | Path) -> dict:
        with open(config_file) as fd:
            enhancement_config = yaml.load(fd, Loader=UnsafeLoader)
            if enhancement_config is None:
                # empty file
                return {}
            enhancement_section = enhancement_config.get(self.prefix, {})
            if not enhancement_section:
                LOG.debug("Config '{}' has no '{}' section or it is empty".format(config_file, self.prefix))
                return {}
            LOG.debug(f"Adding enhancement configuration from file: {config_file}")
        return enhancement_section

    def find_match(self, **query_dict):
        """Find a match."""
        try:
            return super(EnhancementDecisionTree, self).find_match(**query_dict)
        except KeyError:
            # give a more understandable error message
            raise KeyError("No enhancement configuration found for %s" %
                           (query_dict.get("uid", None),))


class Enhancer:
    """Helper class to get enhancement information for images."""

    def __init__(self, enhancement_config_file=None):
        """Initialize an Enhancer instance.

        Args:
            enhancement_config_file: The enhancement configuration to apply, False to leave as is.
        """
        self.enhancement_config_file = enhancement_config_file
        # Set enhancement_config_file to False for no enhancements
        if self.enhancement_config_file is None:
            # it wasn't specified in the config or in the kwargs, we should
            # provide a default
            config_fn = os.path.join("enhancements", "generic.yaml")
            paths = get_entry_points_config_dirs("satpy.enhancements")
            self.enhancement_config_file = config_search_paths(config_fn, search_dirs=paths)

        if not self.enhancement_config_file:
            # They don't want any automatic enhancements
            self.enhancement_tree = None
        else:
            if not isinstance(self.enhancement_config_file, (list, tuple)):
                self.enhancement_config_file = [self.enhancement_config_file]

            self.enhancement_tree = EnhancementDecisionTree(*self.enhancement_config_file)

        self.sensor_enhancement_configs = []

    def get_sensor_enhancement_config(self, sensor):
        """Get the sensor-specific config."""
        if isinstance(sensor, str):
            # one single sensor
            sensor = [sensor]

        paths = get_entry_points_config_dirs("satpy.enhancements")
        for sensor_name in sensor:
            config_fn = os.path.join("enhancements", sensor_name + ".yaml")
            config_files = config_search_paths(config_fn, search_dirs=paths)
            # Note: Enhancement configuration files can't overwrite individual
            # options, only entire sections are overwritten
            for config_file in config_files:
                yield config_file

    def add_sensor_enhancements(self, sensor):
        """Add sensor-specific enhancements."""
        # XXX: Should we just load all enhancements from the base directory?
        new_configs = []
        for config_file in self.get_sensor_enhancement_config(sensor):
            if config_file not in self.sensor_enhancement_configs:
                self.sensor_enhancement_configs.append(config_file)
                new_configs.append(config_file)

        if new_configs:
            self.enhancement_tree.add_config_to_tree(*new_configs)

    def apply(self, img, **info):
        """Apply the enhancements."""
        enh_kwargs = self.enhancement_tree.find_match(**info)

        backup_id = f"<name={info.get('name')}, calibration={info.get('calibration')}>"
        data_id = info.get("_satpy_id", backup_id)
        LOG.debug(f"Data for {data_id} will be enhanced with options:\n\t{enh_kwargs['operations']}")
        for operation in enh_kwargs["operations"]:
            fun = operation["method"]
            args = operation.get("args", [])
            kwargs = operation.get("kwargs", {})
            fun(img, *args, **kwargs)
