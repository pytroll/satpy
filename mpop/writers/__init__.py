#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Shared objects of the various writer classes.

For now, this includes enhancement configuration utilities.
"""

import logging
import ConfigParser
from mpop import PACKAGE_CONFIG_PATH
import os

LOG = logging.getLogger(__name__)


class EnhancementDecisionTree(object):
    any_key = None

    def __init__(self, *config_files, **kwargs):
        self.attrs = kwargs.pop("attrs", ("name", "platform", "sensor", "standard_name", "units",))
        self.prefix = kwargs.pop("prefix", "enhancement:")
        self.tree = {}
        self.add_config_to_tree(*config_files)

    def add_config_to_tree(self, *config_files):
        conf = ConfigParser.ConfigParser(allow_no_value=True)
        for fn in config_files:
            if isinstance(fn, str):
                conf.read(fn)
            else:
                conf.readfp(fn)

        self._build_tree(conf)

    def _build_tree(self, conf):
        for section_name in conf.sections():
            if not section_name.startswith(self.prefix):
                continue
            attrs = dict(conf.items(section_name))
            # Set a path in the tree for each section in the configuration files
            curr_level = self.tree
            for attr in self.attrs:
                # or None is necessary if they have empty strings
                this_attr = attrs.get(attr, self.any_key) or None
                if attr == self.attrs[-1]:
                    # if we are at the last attribute, then assign the value
                    # set the dictionary of attributes because the config is not persistent
                    curr_level[this_attr] = attrs
                elif this_attr not in curr_level:
                    curr_level[this_attr] = {}
                curr_level = curr_level[this_attr]

    def _find_match(self, curr_level, attrs, kwargs):
        if len(attrs) == 0:
            # we're at the bottom level, we must have found something
            return curr_level

        match = None
        if attrs[0] in kwargs and kwargs[attrs[0]] in curr_level:
            # we know what we're searching for, try to find a pattern that uses this attribute
            match = self._find_match(curr_level[kwargs[attrs[0]]], attrs[1:], kwargs)

        if match is None and self.any_key in curr_level:
            # if we couldn't find it using the attribute then continue with the other attributes down the 'any' path
            match = self._find_match(curr_level[self.any_key], attrs[1:], kwargs)
        return match

    def find_match(self, **kwargs):
        try:
            match = self._find_match(self.tree, self.attrs, kwargs)
        except StandardError:
            LOG.debug("Match exception:", exc_info=True)
            LOG.error("Error when finding matching enhancement section")
            match = None

        if match is None:
            # only possible if no default section was provided
            raise KeyError("No enhancement configuration found for %s" % (kwargs.get("uid", None),))
        return match

class Enhancer(object):

    """
    Helper class to get enhancement information for images.
    """

    def __init__(self, **kwargs):
        self.ppp_config_dir = kwargs.get("ppp_config_dir", PACKAGE_CONFIG_PATH)
        self.enhancement_config = kwargs.get("enhancement_config", None)
        # FIXME I don't like this, we should have a special enhancement_config value to say not to apply any enhancement
        if self.enhancement_config is None and "enhancement_config" not in kwargs:
            # it wasn't specified in the config or in the kwargs, we should provide a default
            self.enhancement_config = os.path.join(self.ppp_config_dir, "enhancements", "generic.cfg")

        if self.enhancement_config is not None:
            self.enhancement_tree = EnhancementDecisionTree(self.enhancement_config)
        else:
            # They don't want any automatic enhancements
            self.enhancement_tree = None

        self.sensor_enhancement_configs = []

    def get_sensor_enhancement_config(self, sensor):
        if isinstance(sensor, str):
            # one single sensor
            sensor = [sensor]

        for sensor_name in sensor:
            config_file = os.path.join(self.ppp_config_dir, "enhancements", sensor_name + ".cfg")
            if os.path.isfile(config_file):
                yield config_file

    def add_sensor_enhancements(self, sensor):
        # XXX: Should we just load all enhancements from the base directory?
        new_configs = []
        for config_file in self.get_sensor_enhancement_config(sensor):
            if config_file not in self.sensor_enhancement_configs.append(config_file):
                self.sensor_enhancement_configs.append(config_file)
                new_configs.append(config_file)

        if new_configs:
            self.enhancement_tree.add_config_to_tree(config_file)

