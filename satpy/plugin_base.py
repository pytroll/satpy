#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2011 SMHI

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""The :mod:`satpy.plugin_base` module defines the plugin API.
"""

import logging
import os
import yaml

from satpy.config import config_search_paths, get_environ_config_dir, recursive_dict_update

try:
    import configparser
except ImportError:
    from six.moves import configparser

LOG = logging.getLogger(__name__)


class Plugin(object):
    """The base plugin class. It is not to be used as is, it has to be
    inherited by other classes.
    """

    def __init__(self,
                 ppp_config_dir=None,
                 default_config_filename=None,
                 config_files=None,
                 **kwargs):
        self.ppp_config_dir = ppp_config_dir or get_environ_config_dir()

        self.default_config_filename = default_config_filename
        self.config_files = config_files
        if self.config_files is None and self.default_config_filename is not None:
            # Specify a default
            self.config_files = config_search_paths(
                self.default_config_filename, self.ppp_config_dir)
        if not isinstance(self.config_files, (list, tuple)):
            self.config_files = [self.config_files]

        self.config = {}
        if self.config_files:
            for config_file in self.config_files:
                self.load_yaml_config(config_file)

    def load_yaml_config(self, conf):
        with open(conf) as fd:
            self.config = recursive_dict_update(self.config, yaml.load(fd))
