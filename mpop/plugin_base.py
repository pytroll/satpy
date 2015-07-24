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

"""The :mod:`mpop.plugin_base` module defines the plugin API.
"""

import os
import logging
from ConfigParser import ConfigParser
from mpop import PACKAGE_CONFIG_PATH

LOG = logging.getLogger(__name__)


class Plugin(object):
    """The base plugin class. It is not to be used as is, it has to be
    inherited by other classes.
    """
    def __init__(self, ppp_config_dir=None, default_config_filename=None, config_file=None, **kwargs):
        self.ppp_config_dir = ppp_config_dir or os.environ.get("PPP_CONFIG_DIR", PACKAGE_CONFIG_PATH)
        self.default_config_filename = default_config_filename
        self.config_file = config_file
        if self.config_file is None and self.default_config_filename is not None:
            # Specify a default
            self.config_file = os.path.join(self.ppp_config_dir, self.default_config_filename)

        if self.config_file:
            conf = ConfigParser()
            conf.read(self.config_file)
            self.load_config(conf)

    def load_config(self, conf):
        # Assumes only one section with "reader:" prefix
        for section_name in conf.sections():
            section_type = section_name.split(":")[0]
            load_func = "load_section_%s" % (section_type,)
            if hasattr(self, load_func):
                getattr(self, load_func)(section_name, dict(conf.items(section_name)))
