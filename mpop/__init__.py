#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2013.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""MPOP Package initializer.
"""

import os
from ConfigParser import ConfigParser
from mpop.version import __version__
from logging import getLogger

LOG = getLogger(__name__)

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep)[:-1])

# FIXME: Use package_resources?
PACKAGE_CONFIG_PATH = os.path.join(BASE_PATH, 'etc')

CONFIG_PATH = os.environ.get('PPP_CONFIG_DIR', PACKAGE_CONFIG_PATH)


def get_config(filename):
    """Blends the different configs, from package defaults to .
    """

    config = ConfigParser()

    paths1 = [filename,
              os.path.join(".", filename),
              os.path.join(CONFIG_PATH, filename),
              os.path.join(PACKAGE_CONFIG_PATH, filename)]

    paths2 = [os.path.basename(filename),
              os.path.join(".", os.path.basename(filename)),
              os.path.join(CONFIG_PATH, os.path.basename(filename)),
              os.path.join(PACKAGE_CONFIG_PATH, os.path.basename(filename))]

    for paths in (paths1, paths2):
        successes = config.read(reversed(paths))
        if successes:
            LOG.debug("Read config from %s", str(successes))
            return config

    LOG.warning("Couldn't file any config file matching %s", filename)

def get_config_path(filename):
    """Get the appropriate path for a filename, in that order: filename, ., PPP_CONFIG_DIR, package's etc dir.
    """

    paths = [filename,
             os.path.join(".", filename),
             os.path.join(CONFIG_PATH, filename),
             os.path.join(PACKAGE_CONFIG_PATH, filename),
             os.path.join(".", os.path.basename(filename)),
             os.path.join(CONFIG_PATH, os.path.basename(filename)),
             os.path.join(PACKAGE_CONFIG_PATH, os.path.basename(filename))]

    for path in paths:
        if os.path.exists(path):
            return path