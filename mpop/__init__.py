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
import glob
from mpop.version import __version__
from logging import getLogger
try:
    import configparser
except ImportError:
    from six.moves import configparser
LOG = getLogger(__name__)

BASE_PATH = os.path.sep.join(os.path.dirname(
    os.path.realpath(__file__)).split(os.path.sep)[:-1])

# FIXME: Use package_resources?
PACKAGE_CONFIG_PATH = os.path.join(BASE_PATH, 'etc')


def get_environ_config_dir(default=PACKAGE_CONFIG_PATH):
    return os.environ.get('PPP_CONFIG_DIR', PACKAGE_CONFIG_PATH)

# FIXME: Old readers still use only this, but this may get updated by Scene
CONFIG_PATH = get_environ_config_dir()


def runtime_import(object_path):
    """Import at runtime
    """
    obj_module, obj_element = object_path.rsplit(".", 1)
    loader = __import__(obj_module, globals(), locals(), [str(obj_element)])
    return getattr(loader, obj_element)


def config_search_paths(filename, *search_dirs, **kwargs):
    # Get the environment variable value every time (could be set dynamically)
    # FIXME: Consider removing the 'magic' environment variable all together
    CONFIG_PATH = get_environ_config_dir()

    paths = [filename, os.path.basename(filename)]
    paths += [os.path.join(search_dir, filename) for search_dir in search_dirs]
    # FUTURE: Remove CONFIG_PATH because it should be included as a search_dir
    paths += [os.path.join(CONFIG_PATH, filename), os.path.join(PACKAGE_CONFIG_PATH, filename)]

    if kwargs.get("check_exists", True):
        paths = [x for x in paths if os.path.isfile(x)]

    return paths


def get_config(filename, *search_dirs, **kwargs):
    """Blends the different configs, from package defaults to .
    """
    config = kwargs.get("config_reader_class", configparser.ConfigParser)()

    paths = config_search_paths(filename, *search_dirs)
    successes = config.read(reversed(paths))
    if successes:
        LOG.debug("Read config from %s", str(successes))
        return config, successes

    LOG.warning("Couldn't file any config file matching %s", filename)
    return None, []


def glob_config(pattern, *search_dirs):
    """Return glob results for all possible configuration locations.

    Note: This method does not check the configuration "base" directory if the pattern includes a subdirectory.
          This is done for performance since this is usually used to find *all* configs for a certain component.
    """
    patterns = config_search_paths(pattern, *search_dirs, check_exists=False)

    for pattern in patterns:
        for path in glob.iglob(pattern):
            yield path


def get_config_path(filename, *search_dirs):
    """Get the appropriate path for a filename, in that order: filename, ., PPP_CONFIG_DIR, package's etc dir.
    """
    paths = config_search_paths(filename, *search_dirs)

    for path in paths:
        if os.path.exists(path):
            return path
