#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016.

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   David Hoese <david.hoese@ssec.wisc.edu>

# This file is part of the satpy.

# satpy is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# satpy is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""SatPy Configuration directory and file handling
"""
import glob
import logging
import os
from collections import Mapping, OrderedDict

from six.moves import configparser
import yaml

LOG = logging.getLogger(__name__)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
# FIXME: Use package_resources?
PACKAGE_CONFIG_PATH = os.path.join(BASE_PATH, 'etc')


def get_environ_config_dir(default=PACKAGE_CONFIG_PATH):
    return os.environ.get('PPP_CONFIG_DIR', default)


def get_environ_ancpath(default='.'):
    return os.environ.get('SATPY_ANCPATH', default)


# FIXME: Old readers still use only this, but this may get updated by Scene
CONFIG_PATH = get_environ_config_dir()


def runtime_import(object_path):
    """Import at runtime."""
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
    paths += [os.path.join(CONFIG_PATH, filename),
              os.path.join(PACKAGE_CONFIG_PATH, filename)]
    paths = [os.path.abspath(path) for path in paths]

    if kwargs.get("check_exists", True):
        paths = [x for x in paths if os.path.isfile(x)]

    paths = list(OrderedDict.fromkeys(paths))
    # flip the order of the list so builtins are loaded first
    return paths[::-1]


def get_config(filename, *search_dirs, **kwargs):
    """Blends the different configs, from package defaults to ."""
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
    """Get the appropriate path for a filename, in that order: filename, ., PPP_CONFIG_DIR, package's etc dir."""
    paths = config_search_paths(filename, *search_dirs)

    for path in paths[::-1]:
        if os.path.exists(path):
            return path


def recursive_dict_update(d, u):
    """Recursive dictionary update.

    Copied from:

        http://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            r = recursive_dict_update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def check_yaml_configs(configs, key, hdr_len):
    """Get a diagnostic for the yaml *configs*.

    *key* is the section to look for to get a name for the config at hand.
    *hdr_len* is the number of lines that can be safely read from the config to
    get a name.
    """
    diagnostic = {}
    for i in configs:
        for fname in i:
            with open(fname) as stream:
                try:
                    res = yaml.load(stream)
                    try:
                        diagnostic[res[key]['name']] = 'ok'
                    except Exception:
                        continue
                except yaml.YAMLError as err:
                    stream.seek(0)
                    lines = ''.join(stream.readline() for line in range(hdr_len))
                    res = yaml.load(lines)
                    if err.context == 'while constructing a Python object':
                        problem = err.problem
                    else:
                        problem = 'error'
                    try:
                        diagnostic[res[key]['name']] = problem
                    except Exception:
                        continue
    return diagnostic


def check_satpy():
    """Check the satpy readers and writers for correct installation."""
    from satpy.readers import configs_for_reader
    from satpy.writers import configs_for_writer
    print('Readers')
    print('=======')
    for reader, res in sorted(check_yaml_configs(configs_for_reader(), 'reader', 5).items()):
        print(reader + ': ' + res)
    print()
    print('Writers')
    print('=======')
    for writer, res in sorted(check_yaml_configs(configs_for_writer(), 'writer', 3).items()):
        print(writer + ': ' + res)
