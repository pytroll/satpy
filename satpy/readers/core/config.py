#!/usr/bin/env python
# Copyright (c) 2015-2025 Satpy developers
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
"""Reader configuration."""
from __future__ import annotations

import logging
import os
import warnings

import yaml
from yaml.loader import BaseLoader, FullLoader, UnsafeLoader

from satpy._config import config_search_paths, get_entry_points_config_dirs, glob_config
from satpy.readers.core.yaml_reader import load_yaml_configs as load_yaml_reader_configs

LOG = logging.getLogger(__name__)

# Old Name -> New Name
PENDING_OLD_READER_NAMES = {"fci_l1c_fdhsi": "fci_l1c_nc", "viirs_l2_cloud_mask_nc": "viirs_edr"}
OLD_READER_NAMES: dict[str, str] = {
    "slstr_l2": "ghrsst_l2",
}


def read_reader_config(config_files, loader=UnsafeLoader):
    """Read the reader `config_files` and return the extracted reader metadata."""
    reader_config = load_yaml_reader_configs(*config_files, loader=loader)
    return reader_config["reader"]


def configs_for_reader(reader=None):
    """Generate reader configuration files for one or more readers.

    Args:
        reader (Optional[str]): Yield configs only for this reader

    Returns: Generator of lists of configuration files

    """
    config_files = _get_configs(reader)

    for config_file in config_files:
        config_basename = os.path.basename(config_file)
        reader_name = os.path.splitext(config_basename)[0]
        paths = get_entry_points_config_dirs("satpy.readers")
        reader_configs = config_search_paths(
            os.path.join("readers", config_basename),
            search_dirs=paths, check_exists=True)

        if not reader_configs:
            # either the reader they asked for does not exist
            # or satpy is improperly configured and can't find its own readers
            raise ValueError("No reader named: {}".format(reader_name))

        yield reader_configs


def _get_configs(reader):
    if reader is not None:
        if not isinstance(reader, (list, tuple)):
            reader = [reader]

        reader = get_valid_reader_names(reader)
        # given a config filename or reader name
        return [r if r.endswith(".yaml") else r + ".yaml" for r in reader]

    paths = get_entry_points_config_dirs("satpy.readers")
    reader_configs = glob_config(os.path.join("readers", "*.yaml"), search_dirs=paths)
    return set(reader_configs)


def available_readers(
        as_dict: bool = False,
        yaml_loader: type[BaseLoader] | type[FullLoader] | type[UnsafeLoader] = yaml.loader.UnsafeLoader,
) -> list[str] | list[dict]:
    """Available readers based on current configuration.

    Args:
        as_dict: Optionally return reader information as a dictionary.
                        Default: False.
        yaml_loader:
            The yaml loader type. Default: ``yaml.loader.UnsafeLoader``.

    Returns:
        List of available reader names. If `as_dict` is `True` then
        a list of dictionaries including additionally reader information is returned.

    """
    readers = []
    for reader_configs in configs_for_reader():
        try:
            reader_info = read_reader_config(reader_configs, loader=yaml_loader)
        except (KeyError, IOError, yaml.YAMLError):
            LOG.debug("Could not import reader config from: %s", reader_configs)
            LOG.debug("Error loading YAML", exc_info=True)
            continue
        readers.append(reader_info if as_dict else reader_info["name"])
    if as_dict:
        readers = sorted(readers, key=lambda reader_info: reader_info["name"])
    else:
        readers = sorted(readers)
    return readers


def get_valid_reader_names(reader):
    """Check for old reader names or readers pending deprecation."""
    new_readers = []
    for reader_name in reader:
        if reader_name in OLD_READER_NAMES:
            raise ValueError(
                "Reader name '{}' has been deprecated, "
                "use '{}' instead.".format(reader_name,
                                           OLD_READER_NAMES[reader_name]))

        if reader_name in PENDING_OLD_READER_NAMES:
            new_name = PENDING_OLD_READER_NAMES[reader_name]
            warnings.warn(
                "Reader name '{}' is being deprecated and will be removed soon."
                "Please use '{}' instead.".format(reader_name, new_name),
                FutureWarning,
                stacklevel=2
            )
            new_readers.append(new_name)
        else:
            new_readers.append(reader_name)

    return new_readers
