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
"""Reader loading."""
import logging

import yaml

from satpy.readers.core.yaml_reader import AbstractYAMLReader

from .config import configs_for_reader

LOG = logging.getLogger(__name__)


def load_readers(filenames=None, reader=None, reader_kwargs=None):
    """Create specified readers and assign files to them.

    Args:
        filenames (Iterable or dict): A sequence of files that will be used to load data from. A ``dict`` object
                                      should map reader names to a list of filenames for that reader.
        reader (str or list): The name of the reader to use for loading the data or a list of names.
        reader_kwargs (dict): Keyword arguments to pass to specific reader instances.
            This can either be a single dictionary that will be passed to all
            reader instances, or a mapping of reader names to dictionaries.  If
            the keys of ``reader_kwargs`` match exactly the list of strings in
            ``reader`` or the keys of filenames, each reader instance will get its
            own keyword arguments accordingly.

    Returns: Dictionary mapping reader name to reader instance

    """
    reader_instances = {}
    if _early_exit(filenames, reader):
        return {}

    reader, filenames, remaining_filenames = _get_reader_and_filenames(reader, filenames)
    (reader_kwargs, reader_kwargs_without_filter) = _get_reader_kwargs(reader, reader_kwargs)

    if reader_kwargs is None:
        reader_kwargs = {}

    for idx, reader_configs in enumerate(configs_for_reader(reader)):
        readers_files = _get_readers_files(filenames, reader, idx, remaining_filenames)
        reader_instance = _get_reader_instance(reader, reader_configs, idx, reader_kwargs)
        if reader_instance is None or not readers_files:
            # Reader initiliasation failed or no files were given
            continue

        loadables = reader_instance.select_files_from_pathnames(readers_files)
        if loadables:
            reader_instance.create_storage_items(
                    loadables,
                    fh_kwargs=reader_kwargs_without_filter[None if reader is None else reader[idx]])
            reader_instances[reader_instance.name] = reader_instance
            remaining_filenames -= set(loadables)

        if not remaining_filenames:
            break

    _check_remaining_files(remaining_filenames)
    _check_reader_instances(reader_instances)
    return reader_instances


def _get_readers_files(filenames, reader, idx, remaining_filenames):
    if isinstance(filenames, dict):
        return set(filenames[reader[idx]])
    return remaining_filenames


def _get_reader_instance(reader, reader_configs, idx, reader_kwargs):
    reader_instance = None
    try:
        reader_instance = load_reader(
            reader_configs,
            **reader_kwargs[None if reader is None else reader[idx]])
    except (KeyError, IOError) as err:
        LOG.info("Cannot use %s", str(reader_configs))
        LOG.debug(str(err))
    except yaml.constructor.ConstructorError as err:
        _log_yaml_error(reader_configs, err)

    return reader_instance


def load_reader(reader_configs, **reader_kwargs):
    """Import and setup the reader from *reader_info*."""
    return AbstractYAMLReader.from_config_files(*reader_configs, **reader_kwargs)


def _log_yaml_error(reader_configs, err):
    LOG.error("Problem with %s", str(reader_configs))
    LOG.error(str(err))


def _early_exit(filenames, reader):
    if not filenames and not reader:
        # used for an empty Scene
        return True
    _check_reader_and_filenames(reader, filenames)
    if not filenames:
        LOG.warning("'filenames' required to create readers and load data")
        return True
    return False


def _check_reader_and_filenames(reader, filenames):
    if reader and filenames is not None and not filenames:
        # user made a mistake in their glob pattern
        raise ValueError("'filenames' was provided but is empty.")


def _get_reader_and_filenames(reader, filenames):
    if reader is None and isinstance(filenames, dict):
        # filenames is a dictionary of reader_name -> filenames
        reader = list(filenames.keys())
        remaining_filenames = set(f for fl in filenames.values() for f in fl)
    elif reader and isinstance(filenames, dict):
        # filenames is a dictionary of reader_name -> filenames
        # but they only want one of the readers
        filenames = filenames[reader]
        remaining_filenames = set(filenames or [])
    else:
        remaining_filenames = set(filenames or [])
    return reader, filenames, remaining_filenames


def _check_remaining_files(remaining_filenames):
    if remaining_filenames:
        LOG.warning("Don't know how to open the following files: {}".format(str(remaining_filenames)))


def _check_reader_instances(reader_instances):
    if not reader_instances:
        raise ValueError("No supported files found")
    if not any(list(r.available_dataset_ids) for r in reader_instances.values()):
        raise ValueError("No dataset could be loaded. Either missing "
                         "requirements (such as Epilog, Prolog) or none of the "
                         "provided files match the filter parameters.")


def _get_reader_kwargs(reader, reader_kwargs):
    """Help load_readers to form reader_kwargs.

    Helper for load_readers to get reader_kwargs and
    reader_kwargs_without_filter in the desirable form.
    """
    reader_kwargs = reader_kwargs or {}

    # ensure one reader_kwargs per reader, None if not provided
    if reader is None:
        reader_kwargs = {None: reader_kwargs}
    elif reader_kwargs.keys() != set(reader):
        reader_kwargs = dict.fromkeys(reader, reader_kwargs)

    reader_kwargs_without_filter = {}
    for (k, v) in reader_kwargs.items():
        reader_kwargs_without_filter[k] = v.copy()
        reader_kwargs_without_filter[k].pop("filter_parameters", None)

    return (reader_kwargs, reader_kwargs_without_filter)
