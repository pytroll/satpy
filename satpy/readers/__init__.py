#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2018 Satpy developers
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
"""Shared objects of the various reader classes."""

import logging
import os
from datetime import datetime, timedelta
import numpy as np

import yaml

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

from satpy.config import (config_search_paths, get_environ_config_dir,
                          glob_config)
from satpy.dataset import DataID, minimal_default_keys_config, create_filtered_query

LOG = logging.getLogger(__name__)


# Old Name -> New Name
OLD_READER_NAMES = {
}


class TooManyResults(KeyError):
    """Special exception when one key maps to multiple items in the container."""

    pass


def _wl_dist(wl_a, wl_b):
    """Return the distance between two requested wavelengths."""
    if isinstance(wl_a, tuple):
        # central wavelength
        wl_a = wl_a[1]
    if isinstance(wl_b, tuple):
        wl_b = wl_b[1]
    if wl_a is None or wl_b is None:
        return 1000.
    return abs(wl_a - wl_b)


def get_best_dataset_key(key, choices):
    """Choose the "best" `DataID` from `choices` based on `key`.

    To see how the keys are sorted, refer to `:meth:satpy.datasets.DataQuery.sort_dataids`.

    This function assumes `choices` has already been filtered to only
    include datasets that match the provided `key`.

    Args:
        key (DataQuery): Query parameters to sort `choices` by.
        choices (iterable): `DataID` objects to sort through to determine
                            the best dataset.

    Returns: List of best `DataID`s from `choices`. If there is more
             than one element this function could not choose between the
             available datasets.

    """
    sorted_choices, distances = key.sort_dataids(choices)
    if len(sorted_choices) == 0 or distances[0] is np.inf:
        return []
    else:
        return [choice for choice, distance in zip(sorted_choices, distances) if distance == distances[0]]


def filter_keys_by_dataset_query(dquery, key_container):
    """Filer provided key iterable by the provided `DataQuery`.

    Note: The `modifiers` attribute of `did` should be `None` to allow for
          **any** modifier in the results.

    Args:
        dquery (DataQuery): Query parameters to match in the `key_container`.
        key_container (iterable): Set, list, tuple, or dict of `DataID`
                                  keys.

    Returns (list): List of keys matching the provided parameters in no
                    specific order.

    """
    return dquery.filter_dataids(key_container)


def get_key(key, key_container, num_results=1, best=True, query=None,
            **kwargs):
    """Get the fully-specified key best matching the provided key.

    Only the best match is returned if `best` is `True` (default). See
    `get_best_dataset_key` for more information on how this is determined.

    `query` is provided as a convenience to filter by multiple parameters
    at once without having to filter by multiple `key` inputs.

    Args:
        key (DataID): DataID of query parameters to use for
                         searching. Any parameter that is `None`
                         is considered a wild card and any match is
                         accepted.
        key_container (dict or set): Container of DataID objects that
                                     uses hashing to quickly access items.
        num_results (int): Number of results to return. Use `0` for all
                           matching results. If `1` then the single matching
                           key is returned instead of a list of length 1.
                           (default: 1)
        best (bool): Sort results to get "best" result first
                     (default: True). See `get_best_dataset_key` for details.
        query (DataQuery): filter for the key which can contain for example:
            resolution (float, int, or list): Resolution of the dataset in
                                            dataset units (typically
                                            meters). This can also be a
                                            list of these numbers.
            calibration (str or list): Dataset calibration
                                    (ex.'reflectance'). This can also be a
                                    list of these strings.
            polarization (str or list): Dataset polarization
                                        (ex.'V'). This can also be a
                                        list of these strings.
            level (number or list): Dataset level (ex. 100). This can also be a
                                    list of these numbers.
            modifiers (list): Modifiers applied to the dataset. Unlike
                            resolution and calibration this is the exact
                            desired list of modifiers for one dataset, not
                            a list of possible modifiers.


    Returns (list or DataID): Matching key(s)

    Raises: KeyError if no matching results or if more than one result is
            found when `num_results` is `1`.

    """
    key = create_filtered_query(key, query)

    res = filter_keys_by_dataset_query(key, key_container)
    if not res:
        raise KeyError("No dataset matching '{}' found".format(str(key)))

    if best:
        res = get_best_dataset_key(key, res)

    if num_results == 1 and not res:
        raise KeyError("No dataset matching '{}' found".format(str(key)))
    elif num_results == 1 and len(res) != 1:
        raise TooManyResults("No unique dataset matching {}".format(str(key)))
    elif num_results == 1:
        return res[0]
    elif num_results == 0:
        return res
    else:
        return res[:num_results]


class DatasetDict(dict):
    """Special dictionary object that can handle dict operations based on dataset name, wavelength, or DataID.

    Note: Internal dictionary keys are `DataID` objects.

    """

    def keys(self, names=False, wavelengths=False):
        """Give currently contained keys."""
        # sort keys so things are a little more deterministic (.keys() is not)
        keys = sorted(super(DatasetDict, self).keys())
        if names:
            return (k.get('name') for k in keys)
        elif wavelengths:
            return (k.get('wavelength') for k in keys)
        else:
            return keys

    def get_key(self, match_key, num_results=1, best=True, **dfilter):
        """Get multiple fully-specified keys that match the provided query.

        Args:
            key (DataID): DataID of query parameters to use for
                          searching. Any parameter that is `None`
                          is considered a wild card and any match is
                          accepted. Can also be a string representing the
                          dataset name or a number representing the dataset
                          wavelength.
            num_results (int): Number of results to return. If `0` return all,
                               if `1` return only that element, otherwise
                               return a list of matching keys.
            **dfilter (dict): See `get_key` function for more information.

        """
        return get_key(match_key, self.keys(), num_results=num_results,
                       best=best, **dfilter)

    def getitem(self, item):
        """Get Node when we know the *exact* DataID."""
        return super(DatasetDict, self).__getitem__(item)

    def __getitem__(self, item):
        """Get item from container."""
        try:
            # short circuit - try to get the object without more work
            return super(DatasetDict, self).__getitem__(item)
        except KeyError:
            key = self.get_key(item)
            return super(DatasetDict, self).__getitem__(key)

    def get(self, key, default=None):
        """Get value with optional default."""
        try:
            key = self.get_key(key)
        except KeyError:
            return default
        return super(DatasetDict, self).get(key, default)

    def __setitem__(self, key, value):
        """Support assigning 'Dataset' objects or dictionaries of metadata."""
        value_dict = value
        if hasattr(value, 'attrs'):
            # xarray.DataArray objects
            value_dict = value.attrs
        # use value information to make a more complete DataID
        if not isinstance(key, DataID):
            if not isinstance(value_dict, dict):
                raise ValueError("Key must be a DataID when value is not an xarray DataArray or dict")
            old_key = key
            try:
                key = self.get_key(key)
            except KeyError:
                if isinstance(old_key, str):
                    new_name = old_key
                else:
                    new_name = value_dict.get("name")
                # this is a new key and it's not a full DataID tuple
                if new_name is None and value_dict.get('wavelength') is None:
                    raise ValueError("One of 'name' or 'wavelength' attrs "
                                     "values should be set.")
                try:
                    id_keys = value_dict['_satpy_id'].id_keys
                except KeyError:
                    try:
                        id_keys = value_dict['_satpy_id_keys']
                    except KeyError:
                        id_keys = minimal_default_keys_config
                value_dict['name'] = new_name
                key = DataID(id_keys, **value_dict)
                if hasattr(value, 'attrs') and 'name' not in value.attrs:
                    value.attrs['name'] = new_name

        # update the 'value' with the information contained in the key
        try:
            new_info = key.to_dict()
        except AttributeError:
            new_info = key
        if isinstance(value_dict, dict):
            value_dict.update(new_info)

        if hasattr(value, 'attrs'):
            if isinstance(key, DataID):
                value.attrs['_satpy_id'] = key

        return super(DatasetDict, self).__setitem__(key, value)

    def contains(self, item):
        """Check contains when we know the *exact* DataID."""
        return super(DatasetDict, self).__contains__(item)

    def __contains__(self, item):
        """Check if item exists in container."""
        try:
            key = self.get_key(item)
        except KeyError:
            return False
        return super(DatasetDict, self).__contains__(key)

    def __delitem__(self, key):
        """Delete item from container."""
        try:
            # short circuit - try to get the object without more work
            return super(DatasetDict, self).__delitem__(key)
        except KeyError:
            key = self.get_key(key)
            return super(DatasetDict, self).__delitem__(key)


def group_files(files_to_sort, reader=None, time_threshold=10,
                group_keys=None, ppp_config_dir=None, reader_kwargs=None):
    """Group series of files by file pattern information.

    By default this will group files by their filename ``start_time``
    assuming it exists in the pattern. By passing the individual
    dictionaries returned by this function to the Scene classes'
    ``filenames``, a series `Scene` objects can be easily created.

    .. versionadded:: 0.12

    Args:
        files_to_sort (iterable): File paths to sort in to group
        reader (str): Reader whose file patterns should be used to sort files.
            This is currently a required keyword argument, but may be optional
            in the future (see inline code comments for details).
        time_threshold (int): Number of seconds used to consider time elements
            in a group as being equal. For example, if the 'start_time' item
            is used to group files then any time within `time_threshold`
            seconds of the first file's 'start_time' will be seen as occurring
            at the same time.
        group_keys (list or tuple): File pattern information to use to group
            files. Keys are sorted in order and only the first key is used when
            comparing datetime elements with `time_threshold` (see above). This
            means it is recommended that datetime values should only come from
            the first key in ``group_keys``. Otherwise, there is a good chance
            that files will not be grouped properly (datetimes being barely
            unequal). Defaults to a reader's ``group_keys`` configuration (set
            in YAML), otherwise ``('start_time',)``.
        ppp_config_dir (str): Root usser configuration directory for Satpy.
            This will be deprecated in the future, but is here for consistency
            with other Satpy features.
        reader_kwargs (dict): Additional keyword arguments to pass to reader
            creation.

    Returns:
        List of dictionaries mapping 'reader' to a list of filenames.
        Each of these dictionaries can be passed as ``filenames`` to
        a `Scene` object.

    """
    # FUTURE: Find the best reader for each filename using `find_files_and_readers`
    if reader is None:
        raise ValueError("'reader' keyword argument is required.")
    elif not isinstance(reader, (list, tuple)):
        reader = [reader]

    # FUTURE: Handle multiple readers
    reader = reader[0]
    reader_configs = list(configs_for_reader(reader, ppp_config_dir))[0]
    reader_kwargs = reader_kwargs or {}
    try:
        reader_instance = load_reader(reader_configs, **reader_kwargs)
    except (KeyError, IOError, yaml.YAMLError) as err:
        LOG.info('Cannot use %s', str(reader_configs))
        LOG.debug(str(err))
        # if reader and (isinstance(reader, str) or len(reader) == 1):
        #     # if it is a single reader then give a more usable error
        #     raise
        raise

    if group_keys is None:
        group_keys = reader_instance.info.get('group_keys', ('start_time',))
    file_keys = []
    # make a copy because filename_items_for_filetype will modify inplace
    files_to_sort = set(files_to_sort)
    for _, filetype_info in reader_instance.sorted_filetype_items():
        for f, file_info in reader_instance.filename_items_for_filetype(files_to_sort, filetype_info):
            group_key = tuple(file_info.get(k) for k in group_keys)
            file_keys.append((group_key, f))

    prev_key = None
    threshold = timedelta(seconds=time_threshold)
    file_groups = {}
    for gk, f in sorted(file_keys):
        # use first element of key as time identifier (if datetime type)
        if prev_key is None:
            is_new_group = True
            prev_key = gk
        elif isinstance(gk[0], datetime):
            # datetimes within threshold difference are "the same time"
            is_new_group = (gk[0] - prev_key[0]) > threshold
        else:
            is_new_group = gk[0] != prev_key[0]

        # compare keys for those that are found for both the key and
        # this is a generator and is not computed until the if statement below
        # when we know that `prev_key` is not None
        vals_not_equal = (this_val != prev_val for this_val, prev_val in zip(gk[1:], prev_key[1:])
                          if this_val is not None and prev_val is not None)
        # if this is a new group based on the first element
        if is_new_group or any(vals_not_equal):
            file_groups[gk] = [f]
            prev_key = gk
        else:
            file_groups[prev_key].append(f)
    sorted_group_keys = sorted(file_groups)
    # passable to Scene as 'filenames'
    return [{reader: file_groups[group_key]} for group_key in sorted_group_keys]


def read_reader_config(config_files, loader=UnsafeLoader):
    """Read the reader `config_files` and return the info extracted."""
    conf = {}
    LOG.debug('Reading %s', str(config_files))
    for config_file in config_files:
        with open(config_file) as fd:
            conf.update(yaml.load(fd.read(), Loader=loader))

    try:
        reader_info = conf['reader']
    except KeyError:
        raise KeyError(
            "Malformed config file {}: missing reader 'reader'".format(
                config_files))
    reader_info['config_files'] = config_files
    return reader_info


def load_reader(reader_configs, **reader_kwargs):
    """Import and setup the reader from *reader_info*."""
    reader_info = read_reader_config(reader_configs)
    reader_instance = reader_info['reader'](config_files=reader_configs, **reader_kwargs)
    return reader_instance


def configs_for_reader(reader=None, ppp_config_dir=None):
    """Generate reader configuration files for one or more readers.

    Args:
        reader (Optional[str]): Yield configs only for this reader
        ppp_config_dir (Optional[str]): Additional configuration directory
            to search for reader configuration files.

    Returns: Generator of lists of configuration files

    """
    search_paths = (ppp_config_dir,) if ppp_config_dir else tuple()
    if reader is not None:
        if not isinstance(reader, (list, tuple)):
            reader = [reader]
        # check for old reader names
        new_readers = []
        for reader_name in reader:
            if reader_name.endswith('.yaml') or reader_name not in OLD_READER_NAMES:
                new_readers.append(reader_name)
                continue

            new_name = OLD_READER_NAMES[reader_name]
            # Satpy 0.11 only displays a warning
            # Satpy 0.13 will raise an exception
            raise ValueError("Reader name '{}' has been deprecated, use '{}' instead.".format(reader_name, new_name))
            # Satpy 0.15 or 1.0, remove exception and mapping

        reader = new_readers
        # given a config filename or reader name
        config_files = [r if r.endswith('.yaml') else r + '.yaml' for r in reader]
    else:
        reader_configs = glob_config(os.path.join('readers', '*.yaml'),
                                     *search_paths)
        config_files = set(reader_configs)

    for config_file in config_files:
        config_basename = os.path.basename(config_file)
        reader_name = os.path.splitext(config_basename)[0]
        reader_configs = config_search_paths(
            os.path.join("readers", config_basename), *search_paths)

        if not reader_configs:
            # either the reader they asked for does not exist
            # or satpy is improperly configured and can't find its own readers
            raise ValueError("No reader named: {}".format(reader_name))

        yield reader_configs


def available_readers(as_dict=False):
    """Available readers based on current configuration.

    Args:
        as_dict (bool): Optionally return reader information as a dictionary.
                        Default: False

    Returns: List of available reader names. If `as_dict` is `True` then
             a list of dictionaries including additionally reader information
             is returned.

    """
    readers = []
    for reader_configs in configs_for_reader():
        try:
            reader_info = read_reader_config(reader_configs)
        except (KeyError, IOError, yaml.YAMLError):
            LOG.warning("Could not import reader config from: %s", reader_configs)
            LOG.debug("Error loading YAML", exc_info=True)
            continue
        readers.append(reader_info if as_dict else reader_info['name'])
    return readers


def find_files_and_readers(start_time=None, end_time=None, base_dir=None,
                           reader=None, sensor=None, ppp_config_dir=None,
                           filter_parameters=None, reader_kwargs=None,
                           missing_ok=False, fs=None):
    """Find files matching the provided parameters.

    Use `start_time` and/or `end_time` to limit found filenames by the times
    in the filenames (not the internal file metadata). Files are matched if
    they fall anywhere within the range specified by these parameters.

    Searching is **NOT** recursive.

    Files may be either on-disk or on a remote file system.  By default,
    files are searched for locally.  Users can search on remote filesystems by
    passing an instance of an implementation of
    `fsspec.spec.AbstractFileSystem` (strictly speaking, any object of a class
    implementing a ``glob`` method works).

    If locating files on a local file system, the returned dictionary
    can be passed directly to the `Scene` object through the `filenames`
    keyword argument.  If it points to a remote file system, it is the
    responsibility of the user to download the files first (directly
    reading from cloud storage is not currently available in Satpy).

    The behaviour of time-based filtering depends on whether or not the filename
    contains information about the end time of the data or not:

      - if the end time is not present in the filename, the start time of the filename
        is used and has to fall between (inclusive) the requested start and end times
      - otherwise, the timespan of the filename has to overlap the requested timespan

    Example usage for querying a s3 filesystem using the s3fs module:

    >>> import s3fs, satpy.readers, datetime
    >>> satpy.readers.find_files_and_readers(
    ...     base_dir="s3://noaa-goes16/ABI-L1b-RadF/2019/321/14/",
    ...     fs=s3fs.S3FileSystem(anon=True),
    ...     reader="abi_l1b",
    ...     start_time=datetime.datetime(2019, 11, 17, 14, 40))
    {'abi_l1b': [...]}

    Args:
        start_time (datetime): Limit used files by starting time.
        end_time (datetime): Limit used files by ending time.
        base_dir (str): The directory to search for files containing the
                        data to load. Defaults to the current directory.
        reader (str or list): The name of the reader to use for loading the data or a list of names.
        sensor (str or list): Limit used files by provided sensors.
        ppp_config_dir (str): The directory containing the configuration
                              files for Satpy.
        filter_parameters (dict): Filename pattern metadata to filter on. `start_time` and `end_time` are
                                  automatically added to this dictionary. Shortcut for
                                  `reader_kwargs['filter_parameters']`.
        reader_kwargs (dict): Keyword arguments to pass to specific reader
                              instances to further configure file searching.
        missing_ok (bool): If False (default), raise ValueError if no files
                            are found.  If True, return empty dictionary if no
                            files are found.
        fs (FileSystem): Optional, instance of implementation of
                         fsspec.spec.AbstractFileSystem (strictly speaking,
                         any object of a class implementing ``.glob`` is
                         enough).  Defaults to searching the local filesystem.

    Returns: Dictionary mapping reader name string to list of filenames

    """
    if ppp_config_dir is None:
        ppp_config_dir = get_environ_config_dir()
    reader_files = {}
    reader_kwargs = reader_kwargs or {}
    filter_parameters = filter_parameters or reader_kwargs.get('filter_parameters', {})
    sensor_supported = False

    if start_time or end_time:
        filter_parameters['start_time'] = start_time
        filter_parameters['end_time'] = end_time
    reader_kwargs['filter_parameters'] = filter_parameters

    for reader_configs in configs_for_reader(reader, ppp_config_dir):
        try:
            reader_instance = load_reader(reader_configs, **reader_kwargs)
        except (KeyError, IOError, yaml.YAMLError) as err:
            LOG.info('Cannot use %s', str(reader_configs))
            LOG.debug(str(err))
            if reader and (isinstance(reader, str) or len(reader) == 1):
                # if it is a single reader then give a more usable error
                raise
            continue

        if not reader_instance.supports_sensor(sensor):
            continue
        elif sensor is not None:
            # sensor was specified and a reader supports it
            sensor_supported = True
        loadables = reader_instance.select_files_from_directory(base_dir, fs)
        if loadables:
            loadables = list(
                reader_instance.filter_selected_filenames(loadables))
        if loadables:
            reader_files[reader_instance.name] = list(loadables)

    if sensor and not sensor_supported:
        raise ValueError("Sensor '{}' not supported by any readers".format(sensor))

    if not (reader_files or missing_ok):
        raise ValueError("No supported files found")
    return reader_files


def load_readers(filenames=None, reader=None, reader_kwargs=None,
                 ppp_config_dir=None):
    """Create specified readers and assign files to them.

    Args:
        filenames (iterable or dict): A sequence of files that will be used to load data from. A ``dict`` object
                                      should map reader names to a list of filenames for that reader.
        reader (str or list): The name of the reader to use for loading the data or a list of names.
        reader_kwargs (dict): Keyword arguments to pass to specific reader instances.
        ppp_config_dir (str): The directory containing the configuration files for satpy.

    Returns: Dictionary mapping reader name to reader instance

    """
    reader_instances = {}
    reader_kwargs = reader_kwargs or {}
    reader_kwargs_without_filter = reader_kwargs.copy()
    reader_kwargs_without_filter.pop('filter_parameters', None)

    if ppp_config_dir is None:
        ppp_config_dir = get_environ_config_dir()

    if not filenames and not reader:
        # used for an empty Scene
        return {}
    elif reader and filenames is not None and not filenames:
        # user made a mistake in their glob pattern
        raise ValueError("'filenames' was provided but is empty.")
    elif not filenames:
        LOG.warning("'filenames' required to create readers and load data")
        return {}
    elif reader is None and isinstance(filenames, dict):
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

    for idx, reader_configs in enumerate(configs_for_reader(reader, ppp_config_dir)):
        if isinstance(filenames, dict):
            readers_files = set(filenames[reader[idx]])
        else:
            readers_files = remaining_filenames

        try:
            reader_instance = load_reader(reader_configs, **reader_kwargs)
        except (KeyError, IOError, yaml.YAMLError) as err:
            LOG.info('Cannot use %s', str(reader_configs))
            LOG.debug(str(err))
            continue

        if not readers_files:
            # we weren't given any files for this reader
            continue
        loadables = reader_instance.select_files_from_pathnames(readers_files)
        if loadables:
            reader_instance.create_filehandlers(loadables, fh_kwargs=reader_kwargs_without_filter)
            reader_instances[reader_instance.name] = reader_instance
            remaining_filenames -= set(loadables)
        if not remaining_filenames:
            break

    if remaining_filenames:
        LOG.warning("Don't know how to open the following files: {}".format(str(remaining_filenames)))
    if not reader_instances:
        raise ValueError("No supported files found")
    elif not any(list(r.available_dataset_ids) for r in reader_instances.values()):
        raise ValueError("No dataset could be loaded. Either missing "
                         "requirements (such as Epilog, Prolog) or none of the "
                         "provided files match the filter parameters.")
    return reader_instances
