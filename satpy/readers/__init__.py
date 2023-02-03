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
from __future__ import annotations

import logging
import os
import pickle  # nosec B403
import warnings
from datetime import datetime, timedelta
from functools import total_ordering

import yaml
from yaml import UnsafeLoader

from satpy._config import config_search_paths, get_entry_points_config_dirs, glob_config

from .yaml_reader import AbstractYAMLReader
from .yaml_reader import load_yaml_configs as load_yaml_reader_configs

LOG = logging.getLogger(__name__)


# Old Name -> New Name
PENDING_OLD_READER_NAMES = {'fci_l1c_fdhsi': 'fci_l1c_nc'}
OLD_READER_NAMES: dict[str, str] = {}


def group_files(files_to_sort, reader=None, time_threshold=10,
                group_keys=None, reader_kwargs=None,
                missing="pass"):
    """Group series of files by file pattern information.

    By default this will group files by their filename ``start_time``
    assuming it exists in the pattern. By passing the individual
    dictionaries returned by this function to the Scene classes'
    ``filenames``, a series `Scene` objects can be easily created.

    Args:
        files_to_sort (iterable): File paths to sort in to group
        reader (str or Collection[str]): Reader or readers whose file patterns
            should be used to sort files.  If not given, try all readers (slow,
            adding a list of readers is strongly recommended).
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
            in YAML), otherwise ``('start_time',)``.  When passing multiple
            readers, passing group_keys is strongly recommended as the
            behaviour without doing so is undefined.
        reader_kwargs (dict): Additional keyword arguments to pass to reader
            creation.
        missing (str): Parameter to control the behavior in the scenario where
            multiple readers were passed, but at least one group does not have
            files associated with every reader.  Valid values are ``"pass"``
            (the default), ``"skip"``, and ``"raise"``.  If set to ``"pass"``,
            groups are passed as-is.  Some groups may have zero files for some
            readers.  If set to ``"skip"``, groups for which one or more
            readers have zero files are skipped (meaning that some files may
            not be associated to any group).  If set to ``"raise"``, raise a
            `FileNotFoundError` in case there are any groups for which one or
            more readers have no files associated.

    Returns:
        List of dictionaries mapping 'reader' to a list of filenames.
        Each of these dictionaries can be passed as ``filenames`` to
        a `Scene` object.

    """
    if reader is not None and not isinstance(reader, (list, tuple)):
        reader = [reader]

    reader_kwargs = reader_kwargs or {}

    reader_files = _assign_files_to_readers(
            files_to_sort, reader, reader_kwargs)

    if reader is None:
        reader = reader_files.keys()

    file_keys = _get_file_keys_for_reader_files(
            reader_files, group_keys=group_keys)

    file_groups = _get_sorted_file_groups(file_keys, time_threshold)

    groups = [{rn: file_groups[group_key].get(rn, []) for rn in reader} for group_key in file_groups]

    return list(_filter_groups(groups, missing=missing))


def _assign_files_to_readers(files_to_sort, reader_names,
                             reader_kwargs):
    """Assign files to readers.

    Given a list of file names (paths), match those to reader instances.

    Internal helper for group_files.

    Args:
        files_to_sort (Collection[str]): Files to assign to readers.
        reader_names (Collection[str]): Readers to consider
        reader_kwargs (Mapping):

    Returns:
        Mapping[str, Tuple[reader, Set[str]]]
        Mapping where the keys are reader names and the values are tuples of
        (reader_configs, filenames).
    """
    files_to_sort = set(files_to_sort)
    reader_dict = {}
    for reader_configs in configs_for_reader(reader_names):
        try:
            reader = load_reader(reader_configs, **reader_kwargs)
        except yaml.constructor.ConstructorError:
            LOG.exception(
                    f"ConstructorError loading {reader_configs!s}, "
                    "probably a missing dependency, skipping "
                    "corresponding reader (if you did not explicitly "
                    "specify the reader, Satpy tries all; performance "
                    "will improve if you pass readers explicitly).")
            continue
        reader_name = reader.info["name"]
        files_matching = set(reader.filter_selected_filenames(files_to_sort))
        files_to_sort -= files_matching
        if files_matching or reader_names is not None:
            reader_dict[reader_name] = (reader, files_matching)
    if files_to_sort:
        raise ValueError("No matching readers found for these files: " +
                         ", ".join(files_to_sort))
    return reader_dict


def _get_file_keys_for_reader_files(reader_files, group_keys=None):
    """From a mapping from _assign_files_to_readers, get file keys.

    Given a mapping where each key is a reader name and each value is a
    tuple of reader instance (typically FileYAMLReader) and a collection
    of files, return a mapping with the same keys, but where the values are
    lists of tuples of (keys, filename), where keys are extracted from the filenames
    according to group_keys and filenames are the names those keys were
    extracted from.

    Internal helper for group_files.

    Returns:
        Mapping[str, List[Tuple[Tuple, str]]], as described.
    """
    file_keys = {}
    for (reader_name, (reader_instance, files_to_sort)) in reader_files.items():
        if group_keys is None:
            group_keys = reader_instance.info.get('group_keys', ('start_time',))
        file_keys[reader_name] = []
        # make a copy because filename_items_for_filetype will modify inplace
        files_to_sort = set(files_to_sort)
        for _, filetype_info in reader_instance.sorted_filetype_items():
            for f, file_info in reader_instance.filename_items_for_filetype(files_to_sort, filetype_info):
                group_key = tuple(file_info.get(k) for k in group_keys)
                if all(g is None for g in group_key):
                    warnings.warn(
                            f"Found matching file {f:s} for reader "
                            "{reader_name:s}, but none of group keys found. "
                            "Group keys requested: " + ", ".join(group_keys),
                            UserWarning)
                file_keys[reader_name].append((group_key, f))
    return file_keys


def _get_sorted_file_groups(all_file_keys, time_threshold):
    """Get sorted file groups.

    Get a list of dictionaries, where each list item consists of a dictionary
    mapping a tuple of keys to a mapping of reader names to files.  The files
    listed in each list item are considered to be grouped within the same time.

    Args:
        all_file_keys, as returned by _get_file_keys_for_reader_files
        time_threshold: temporal threshold

    Returns:
        List[Mapping[Tuple, Mapping[str, List[str]]]], as described

    Internal helper for group_files.
    """
    # flatten to get an overall sorting; put the name in the middle in the
    # interest of sorting
    flat_keys = ((v[0], rn, v[1]) for (rn, vL) in all_file_keys.items() for v in vL)
    prev_key = None
    threshold = timedelta(seconds=time_threshold)
    # file_groups is sorted, because dictionaries are sorted by insertion
    # order in Python 3.7+
    file_groups = {}
    for gk, rn, f in sorted(flat_keys):
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
            file_groups[gk] = {rn: [f]}
            prev_key = gk
        else:
            if rn not in file_groups[prev_key]:
                file_groups[prev_key][rn] = [f]
            else:
                file_groups[prev_key][rn].append(f)
    return file_groups


def _filter_groups(groups, missing="pass"):
    """Filter multi-reader group-files behavior.

    Helper for `group_files`.  When `group_files` is called with multiple
    readers, make sure that the desired behaviour for missing files is
    enforced: if missing is ``"raise"``, raise an exception if at least one
    group has at least one reader without files; if it is ``"skip"``, remove
    those.  If it is ``"pass"``, do nothing.  Yields groups to be kept.

    Args:
        groups (List[Mapping[str, List[str]]]):
            groups as found by `group_files`.
        missing (str):
            String controlling behaviour, see documentation above.

    Yields:
        ``Mapping[str:, List[str]]``: groups to be retained
    """
    if missing == "pass":
        yield from groups
        return
    if missing not in ("raise", "skip"):
        raise ValueError("Invalid value for ``missing`` argument.  Expected "
                         f"'raise', 'skip', or 'pass', got {missing!r}")
    for (i, grp) in enumerate(groups):
        readers_without_files = _get_keys_with_empty_values(grp)
        if readers_without_files:
            if missing == "raise":
                raise FileNotFoundError(
                        f"when grouping files, group at index {i:d} "
                        "had no files for readers: " +
                        ", ".join(readers_without_files))
        else:
            yield grp


def _get_keys_with_empty_values(grp):
    """Find mapping keys where values have length zero.

    Helper for `_filter_groups`, which is in turn a helper for `group_files`.
    Given a mapping key -> Collection[Any], return the keys where the length of the
    collection is zero.

    Args:
        grp (Mapping[Any, Collection[Any]]): dictionary to check

    Returns:
        set of keys
    """
    empty = set()
    for (k, v) in grp.items():
        if len(v) == 0:  # explicit check to ensure failure if not a collection
            empty.add(k)
    return empty


def read_reader_config(config_files, loader=UnsafeLoader):
    """Read the reader `config_files` and return the extracted reader metadata."""
    reader_config = load_yaml_reader_configs(*config_files, loader=loader)
    return reader_config['reader']


def load_reader(reader_configs, **reader_kwargs):
    """Import and setup the reader from *reader_info*."""
    return AbstractYAMLReader.from_config_files(*reader_configs, **reader_kwargs)


def configs_for_reader(reader=None):
    """Generate reader configuration files for one or more readers.

    Args:
        reader (Optional[str]): Yield configs only for this reader

    Returns: Generator of lists of configuration files

    """
    if reader is not None:
        if not isinstance(reader, (list, tuple)):
            reader = [reader]

        reader = get_valid_reader_names(reader)
        # given a config filename or reader name
        config_files = [r if r.endswith('.yaml') else r + '.yaml' for r in reader]
    else:
        paths = get_entry_points_config_dirs('satpy.readers')
        reader_configs = glob_config(os.path.join('readers', '*.yaml'), search_dirs=paths)
        config_files = set(reader_configs)

    for config_file in config_files:
        config_basename = os.path.basename(config_file)
        reader_name = os.path.splitext(config_basename)[0]
        paths = get_entry_points_config_dirs('satpy.readers')
        reader_configs = config_search_paths(
            os.path.join("readers", config_basename),
            search_dirs=paths, check_exists=True)

        if not reader_configs:
            # either the reader they asked for does not exist
            # or satpy is improperly configured and can't find its own readers
            raise ValueError("No reader named: {}".format(reader_name))

        yield reader_configs


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
            warnings.warn("Reader name '{}' is being deprecated and will be removed soon."
                          "Please use '{}' instead.".format(reader_name, new_name),
                          FutureWarning)
            new_readers.append(new_name)
        else:
            new_readers.append(reader_name)

    return new_readers


def available_readers(as_dict=False, yaml_loader=UnsafeLoader):
    """Available readers based on current configuration.

    Args:
        as_dict (bool): Optionally return reader information as a dictionary.
                        Default: False.
        yaml_loader (Optional[Union[yaml.BaseLoader, yaml.FullLoader, yaml.UnsafeLoader]]):
            The yaml loader type. Default: ``yaml.UnsafeLoader``.

    Returns:
        Union[list[str], list[dict]]: List of available reader names. If `as_dict` is `True` then
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
        readers.append(reader_info if as_dict else reader_info['name'])
    if as_dict:
        readers = sorted(readers, key=lambda reader_info: reader_info['name'])
    else:
        readers = sorted(readers)
    return readers


def find_files_and_readers(start_time=None, end_time=None, base_dir=None,
                           reader=None, sensor=None,
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
        filter_parameters (dict): Filename pattern metadata to filter on. `start_time` and `end_time` are
                                  automatically added to this dictionary. Shortcut for
                                  `reader_kwargs['filter_parameters']`.
        reader_kwargs (dict): Keyword arguments to pass to specific reader
                              instances to further configure file searching.
        missing_ok (bool): If False (default), raise ValueError if no files
                            are found.  If True, return empty dictionary if no
                            files are found.
        fs (:class:`fsspec.spec.AbstractFileSystem`): Optional, instance of implementation of
            :class:`fsspec.spec.AbstractFileSystem` (strictly speaking, any object of a class implementing
            ``.glob`` is enough).  Defaults to searching the local filesystem.

    Returns:
        dict: Dictionary mapping reader name string to list of filenames

    """
    reader_files = {}
    reader_kwargs = reader_kwargs or {}
    filter_parameters = filter_parameters or reader_kwargs.get('filter_parameters', {})
    sensor_supported = False

    if start_time or end_time:
        filter_parameters['start_time'] = start_time
        filter_parameters['end_time'] = end_time
    reader_kwargs['filter_parameters'] = filter_parameters

    for reader_configs in configs_for_reader(reader):
        (reader_instance, loadables, this_sensor_supported) = _get_loadables_for_reader_config(
                base_dir, reader, sensor, reader_configs, reader_kwargs, fs)
        sensor_supported = sensor_supported or this_sensor_supported
        if loadables:
            reader_files[reader_instance.name] = list(loadables)

    if sensor and not sensor_supported:
        raise ValueError("Sensor '{}' not supported by any readers".format(sensor))

    if not (reader_files or missing_ok):
        raise ValueError("No supported files found")
    return reader_files


def _get_loadables_for_reader_config(base_dir, reader, sensor, reader_configs,
                                     reader_kwargs, fs):
    """Get loadables for reader configs.

    Helper for find_files_and_readers.

    Args:
        base_dir: as for `find_files_and_readers`
        reader: as for `find_files_and_readers`
        sensor: as for `find_files_and_readers`
        reader_configs: reader metadata such as returned by
            `configs_for_reader`.
        reader_kwargs: Keyword arguments to be passed to reader.
        fs (FileSystem): as for `find_files_and_readers`
    """
    sensor_supported = False
    try:
        reader_instance = load_reader(reader_configs, **reader_kwargs)
    except (KeyError, IOError, yaml.YAMLError) as err:
        LOG.info('Cannot use %s', str(reader_configs))
        LOG.debug(str(err))
        if reader and (isinstance(reader, str) or len(reader) == 1):
            # if it is a single reader then give a more usable error
            raise
        return (None, [], False)

    if not reader_instance.supports_sensor(sensor):
        return (reader_instance, [], False)
    if sensor is not None:
        # sensor was specified and a reader supports it
        sensor_supported = True
    loadables = reader_instance.select_files_from_directory(base_dir, fs)
    if loadables:
        loadables = list(
            reader_instance.filter_selected_filenames(loadables))
    return (reader_instance, loadables, sensor_supported)


def load_readers(filenames=None, reader=None, reader_kwargs=None):
    """Create specified readers and assign files to them.

    Args:
        filenames (iterable or dict): A sequence of files that will be used to load data from. A ``dict`` object
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

    for idx, reader_configs in enumerate(configs_for_reader(reader)):
        if isinstance(filenames, dict):
            readers_files = set(filenames[reader[idx]])
        else:
            readers_files = remaining_filenames

        try:
            reader_instance = load_reader(
                    reader_configs,
                    **reader_kwargs[None if reader is None else reader[idx]])
        except (KeyError, IOError, yaml.YAMLError) as err:
            LOG.info('Cannot use %s', str(reader_configs))
            LOG.debug(str(err))
            continue

        if not readers_files:
            # we weren't given any files for this reader
            continue
        loadables = reader_instance.select_files_from_pathnames(readers_files)
        if loadables:
            reader_instance.create_filehandlers(
                    loadables,
                    fh_kwargs=reader_kwargs_without_filter[None if reader is None else reader[idx]])
            reader_instances[reader_instance.name] = reader_instance
            remaining_filenames -= set(loadables)
        if not remaining_filenames:
            break

    _check_remaining_files(remaining_filenames)
    _check_reader_instances(reader_instances)
    return reader_instances


def _early_exit(filenames, reader):
    if not filenames and not reader:
        # used for an empty Scene
        return True
    if reader and filenames is not None and not filenames:
        # user made a mistake in their glob pattern
        raise ValueError("'filenames' was provided but is empty.")
    if not filenames:
        LOG.warning("'filenames' required to create readers and load data")
        return True
    return False


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
        reader_kwargs_without_filter[k].pop('filter_parameters', None)

    return (reader_kwargs, reader_kwargs_without_filter)


@total_ordering
class FSFile(os.PathLike):
    """Implementation of a PathLike file object, that can be opened.

    Giving the filenames to :class:`Scene` with valid transfer protocols will automatically
    use this class so manual usage of this class is needed mainly for fine-grained control.

    This class is made to be used in conjuction with fsspec or s3fs. For example::

        from satpy import Scene

        import fsspec
        filename = 'noaa-goes16/ABI-L1b-RadC/2019/001/17/*_G16_s20190011702186*'

        the_files = fsspec.open_files("simplecache::s3://" + filename, s3={'anon': True})

        from satpy.readers import FSFile
        fs_files = [FSFile(open_file) for open_file in the_files]

        scn = Scene(filenames=fs_files, reader='abi_l1b')
        scn.load(['true_color_raw'])

    """

    def __init__(self, file, fs=None):
        """Initialise the FSFile instance.

        Args:
            file (str, Pathlike, or OpenFile):
                String, object implementing the `os.PathLike` protocol, or
                an `fsspec.OpenFile` instance.  If passed an instance of
                `fsspec.OpenFile`, the following argument ``fs`` has no
                effect.
            fs (fsspec filesystem, optional)
                Object implementing the fsspec filesystem protocol.
        """
        self._fs_open_kwargs = _get_fs_open_kwargs(file)
        try:
            self._file = file.path
            self._fs = file.fs
        except AttributeError:
            self._file = file
            self._fs = fs

    def __str__(self):
        """Return the string version of the filename."""
        return os.fspath(self._file)

    def __fspath__(self):
        """Comply with PathLike."""
        return os.fspath(self._file)

    def __repr__(self):
        """Representation of the object."""
        return '<FSFile "' + str(self._file) + '">'

    def open(self, *args, **kwargs):
        """Open the file.

        This is read-only.
        """
        fs_open_kwargs = self._update_with_fs_open_kwargs(kwargs)
        try:
            return self._fs.open(self._file, *args, **fs_open_kwargs)
        except AttributeError:
            return open(self._file, *args, **kwargs)

    def _update_with_fs_open_kwargs(self, user_kwargs):
        """Complement keyword arguments for opening a file via file system."""
        kwargs = user_kwargs.copy()
        kwargs.update(self._fs_open_kwargs)
        return kwargs

    def __lt__(self, other):
        """Implement ordering.

        Ordering is defined by the string representation of the filename,
        without considering the file system.
        """
        return os.fspath(self) < os.fspath(other)

    def __eq__(self, other):
        """Implement equality comparisons.

        Two FSFile instances are considered equal if they have the same
        filename and the same file system.
        """
        return (isinstance(other, FSFile) and
                self._file == other._file and
                self._fs == other._fs)

    def __hash__(self):
        """Implement hashing.

        Make FSFile objects hashable, so that they can be used in sets.  Some
        parts of satpy and perhaps others use sets of filenames (strings or
        pathlib.Path), or maybe use them as dictionary keys.  This requires
        them to be hashable.  To ensure FSFile can work as a drop-in
        replacement for strings of Path objects to represent the location of
        blob of data, FSFile should be hashable too.

        Returns the hash, computed from the hash of the filename and the hash
        of the filesystem.
        """
        try:
            fshash = hash(self._fs)
        except TypeError:  # fsspec < 0.8.8 for CachingFileSystem
            fshash = hash(pickle.dumps(self._fs))  # nosec B403
        return hash(self._file) ^ fshash


def _get_fs_open_kwargs(file):
    """Get keyword arguments for opening a file via file system.

    For example compression.
    """
    return {
        "compression": _get_compression(file)
    }


def _get_compression(file):
    try:
        return file.compression
    except AttributeError:
        return None


def open_file_or_filename(unknown_file_thing):
    """Try to open the *unknown_file_thing*, otherwise return the filename."""
    try:
        f_obj = unknown_file_thing.open()
    except AttributeError:
        f_obj = unknown_file_thing
    return f_obj
