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

"""Grouping functionality for the readers."""

import datetime as dt
import logging
import warnings

import yaml

from .config import configs_for_reader
from .loading import load_reader

LOG = logging.getLogger(__name__)


def group_files(files_to_sort, reader=None, time_threshold=10,
                group_keys=None, reader_kwargs=None,
                missing="pass"):
    """Group series of files by file pattern information.

    By default this will group files by their filename ``start_time``
    assuming it exists in the pattern. By passing the individual
    dictionaries returned by this function to the Scene classes'
    ``filenames``, a series `Scene` objects can be easily created.

    Args:
        files_to_sort (Iterable): File paths to sort in to group
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


def _assign_files_to_readers(files_to_sort, reader_names,  # noqa: D417
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
            group_keys = reader_instance.info.get("group_keys", ("start_time",))
        file_keys[reader_name] = []
        # make a copy because filename_items_for_filetype will modify inplace
        files_to_sort = set(files_to_sort)
        _walk_through_sorted_filetype_items(reader_instance, file_keys, files_to_sort, group_keys, reader_name)
    return file_keys


def _walk_through_sorted_filetype_items(reader_instance, file_keys, files_to_sort, group_keys, reader_name):
    for _, filetype_info in reader_instance.sorted_filetype_items():
        for f, file_info in reader_instance.filename_items_for_filetype(files_to_sort, filetype_info):
            _update_file_keys(file_keys, group_keys, file_info, f, reader_name)


def _update_file_keys(file_keys, group_keys, file_info, f, reader_name):
    group_key = tuple(file_info.get(k) for k in group_keys)
    if all(g is None for g in group_key):
        warnings.warn(
            f"Found matching file {f:s} for reader "
            f"{reader_name:s}, but none of group keys found. "
            "Group keys requested: " + ", ".join(group_keys),
            UserWarning,
            stacklevel=5
        )
    file_keys[reader_name].append((group_key, f))


def _get_sorted_file_groups(all_file_keys, time_threshold):  # noqa: D417
    """Get sorted file groups.

    Get a list of dictionaries, where each list item consists of a dictionary
    mapping a tuple of keys to a mapping of reader names to files.  The files
    listed in each list item are considered to be grouped within the same time.

    Args:
        all_file_keys (Iterable): as returned by _get_file_keys_for_reader_files
        time_threshold (numbers.Number): temporal threshold in seconds

    Returns:
        List[Mapping[Tuple, Mapping[str, List[str]]]], as described

    Internal helper for group_files.
    """
    # flatten to get an overall sorting; put the name in the middle in the
    # interest of sorting
    flat_keys = ((v[0], rn, v[1]) for (rn, vL) in all_file_keys.items() for v in vL)
    prev_key = None
    threshold = dt.timedelta(seconds=time_threshold)
    # file_groups is sorted, because dictionaries are sorted by insertion
    # order in Python 3.7+
    file_groups = {}
    for gk, rn, f in sorted(flat_keys):
        # use first element of key as time identifier (if datetime type)
        if prev_key is None:
            is_new_group = True
            prev_key = gk
        else:
            is_new_group = _get_group_status(gk, prev_key, threshold)

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
            _update_existing_group(file_groups, rn, prev_key, f)
    return file_groups


def _get_group_status(gk, prev_key, threshold):
    if isinstance(gk[0], dt.datetime):
        # datetimes within threshold difference are "the same time"
        return (gk[0] - prev_key[0]) > threshold
    return gk[0] != prev_key[0]


def _update_existing_group(file_groups, rn, prev_key, f):
    if rn not in file_groups[prev_key]:
        file_groups[prev_key][rn] = [f]
    else:
        file_groups[prev_key][rn].append(f)


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
            _check_raise_missing(missing, i, readers_without_files)
        else:
            yield grp


def _check_raise_missing(missing, i, readers_without_files):
    if missing == "raise":
        raise FileNotFoundError(
            f"when grouping files, group at index {i:d} "
            "had no files for readers: " +
            ", ".join(readers_without_files))


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
        start_time (datetime.datetime): Limit used files by starting time.
        end_time (datetime.datetime): Limit used files by ending time.
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
    filter_parameters = filter_parameters or reader_kwargs.get("filter_parameters", {})
    sensor_supported = False

    _set_filter_times(filter_parameters, start_time, end_time)
    reader_kwargs["filter_parameters"] = filter_parameters

    for reader_configs in configs_for_reader(reader):
        (reader_instance, loadables, this_sensor_supported) = _get_loadables_for_reader_config(
                base_dir, reader, sensor, reader_configs, reader_kwargs, fs)
        sensor_supported = sensor_supported or this_sensor_supported
        _update_reader_files(reader_files, reader_instance, loadables)

    _check_sensor_status(sensor, sensor_supported)
    _check_reader_file_status(reader_files, missing_ok)

    return reader_files


def _set_filter_times(filter_parameters, start_time, end_time):
    if start_time or end_time:
        filter_parameters["start_time"] = start_time
        filter_parameters["end_time"] = end_time


def _update_reader_files(reader_files, reader_instance, loadables):
    if loadables:
        reader_files[reader_instance.name] = list(loadables)


def _check_sensor_status(sensor, sensor_supported):
    if sensor and not sensor_supported:
        raise ValueError("Sensor '{}' not supported by any readers".format(sensor))


def _check_reader_file_status(reader_files, missing_ok):
    if not (reader_files or missing_ok):
        raise ValueError("No supported files found")


def _get_loadables_for_reader_config(base_dir, reader, sensor, reader_configs,
                                     reader_kwargs, fs):
    """Get loadables for reader configs.

    Helper for find_files_and_readers.

    Args:
        base_dir (str): as for `find_files_and_readers`
        reader (str): as for `find_files_and_readers`
        sensor (str): as for `find_files_and_readers`
        reader_configs (dict): reader metadata such as returned by
            `configs_for_reader`.
        reader_kwargs (dict): Keyword arguments to be passed to reader.
        fs (fsspec.spec.AbstractFileSystem): as for `find_files_and_readers`
    """
    sensor_supported = False
    reader_instance = _get_reader_instance(reader, reader_configs, **reader_kwargs)
    if isinstance(reader_instance, tuple):
        return reader_instance

    if not reader_instance.supports_sensor(sensor):
        return (reader_instance, [], False)
    if sensor is not None:
        # sensor was specified and a reader supports it
        sensor_supported = True

    loadables = _get_loadables_from_reader(reader_instance, base_dir, fs)
    return (reader_instance, loadables, sensor_supported)


def _get_reader_instance(reader, reader_configs, **reader_kwargs):
    try:
        return load_reader(reader_configs, **reader_kwargs)
    except (KeyError, IOError, yaml.YAMLError) as err:
        LOG.info("Cannot use %s", str(reader_configs))
        LOG.debug(str(err))
        if _is_single_reader(reader):
            # if it is a single reader then give a more usable error
            raise
        return (None, [], False)


def _is_single_reader(reader):
    return reader and (isinstance(reader, str) or len(reader) == 1)


def _get_loadables_from_reader(reader_instance, base_dir, fs):
    loadables = reader_instance.select_files_from_directory(base_dir, fs)
    if loadables:
        loadables = list(
            reader_instance.filter_selected_filenames(loadables))
    return loadables
