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
import numbers
import os
from datetime import datetime, timedelta

import six
import yaml

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

from satpy.config import (config_search_paths, get_environ_config_dir,
                          glob_config)
from satpy.dataset import DATASET_KEYS, DatasetID
from satpy import CALIBRATION_ORDER

try:
    import configparser  # noqa
except ImportError:
    from six.moves import configparser  # noqa

LOG = logging.getLogger(__name__)


# Old Name -> New Name
OLD_READER_NAMES = {
}


class TooManyResults(KeyError):
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
    """Choose the "best" `DatasetID` from `choices` based on `key`.

    The best key is chosen based on the follow criteria:

        1. Central wavelength is nearest to the `key` wavelength if
           specified.
        2. Least modified dataset if `modifiers` is `None` in `key`.
           Otherwise, the modifiers are ignored.
        3. Highest calibration if `calibration` is `None` in `key`.
           Calibration priority is chosen by `satpy.CALIBRATION_ORDER`.
        4. Best resolution (smallest number) if `resolution` is `None`
           in `key`. Otherwise, the resolution is ignored.

    This function assumes `choices` has already been filtered to only
    include datasets that match the provided `key`.

    Args:
        key (DatasetID): Query parameters to sort `choices` by.
        choices (iterable): `DatasetID` objects to sort through to determine
                            the best dataset.

    Returns: List of best `DatasetID`s from `choices`. If there is more
             than one element this function could not choose between the
             available datasets.

    """
    # Choose the wavelength closest to the choice
    if key.wavelength is not None and choices:
        # find the dataset with a central wavelength nearest to the
        # requested wavelength
        nearest_wl = min([_wl_dist(key.wavelength, x.wavelength)
                          for x in choices if x.wavelength is not None])
        choices = [c for c in choices
                   if _wl_dist(key.wavelength, c.wavelength) == nearest_wl]
    if key.modifiers is None and choices:
        num_modifiers = min(len(x.modifiers or tuple()) for x in choices)
        choices = [c for c in choices if len(
            c.modifiers or tuple()) == num_modifiers]
    if key.calibration is None and choices:
        best_cal = [x.calibration for x in choices if x.calibration]
        if best_cal:
            best_cal = min(best_cal, key=lambda x: CALIBRATION_ORDER[x])
            choices = [c for c in choices if c.calibration == best_cal]
    if key.resolution is None and choices:
        low_res = [x.resolution for x in choices if x.resolution]
        if low_res:
            low_res = min(low_res)
            choices = [c for c in choices if c.resolution == low_res]
    if key.level is None and choices:
        low_level = [x.level for x in choices if x.level]
        if low_level:
            low_level = max(low_level)
            choices = [c for c in choices if c.level == low_level]

    return choices


def filter_keys_by_dataset_id(did, key_container):
    """Filer provided key iterable by the provided `DatasetID`.

    Note: The `modifiers` attribute of `did` should be `None` to allow for
          **any** modifier in the results.

    Args:
        did (DatasetID): Query parameters to match in the `key_container`.
        key_container (iterable): Set, list, tuple, or dict of `DatasetID`
                                  keys.

    Returns (list): List of keys matching the provided parameters in no
                    specific order.

    """
    keys = iter(key_container)

    for key in DATASET_KEYS:
        if getattr(did, key) is not None:
            if key == "wavelength":
                keys = [k for k in keys
                        if (getattr(k, key) is not None and
                            DatasetID.wavelength_match(getattr(k, key),
                                                       getattr(did, key)))]
            else:
                keys = [k for k in keys
                        if getattr(k, key) is not None and getattr(k, key)
                        == getattr(did, key)]

    return keys


def get_key(key, key_container, num_results=1, best=True,
            resolution=None, calibration=None, polarization=None,
            level=None, modifiers=None):
    """Get the fully-specified key best matching the provided key.

    Only the best match is returned if `best` is `True` (default). See
    `get_best_dataset_key` for more information on how this is determined.

    The `resolution` and other identifier keywords are provided as a
    convenience to filter by multiple parameters at once without having
    to filter by multiple `key` inputs.

    Args:
        key (DatasetID): DatasetID of query parameters to use for
                         searching. Any parameter that is `None`
                         is considered a wild card and any match is
                         accepted.
        key_container (dict or set): Container of DatasetID objects that
                                     uses hashing to quickly access items.
        num_results (int): Number of results to return. Use `0` for all
                           matching results. If `1` then the single matching
                           key is returned instead of a list of length 1.
                           (default: 1)
        best (bool): Sort results to get "best" result first
                     (default: True). See `get_best_dataset_key` for details.
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


    Returns (list or DatasetID): Matching key(s)

    Raises: KeyError if no matching results or if more than one result is
            found when `num_results` is `1`.

    """
    if isinstance(key, numbers.Number):
        # we want this ID to act as a query so we set modifiers to None
        # meaning "we don't care how many modifiers it has".
        key = DatasetID(wavelength=key, modifiers=None)
    elif isinstance(key, (str, six.text_type)):
        # ID should act as a query (see wl comment above)
        key = DatasetID(name=key, modifiers=None)
    elif not isinstance(key, DatasetID):
        raise ValueError("Expected 'DatasetID', str, or number dict key, "
                         "not {}".format(str(type(key))))

    res = filter_keys_by_dataset_id(key, key_container)

    # further filter by other parameters
    if resolution is not None:
        if not isinstance(resolution, (list, tuple)):
            resolution = (resolution, )
        res = [k for k in res
               if k.resolution is not None and k.resolution in resolution]
    if polarization is not None:
        if not isinstance(polarization, (list, tuple)):
            polarization = (polarization, )
        res = [k for k in res
               if k.polarization is not None and k.polarization in
               polarization]
    if calibration is not None:
        if not isinstance(calibration, (list, tuple)):
            calibration = (calibration, )
        res = [k for k in res
               if k.calibration is not None and k.calibration in calibration]
    if level is not None:
        if not isinstance(level, (list, tuple)):
            level = (level, )
        res = [k for k in res
               if k.level is not None and k.level in level]
    if modifiers is not None:
        res = [k for k in res
               if k.modifiers is not None and k.modifiers == modifiers]

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

    """Special dictionary object that can handle dict operations based on
    dataset name, wavelength, or DatasetID.

    Note: Internal dictionary keys are `DatasetID` objects.
    """

    def __init__(self, *args, **kwargs):
        super(DatasetDict, self).__init__(*args, **kwargs)

    def keys(self, names=False, wavelengths=False):
        # sort keys so things are a little more deterministic (.keys() is not)
        keys = sorted(super(DatasetDict, self).keys())
        if names:
            return (k.name for k in keys)
        elif wavelengths:
            return (k.wavelength for k in keys)
        else:
            return keys

    def get_key(self, match_key, num_results=1, best=True, **dfilter):
        """Get multiple fully-specified keys that match the provided query.

        Args:
            key (DatasetID): DatasetID of query parameters to use for
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
        """Get Node when we know the *exact* DatasetID."""
        return super(DatasetDict, self).__getitem__(item)

    def __getitem__(self, item):
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
        """Support assigning 'Dataset' objects or dictionaries of metadata.
        """
        d = value
        if hasattr(value, 'attrs'):
            # xarray.DataArray objects
            d = value.attrs
        # use value information to make a more complete DatasetID
        if not isinstance(key, DatasetID):
            if not isinstance(d, dict):
                raise ValueError("Key must be a DatasetID when value is not an xarray DataArray or dict")
            old_key = key
            try:
                key = self.get_key(key)
            except KeyError:
                if isinstance(old_key, (str, six.text_type)):
                    new_name = old_key
                else:
                    new_name = d.get("name")
                # this is a new key and it's not a full DatasetID tuple
                key = DatasetID(name=new_name,
                                resolution=d.get("resolution"),
                                wavelength=d.get("wavelength"),
                                polarization=d.get("polarization"),
                                calibration=d.get("calibration"),
                                level=d.get("level"),
                                modifiers=d.get("modifiers", tuple()))
                if key.name is None and key.wavelength is None:
                    raise ValueError("One of 'name' or 'wavelength' attrs "
                                     "values should be set.")

        # update the 'value' with the information contained in the key
        if isinstance(d, dict):
            d["name"] = key.name
            # XXX: What should users be allowed to modify?
            d["resolution"] = key.resolution
            d["calibration"] = key.calibration
            d["polarization"] = key.polarization
            d["level"] = key.level
            d["modifiers"] = key.modifiers
            # you can't change the wavelength of a dataset, that doesn't make
            # sense
            if "wavelength" in d and d["wavelength"] != key.wavelength:
                raise TypeError("Can't change the wavelength of a dataset")

        return super(DatasetDict, self).__setitem__(key, value)

    def contains(self, item):
        """Check contains when we know the *exact* DatasetID."""
        return super(DatasetDict, self).__contains__(item)

    def __contains__(self, item):
        try:
            key = self.get_key(item)
        except KeyError:
            return False
        return super(DatasetDict, self).__contains__(key)

    def __delitem__(self, key):
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
            This
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
    for filetype, filetype_info in reader_instance.sorted_filetype_items():
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
    """Generator of reader configuration files for one or more readers

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
        reader_configs = config_search_paths(
            os.path.join("readers", config_basename), *search_paths)

        if not reader_configs:
            # either the reader they asked for does not exist
            # or satpy is improperly configured and can't find its own readers
            raise ValueError("No reader(s) named: {}".format(reader))

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
                           filter_parameters=None, reader_kwargs=None):
    """Find on-disk files matching the provided parameters.

    Use `start_time` and/or `end_time` to limit found filenames by the times
    in the filenames (not the internal file metadata). Files are matched if
    they fall anywhere within the range specified by these parameters.

    Searching is **NOT** recursive.

    The returned dictionary can be passed directly to the `Scene` object
    through the `filenames` keyword argument.

    The behaviour of time-based filtering depends on whether or not the filename
    contains information about the end time of the data or not:

      - if the end time is not present in the filename, the start time of the filename
        is used and has to fall between (inclusive) the requested start and end times
      - otherwise, the timespan of the filename has to overlap the requested timespan

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
        loadables = reader_instance.select_files_from_directory(base_dir)
        if loadables:
            loadables = list(
                reader_instance.filter_selected_filenames(loadables))
        if loadables:
            reader_files[reader_instance.name] = list(loadables)

    if sensor and not sensor_supported:
        raise ValueError("Sensor '{}' not supported by any readers".format(sensor))

    if not reader_files:
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

        if readers_files:
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
