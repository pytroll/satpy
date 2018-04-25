#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2017.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Shared objects of the various reader classes."""

import logging
import numbers
import os

import six
import yaml

from satpy.config import (config_search_paths, get_environ_config_dir,
                          glob_config)
from satpy.dataset import DATASET_KEYS, DatasetID
from satpy import CALIBRATION_ORDER

try:
    import configparser
except ImportError:
    from six.moves import configparser

LOG = logging.getLogger(__name__)


def _wl_dist(wl_a, wl_b):
    """Return the distance between two requested wavelengths."""
    if isinstance(wl_a, tuple):
        # central wavelength
        wl_a = wl_a[1]
    if isinstance(wl_b, tuple):
        wl_b = wl_b[1]
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
        choices (iterable): `DatasetID`s to sort through to determine
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
            low_res = min(best_cal, key=lambda x: CALIBRATION_ORDER[x])
            choices = [c for c in choices if c.resolution == low_res]
    if key.resolution is None and choices:
        low_res = [x.resolution for x in choices if x.resolution]
        if low_res:
            low_res = min(low_res)
            choices = [c for c in choices if c.resolution == low_res]
    return choices


class MalformedConfigError(IOError):
    pass


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

    def get_key(self, key):
        """Get the fully-specified key best matching the provided key.

        Only the best match is returned. See `get_best_dataset_key` for more
        information on how this is determined.

        Args:
            key (DatasetID): DatasetID of query parameters to use for
                             searching. Any parameter that is `None`
                             is considered a wild card and any match is
                             accepted.

        """
        if isinstance(key, numbers.Number):
            # we want this ID to act as a query so we set modifiers to None
            # meaning "we don't care how many modifiers it has".
            key = DatasetID(wavelength=key, modifiers=None)
        elif isinstance(key, str):
            # ID should act as a query (see wl comment above)
            key = DatasetID(name=key, modifiers=None)

        res = self.get_keys_by_datasetid(key)
        if not res:
            raise KeyError("No dataset matching '{}' found".format(str(key)))
        elif len(res) == 1:
            return res[0]

        # more than one dataset matched
        res = get_best_dataset_key(key, res)
        if len(res) != 1:
            raise KeyError("No unique dataset matching " + str(key))
        return res[0]

    def get_keys(self,
                 name_or_wl,
                 resolution=None,
                 polarization=None,
                 calibration=None,
                 modifiers=None):
        """Get multiple fully-specified keys that match the provided query.

        Returned keys are not ordered by "best" choice like `get_key` is
        (best resolution, nearest wavelength, highest calibration, etc).

        Args:
            name_or_wl (str or float): Name of datasets or relative wavelength
                                       for the datasets.
            resolution (float, int, or list): Resolution of the dataset in
                                              dataset units (typically
                                              meters). This can also be a
                                              list of these numbers.
            calibration (str or list): Dataset calibration
                                       (ex.'reflectance'). This can also be a
                                       list of these strings.
            modifiers (list): Modifiers applied to the dataset. Unlike
                              resolution and calibration this is the exact
                              desired list of modifiers for one dataset, not
                              a list of possible modifiers.

        """
        # Get things that match at least the name_or_wl
        if isinstance(name_or_wl, numbers.Number):
            keys = [k for k in self.keys()
                    if DatasetID.wavelength_match(k.wavelength, name_or_wl)]
        elif isinstance(name_or_wl, (str, six.text_type)):
            keys = [k for k in self.keys()
                    if DatasetID.name_match(k.name, name_or_wl)]
        else:
            raise TypeError("First argument must be a wavelength or name")

        if resolution is not None:
            if not isinstance(resolution, (list, tuple)):
                resolution = (resolution, )
            keys = [k for k in keys
                    if k.resolution is not None and k.resolution in resolution]
        if polarization is not None:
            if not isinstance(polarization, (list, tuple)):
                polarization = (polarization, )
            keys = [k for k in keys
                    if k.polarization is not None and k.polarization in
                    polarization]
        if calibration is not None:
            if not isinstance(calibration, (list, tuple)):
                calibration = (calibration, )
            keys = [
                k for k in keys
                if k.calibration is not None and k.calibration in calibration
            ]
        if modifiers is not None:
            keys = [
                k for k in keys
                if k.modifiers is not None and k.modifiers == modifiers
            ]

        return keys

    def get_keys_by_datasetid(self, did):
        keys = self.keys()
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

    def __getitem__(self, item):
        try:
            # short circuit - try to get the object without more work
            return super(DatasetDict, self).__getitem__(item)
        except KeyError:
            key = self.get_key(item)
            return super(DatasetDict, self).__getitem__(key)

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
                                modifiers=d.get("modifiers", tuple()))
                if key.name is None and key.wavelength is None:
                    raise ValueError(
                        "One of 'name' or 'wavelength' attrs values should be set.")

        # update the 'value' with the information contained in the key
        if isinstance(d, dict):
            d["name"] = key.name
            # XXX: What should users be allowed to modify?
            d["resolution"] = key.resolution
            d["calibration"] = key.calibration
            d["polarization"] = key.polarization
            d["modifiers"] = key.modifiers
            # you can't change the wavelength of a dataset, that doesn't make
            # sense
            if "wavelength" in d and d["wavelength"] != key.wavelength:
                raise TypeError("Can't change the wavelength of a dataset")

        return super(DatasetDict, self).__setitem__(key, value)

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


def read_reader_config(config_files, loader=yaml.Loader):
    """Read the reader *config_files* and return the info extracted.
    """

    conf = {}
    LOG.debug('Reading ' + str(config_files))
    for config_file in config_files:
        with open(config_file) as fd:
            conf.update(yaml.load(fd.read(), loader))

    try:
        reader_info = conf['reader']
    except KeyError:
        raise MalformedConfigError(
            "Malformed config file {}: missing reader 'reader'".format(
                config_files))
    reader_info['config_files'] = config_files
    return reader_info


def load_reader(reader_configs, **reader_kwargs):
    """Import and setup the reader from *reader_info*
    """
    reader_info = read_reader_config(reader_configs)
    reader_instance = reader_info['reader'](
        config_files=reader_configs,
        **reader_kwargs
    )
    return reader_instance


def configs_for_reader(reader=None, ppp_config_dir=None):
    """Generator of reader configuration files for one or more readers

    Args:
        reader (Optional[str]): Yield configs only for this reader
        ppp_config_dir (Optional[str]): Additional configuration directory
            to search for reader configuration files.

    Returns: Generator of lists of configuration files

    """
    if reader is not None:
        if not isinstance(reader, (list, tuple)):
            reader = [reader]
        # given a config filename or reader name
        config_files = [r if r.endswith('.yaml') else r + '.yaml' for r in reader]
    else:
        reader_configs = glob_config(os.path.join('readers', '*.yaml'), ppp_config_dir)
        config_files = set(reader_configs)

    for config_file in config_files:
        config_basename = os.path.basename(config_file)
        reader_configs = config_search_paths(
            os.path.join("readers", config_basename), ppp_config_dir)

        if not reader_configs:
            LOG.warning("No reader configs found for '%s'", reader)
            continue

        yield reader_configs


def find_files_and_readers(start_time=None, end_time=None, base_dir=None,
                           reader=None, sensor=None, ppp_config_dir=get_environ_config_dir(),
                           filter_parameters=None, reader_kwargs=None):
    """Find on-disk files matching the provided parameters.

    Use `start_time` and/or `end_time` to limit found filenames by the times
    in the filenames (not the internal file metadata). Files are matched if
    they fall anywhere within the range specified by these parameters.

    Searching is **NOT** recursive.

    The returned dictionary can be passed directly to the `Scene` object
    through the `filenames` keyword argument.

    Args:
        start_time (datetime): Limit used files by starting time.
        end_time (datetime): Limit used files by ending time.
        base_dir (str): The directory to search for files containing the
                        data to load. Defaults to the current directory.
        reader (str or list): The name of the reader to use for loading the data or a list of names.
        sensor (str or list): Limit used files by provided sensors.
        ppp_config_dir (str): The directory containing the configuration
                              files for SatPy.
        filter_parameters (dict): Filename pattern metadata to filter on. `start_time` and `end_time` are
                                  automatically added to this dictionary. Shortcut for
                                  `reader_kwargs['filter_parameters']`.
        reader_kwargs (dict): Keyword arguments to pass to specific reader
                              instances to further configure file searching.

    Returns: Dictionary mapping reader name string to list of filenames

    """
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
        except (KeyError, MalformedConfigError, yaml.YAMLError) as err:
            LOG.info('Cannot use %s', str(reader_configs))
            LOG.debug(str(err))
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
                 ppp_config_dir=get_environ_config_dir()):
    """Create specified readers and assign files to them.

    Args:
        filenames (iterable or dict): A sequence of files that will be used to load data from. A ``dict`` object
                                      should map reader names to a list of filenames for that reader.
        reader (str or list): The name of the reader to use for loading the data or a list of names.
        filter_parameters (dict): Specify loaded file filtering parameters.
                                  Shortcut for `reader_kwargs['filter_parameters']`.
        reader_kwargs (dict): Keyword arguments to pass to specific reader instances.
        ppp_config_dir (str): The directory containing the configuration files for satpy.

    Returns: Dictionary mapping reader name to reader instance

    """
    reader_instances = {}
    reader_kwargs = reader_kwargs or {}

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
    else:
        remaining_filenames = set(filenames or [])

    for idx, reader_configs in enumerate(configs_for_reader(reader, ppp_config_dir)):
        if isinstance(filenames, dict):
            readers_files = set(filenames[reader[idx]])
        else:
            readers_files = remaining_filenames

        try:
            reader_instance = load_reader(reader_configs, **reader_kwargs)
        except (KeyError, MalformedConfigError, yaml.YAMLError) as err:
            LOG.info('Cannot use %s', str(reader_configs))
            LOG.debug(str(err))
            continue

        if readers_files:
            loadables = reader_instance.select_files_from_pathnames(readers_files)
        if loadables:
            reader_instance.create_filehandlers(loadables)
            reader_instances[reader_instance.name] = reader_instance
            remaining_filenames -= set(loadables)
        if not remaining_filenames:
            break

    if remaining_filenames:
        LOG.warning(
            "Don't know how to open the following files: {}".format(str(
                remaining_filenames)))
    if not reader_instances:
        raise ValueError("No supported files found")
    return reader_instances
