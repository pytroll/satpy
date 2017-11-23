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

try:
    import configparser
except ImportError:
    from six.moves import configparser

LOG = logging.getLogger(__name__)


class MalformedConfigError(Exception):
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
        if isinstance(key, DatasetID):
            res = self.get_keys_by_datasetid(key)
            if not res:
                return None
            elif len(res) == 1:
                return res[0]

            # more than one dataset matched
            res = self.get_best_choice(key, res)
            if len(res) != 1:
                raise KeyError("No unique dataset matching " + str(key))
            return res[0]
        # get by wavelength
        elif isinstance(key, numbers.Number):
            for k in self.keys():
                if k.wavelength is not None and \
                        DatasetID.wavelength_match(k.wavelength, key):
                    return k
        # get by name
        else:
            for k in self.keys():
                if DatasetID.name_match(k.name, key):
                    return k

    def get_keys(self,
                 name_or_wl,
                 resolution=None,
                 polarization=None,
                 calibration=None,
                 modifiers=None):
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

    def get_best_choice(self, key, choices):
        if key.modifiers is None and choices:
            num_modifiers = min(len(x.modifiers or tuple()) for x in choices)
            choices = [c for c in choices if len(
                c.modifiers or tuple()) == num_modifiers]
        if key.resolution is None and choices:
            low_res = [x.resolution for x in choices if x.resolution]
            if low_res:
                low_res = min(low_res)
                choices = [c for c in choices if c.resolution == low_res]
        return choices

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

    def get_item(self,
                 name_or_wl,
                 resolution=None,
                 polarization=None,
                 calibration=None,
                 modifiers=None):
        keys = self.get_keys(name_or_wl,
                             resolution=resolution,
                             polarization=polarization,
                             calibration=calibration,
                             modifiers=modifiers)
        if not keys:
            raise KeyError("No keys found matching provided filters")

        return self[keys[0]]

    def __getitem__(self, item):
        key = self.get_key(item)
        if key is None:
            raise KeyError("No dataset matching '{}' found".format(str(item)))
        return super(DatasetDict, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Support assigning 'Dataset' objects or dictionaries of metadata.
        """
        d = value.info if hasattr(value, 'info') else value
        if not isinstance(key, DatasetID):
            old_key = key
            key = self.get_key(key)
            if key is None:
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
                        "One of 'name' or 'wavelength' info values should be set.")

        # update the 'value' with the information contained in the key
        if hasattr(d, '__setitem__'):
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
        key = self.get_key(item)
        return super(DatasetDict, self).__contains__(key)

    def __delitem__(self, key):
        key = self.get_key(key)
        return super(DatasetDict, self).__delitem__(key)


def read_reader_config(config_files):
    """Read the reader *config_files* and return the info extracted.
    """

    conf = {}
    LOG.debug('Reading ' + str(config_files))
    for config_file in config_files:
        with open(config_file) as fd:
            conf.update(yaml.load(fd.read()))

    try:
        reader_info = conf['reader']
    except KeyError:
        raise MalformedConfigError(
            "Malformed config file {}: missing reader 'reader'".format(
                config_files))
    reader_info['config_files'] = config_files
    return reader_info


def load_reader(reader_configs, metadata=None, **reader_kwargs):
    """Import and setup the reader from *reader_info*
    """
    reader_info = read_reader_config(reader_configs)
    reader_instance = reader_info['reader'](
        config_files=reader_configs,
        metadata=metadata,
        **reader_kwargs
    )
    return reader_instance


class ReaderFinder(object):
    """Find readers given a scene, filenames, sensors, and/or a reader_name
    """

    def __init__(self,
                 ppp_config_dir=get_environ_config_dir(),
                 base_dir=None,
                 start_time=None,
                 end_time=None,
                 area=None):
        """Find readers.

        If both *filenames* and *base_dir* are provided, only *filenames* is
        used.
        """
        self.ppp_config_dir = ppp_config_dir
        self.base_dir = base_dir
        self.start_time = start_time
        self.end_time = end_time
        self.area = area

    def __call__(self, filenames=None, sensor=None, reader=None,
                 reader_kwargs=None, metadata=None):
        reader_instances = {}
        reader_kwargs = reader_kwargs or {}

        if not filenames and reader is None and not self.base_dir:
            # we weren't given anything to search through
            LOG.info("Not enough information provided to find readers.")
            return reader_instances

        if reader is not None:
            # given a config filename or reader name
            if not reader.endswith(".yaml"):
                reader += ".yaml"
            config_files = [reader]
        else:
            config_files = set(self.config_files())
        # FUTURE: Allow for a reader instance to be passed

        sensor_supported = False
        remaining_filenames = set(
            filenames) if filenames is not None else set()
        for config_file in config_files:
            config_basename = os.path.basename(config_file)
            reader_configs = config_search_paths(
                os.path.join("readers", config_basename), self.ppp_config_dir)

            if not reader_configs:
                LOG.warning("No reader configs found for '%s'", reader)
                continue

            try:
                reader_instance = load_reader(reader_configs,
                                              start_time=self.start_time,
                                              end_time=self.end_time,
                                              area=self.area,
                                              metadata=metadata,
                                              **reader_kwargs)
            except (KeyError, MalformedConfigError, yaml.YAMLError) as err:
                LOG.info('Cannot use %s', str(reader_configs))
                LOG.debug(str(err))
                continue

            if not reader_instance.supports_sensor(sensor):
                continue
            elif sensor is not None:
                # sensor was specified and a reader supports it
                sensor_supported = True
            if remaining_filenames:
                loadables = reader_instance.select_files_from_pathnames(
                    remaining_filenames)
            else:
                loadables = reader_instance.select_files_from_directory(
                    self.base_dir)
            if loadables:
                reader_instance.create_filehandlers(loadables)
                reader_instances[reader_instance.name] = reader_instance
                remaining_filenames -= set(loadables)
            if filenames is not None and not remaining_filenames:
                # we were given filenames to look through and found a reader
                # for all of them
                break

        if sensor and not sensor_supported:
            LOG.warning(
                "Sensor '{}' not supported by any readers".format(sensor))

        if remaining_filenames:
            LOG.warning(
                "Don't know how to open the following files: {}".format(str(
                    remaining_filenames)))
        if not reader_instances:
            raise ValueError("No supported files found")
        return reader_instances

    def config_files(self):
        return glob_config(
            os.path.join("readers", "*.yaml"), self.ppp_config_dir)
