#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2011 SMHI

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`mpop.plugin_base` module defines the plugin API.
"""

import os
import weakref
import numbers
import logging
from ConfigParser import ConfigParser
from trollsift import parser
from mpop import PACKAGE_CONFIG_PATH
from mpop.writers import EnhancementDecisionTree

LOG = logging.getLogger(__name__)


class Plugin(object):
    """The base plugin class. It is not to be used as is, it has to be
    inherited by other classes.
    """
    pass

class Reader(Plugin):
    """Reader plugins. They should have a *pformat* attribute, and implement
    the *load* method. This is an abstract class to be inherited.
    """
    def __init__(self, name, config_file=None,
                 file_patterns=None,
                 filenames=None,
                 description="",
                 start_time=None,
                 end_time=None,
                 area=None,
                 **kwargs):
        """The reader plugin takes as input a satellite scene to fill in.
        
        Arguments:
        - `scene`: the scene to fill.
        """
        Plugin.__init__(self)
        self.name = name
        self.config_file = config_file
        self.file_patterns = file_patterns
        self.filenames = filenames or []
        self.description = description
        self.start_time = start_time
        self.end_time = end_time
        self.area = area
        del kwargs

        self.channels = {}

        if self.config_file:
            self.load_config()

    def add_filenames(self, *filenames):
        self.filenames |= set(filenames)

    @property
    def channel_names(self):
        """Names of all channels configured for this reader.
        """
        return sorted(self.channels.keys())

    @property
    def sensor_names(self):
        """Sensors supported by this reader.
        """
        return set([sensor_name for chn_info in self.channels.values()
                    for sensor_name in chn_info["sensor"].split(",")])

    def load_config(self):
        conf = ConfigParser()
        conf.read(self.config_file)
        # Assumes only one section with "reader:" prefix
        info = {}
        for section_name in conf.sections():
            # Can't load the reader section because any options not specified in keywords don't use the default
            # if section_name.startswith("reader:"):
            #     info.update(dict(conf.items(section_name)))
            if section_name.startswith("channel:"):
                channel_info = self.parse_channel_section(section_name, dict(conf.items(section_name)))
                self.channels[channel_info["name"]] = channel_info
        return info

    def parse_channel_section(self, section_name, section_options):
        # Allow subclasses to make up their own rules about channels, but this is a good starting point
        if "file_patterns" in section_options:
            section_options["file_patterns"] = section_options["file_patterns"].split(",")
        if "wavelength_range" in section_options:
            section_options["wavelength_range"] = [float(wl) for wl in section_options["wavelength_range"].split(",")]
        return section_options

    def get_channel(self, key):
        """Get the channel corresponding to *key*, either by name or centerwavelength.
        """
        # get by wavelength
        if isinstance(key, numbers.Number):
            channels = [chn for chn in self.channels.values()
                        if("wavelength_range" in chn and
                           chn["wavelength_range"][0] <= key <=chn["wavelength_range"][2])]
            channels = sorted(channels,
                              lambda ch1, ch2:
                              cmp(abs(ch1["wavelength_range"][1] - key),
                                  abs(ch2["wavelength_range"][1] - key)))

            if not channels:
                raise KeyError("Can't find any projectable at %gum" % key)
            return channels[0]
        # get by name
        else:
            return self.channels[key]

    def load(self, channels_to_load):
        """Loads the *channels_to_load* into the scene object.
        """
        raise NotImplementedError


class Writer(Plugin):
    """Writer plugins. They must implement the *save* method. This is an
    abstract class to be inherited.
    """
    ptype = "writer"

    def __init__(self, config_file=None, **kwargs):
        self.config_file = config_file
        if config_file is not None:
            # If given a config file then load defaults from it
            self.config_options = self.load_config()
        else:
            self.config_options = {}

        self.options = self.config_options.copy()
        self.options.update(kwargs)
        self.ppp_config_dir = self.options.get("ppp_config_dir", PACKAGE_CONFIG_PATH)

        self.name = self.options.get("name", None)
        if self.name is None:
            raise ValueError("Writer 'name' not set by config file or user")

        self.file_pattern = self.options.get("file_pattern", None)
        # Set a way to create filenames if we were given a pattern
        self.filename_parser = parser.Parser(self.file_pattern) if self.file_pattern else None

        self.enhancement_config = self.options.get("enhancement_config", None)
        if self.enhancement_config is None and "enhancement_config" not in self.options:
            # it wasn't specified in the config or in the kwargs, we should provide a default
            self.enhancement_config = os.path.join(self.ppp_config_dir, "enhancements", "generic.cfg")

        if self.enhancement_config is not None:
            self.enhancement_tree = EnhancementDecisionTree(self.enhancement_config)
        else:
            # They don't want any automatic enhancements
            self.enhancement_tree = None

        self.sensor_enhancement_configs = []

    def get_sensor_enhancement_config(self, sensor):
        if isinstance(sensor, str):
            # one single sensor
            sensor = [sensor]

        for sensor_name in sensor:
            config_file = os.path.join(self.ppp_config_dir, "enhancements", sensor_name + ".cfg")
            if os.path.isfile(config_file):
                yield config_file

    def add_sensor_enhancements(self, sensor):
        # XXX: Should we just load all enhancements from the base directory?
        new_configs = []
        for config_file in self.get_sensor_enhancement_config(sensor):
            if config_file not in self.sensor_enhancement_configs.append(config_file):
                self.sensor_enhancement_configs.append(config_file)
                new_configs.append(config_file)

        if new_configs:
            self.enhancement_tree.add_config_to_tree(config_file)

    def load_config(self):
        conf = ConfigParser()
        conf.read(self.config_file)
        for section_name in conf.sections():
            if section_name.startswith("writer:"):
                options = dict(conf.items(section_name))
                return options
        LOG.warning("No 'writer:' section found in config file: %s", self.config_file)
        return {}

    def get_filename(self, **kwargs):
        if self.filename_parser is None:
            raise RuntimeError("No filename pattern or specific filename provided")
        return self.filename_parser.compose(kwargs)

    def _determine_mode(self, dataset):
        if "mode" in dataset.info:
            return dataset.info["mode"]

        ndim = dataset.data.ndim
        default_modes = {
            2: "L",
            3: "RGB",
            4: "RGBA",
        }
        if ndim in default_modes:
            return default_modes[ndim]
        else:
            raise RuntimeError("Can't determine 'mode' of dataset: %s" % (dataset.info.get("name", None),))

    def save_dataset(self, dataset, fill_value=None, **kwargs):
        """Saves the *dataset* to a given *filename*.
        """
        fill_value = fill_value if fill_value is not None else self.fill_value
        mode = self._determine_mode(dataset)

        if self.enhancement_tree is None:
            raise RuntimeError("No enhancement configuration files found or specified, can not automatically enhance dataset")

        # Load any additional enhancement configs that are specific to this datasets sensors
        if dataset.info.get("sensor", None):
            self.add_sensor_enhancements(dataset.info["sensor"])

        enh_kwargs = self.enhancement_tree.find_match(**dataset.info)

        # Create an image for enhancement
        img = dataset.to_image(mode=mode, fill_value=fill_value)
        LOG.debug("Enhancement configuration options: %s" % (str(enh_kwargs),))
        img.enhance(**enh_kwargs)

        # if dataset.data.ndim == 3:
        #     img.enhance(stretch="histogram")
        # else:
        #     img.enhance(stretch="linear")

        self.save_image(img, dataset.info, **kwargs)

    def save_image(self, img, metadata, *args, **kwargs):
        raise NotImplementedError("Writer '%s' has not implemented image saving" % (self.name,))

