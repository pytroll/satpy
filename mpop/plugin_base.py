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
import numbers
import logging
from ConfigParser import ConfigParser
from trollsift import parser
from mpop.writers import Enhancer
from mpop import PACKAGE_CONFIG_PATH

LOG = logging.getLogger(__name__)


class Plugin(object):
    """The base plugin class. It is not to be used as is, it has to be
    inherited by other classes.
    """
    def __init__(self, ppp_config_dir=None, default_config_filename=None, config_file=None, **kwargs):
        self.ppp_config_dir = ppp_config_dir or os.environ.get("PPP_CONFIG_DIR", PACKAGE_CONFIG_PATH)
        self.default_config_filename = default_config_filename
        self.config_file = config_file
        if self.config_file is None and self.default_config_filename is not None:
            # Specify a default
            self.config_file = os.path.join(self.ppp_config_dir, self.default_config_filename)

        if self.config_file:
            conf = ConfigParser()
            conf.read(self.config_file)
            self.load_config(conf)

    def load_config(self, conf):
        # Assumes only one section with "reader:" prefix
        for section_name in conf.sections():
            section_type = section_name.split(":")[0]
            load_func = "load_section_%s" % (section_type,)
            if hasattr(self, load_func):
                getattr(self, load_func)(section_name, dict(conf.items(section_name)))


class Reader(Plugin):
    """Reader plugins. They should have a *pformat* attribute, and implement
    the *load* method. This is an abstract class to be inherited.
    """
    def __init__(self, name,
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
        # Hold information about channels
        self.channels = {}

        # Load the config
        Plugin.__init__(self, **kwargs)

        # Use options from the config file if they weren't passed as arguments
        self.name = self.config_options.get("name", None) if name is None else name
        self.file_patterns = self.config_options.get("file_patterns", None) if file_patterns is None else file_patterns
        self.filenames = self.config_options.get("filenames", []) if filenames is None else filenames
        self.description = self.config_options.get("description", None) if description is None else description

        # These can't be provided by a configuration file
        self.start_time = start_time
        self.end_time = end_time
        self.area = area

        if self.name is None:
            raise ValueError("Reader 'name' not provided")

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

    def load_section_reader(self, section_name, section_options):
        self.config_options = section_options

    def load_section_channel(self, section_name, section_options):
        # Allow subclasses to make up their own rules about channels, but this is a good starting point
        if "file_patterns" in section_options:
            section_options["file_patterns"] = section_options["file_patterns"].split(",")
        if "wavelength_range" in section_options:
            section_options["wavelength_range"] = [float(wl) for wl in section_options["wavelength_range"].split(",")]

        self.channels[section_options["name"]] = section_options

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
    """Writer plugins. They must implement the *save_image* method. This is an
    abstract class to be inherited.
    """

    def __init__(self, name=None, fill_value=None, file_pattern=None, enhancement_config=None, **kwargs):
        # Load the config
        Plugin.__init__(self, **kwargs)

        # Use options from the config file if they weren't passed as arguments
        self.name = self.config_options.get("name", None) if name is None else name
        self.fill_value = self.config_options.get("fill_value", None) if fill_value is None else fill_value
        self.file_pattern = self.config_options.get("file_pattern", None) if file_pattern is None else file_pattern
        enhancement_config = self.config_options.get("enhancement_config", None) if enhancement_config is None else enhancement_config

        if self.name is None:
            raise ValueError("Writer 'name' not provided")
        if self.fill_value:
            self.fill_value = float(self.fill_value)

        self.create_filename_parser()
        self.enhancer = Enhancer(ppp_config_dir=self.ppp_config_dir, enhancement_config_file=enhancement_config)

    def create_filename_parser(self):
        # just in case a writer needs more complex file patterns
        # Set a way to create filenames if we were given a pattern
        self.filename_parser = parser.Parser(self.file_pattern) if self.file_pattern else None

    def load_section_writer(self, section_name, section_options):
        self.config_options = section_options

    def get_filename(self, **kwargs):
        if self.filename_parser is None:
            raise RuntimeError("No filename pattern or specific filename provided")
        return self.filename_parser.compose(kwargs)

    def save_dataset(self, dataset, fill_value=None, **kwargs):
        """Saves the *dataset* to a given *filename*.
        """
        fill_value = fill_value if fill_value is not None else self.fill_value
        img = dataset.get_enhanced_image(self.enhancer, fill_value)
        self.save_image(img, **kwargs)

    def save_image(self, img, *args, **kwargs):
        raise NotImplementedError("Writer '%s' has not implemented image saving" % (self.name,))

