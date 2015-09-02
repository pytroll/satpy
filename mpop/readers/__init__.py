#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Shared objects of the various reader classes.

"""

from mpop.plugin_base import Plugin
import logging
import numbers

LOG = logging.getLogger(__name__)


class Reader(Plugin):
    """Reader plugins. They should have a *pformat* attribute, and implement
    the *load* method. This is an abstract class to be inherited.
    """
    def __init__(self, name=None,
                 file_patterns=None,
                 filenames=None,
                 description="",
                 start_time=None,
                 end_time=None,
                 area=None,
                 sensor=None,
                 **kwargs):
        """The reader plugin takes as input a satellite scene to fill in.

        Arguments:
        - `scene`: the scene to fill.
        """
        # Hold information about datasets
        self.datasets = {}

        # Load the config
        super(Reader, self).__init__(**kwargs)

        # Use options from the config file if they weren't passed as arguments
        self.name = self.config_options.get("name", None) if name is None else name
        self.file_patterns = self.config_options.get("file_patterns", None) if file_patterns is None else file_patterns
        self.filenames = self.config_options.get("filenames", []) if filenames is None else filenames
        self.description = self.config_options.get("description", None) if description is None else description
        self.sensor = self.config_options.get("sensor", "").split(",") if sensor is None else set(sensor)

        # These can't be provided by a configuration file
        self.start_time = start_time
        self.end_time = end_time
        self.area = area

        if self.name is None:
            raise ValueError("Reader 'name' not provided")

    def add_filenames(self, *filenames):
        self.filenames |= set(filenames)

    @property
    def dataset_names(self):
        """Names of all datasets configured for this reader.
        """
        return sorted(self.datasets.keys())

    @property
    def sensor_names(self):
        """Sensors supported by this reader.
        """
        sensors = set()
        for ds_info in self.datasets.values():
            if "sensor" in ds_info:
                sensors |= set(ds_info["sensor"].split(","))
        return sensors | self.sensor

    def load_section_reader(self, section_name, section_options):
        self.config_options = section_options

    def load_section_dataset(self, section_name, section_options):
        name = section_options.get("name", section_name.split(":")[-1])
        section_options["name"] = name

        # Allow subclasses to make up their own rules about datasets, but this is a good starting point
        if "file_patterns" in section_options:
            section_options["file_patterns"] = section_options["file_patterns"].split(",")
        if "wavelength_range" in section_options:
            section_options["wavelength_range"] = [float(wl) for wl in section_options["wavelength_range"].split(",")]
        if "calibration_level" in section_options:
            section_options["calibration_level"] = int(section_options["calibration_level"])

        self.datasets[name] = section_options

    def get_dataset(self, key):
        """Get the dataset corresponding to *key*, either by name or centerwavelength.
        """
        # get by wavelength
        if isinstance(key, numbers.Number):
            datasets = [ds for ds in self.datasets.values()
                        if("wavelength_range" in ds and
                           ds["wavelength_range"][0] <= key <= ds["wavelength_range"][2])]
            datasets = sorted(datasets,
                              lambda ch1, ch2:
                              cmp(abs(ch1["wavelength_range"][1] - key),
                                  abs(ch2["wavelength_range"][1] - key)))

            if not datasets:
                raise KeyError("Can't find any projectable at %gum" % key)
            return datasets[0]
        # get by name
        else:
            return self.datasets[key]

    def load(self, datasets_to_load):
        """Loads the *datasets_to_load* into the scene object.
        """
        raise NotImplementedError

