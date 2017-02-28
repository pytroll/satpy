#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

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
"""Shared objects of the various writer classes.

For now, this includes enhancement configuration utilities.
"""

import glob
import json
import logging
import os

import numpy as np

from satpy.config import config_search_paths, get_environ_config_dir
from satpy.plugin_base import Plugin
from trollimage.image import Image
from trollsift import parser

try:
    import configparser
except ImportError:
    from six.moves import configparser

LOG = logging.getLogger(__name__)


def _determine_mode(dataset):
    if "mode" in dataset.info:
        return dataset.info["mode"]

    if dataset.ndim == 2:
        return "L"
    elif dataset.shape[0] == 2:
        return "LA"
    elif dataset.shape[0] == 3:
        return "RGB"
    elif dataset.shape[0] == 4:
        return "RGBA"
    else:
        raise RuntimeError("Can't determine 'mode' of dataset: %s" %
                           (dataset.id,))


def add_overlay(orig, area, coast_dir, color=(0, 0, 0), width=0.5, resolution=None):
    """Add coastline and political borders to image, using *color* (tuple
    of integers between 0 and 255).
    Warning: Loses the masks !
    *resolution* is chosen automatically if None (default), otherwise it should be one of:
    +-----+-------------------------+---------+
    | 'f' | Full resolution         | 0.04 km |
    | 'h' | High resolution         | 0.2 km  |
    | 'i' | Intermediate resolution | 1.0 km  |
    | 'l' | Low resolution          | 5.0 km  |
    | 'c' | Crude resolution        | 25  km  |
    +-----+-------------------------+---------+
    """
    img = orig.pil_image()

    if area is None:
        raise ValueError("Area of image is None, can't add overlay.")

    from satpy.resample import get_area_def
    if isinstance(area, str):
        area = get_area_def(area)
    LOG.info("Add coastlines and political borders to image.")

    if resolution is None:

        x_resolution = ((area.area_extent[2] -
                         area.area_extent[0]) /
                        area.x_size)
        y_resolution = ((area.area_extent[3] -
                         area.area_extent[1]) /
                        area.y_size)
        res = min(x_resolution, y_resolution)

        if res > 25000:
            resolution = "c"
        elif res > 5000:
            resolution = "l"
        elif res > 1000:
            resolution = "i"
        elif res > 200:
            resolution = "h"
        else:
            resolution = "f"

        LOG.debug("Automagically choose resolution " + resolution)

    from pycoast import ContourWriterAGG
    cw_ = ContourWriterAGG(coast_dir)
    cw_.add_coastlines(img, area, outline=color,
                       resolution=resolution, width=width)
    cw_.add_borders(img, area, outline=color,
                    resolution=resolution, width=width)

    arr = np.array(img)

    if len(orig.channels) == 1:
        orgi.channels[0] = np.ma.array(arr[:, :] / 255.0)
    else:
        for idx in range(len(orig.channels)):
            orig.channels[idx] = np.ma.array(arr[:, :, idx] / 255.0)


def get_enhanced_image(dataset,
                       enhancer=None,
                       fill_value=None,
                       ppp_config_dir=None,
                       enhancement_config_file=None,
                       overlay=None):
    mode = _determine_mode(dataset)

    if ppp_config_dir is None:
        ppp_config_dir = get_environ_config_dir()

    if enhancer is None:
        enhancer = Enhancer(ppp_config_dir, enhancement_config_file)

    if enhancer.enhancement_tree is None:
        raise RuntimeError(
            "No enhancement configuration files found or specified, cannot"
            " automatically enhance dataset")

    if dataset.info.get("sensor", None):
        enhancer.add_sensor_enhancements(dataset.info["sensor"])

    # Create an image for enhancement
    img = to_image(dataset, mode=mode, fill_value=fill_value)
    enhancer.apply(img, **dataset.info)

    img.info.update(dataset.info)

    if overlay is not None:
        add_overlay(img, dataset.info['area'], **overlay)
    return img


def show(dataset, **kwargs):
    """Display the dataset as an image.
    """
    if not dataset.is_loaded():
        raise ValueError("Dataset not loaded, cannot display.")

    img = get_enhanced_image(dataset, **kwargs)
    img.show()


def to_image(dataset, copy=True, **kwargs):
    # Only add keywords if they are present
    for key in ["mode", "fill_value", "palette"]:
        if key in dataset.info:
            kwargs.setdefault(key, dataset.info[key])

    if dataset.ndim == 2:
        return Image([dataset], copy=copy, **kwargs)
    elif dataset.ndim == 3:
        return Image([band for band in dataset], copy=copy, **kwargs)
    else:
        raise ValueError(
            "Don't know how to convert array with ndim %d to image" %
            dataset.ndim)


class Writer(Plugin):
    """Writer plugins. They must implement the *save_image* method. This is an
    abstract class to be inherited.
    """

    def __init__(self,
                 name=None,
                 fill_value=None,
                 file_pattern=None,
                 base_dir=None,
                 **kwargs):
        # Load the config
        Plugin.__init__(self, **kwargs)

        # Use options from the config file if they weren't passed as arguments
        self.name = self.config_options.get("name",
                                            None) if name is None else name
        self.fill_value = self.config_options.get(
            "fill_value", None) if fill_value is None else fill_value
        self.file_pattern = self.config_options.get(
            "file_pattern", None) if file_pattern is None else file_pattern

        if self.name is None:
            raise ValueError("Writer 'name' not provided")
        if self.fill_value:
            self.fill_value = float(self.fill_value)

        self.create_filename_parser(base_dir)

    def create_filename_parser(self, base_dir):
        # just in case a writer needs more complex file patterns
        # Set a way to create filenames if we were given a pattern
        if base_dir and self.file_pattern:
            file_pattern = os.path.join(base_dir, self.file_pattern)
        else:
            file_pattern = self.file_pattern
        self.filename_parser = parser.Parser(
            file_pattern) if file_pattern else None

    def load_section_writer(self, section_name, section_options):
        self.config_options = section_options

    def get_filename(self, **kwargs):
        if self.filename_parser is None:
            raise RuntimeError(
                "No filename pattern or specific filename provided")
        return self.filename_parser.compose(kwargs)

    def save_datasets(self, datasets, **kwargs):
        """Save all datasets to one or more files.

        Subclasses can use this method to save all datasets to one single
        file or optimize the writing of individual datasets. By default
        this simply calls `save_dataset` for each dataset provided.
        """
        for ds in datasets:
            self.save_dataset(ds, **kwargs)

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Saves the *dataset* to a given *filename*.
        """
        raise NotImplementedError(
            "Writer '%s' has not implemented dataset saving" % (self.name, ))


class ImageWriter(Writer):

    def __init__(self,
                 name=None,
                 fill_value=None,
                 file_pattern=None,
                 enhancement_config=None,
                 base_dir=None,
                 **kwargs):
        Writer.__init__(self, name, fill_value, file_pattern, base_dir,
                        **kwargs)
        enhancement_config = self.config_options.get(
            "enhancement_config",
            None) if enhancement_config is None else enhancement_config

        self.enhancer = Enhancer(ppp_config_dir=self.ppp_config_dir,
                                 enhancement_config_file=enhancement_config)

    def save_dataset(self, dataset, filename=None, fill_value=None, overlay=None, **kwargs):
        """Saves the *dataset* to a given *filename*.
        """
        fill_value = fill_value if fill_value is not None else self.fill_value
        img = get_enhanced_image(
            dataset, self.enhancer, fill_value, overlay=overlay)
        self.save_image(img, filename=filename, **kwargs)

    def save_image(self, img, filename=None, **kwargs):
        raise NotImplementedError(
            "Writer '%s' has not implemented image saving" % (self.name, ))


class EnhancementDecisionTree(object):
    any_key = None

    def __init__(self, *config_files, **kwargs):
        self.attrs = kwargs.pop("attrs", ("name",
                                          "platform",
                                          "sensor",
                                          "standard_name",
                                          "units", ))
        self.prefix = kwargs.pop("prefix", "enhancement:")
        self.tree = {}
        self.add_config_to_tree(*config_files)

    def add_config_to_tree(self, *config_files):
        conf = configparser.ConfigParser(allow_no_value=True)
        for fn in config_files:
            if isinstance(fn, str):
                conf.read(fn)
            else:
                conf.readfp(fn)

        self._build_tree(conf)

    def _build_tree(self, conf):
        for section_name in conf.sections():
            if not section_name.startswith(self.prefix):
                continue
            attrs = dict(conf.items(section_name))
            # Set a path in the tree for each section in the configuration
            # files
            curr_level = self.tree
            for attr in self.attrs:
                # or None is necessary if they have empty strings
                this_attr = attrs.get(attr, self.any_key) or None
                if attr == self.attrs[-1]:
                    # if we are at the last attribute, then assign the value
                    # set the dictionary of attributes because the config is
                    # not persistent
                    curr_level[this_attr] = attrs
                elif this_attr not in curr_level:
                    curr_level[this_attr] = {}
                curr_level = curr_level[this_attr]

    def _find_match(self, curr_level, attrs, kwargs):
        if len(attrs) == 0:
            # we're at the bottom level, we must have found something
            return curr_level

        match = None
        try:
            if attrs[0] in kwargs and kwargs[attrs[0]] in curr_level:
                # we know what we're searching for, try to find a pattern
                # that uses this attribute
                match = self._find_match(curr_level[kwargs[attrs[0]]],
                                         attrs[1:], kwargs)
        except TypeError:
            # we don't handle multiple values (for example sensor) atm.
            LOG.debug("Strange stuff happening in decision tree for %s: %s",
                      attrs[0], kwargs[attrs[0]])

        if match is None and self.any_key in curr_level:
            # if we couldn't find it using the attribute then continue with
            # the other attributes down the 'any' path
            match = self._find_match(curr_level[self.any_key], attrs[1:],
                                     kwargs)
        return match

    def find_match(self, **kwargs):
        try:
            match = self._find_match(self.tree, self.attrs, kwargs)
        except StandardError:
            LOG.debug("Match exception:", exc_info=True)
            LOG.error("Error when finding matching enhancement section")

        if match is None:
            # only possible if no default section was provided
            raise KeyError("No enhancement configuration found for %s" %
                           (kwargs.get("uid", None), ))
        for key, val in match.items():
            try:
                match[key] = json.loads(val)
            except TypeError:
                match[key] = val
            except ValueError:
                pass
        return match


class Enhancer(object):
    """
    Helper class to get enhancement information for images.
    """

    def __init__(self, ppp_config_dir=None, enhancement_config_file=None):
        """
        Args:
            ppp_config_dir: Points to the base configuration directory
            enhancement_config_file: The enhancement configuration to
                apply, False to leave as is.
        """
        self.ppp_config_dir = ppp_config_dir or get_environ_config_dir()
        self.enhancement_config_file = enhancement_config_file
        # Set enhancement_config_file to False for no enhancements
        if self.enhancement_config_file is None:
            # it wasn't specified in the config or in the kwargs, we should
            # provide a default
            config_fn = os.path.join("enhancements", "generic.cfg")
            self.enhancement_config_file = config_search_paths(
                config_fn, self.ppp_config_dir)
        if not isinstance(self.enhancement_config_file, (list, tuple)):
            self.enhancement_config_file = [self.enhancement_config_file]

        if self.enhancement_config_file:
            self.enhancement_tree = EnhancementDecisionTree(
                *self.enhancement_config_file)
        else:
            # They don't want any automatic enhancements
            self.enhancement_tree = None

        self.sensor_enhancement_configs = []

    def get_sensor_enhancement_config(self, sensor):
        if isinstance(sensor, str):
            # one single sensor
            sensor = [sensor]

        for sensor_name in sensor:
            config_fn = os.path.join("enhancements", sensor_name + ".cfg")
            config_files = config_search_paths(config_fn, self.ppp_config_dir)
            # Note: Enhancement configuration files can't overwrite individual
            # options, only entire sections are overwritten
            for config_file in config_files:
                yield config_file

    def add_sensor_enhancements(self, sensor):
        # XXX: Should we just load all enhancements from the base directory?
        new_configs = []
        for config_file in self.get_sensor_enhancement_config(sensor):
            if config_file not in self.sensor_enhancement_configs.append(
                    config_file):
                self.sensor_enhancement_configs.append(config_file)
                new_configs.append(config_file)

        if new_configs:
            self.enhancement_tree.add_config_to_tree(*new_configs)

    def apply(self, img, **info):
        enh_kwargs = self.enhancement_tree.find_match(**info)
        LOG.debug("Enhancement configuration options: %s" %
                  (str(enh_kwargs), ))
        img.enhance(**enh_kwargs)
