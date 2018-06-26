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
"""Shared objects of the various writer classes.

For now, this includes enhancement configuration utilities.
"""

import logging
import os

import numpy as np
import yaml
import dask
import dask.array as da
import xarray as xr

from satpy.config import (config_search_paths, glob_config,
                          get_environ_config_dir, recursive_dict_update)
from satpy import CHUNK_SIZE
from satpy.plugin_base import Plugin
from satpy.resample import get_area_def

from trollsift import parser

from trollimage.xrimage import XRImage

LOG = logging.getLogger(__name__)


def read_writer_config(config_files, loader=yaml.Loader):
    """Read the writer `config_files` and return the info extracted."""

    conf = {}
    LOG.debug('Reading %s', str(config_files))
    for config_file in config_files:
        with open(config_file) as fd:
            conf.update(yaml.load(fd.read(), loader))

    try:
        writer_info = conf['writer']
    except KeyError:
        raise KeyError(
            "Malformed config file {}: missing writer 'writer'".format(
                config_files))
    writer_info['config_files'] = config_files
    return writer_info


def load_writer_configs(writer_configs, ppp_config_dir,
                        **writer_kwargs):
    """Load the writer from the provided `writer_configs`."""
    try:
        writer_info = read_writer_config(writer_configs)
        writer_class = writer_info['writer']
    except (ValueError, KeyError, yaml.YAMLError):
        raise ValueError("Invalid writer configs: "
                         "'{}'".format(writer_configs))
    init_kwargs, kwargs = writer_class.separate_init_kwargs(writer_kwargs)
    writer = writer_class(ppp_config_dir=ppp_config_dir,
                          config_files=writer_configs,
                          **init_kwargs)
    return writer, kwargs


def load_writer(writer, ppp_config_dir=None, **writer_kwargs):
    """Find and load writer `writer` in the available configuration files."""
    if ppp_config_dir is None:
        ppp_config_dir = get_environ_config_dir()

    config_fn = writer + ".yaml" if "." not in writer else writer
    config_files = config_search_paths(
        os.path.join("writers", config_fn), ppp_config_dir)
    writer_kwargs.setdefault("config_files", config_files)
    if not writer_kwargs['config_files']:
        raise ValueError("Unknown writer '{}'".format(writer))

    try:
        return load_writer_configs(writer_kwargs['config_files'],
                                   ppp_config_dir=ppp_config_dir,
                                   **writer_kwargs)
    except ValueError:
        raise ValueError("Writer '{}' does not exist or could not be "
                         "loaded".format(writer))


def configs_for_writer(writer=None, ppp_config_dir=None):
    """Generator of writer configuration files for one or more writers

    Args:
        writer (Optional[str]): Yield configs only for this writer
        ppp_config_dir (Optional[str]): Additional configuration directory
            to search for writer configuration files.

    Returns: Generator of lists of configuration files

    """
    search_paths = (ppp_config_dir,) if ppp_config_dir else tuple()
    if writer is not None:
        if not isinstance(writer, (list, tuple)):
            writer = [writer]
        # given a config filename or writer name
        config_files = [w if w.endswith('.yaml') else w + '.yaml' for w in writer]
    else:
        writer_configs = glob_config(os.path.join('writers', '*.yaml'),
                                     *search_paths)
        config_files = set(writer_configs)

    for config_file in config_files:
        config_basename = os.path.basename(config_file)
        writer_configs = config_search_paths(
            os.path.join("writers", config_basename), *search_paths)

        if not writer_configs:
            LOG.warning("No writer configs found for '%s'", writer)
            continue

        yield writer_configs


def available_writers(as_dict=False):
    """Available writers based on current configuration.

    Args:
        as_dict (bool): Optionally return writer information as a dictionary.
                        Default: False

    Returns: List of available writer names. If `as_dict` is `True` then
             a list of dictionaries including additionally writer information
             is returned.

    """
    writers = []
    for writer_configs in configs_for_writer():
        try:
            writer_info = read_writer_config(writer_configs)
        except (KeyError, IOError, yaml.YAMLError):
            LOG.warning("Could not import writer config from: %s", writer_configs)
            LOG.debug("Error loading YAML", exc_info=True)
            continue
        writers.append(writer_info if as_dict else writer_info['name'])
    return writers


def _determine_mode(dataset):
    if "mode" in dataset.attrs:
        return dataset.attrs["mode"]

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
                           str(dataset))


def add_overlay(orig, area, coast_dir, color=(0, 0, 0), width=0.5, resolution=None,
                level_coast=1, level_borders=1):
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

    if area is None:
        raise ValueError("Area of image is None, can't add overlay.")

    from pycoast import ContourWriterAGG
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

        LOG.debug("Automagically choose resolution %s", resolution)

    img = orig.pil_image()
    cw_ = ContourWriterAGG(coast_dir)
    cw_.add_coastlines(img, area, outline=color,
                       resolution=resolution, width=width, level=level_coast)
    cw_.add_borders(img, area, outline=color,
                    resolution=resolution, width=width, level=level_borders)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    orig.data = xr.DataArray(arr, dims=['y', 'x', 'bands'],
                             coords={'y': orig.data.coords['y'],
                                     'x': orig.data.coords['x'],
                                     'bands': list(img.mode)})


def add_text(orig, dc, img, text=None):
    """
    Add text to an image using the pydecorate function add_text
    All the features in pydecorate are available

    See documentation of pydecorate
    """
    LOG.info("Add text to image.")

    dc.add_text(**text)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    orig.data = xr.DataArray(arr, dims=['y', 'x', 'bands'],
                             coords={'y': orig.data.coords['y'],
                                     'x': orig.data.coords['x'],
                                     'bands': list(img.mode)})


def add_logo(orig, dc, img, logo=None):
    """
    Add logos or other images to an image using the pydecorate function add_logo
    All the features in pydecorate are available

    See documentation of pydecorate
    """
    LOG.info("Add logo to image.")

    dc.add_logo(**logo)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    orig.data = xr.DataArray(arr, dims=['y', 'x', 'bands'],
                             coords={'y': orig.data.coords['y'],
                                     'x': orig.data.coords['x'],
                                     'bands': list(img.mode)})


def add_decorate(orig, **decorate):
    """Decorate an image with text and/or logos/images.

    This call adds text/logos in order as given in the input to keep the
    alignment features available in pydecorate.

    An example of the decorate config::

        decorate = {
            'decorate': [
                {'logo': {'logo_path': <path to a logo>, 'height': 143, 'bg': 'white', 'bg_opacity': 255}},
                {'text': {'txt': start_time_txt,
                          'align': {'top_bottom': 'bottom', 'left_right': 'right'},
                          'font': <path to ttf font>,
                          'font_size': 22,
                          'height': 30,
                          'bg': 'black',
                          'bg_opacity': 255,
                          'line': 'white'}}
            ]
        }

    Any numbers of text/logo in any order can be added to the decorate list,
    but the order of the list is kept as described above.

    Note that a feature given in one element, eg. bg (which is the background color)
    will also apply on the next elements  unless a new value is given.

    align is a special keyword telling where in the image to start adding features, top_bottom is either top or bottom
    and left_right is either left or right.
    """
    LOG.info("Decorate image.")

    # Need to create this here to possible keep the alignment
    # when adding text and/or logo with pydecorate
    img_orig = orig.pil_image()
    from pydecorate import DecoratorAGG
    dc = DecoratorAGG(img_orig)

    # decorate need to be a list to maintain the alignment
    # as ordered in the list
    if 'decorate' in decorate:
        for dec in decorate['decorate']:
            if 'logo' in dec:
                add_logo(orig, dc, img_orig, logo=dec['logo'])
            elif 'text' in dec:
                add_text(orig, dc, img_orig, text=dec['text'])


def get_enhanced_image(dataset,
                       enhancer=None,
                       fill_value=None,
                       ppp_config_dir=None,
                       enhancement_config_file=None,
                       overlay=None,
                       decorate=None):
    mode = _determine_mode(dataset)
    if ppp_config_dir is None:
        ppp_config_dir = get_environ_config_dir()

    if enhancer is None:
        enhancer = Enhancer(ppp_config_dir, enhancement_config_file)

    # Create an image for enhancement
    img = to_image(dataset, mode=mode, fill_value=fill_value)

    if enhancer.enhancement_tree is None:
        LOG.debug("No enhancement being applied to dataset")
    else:
        if dataset.attrs.get("sensor", None):
            enhancer.add_sensor_enhancements(dataset.attrs["sensor"])

        enhancer.apply(img, **dataset.attrs)

    if overlay is not None:
        add_overlay(img, dataset.attrs['area'], **overlay)

    if decorate is not None:
        add_decorate(img, **decorate)

    return img


def show(dataset, **kwargs):
    """Display the dataset as an image.
    """
    img = get_enhanced_image(dataset.squeeze(), **kwargs)
    img.show()
    return img


def to_image(dataset, copy=False, **kwargs):
    # Only add keywords if they are present
    for key in ["mode", "fill_value", "palette"]:
        if key in dataset.attrs:
            kwargs.setdefault(key, dataset.attrs[key])
    dataset = dataset.squeeze()

    if dataset.ndim < 2:
        raise ValueError("Need at least a 2D array to make an image.")
    else:
        return XRImage(dataset)


class Writer(Plugin):

    """Writer plugins. They must implement the *save_image* method. This is an
    abstract class to be inherited.
    """

    def __init__(self,
                 name=None,
                 file_pattern=None,
                 base_dir=None,
                 **kwargs):
        # Load the config
        Plugin.__init__(self, **kwargs)
        self.info = self.config['writer']

        # Use options from the config file if they weren't passed as arguments
        self.name = self.info.get("name",
                                  None) if name is None else name
        self.file_pattern = self.info.get(
            "file_pattern", None) if file_pattern is None else file_pattern

        if self.name is None:
            raise ValueError("Writer 'name' not provided")

        self.filename_parser = self.create_filename_parser(base_dir)

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs = {}
        kwargs = kwargs.copy()
        for kw in ['base_dir', 'file_pattern']:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)
        return init_kwargs, kwargs

    def create_filename_parser(self, base_dir):
        # just in case a writer needs more complex file patterns
        # Set a way to create filenames if we were given a pattern
        if base_dir and self.file_pattern:
            file_pattern = os.path.join(base_dir, self.file_pattern)
        else:
            file_pattern = self.file_pattern
        return parser.Parser(file_pattern) if file_pattern else None

    def get_filename(self, **kwargs):
        if self.filename_parser is None:
            raise RuntimeError(
                "No filename pattern or specific filename provided")
        return self.filename_parser.compose(kwargs)

    def save_datasets(self, datasets, compute=True, **kwargs):
        """Save all datasets to one or more files.

        Subclasses can use this method to save all datasets to one single
        file or optimize the writing of individual datasets. By default
        this simply calls `save_dataset` for each dataset provided.

        Args:
            datasets (iterable): Iterable of `xarray.DataArray` objects to
                                 save using this writer.
            compute (bool): If `True` (default), compute all of the saves to
                            disk. If `False` then the return value is either
                            a `dask.delayed.Delayed` object or two lists to
                            be passed to a `dask.array.store` call.
                            See return values below for more details.
            **kwargs: Keyword arguments to pass to `save_dataset`. See that
                      documentation for more details.

        Returns:
            Value returned depends on `compute` keyword argument. If
            `compute` is `True` the value is the result of a either a
            `dask.array.store` operation or a `dask.delayed.Delayed` compute,
            typically this is `None`. If `compute` is `False` then the
            result is either a `dask.delayed.Delayed` object that can be
            computed with `delayed.compute()` or a two element tuple of
            sources and targets to be passed to `dask.array.store`. If
            `targets` is provided then it is the caller's responsibility to
            close any objects that have a "close" method.

        """
        sources = []
        targets = []
        for ds in datasets:
            res = self.save_dataset(ds, compute=False, **kwargs)
            if isinstance(res, tuple):
                # source, target to be passed to da.store
                sources.append(res[0])
                targets.append(res[1])
            else:
                # delayed object
                sources.append(res)

        # we have targets, we should save sources to targets
        if targets and compute:
            LOG.info("Computing and writing results...")
            res = da.store(sources, targets)
            for target in targets:
                if hasattr(target, 'close'):
                    target.close()
            return res
        elif targets:
            return sources, targets

        delayed = dask.delayed(sources)
        if compute:
            LOG.info("Computing and writing results...")
            return delayed.compute()
        return delayed

    def save_dataset(self, dataset, filename=None, fill_value=None,
                     compute=True, **kwargs):
        """Saves the ``dataset`` to a given ``filename``.

        This method must be overloaded by the subclass.

        Args:
            dataset (xarray.DataArray): Dataset to save using this writer.
            filename (str): Optionally specify the filename to save this
                            dataset to. If not provided then `file_pattern`
                            which can be provided to the init method will be
                            used and formatted by dataset attributes.
            fill_value (int or float): Replace invalid values in the dataset
                                       with this fill value if applicable to
                                       this writer.
            compute (bool): If `True` (default), compute and save the dataset.
                            If `False` return either a `dask.delayed.Delayed`
                            object or tuple of (source, target). See the
                            return values below for more information.
            **kwargs: Other keyword arguments for this particular writer.

        Returns:
            Value returned depends on `compute`. If `compute` is `True` then
            the return value is the result of computing a
            `dask.delayed.Delayed` object or running `dask.array.store`. If
            `compute` is `False` then the returned value is either a
            `dask.delayed.Delayed` object that can be computed using
            `delayed.compute()` or a tuple of (source, target) that should be
            passed to `dask.array.store`. If target is provided the the caller
            is responsible for calling `target.close()` if the target has
            this method.

        """
        raise NotImplementedError(
            "Writer '%s' has not implemented dataset saving" % (self.name, ))


class ImageWriter(Writer):

    def __init__(self,
                 name=None,
                 file_pattern=None,
                 enhancement_config=None,
                 base_dir=None,
                 **kwargs):
        Writer.__init__(self, name, file_pattern, base_dir,
                        **kwargs)
        enhancement_config = self.info.get(
            "enhancement_config",
            None) if enhancement_config is None else enhancement_config

        self.enhancer = Enhancer(ppp_config_dir=self.ppp_config_dir,
                                 enhancement_config_file=enhancement_config)

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs, kwargs = super(ImageWriter, cls).separate_init_kwargs(
            kwargs)
        for kw in ['enhancement_config']:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)
        return init_kwargs, kwargs

    def save_dataset(self, dataset, filename=None, fill_value=None,
                     overlay=None, decorate=None, compute=True, **kwargs):
        """Saves the ``dataset`` to a given ``filename``.

        This method creates an enhanced image using `get_enhanced_image`. The
        image is then passed to `save_image`. See both of these functions for
        more details on the arguments passed to this method.

        """
        img = get_enhanced_image(
            dataset.squeeze(), self.enhancer, fill_value, overlay=overlay,
            decorate=decorate)
        return self.save_image(img, filename=filename, compute=compute,
                               **kwargs)

    def save_image(self, img, filename=None, compute=True, **kwargs):
        """Save Image object to a given ``filename``.

        Args:
            img (trollimage.xrimage.XRImage): Image object to save to disk.
            filename (str): Optionally specify the filename to save this
                            dataset to. It may include string formatting
                            patterns that will be filled in by dataset
                            attributes.
            compute (bool): If `True` (default), compute and save the dataset.
                            If `False` return either a `dask.delayed.Delayed`
                            object or tuple of (source, target). See the
                            return values below for more information.
            **kwargs: Other keyword arguments to pass to this writer.

        Returns:
            Value returned depends on `compute`. If `compute` is `True` then
            the return value is the result of computing a
            `dask.delayed.Delayed` object or running `dask.array.store`. If
            `compute` is `False` then the returned value is either a
            `dask.delayed.Delayed` object that can be computed using
            `delayed.compute()` or a tuple of (source, target) that should be
            passed to `dask.array.store`. If target is provided the the caller
            is responsible for calling `target.close()` if the target has
            this method.

        """
        raise NotImplementedError(
            "Writer '%s' has not implemented image saving" % (self.name, ))


class DecisionTree(object):
    any_key = None

    def __init__(self, decision_dicts, attrs, **kwargs):
        self.attrs = attrs
        self.tree = {}
        if not isinstance(decision_dicts, (list, tuple)):
            decision_dicts = [decision_dicts]
        self.add_config_to_tree(*decision_dicts)

    def add_config_to_tree(self, *decision_dicts):
        conf = {}
        for decision_dict in decision_dicts:
            conf = recursive_dict_update(conf, decision_dict)
        self._build_tree(conf)

    def _build_tree(self, conf):
        for section_name, attrs in conf.items():
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
        except (KeyError, IndexError, ValueError):
            LOG.debug("Match exception:", exc_info=True)
            LOG.error("Error when finding matching decision section")

        if match is None:
            # only possible if no default section was provided
            raise KeyError("No decision section found for %s" %
                           (kwargs.get("uid", None), ))
        return match


class EnhancementDecisionTree(DecisionTree):

    def __init__(self, *decision_dicts, **kwargs):
        attrs = kwargs.pop("attrs", ("name",
                                     "platform_name",
                                     "sensor",
                                     "standard_name",
                                     "units",))
        self.prefix = kwargs.pop("config_section", "enhancements")
        super(EnhancementDecisionTree, self).__init__(
            decision_dicts, attrs, **kwargs)

    def add_config_to_tree(self, *decision_dict):
        conf = {}
        for config_file in decision_dict:
            if os.path.isfile(config_file):
                with open(config_file) as fd:
                    enhancement_section = yaml.load(fd).get(self.prefix, {})
                    if not enhancement_section:
                        LOG.debug("Config '{}' has no '{}' section or it is empty".format(config_file, self.prefix))
                        continue
                    conf = recursive_dict_update(conf, enhancement_section)
            elif isinstance(config_file, dict):
                conf = recursive_dict_update(conf, config_file)
            else:
                LOG.debug("Loading enhancement config string")
                d = yaml.load(config_file)
                if not isinstance(d, dict):
                    raise ValueError(
                        "YAML file doesn't exist or string is not YAML dict: {}".format(config_file))
                conf = recursive_dict_update(conf, d)

        self._build_tree(conf)

    def find_match(self, **kwargs):
        try:
            return super(EnhancementDecisionTree, self).find_match(**kwargs)
        except KeyError:
            # give a more understandable error message
            raise KeyError("No enhancement configuration found for %s" %
                           (kwargs.get("uid", None), ))


class Enhancer(object):

    """Helper class to get enhancement information for images."""

    def __init__(self, ppp_config_dir=None, enhancement_config_file=None):
        """Initialize an Enhancer instance.

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
            config_fn = os.path.join("enhancements", "generic.yaml")
            self.enhancement_config_file = config_search_paths(
                config_fn, self.ppp_config_dir)

        if not self.enhancement_config_file:
            # They don't want any automatic enhancements
            self.enhancement_tree = None
        else:
            if not isinstance(self.enhancement_config_file, (list, tuple)):
                self.enhancement_config_file = [self.enhancement_config_file]

            self.enhancement_tree = EnhancementDecisionTree(
                *self.enhancement_config_file)

        self.sensor_enhancement_configs = []

    def get_sensor_enhancement_config(self, sensor):
        if isinstance(sensor, str):
            # one single sensor
            sensor = [sensor]

        for sensor_name in sensor:
            config_fn = os.path.join("enhancements", sensor_name + ".yaml")
            config_files = config_search_paths(config_fn, self.ppp_config_dir)
            # Note: Enhancement configuration files can't overwrite individual
            # options, only entire sections are overwritten
            for config_file in config_files:
                yield config_file

    def add_sensor_enhancements(self, sensor):
        # XXX: Should we just load all enhancements from the base directory?
        new_configs = []
        for config_file in self.get_sensor_enhancement_config(sensor):
            if config_file not in self.sensor_enhancement_configs:
                self.sensor_enhancement_configs.append(config_file)
                new_configs.append(config_file)

        if new_configs:
            self.enhancement_tree.add_config_to_tree(*new_configs)

    def apply(self, img, **info):
        enh_kwargs = self.enhancement_tree.find_match(**info)

        LOG.debug("Enhancement configuration options: %s" %
                  (str(enh_kwargs['operations']), ))
        for operation in enh_kwargs['operations']:
            fun = operation['method']
            args = operation.get('args', [])
            kwargs = operation.get('kwargs', {})
            fun(img, *args, **kwargs)
        # img.enhance(**enh_kwargs)
