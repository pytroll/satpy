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
import warnings

import dask.array as da
import numpy as np
import xarray as xr
import yaml

try:
    from yaml import UnsafeLoader
except ImportError:
    from yaml import Loader as UnsafeLoader

from satpy.config import (config_search_paths, glob_config,
                          get_environ_config_dir, recursive_dict_update)
from satpy import CHUNK_SIZE
from satpy.plugin_base import Plugin
from satpy.resample import get_area_def

from trollsift import parser

from trollimage.xrimage import XRImage

LOG = logging.getLogger(__name__)


def read_writer_config(config_files, loader=UnsafeLoader):
    """Read the writer `config_files` and return the info extracted."""

    conf = {}
    LOG.debug('Reading %s', str(config_files))
    for config_file in config_files:
        with open(config_file) as fd:
            conf.update(yaml.load(fd.read(), Loader=loader))

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
                level_coast=1, level_borders=1, fill_value=None,
                grid=None):
    """Add coastline, political borders and grid(graticules) to image.

    Uses ``color`` for feature colors where ``color`` is a 3-element tuple
    of integers between 0 and 255 representing (R, G, B).

    .. warning::

        This function currently loses the data mask (alpha band).

    ``resolution`` is chosen automatically if None (default), otherwise it should be one of:

    +-----+-------------------------+---------+
    | 'f' | Full resolution         | 0.04 km |
    | 'h' | High resolution         | 0.2 km  |
    | 'i' | Intermediate resolution | 1.0 km  |
    | 'l' | Low resolution          | 5.0 km  |
    | 'c' | Crude resolution        | 25  km  |
    +-----+-------------------------+---------+

    ``grid`` is a dictionary with key values as documented in detail in pycoast

    eg. overlay={'grid': {'major_lonlat': (10, 10),
                          'write_text': False,
                          'outline': (224, 224, 224),
                          'width': 0.5}}

    Here major_lonlat is plotted every 10 deg for both longitude and latitude,
    no labels for the grid lines are plotted, the color used for the grid lines
    is light gray, and the width of the gratucules is 0.5 pixels.

    For grid if aggdraw is used, font option is mandatory, if not write_text is set to False
    eg. font = aggdraw.Font('black', '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',
                            opacity=127, size=16)
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

    if hasattr(orig, 'convert'):
        # image must be in RGB space to work with pycoast/pydecorate
        orig = orig.convert('RGBA' if orig.mode.endswith('A') else 'RGB')
    elif not orig.mode.startswith('RGB'):
        raise RuntimeError("'trollimage' 1.6+ required to support adding "
                           "overlays/decorations to non-RGB data.")
    img = orig.pil_image(fill_value=fill_value)
    cw_ = ContourWriterAGG(coast_dir)
    cw_.add_coastlines(img, area, outline=color,
                       resolution=resolution, width=width, level=level_coast)
    cw_.add_borders(img, area, outline=color,
                    resolution=resolution, width=width, level=level_borders)
    # Only add grid if major_lonlat is given.
    if grid and 'major_lonlat' in grid and grid['major_lonlat']:
        major_lonlat = grid.pop('major_lonlat')
        minor_lonlat = grid.pop('minor_lonlat', major_lonlat)

        cw_.add_grid(img, area, major_lonlat, minor_lonlat, **grid)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    new_data = xr.DataArray(arr, dims=['y', 'x', 'bands'],
                            coords={'y': orig.data.coords['y'],
                                    'x': orig.data.coords['x'],
                                    'bands': list(img.mode)},
                            attrs=orig.data.attrs)
    return XRImage(new_data)


def add_text(orig, dc, img, text=None):
    """Add text to an image using the pydecorate package.

    All the features of pydecorate's ``add_text`` are available.
    See documentation of :doc:`pydecorate:index` for more info.

    """
    LOG.info("Add text to image.")

    dc.add_text(**text)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    new_data = xr.DataArray(arr, dims=['y', 'x', 'bands'],
                            coords={'y': orig.data.coords['y'],
                                    'x': orig.data.coords['x'],
                                    'bands': list(img.mode)},
                            attrs=orig.data.attrs)
    return XRImage(new_data)


def add_logo(orig, dc, img, logo=None):
    """Add logos or other images to an image using the pydecorate package.

    All the features of pydecorate's ``add_logo`` are available.
    See documentation of :doc:`pydecorate:index` for more info.

    """
    LOG.info("Add logo to image.")

    dc.add_logo(**logo)

    arr = da.from_array(np.array(img) / 255.0, chunks=CHUNK_SIZE)

    new_data = xr.DataArray(arr, dims=['y', 'x', 'bands'],
                            coords={'y': orig.data.coords['y'],
                                    'x': orig.data.coords['x'],
                                    'bands': list(img.mode)},
                            attrs=orig.data.attrs)
    return XRImage(new_data)


def add_decorate(orig, fill_value=None, **decorate):
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
    if hasattr(orig, 'convert'):
        # image must be in RGB space to work with pycoast/pydecorate
        orig = orig.convert('RGBA' if orig.mode.endswith('A') else 'RGB')
    elif not orig.mode.startswith('RGB'):
        raise RuntimeError("'trollimage' 1.6+ required to support adding "
                           "overlays/decorations to non-RGB data.")
    img_orig = orig.pil_image(fill_value=fill_value)
    from pydecorate import DecoratorAGG
    dc = DecoratorAGG(img_orig)

    # decorate need to be a list to maintain the alignment
    # as ordered in the list
    img = orig
    if 'decorate' in decorate:
        for dec in decorate['decorate']:
            if 'logo' in dec:
                img = add_logo(img, dc, img_orig, logo=dec['logo'])
            elif 'text' in dec:
                img = add_text(img, dc, img_orig, text=dec['text'])
    return img


def get_enhanced_image(dataset, ppp_config_dir=None, enhance=None, enhancement_config_file=None,
                       overlay=None, decorate=None, fill_value=None):
    """Get an enhanced version of `dataset` as an :class:`~trollimage.xrimage.XRImage` instance.

    Args:
        dataset (xarray.DataArray): Data to be enhanced and converted to an image.
        ppp_config_dir (str): Root configuration directory.
        enhance (bool or Enhancer): Whether to automatically enhance
            data to be more visually useful and to fit inside the file
            format being saved to. By default this will default to using
            the enhancement configuration files found using the default
            :class:`~satpy.writers.Enhancer` class. This can be set to
            `False` so that no enhancments are performed. This can also
            be an instance of the :class:`~satpy.writers.Enhancer` class
            if further custom enhancement is needed.
        enhancement_config_file (str): Deprecated.
        overlay (dict): Options for image overlays. See :func:`add_overlay`
            for available options.
        decorate (dict): Options for decorating the image. See
            :func:`add_decorate` for available options.
        fill_value (int or float): Value to use when pixels are masked or
            invalid. Default of `None` means to create an alpha channel.
            See :meth:`~trollimage.xrimage.XRImage.finalize` for more
            details. Only used when adding overlays or decorations. Otherwise
            it is up to the caller to "finalize" the image before using it
            except if calling ``img.show()`` or providing the image to
            a writer as these will finalize the image.

    .. versionchanged:: 0.10

        Deprecated `enhancement_config_file` and 'enhancer' in favor of
        `enhance`. Pass an instance of the `Enhancer` class to `enhance`
        instead.

    """
    if ppp_config_dir is None:
        ppp_config_dir = get_environ_config_dir()

    if enhancement_config_file is not None:
        warnings.warn("'enhancement_config_file' has been deprecated. Pass an instance of the "
                      "'Enhancer' class to the 'enhance' keyword argument instead.", DeprecationWarning)

    if enhance is False:
        # no enhancement
        enhancer = None
    elif enhance is None or enhance is True:
        # default enhancement
        enhancer = Enhancer(ppp_config_dir, enhancement_config_file)
    else:
        # custom enhancer
        enhancer = enhance

    # Create an image for enhancement
    img = to_image(dataset)

    if enhancer is None or enhancer.enhancement_tree is None:
        LOG.debug("No enhancement being applied to dataset")
    else:
        if dataset.attrs.get("sensor", None):
            enhancer.add_sensor_enhancements(dataset.attrs["sensor"])

        enhancer.apply(img, **dataset.attrs)

    if overlay is not None:
        img = add_overlay(img, dataset.attrs['area'], fill_value=fill_value, **overlay)

    if decorate is not None:
        img = add_decorate(img, fill_value=fill_value, **decorate)

    return img


def show(dataset, **kwargs):
    """Display the dataset as an image.
    """
    img = get_enhanced_image(dataset.squeeze(), **kwargs)
    img.show()
    return img


def to_image(dataset):
    """convert ``dataset`` into a :class:`~trollimage.xrimage.XRImage` instance.

    Convert the ``dataset`` into an instance of the
    :class:`~trollimage.xrimage.XRImage` class.  This function makes no other
    changes.  To get an enhanced image, possibly with overlays and decoration,
    see :func:`~get_enhanced_image`.

    Args:
        dataset (xarray.DataArray): Data to be converted to an image.

    Returns:
        Instance of :class:`~trollimage.xrimage.XRImage`.
    """
    dataset = dataset.squeeze()
    if dataset.ndim < 2:
        raise ValueError("Need at least a 2D array to make an image.")
    else:
        return XRImage(dataset)


def split_results(results):
    """Get sources, targets and delayed objects to separate lists from a
    list of results collected from (multiple) writer(s)."""
    from dask.delayed import Delayed

    def flatten(results):
        out = []
        if isinstance(results, (list, tuple)):
            for itm in results:
                out.extend(flatten(itm))
            return out
        return [results]

    sources = []
    targets = []
    delayeds = []

    for res in flatten(results):
        if isinstance(res, da.Array):
            sources.append(res)
        elif isinstance(res, Delayed):
            delayeds.append(res)
        else:
            targets.append(res)
    return sources, targets, delayeds


def compute_writer_results(results):
    """Compute all the given dask graphs `results` so that the files are
    saved.

    Args:
        results (iterable): Iterable of dask graphs resulting from calls to
                            `scn.save_datasets(..., compute=False)`
    """
    if not results:
        return

    sources, targets, delayeds = split_results(results)

    # one or more writers have targets that we need to close in the future
    if targets:
        delayeds.append(da.store(sources, targets, compute=False))

    if delayeds:
        da.compute(delayeds)

    if targets:
        for target in targets:
            if hasattr(target, 'close'):
                target.close()


class Writer(Plugin):
    """Base Writer class for all other writers.

    A minimal writer subclass should implement the `save_dataset` method.
    """

    def __init__(self, name=None, filename=None, base_dir=None, **kwargs):
        """Initialize the writer object.

        Args:
            name (str): A name for this writer for log and error messages.
                If this writer is configured in a YAML file its name should
                match the name of the YAML file. Writer names may also appear
                in output file attributes.
            filename (str): Filename to save data to. This filename can and
                should specify certain python string formatting fields to
                differentiate between data written to the files. Any
                attributes provided by the ``.attrs`` of a DataArray object
                may be included. Format and conversion specifiers provided by
                the :class:`trollsift <trollsift.parser.StringFormatter>`
                package may also be used. Any directories in the provided
                pattern will be created if they do not exist. Example::

                    {platform_name}_{sensor}_{name}_{start_time:%Y%m%d_%H%M%S.tif

            base_dir (str):
                Base destination directories for all created files.
            kwargs (dict): Additional keyword arguments to pass to the
                :class:`~satpy.plugin_base.Plugin` class.

            """
        # Load the config
        Plugin.__init__(self, **kwargs)
        self.info = self.config.get('writer', {})

        if 'file_pattern' in self.info:
            warnings.warn("Writer YAML config is using 'file_pattern' which "
                          "has been deprecated, use 'filename' instead.")
            self.info['filename'] = self.info.pop('file_pattern')

        if 'file_pattern' in kwargs:
            warnings.warn("'file_pattern' has been deprecated, use 'filename' instead.", DeprecationWarning)
            filename = kwargs.pop('file_pattern')

        # Use options from the config file if they weren't passed as arguments
        self.name = self.info.get("name", None) if name is None else name
        self.file_pattern = self.info.get("filename", None) if filename is None else filename

        if self.name is None:
            raise ValueError("Writer 'name' not provided")

        self.filename_parser = self.create_filename_parser(base_dir)

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        """Helper class method to separate arguments between init and save methods.

        Currently the :class:`~satpy.scene.Scene` is passed one set of
        arguments to represent the Writer creation and saving steps. This is
        not preferred for Writer structure, but provides a simpler interface
        to users. This method splits the provided keyword arguments between
        those needed for initialization and those needed for the ``save_dataset``
        and ``save_datasets`` method calls.

        Writer subclasses should try to prefer keyword arguments only for the
        save methods only and leave the init keyword arguments to the base
        classes when possible.

        """
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs = {}
        kwargs = kwargs.copy()
        for kw in ['base_dir', 'filename', 'file_pattern']:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)
        return init_kwargs, kwargs

    def create_filename_parser(self, base_dir):
        """Create a :class:`trollsift.parser.Parser` object for later use."""
        # just in case a writer needs more complex file patterns
        # Set a way to create filenames if we were given a pattern
        if base_dir and self.file_pattern:
            file_pattern = os.path.join(base_dir, self.file_pattern)
        else:
            file_pattern = self.file_pattern
        return parser.Parser(file_pattern) if file_pattern else None

    def get_filename(self, **kwargs):
        """Create a filename where output data will be saved.

        Args:
            kwargs (dict): Attributes and other metadata to use for formatting
                the previously provided `filename`.

        """
        if self.filename_parser is None:
            raise RuntimeError("No filename pattern or specific filename provided")
        output_filename = self.filename_parser.compose(kwargs)
        dirname = os.path.dirname(output_filename)
        if dirname and not os.path.isdir(dirname):
            LOG.info("Creating output directory: {}".format(dirname))
            os.makedirs(dirname)
        return output_filename

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
        results = []
        for ds in datasets:
            results.append(self.save_dataset(ds, compute=False, **kwargs))

        if compute:
            LOG.info("Computing and writing results...")
            return compute_writer_results([results])

        targets, sources, delayeds = split_results([results])
        if delayeds:
            # This writer had only delayed writes
            return delayeds
        else:
            return targets, sources

    def save_dataset(self, dataset, filename=None, fill_value=None,
                     compute=True, **kwargs):
        """Saves the ``dataset`` to a given ``filename``.

        This method must be overloaded by the subclass.

        Args:
            dataset (xarray.DataArray): Dataset to save using this writer.
            filename (str): Optionally specify the filename to save this
                            dataset to. If not provided then `filename`
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
    """Base writer for image file formats."""

    def __init__(self, name=None, filename=None, base_dir=None, enhance=None, enhancement_config=None, **kwargs):
        """Initialize image writer object.

        Args:
            name (str): A name for this writer for log and error messages.
                If this writer is configured in a YAML file its name should
                match the name of the YAML file. Writer names may also appear
                in output file attributes.
            filename (str): Filename to save data to. This filename can and
                should specify certain python string formatting fields to
                differentiate between data written to the files. Any
                attributes provided by the ``.attrs`` of a DataArray object
                may be included. Format and conversion specifiers provided by
                the :class:`trollsift <trollsift.parser.StringFormatter>`
                package may also be used. Any directories in the provided
                pattern will be created if they do not exist. Example::

                    {platform_name}_{sensor}_{name}_{start_time:%Y%m%d_%H%M%S.tif

            base_dir (str):
                Base destination directories for all created files.
            enhance (bool or Enhancer): Whether to automatically enhance
                data to be more visually useful and to fit inside the file
                format being saved to. By default this will default to using
                the enhancement configuration files found using the default
                :class:`~satpy.writers.Enhancer` class. This can be set to
                `False` so that no enhancments are performed. This can also
                be an instance of the :class:`~satpy.writers.Enhancer` class
                if further custom enhancement is needed.
            enhancement_config (str): Deprecated.

            kwargs (dict): Additional keyword arguments to pass to the
                :class:`~satpy.writer.Writer` base class.

        .. versionchanged:: 0.10

            Deprecated `enhancement_config_file` and 'enhancer' in favor of
            `enhance`. Pass an instance of the `Enhancer` class to `enhance`
            instead.

        """
        super(ImageWriter, self).__init__(name, filename, base_dir, **kwargs)
        if enhancement_config is not None:
            warnings.warn("'enhancement_config' has been deprecated. Pass an instance of the "
                          "'Enhancer' class to the 'enhance' keyword argument instead.", DeprecationWarning)
        else:
            enhancement_config = self.info.get("enhancement_config", None)

        if enhance is False:
            # No enhancement
            self.enhancer = False
        elif enhance is None or enhance is True:
            # default enhancement
            self.enhancer = Enhancer(ppp_config_dir=self.ppp_config_dir, enhancement_config_file=enhancement_config)
        else:
            # custom enhancer
            self.enhancer = enhance

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs, kwargs = super(ImageWriter, cls).separate_init_kwargs(kwargs)
        for kw in ['enhancement_config', 'enhance']:
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
        img = get_enhanced_image(dataset.squeeze(), enhance=self.enhancer, overlay=overlay,
                                 decorate=decorate, fill_value=fill_value)
        return self.save_image(img, filename=filename, compute=compute, fill_value=fill_value, **kwargs)

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
        raise NotImplementedError("Writer '%s' has not implemented image saving" % (self.name,))


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
                    enhancement_config = yaml.load(fd, Loader=UnsafeLoader)
                    if enhancement_config is None:
                        # empty file
                        continue
                    enhancement_section = enhancement_config.get(
                        self.prefix, {})
                    if not enhancement_section:
                        LOG.debug("Config '{}' has no '{}' section or it is empty".format(config_file, self.prefix))
                        continue
                    conf = recursive_dict_update(conf, enhancement_section)
            elif isinstance(config_file, dict):
                conf = recursive_dict_update(conf, config_file)
            else:
                LOG.debug("Loading enhancement config string")
                d = yaml.load(config_file, Loader=UnsafeLoader)
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
            enhancement_config_file: The enhancement configuration to apply, False to leave as is.
        """
        self.ppp_config_dir = ppp_config_dir or get_environ_config_dir()
        self.enhancement_config_file = enhancement_config_file
        # Set enhancement_config_file to False for no enhancements
        if self.enhancement_config_file is None:
            # it wasn't specified in the config or in the kwargs, we should
            # provide a default
            config_fn = os.path.join("enhancements", "generic.yaml")
            self.enhancement_config_file = config_search_paths(config_fn, self.ppp_config_dir)

        if not self.enhancement_config_file:
            # They don't want any automatic enhancements
            self.enhancement_tree = None
        else:
            if not isinstance(self.enhancement_config_file, (list, tuple)):
                self.enhancement_config_file = [self.enhancement_config_file]

            self.enhancement_tree = EnhancementDecisionTree(*self.enhancement_config_file)

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
