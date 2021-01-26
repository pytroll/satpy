#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2021 Satpy developers
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
"""Functions and utilities for downloading ancillary data.

TODO: Put examples here or on a new sphinx page?

"""

import logging
import satpy
import unittest.mock

try:
    import pooch
except ImportError:
    # TODO: Implement DumpPooch for local files only
    pooch = None

logger = logging.getLogger(__name__)

FILE_REGISTRY = {}
FILE_URLS = {}


def register_file(url, filename, component_type=None, component_name=None, known_hash=None):
    """Register file for future retrieval.

    This function only prepares Satpy to be able to download and cache the
    provided file. It will not download the file. See
    :func:`satpy.data_download.retrieve` for more information.

    Args:
        url (str): URL where remote file can be downloaded.
        filename (str): Filename used to identify and store the downloaded
            file as.
        component_type (str or None): Name of the type of Satpy component that
            will use this file. Typically "readers", "composites", "writers",
            or "enhancements" for consistency. This will be prepended to the
            filename when storing the data in the cache.
        component_name (str or None): Name of the Satpy component that will
            use this file. In most cases this will be the name of the Python
            class instead of the name of the instance
            (ex. StaticImageCompositor versus '_night_background'). This will be
            prepended to the filename when storing the data in the cache.
        known_hash (str): Hash used to verify the file is downloaded correctly.
            See https://www.fatiando.org/pooch/v1.3.0/beginner.html#hashes
            for more information. If not provided then the file is not checked.

    Returns:
        Cache key that can be used to retrieve the file later. The cache key
        consists of the ``component_type``, ``component_name``, and provided
        ``filename``. This should be passed to
        :func:`satpy.data_download_retrieve` when the file will be used.

    """
    if known_hash is None:
        # https://www.fatiando.org/pooch/v1.3.0/advanced.html#bypassing-the-hash-check
        known_hash = unittest.mock.ANY
    fname = _generate_filename(filename, component_type, component_name)

    global FILE_REGISTRY
    global FILE_URLS
    FILE_REGISTRY[fname] = known_hash
    FILE_URLS[fname] = url
    return fname


def _generate_filename(filename, component_type, component_name):
    if filename is None:
        return None
    path = filename
    if component_name:
        path = '/'.join([component_name, path])
    if component_type:
        path = '/'.join([component_type, path])
    return path


# def retrieve(url, filename=None, component_type=None, component_name=None,
#              known_hash=None, pooch_kwargs=None):
#     if pooch is None:
#         raise ImportError("Extra dependency library 'pooch' is required to "
#                           "download data files.")
#     pooch_kwargs = pooch_kwargs or {}
#
#     path = satpy.config.get('data_dir')
#     fname = register_file(url, filename, component_type, component_name,
#                           known_hash)
#     return pooch.retrieve(url, known_hash, fname=fname, path=path,
#                           **pooch_kwargs)


def retrieve(cache_key, pooch_kwargs=None):
    """Download and cache the file associated with the provided ``cache_key``.

    Cache location is controlled by the config ``data_dir`` key. See
    :ref:`data_dir_setting` for more information.

    Args:
        cache_key (str): Cache key returned by
            :func:`~satpy.data_download.register_file`.
        pooch_kwargs (dict or None): Extra keyword arguments to pass to
            :meth:`pooch.Pooch.fetch`.

    Returns:
        Local path of the cached file.


    """
    if pooch is None:
        raise ImportError("Extra dependency library 'pooch' is required to "
                          "download data files.")
    pooch_kwargs = pooch_kwargs or {}

    path = satpy.config.get('data_dir')
    # reuse data directory as the default URL where files can be downloaded from
    pooch_obj = pooch.create(path, path, registry=FILE_REGISTRY,
                             urls=FILE_URLS)
    return pooch_obj.fetch(cache_key, **pooch_kwargs)


def retrieve_all(pooch_kwargs=None):
    """Find cache-able data files for Satpy and download them.

    The typical use case for this function is to download all ancillary files
    before going to an environment/system that does not have internet access.

    """
    if pooch is None:
        raise ImportError("Extra dependency library 'pooch' is required to "
                          "download data files.")
    if pooch_kwargs is None:
        pooch_kwargs = {}

    _find_registerable_files()
    path = satpy.config.get('data_dir')
    pooch_obj = pooch.create(path, path, registry=FILE_REGISTRY,
                             urls=FILE_URLS)
    for fname in FILE_REGISTRY:
        logger.info("Downloading extra data file '%s'...", fname)
        pooch_obj.fetch(fname, **pooch_kwargs)
    logger.info("Done downloading all extra files.")


def _find_registerable_files():
    """Load all Satpy components so they can be downloaded."""
    _find_registerable_files_compositors()
    # TODO: Readers, writers


def _find_registerable_files_compositors():
    """Load all compositor configs so that files are registered.

    Compositor objects should register files when they are initialized.

    """
    from satpy.composites.config_loader import CompositorLoader
    composite_loader = CompositorLoader()
    all_sensor_names = ['viirs', 'seviri']  # FIXME: Find a way to actually get these
    composite_loader.load_compositors(all_sensor_names)
