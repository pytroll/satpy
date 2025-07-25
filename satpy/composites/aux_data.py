# Copyright (c) 2015-2025 Satpy developers
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

"""Compositors using auxiliary data."""

from __future__ import annotations

import logging
import os

import satpy
from satpy.aux_download import DataDownloadMixin

from .core import GenericCompositor

LOG = logging.getLogger(__name__)


class StaticImageCompositor(GenericCompositor, DataDownloadMixin):
    """A compositor that loads a static image from disk.

    Environment variables in the filename are automatically expanded.

    """

    def __init__(self, name, filename=None, url=None, known_hash=None, area=None,  # noqa: D417
                 **kwargs):
        """Collect custom configuration values.

        Args:
            filename (str): Name to use when storing and referring to the file
                in the ``data_dir`` cache. If ``url`` is provided (preferred),
                then this is used as the filename in the cache and will be
                appended to ``<data_dir>/composites/<class_name>/``. If
                ``url`` is provided and ``filename`` is not then the
                ``filename`` will be guessed from the ``url``.
                If ``url`` is not provided, then it is assumed ``filename``
                refers to a local file. If the ``filename`` does not come with
                an absolute path, ``data_dir`` will be used as the directory path.
                Environment variables are expanded.
            url (str): URL to remote file. When the composite is created the
                file will be downloaded and cached in Satpy's ``data_dir``.
                Environment variables are expanded.
            known_hash (str or None): Hash of the remote file used to verify
                a successful download. If not provided then the download will
                not be verified. See :func:`satpy.aux_download.register_file`
                for more information.
            area (str): Name of area definition for the image.  Optional
                for images with built-in area definitions (geotiff).

        Use cases:
            1. url + no filename:
               Satpy determines the filename based on the filename in the URL,
               then downloads the URL, and saves it to <data_dir>/<filename>.
               If the file already exists and known_hash is also provided, then the pooch
               library compares the hash of the file to the known_hash. If it does not
               match, then the URL is re-downloaded. If it matches then no download.
            2. url + relative filename:
               Same as case 1 but filename is already provided so download goes to
               <data_dir>/<filename>. Same hashing behavior. This does not check for an
               absolute path.
            3. No url + absolute filename:
               No download, filename is passed directly to generic_image reader. No hashing
               is done.
            4. No url + relative filename:
               Check if <data_dir>/<filename> exists. If it does then make filename an
               absolute path. If it doesn't, then keep it as is and let the exception at
               the bottom of the method get raised.
        """
        filename, url = self._get_cache_filename_and_url(filename, url)
        self._cache_filename = filename
        self._url = url
        self._known_hash = known_hash
        self.area = None
        if area is not None:
            from satpy.area import get_area_def
            self.area = get_area_def(area)

        super(StaticImageCompositor, self).__init__(name, **kwargs)
        cache_keys = self.register_data_files([])
        self._cache_key = cache_keys[0]

    @staticmethod
    def _check_relative_filename(filename):
        data_dir = satpy.config.get("data_dir")
        path = os.path.join(data_dir, filename)

        return path if os.path.exists(path) else filename

    def _get_cache_filename_and_url(self, filename, url):
        filename = self._check_filename(filename, url)
        url, filename = self._check_url(url, filename)

        return filename, url

    def _check_filename(self, filename, url):
        if filename:
            filename = os.path.expanduser(os.path.expandvars(filename))
            if not os.path.isabs(filename) and not url:
                filename = self._check_relative_filename(filename)
        return filename

    def _check_url(self, url, filename):
        if url:
            url = os.path.expandvars(url)
            if not filename:
                filename = os.path.basename(url)
        elif not filename or not os.path.isabs(filename):
            raise ValueError("StaticImageCompositor needs a remote 'url', "
                             "or absolute path to 'filename', "
                             "or an existing 'filename' relative to Satpy's 'data_dir'.")
        return url, filename

    def register_data_files(self, data_files):
        """Tell Satpy about files we may want to download."""
        if os.path.isabs(self._cache_filename):
            return [None]
        return super().register_data_files([{
            "url": self._url,
            "known_hash": self._known_hash,
            "filename": self._cache_filename,
        }])

    def _retrieve_data_file(self):
        from satpy.aux_download import retrieve
        if os.path.isabs(self._cache_filename):
            return self._cache_filename
        return retrieve(self._cache_key)

    def __call__(self, *args, **kwargs):
        """Call the compositor."""
        from satpy import Scene
        local_file = self._retrieve_data_file()
        scn = Scene(reader="generic_image", filenames=[local_file])
        scn.load(["image"])
        img = scn["image"]
        # use compositor parameters as extra metadata
        # most important: set 'name' of the image
        img.attrs.update(self.attrs)
        # Check for proper area definition.  Non-georeferenced images
        # do not have `area` in the attributes
        if "area" not in img.attrs:
            if self.area is None:
                raise AttributeError("Area definition needs to be configured")
            img.attrs["area"] = self.area
        img.attrs["sensor"] = None
        img.attrs["mode"] = "".join(img.bands.data)
        img.attrs.pop("modifiers", None)
        img.attrs.pop("calibration", None)

        return img
