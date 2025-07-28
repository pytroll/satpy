#!/usr/bin/env python
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
"""Helper functions for remote reading."""

import os
import pathlib
import pickle  # nosec B403
from functools import total_ordering

import fsspec


@total_ordering
class FSFile(os.PathLike):
    """Implementation of a PathLike file object, that can be opened.

    Giving the filenames to :class:`Scene <satpy.scene.Scene>` with valid transfer protocols will automatically
    use this class so manual usage of this class is needed mainly for fine-grained control.

    This class is made to be used in conjuction with fsspec or s3fs. For example::

        from satpy import Scene

        import fsspec
        filename = 'noaa-goes16/ABI-L1b-RadC/2019/001/17/*_G16_s20190011702186*'

        the_files = fsspec.open_files("simplecache::s3://" + filename, s3={'anon': True})

        from satpy.readers.core.remote import FSFile
        fs_files = [FSFile(open_file) for open_file in the_files]

        scn = Scene(filenames=fs_files, reader='abi_l1b')
        scn.load(['true_color_raw'])

    """

    def __init__(
            self,
            file: os.PathLike | fsspec.core.OpenFile | str,
            fs: fsspec.spec.AbstractFileSystem | None = None,
    ):
        """Initialise the FSFile instance.

        Args:
            file:
                String, object implementing the :class:`os.PathLike` protocol, or
                an :class:`~fsspec.core.OpenFile` instance.  If passed an instance of
                :class:`~fsspec.core.OpenFile`, the following argument ``fs`` has no
                effect.
            fs:
                Object implementing the fsspec filesystem protocol.
        """
        self._fs_open_kwargs = _get_fs_open_kwargs(file)
        if hasattr(file, "path") and hasattr(file, "fs"):
            self._file = file.path
            self._fs = file.fs
        else:
            self._file = file
            self._fs = fs

    def __str__(self):
        """Return the string version of the filename."""
        return os.fspath(self._file)

    def __fspath__(self):
        """Comply with PathLike."""
        return os.fspath(self._file)

    def __repr__(self):
        """Representation of the object."""
        return '<FSFile "' + str(self._file) + '">'

    @property
    def fs(self):
        """Return the underlying private filesystem attribute."""
        return self._fs

    def open(self, *args, **kwargs):  # noqa: A003
        """Open the file.

        This is read-only.
        """
        fs_open_kwargs = self._update_with_fs_open_kwargs(kwargs)
        try:
            return self._fs.open(self._file, *args, **fs_open_kwargs)
        except AttributeError:
            return open(self._file, *args, **kwargs)

    def _update_with_fs_open_kwargs(self, user_kwargs):
        """Complement keyword arguments for opening a file via file system."""
        kwargs = user_kwargs.copy()
        kwargs.update(self._fs_open_kwargs)
        return kwargs

    def __lt__(self, other):
        """Implement ordering.

        Ordering is defined by the string representation of the filename,
        without considering the file system.
        """
        return os.fspath(self) < os.fspath(other)

    def __eq__(self, other):
        """Implement equality comparisons.

        Two FSFile instances are considered equal if they have the same
        filename and the same file system.
        """
        return (isinstance(other, FSFile) and
                self._file == other._file and
                self._fs == other._fs)

    def __hash__(self):
        """Implement hashing.

        Make FSFile objects hashable, so that they can be used in sets.  Some
        parts of satpy and perhaps others use sets of filenames (strings or
        pathlib.Path), or maybe use them as dictionary keys.  This requires
        them to be hashable.  To ensure FSFile can work as a drop-in
        replacement for strings of Path objects to represent the location of
        blob of data, FSFile should be hashable too.

        Returns the hash, computed from the hash of the filename and the hash
        of the filesystem.
        """
        try:
            fshash = hash(self._fs)
        except TypeError:  # fsspec < 0.8.8 for CachingFileSystem
            fshash = hash(pickle.dumps(self._fs))  # nosec B403
        return hash(self._file) ^ fshash


def _get_fs_open_kwargs(file):
    """Get keyword arguments for opening a file via file system.

    For example compression.
    """
    return {
        "compression": _get_compression(file)
    }


def _get_compression(file):
    try:
        return file.compression
    except AttributeError:
        return None


def open_file_or_filename(unknown_file_thing, mode=None):
    """Try to open the provided file "thing" if needed, otherwise return the filename or Path.

    This wraps the logic of getting something like an fsspec OpenFile object
    that is not directly supported by most reading libraries and making it
    usable. If a :class:`pathlib.Path` object or something that is not
    open-able is provided then that object is passed along. In the case of
    fsspec OpenFiles their ``.open()`` method is called and the result returned.

    """
    if isinstance(unknown_file_thing, pathlib.Path):
        f_obj = unknown_file_thing
    else:
        try:
            if mode is None:
                f_obj = unknown_file_thing.open()
            else:
                f_obj = unknown_file_thing.open(mode=mode)
        except AttributeError:
            f_obj = unknown_file_thing
    return f_obj
