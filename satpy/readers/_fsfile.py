from __future__ import annotations

import os
import pickle  # nosec B403
from functools import total_ordering


@total_ordering
class FSFile(os.PathLike):
    """Implementation of a PathLike file object, that can be opened.

    Giving the filenames to :class:`Scene` with valid transfer protocols will automatically
    use this class so manual usage of this class is needed mainly for fine-grained control.

    This class is made to be used in conjuction with fsspec or s3fs. For example::

        from satpy import Scene

        import fsspec
        filename = 'noaa-goes16/ABI-L1b-RadC/2019/001/17/*_G16_s20190011702186*'

        the_files = fsspec.open_files("simplecache::s3://" + filename, s3={'anon': True})

        from satpy.readers import FSFile
        fs_files = [FSFile(open_file) for open_file in the_files]

        scn = Scene(filenames=fs_files, reader='abi_l1b')
        scn.load(['true_color_raw'])

    """

    def __init__(self, file, fs=None):
        """Initialise the FSFile instance.

        Args:
            file (str, Pathlike, or OpenFile):
                String, object implementing the `os.PathLike` protocol, or
                an `fsspec.OpenFile` instance.  If passed an instance of
                `fsspec.OpenFile`, the following argument ``fs`` has no
                effect.
            fs (fsspec filesystem, optional)
                Object implementing the fsspec filesystem protocol.
        """
        self._fs_open_kwargs = _get_fs_open_kwargs(file)
        try:
            self._file = file.path
            self._fs = file.fs
        except AttributeError:
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

    def open(self, *args, **kwargs):
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
