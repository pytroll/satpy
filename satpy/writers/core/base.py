# Copyright (c) 2025 Satpy developers
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
"""Shared objects and base classes for writers."""
from __future__ import annotations

import logging
import os
import warnings

from satpy.aux_download import DataDownloadMixin
from satpy.plugin_base import Plugin
from satpy.writers.core.compute import compute_writer_results, split_results

LOG = logging.getLogger(__name__)


class Writer(Plugin, DataDownloadMixin):
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

                    {platform_name}_{sensor}_{name}_{start_time:%Y%m%d_%H%M%S}.tif

            base_dir (str):
                Base destination directories for all created files.
            kwargs (dict): Additional keyword arguments to pass to the
                :class:`~satpy.plugin_base.Plugin` class.

        """
        # Load the config
        Plugin.__init__(self, **kwargs)
        self.info = self.config.get("writer", {})

        if "file_pattern" in self.info:
            warnings.warn(
                "Writer YAML config is using 'file_pattern' which "
                "has been deprecated, use 'filename' instead.",
                stacklevel=2
            )
            self.info["filename"] = self.info.pop("file_pattern")

        if "file_pattern" in kwargs:
            warnings.warn(
                "'file_pattern' has been deprecated, use 'filename' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            filename = kwargs.pop("file_pattern")

        # Use options from the config file if they weren't passed as arguments
        self.name = self.info.get("name", None) if name is None else name
        self.file_pattern = self.info.get("filename", None) if filename is None else filename

        if self.name is None:
            raise ValueError("Writer 'name' not provided")

        self.filename_parser = self.create_filename_parser(base_dir)
        self.register_data_files()

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        """Help separating arguments between init and save methods.

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
        for kw in ["base_dir", "filename", "file_pattern"]:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)
        return init_kwargs, kwargs

    def create_filename_parser(self, base_dir):
        """Create a :class:`trollsift.parser.Parser` object for later use."""
        from trollsift import parser

        # just in case a writer needs more complex file patterns
        # Set a way to create filenames if we were given a pattern
        if base_dir and self.file_pattern:
            file_pattern = os.path.join(base_dir, self.file_pattern)
        else:
            file_pattern = self.file_pattern
        return parser.Parser(file_pattern) if file_pattern else None

    @staticmethod
    def _prepare_metadata_for_filename_formatting(attrs):
        if isinstance(attrs.get("sensor"), set):
            attrs["sensor"] = "-".join(sorted(attrs["sensor"]))

    def get_filename(self, **kwargs):
        """Create a filename where output data will be saved.

        Args:
            kwargs (dict): Attributes and other metadata to use for formatting
                the previously provided `filename`.

        """
        if self.filename_parser is None:
            raise RuntimeError("No filename pattern or specific filename provided")
        self._prepare_metadata_for_filename_formatting(kwargs)
        output_filename = self.filename_parser.compose(kwargs)
        dirname = os.path.dirname(output_filename)
        if dirname and not os.path.isdir(dirname):
            LOG.info("Creating output directory: {}".format(dirname))
            os.makedirs(dirname, exist_ok=True)
        return output_filename

    def save_datasets(self, datasets, compute=True, **kwargs):
        """Save all datasets to one or more files.

        Subclasses can use this method to save all datasets to one single
        file or optimize the writing of individual datasets. By default
        this simply calls `save_dataset` for each dataset provided.

        Args:
            datasets (Iterable): Iterable of `xarray.DataArray` objects to
                                 save using this writer.
            compute (bool): If `True` (default), compute all the saves to
                            disk. If `False` then the return value is either
                            a :doc:`dask:delayed` object or two lists to
                            be passed to a :func:`dask.array.store` call.
                            See return values below for more details.
            **kwargs: Keyword arguments to pass to `save_dataset`. See that
                      documentation for more details.

        Returns:
            Value returned depends on `compute` keyword argument. If
            `compute` is `True` the value is the result of either a
            :func:`dask.array.store` operation or a :doc:`dask:delayed`
            compute, typically this is `None`. If `compute` is `False` then
            the result is either a :doc:`dask:delayed` object that can be
            computed with `delayed.compute()` or a two element tuple of
            sources and targets to be passed to :func:`dask.array.store`. If
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
                     compute=True, units=None, **kwargs):
        """Save the ``dataset`` to a given ``filename``.

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
                            If `False` return either a :doc:`dask:delayed`
                            object or tuple of (source, target). See the
                            return values below for more information.
            units (str or None): If not None, will convert the dataset to
                                    the given unit using pint-xarray before
                                    saving. Default is not to do any
                                    conversion.
            **kwargs: Other keyword arguments for this particular writer.

        Returns:
            Value returned depends on `compute`. If `compute` is `True` then
            the return value is the result of computing a
            :doc:`dask:delayed` object or running :func:`dask.array.store`.
            If `compute` is `False` then the returned value is either a
            :doc:`dask:delayed` object that can be computed using
            `delayed.compute()` or a tuple of (source, target) that should be
            passed to :func:`dask.array.store`. If target is provided the
            caller is responsible for calling `target.close()` if the target
            has this method.

        """
        raise NotImplementedError(
            "Writer '%s' has not implemented dataset saving" % (self.name, ))
