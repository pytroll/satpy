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
"""Shared objects for writing image-like output."""
from __future__ import annotations

import typing

from satpy.enhancements.enhancer import get_enhanced_image
from satpy.writers.core.base import Writer

if typing.TYPE_CHECKING:
    from trollimage.xrimage import XRImage


class ImageWriter(Writer):
    """Base writer for image file formats."""

    def __init__(self, name=None, filename=None, base_dir=None, enhance=None, **kwargs):
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

                    {platform_name}_{sensor}_{name}_{start_time:%Y%m%d_%H%M%S}.tif

            base_dir (str | Path):
                Base destination directories for all created files.
            enhance (bool or Enhancer): Whether to automatically enhance
                data to be more visually useful and to fit inside the file
                format being saved to. By default, this will default to using
                the enhancement configuration files found using the default
                :class:`~satpy.enhancements.enhancer.Enhancer` class. This can be set to
                `False` so that no enhancements are performed. This can also
                be an instance of the :class:`~satpy.enhancements.enhancer.Enhancer` class
                if further custom enhancement is needed.

            kwargs (dict): Additional keyword arguments to pass to the
                :class:`~satpy.writers.core.base.Writer` base class.

        .. versionchanged:: 0.10

            Deprecated `enhancement_config_file` and 'enhancer' in favor of
            `enhance`. Pass an instance of the `Enhancer` class to `enhance`
            instead.

        """
        super().__init__(name, filename, base_dir, **kwargs)
        if enhance is False:
            # No enhancement
            self.enhancer = False
        elif enhance is None or enhance is True:
            # default enhancement
            from satpy.enhancements.enhancer import Enhancer

            enhancement_config = self.info.get("enhancement_config", None)
            self.enhancer = Enhancer(enhancement_config_file=enhancement_config)
        else:
            # custom enhancer
            self.enhancer = enhance

    @classmethod
    def separate_init_kwargs(cls, kwargs):
        """Separate the init kwargs."""
        # FUTURE: Don't pass Scene.save_datasets kwargs to init and here
        init_kwargs, kwargs = super(ImageWriter, cls).separate_init_kwargs(kwargs)
        for kw in ["enhancement_config", "enhance"]:
            if kw in kwargs:
                init_kwargs[kw] = kwargs.pop(kw)
        return init_kwargs, kwargs

    def save_dataset(self, dataset, filename=None, fill_value=None,
                     overlay=None, decorate=None, compute=True, units=None, **kwargs):
        """Save the ``dataset`` to a given ``filename``.

        This method creates an enhanced image using :func:`~satpy.enhancements.enhancer.get_enhanced_image`.
        The image is then passed to :meth:`save_image`. See both of these
        functions for more details on the arguments passed to this method.

        """
        if units is not None:
            import pint_xarray  # noqa
            dataset = dataset.pint.quantify().pint.to(units).pint.dequantify()
        img = get_enhanced_image(dataset.squeeze(), enhance=self.enhancer, overlay=overlay,
                                 decorate=decorate, fill_value=fill_value)
        return self.save_image(img, filename=filename, compute=compute, fill_value=fill_value, **kwargs)

    def save_image(
            self,
            img: XRImage,
            filename: str | None = None,
            compute: bool = True,
            **kwargs
    ):
        """Save Image object to a given ``filename``.

        Args:
            img (trollimage.xrimage.XRImage): Image object to save to disk.
            filename (str): Optionally specify the filename to save this
                            dataset to. It may include string formatting
                            patterns that will be filled in by dataset
                            attributes.
            compute (bool): If `True` (default), compute and save the dataset.
                            If `False` return either a :doc:`dask:delayed`
                            object or tuple of (source, target). See the
                            return values below for more information.
            **kwargs: Other keyword arguments to pass to this writer.

        Returns:
            Value returned depends on `compute`. If `compute` is `True` then
            the return value is the result of computing a
            :doc:`dask:delayed` object or running :func:`dask.array.store`.
            If `compute` is `False` then the returned value is either a
            :doc:`dask:delayed` object that can be computed using
            `delayed.compute()` or a tuple of (source, target) that should be
            passed to :func:`dask.array.store`. If target is provided the the
            caller is responsible for calling `target.close()` if the target
            has this method.

        """
        raise NotImplementedError("Writer '%s' has not implemented image saving" % (self.name,))
