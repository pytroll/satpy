#!/usr/bin/python
# Copyright (c) 2015.
#

# Author(s):
#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
"""
"""

import logging

from satpy.writers import ImageWriter

LOG = logging.getLogger(__name__)


class PillowWriter(ImageWriter):
    def __init__(self, **kwargs):
        ImageWriter.__init__(
            self,
            default_config_filename="writers/simple_image.yaml",
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
            **kwargs: Keyword arguments to pass to the images `save` method.

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
        filename = filename or self.get_filename(**img.data.attrs)

        LOG.debug("Saving to image: %s", filename)
        return img.save(filename, compute=compute, **kwargs)
