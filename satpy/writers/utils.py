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
"""Utilities for writers."""
from __future__ import annotations

import logging

LOG = logging.getLogger(__name__)


def get_enhanced_image(dataset, enhance=None, overlay=None, decorate=None,
                       fill_value=None):
    """Get an enhanced version of `dataset` as an :class:`~trollimage.xrimage.XRImage` instance.

    Args:
        dataset (xarray.DataArray): Data to be enhanced and converted to an image.
        enhance (bool or satpy.enhancements.enhancer.Enhancer): Whether to automatically enhance
            data to be more visually useful and to fit inside the file
            format being saved to. By default, this will default to using
            the enhancement configuration files found using the default
            :class:`~satpy.enhancements.enhancer.Enhancer` class. This can be set to
            `False` so that no enhancments are performed. This can also
            be an instance of the :class:`~satpy.enhancements.enhancer.Enhancer` class
            if further custom enhancement is needed.
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
    """
    from trollimage.xrimage import XRImage

    if enhance is False:
        # no enhancement
        enhancer = None
    elif enhance is None or enhance is True:
        # default enhancement
        from satpy.enhancements.enhancer import Enhancer

        enhancer = Enhancer()
    else:
        # custom enhancer
        enhancer = enhance

    # Create an image for enhancement
    img = XRImage(dataset)

    if enhancer is None or enhancer.enhancement_tree is None:
        LOG.debug("No enhancement being applied to dataset")
    else:
        if dataset.attrs.get("sensor", None):
            enhancer.add_sensor_enhancements(dataset.attrs["sensor"])

        enhancer.apply(img, **dataset.attrs)

    if overlay is not None:
        from satpy.writers.overlay_utils import add_overlay

        img = add_overlay(img, dataset.attrs["area"], fill_value=fill_value, **overlay)

    if decorate is not None:
        from satpy.writers.overlay_utils import add_decorate

        img = add_decorate(img, fill_value=fill_value, **decorate)

    return img
