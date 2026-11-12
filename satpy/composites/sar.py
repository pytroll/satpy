"""Composite classes for the VIIRS instrument."""

import logging

import numpy as np

from satpy.composites.core import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


def overlay(top, bottom, maxval=None):
    """Blending two layers.

    from: https://docs.gimp.org/en/gimp-concepts-layer-modes.html
    """
    if maxval is None:
        maxval = np.maximum(top.max(), bottom.max())

    res = ((2 * top / maxval - 1) * bottom + 2 * top) * bottom / maxval
    return res.clip(min=0)


def soft_light(top, bottom, maxval):
    """Apply soft light.

    http://www.pegtop.net/delphi/articles/blendmodes/softlight.htm
    """
    a = top / maxval
    b = bottom / maxval
    return (2*a*b + a*a * (1 - 2*b)) * maxval


class SARIce(GenericCompositor):
    """The SAR Ice composite."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR Ice composite."""
        (mhh, mhv) = projectables
        ch1attrs = mhh.attrs
        ch2attrs = mhv.attrs
        mhh = np.sqrt(mhh + 0.002) - 0.04
        mhv = np.sqrt(mhv + 0.002) - 0.04
        mhh.attrs = ch1attrs
        mhv.attrs = ch2attrs
        green = overlay(mhh, mhv, 30) * 1000
        green.attrs = combine_metadata(mhh, mhv)

        return super(SARIce, self).__call__((mhv, green, mhh), *args, **kwargs)


def _square_root_channels(*projectables):
    """Return the square root of the channels, preserving the attributes."""
    results = []
    for projectable in projectables:
        attrs = projectable.attrs
        projectable = np.sqrt(projectable)
        projectable.attrs = attrs
        results.append(projectable)
    return results


class SARIceLegacy(GenericCompositor):
    """The SAR Ice composite, legacy version with dynamic stretching."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR RGB composite."""
        mhh, mhv = _square_root_channels(*projectables)
        green = overlay(mhh, mhv)
        green.attrs = combine_metadata(mhh, mhv)

        return super(SARIceLegacy, self).__call__((mhv, green, mhh), *args, **kwargs)


class SARIceLog(GenericCompositor):
    """The SAR Ice composite, using log-scale data."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR Ice Log composite."""
        mhh, mhv = projectables
        mhh = mhh.clip(-40)
        mhv = mhv.clip(-38)
        green = soft_light(mhh + 100, mhv + 100, 100) - 100
        green.attrs = combine_metadata(mhh, mhv)

        return super().__call__((mhv, green, mhh), *args, **kwargs)


class SARRGB(GenericCompositor):
    """The SAR RGB composite."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR RGB composite."""
        mhh, mhv = _square_root_channels(*projectables)
        green = overlay(mhh, mhv)
        green.attrs = combine_metadata(mhh, mhv)

        return super(SARRGB, self).__call__((-mhv, -green, -mhh), *args, **kwargs)


class SARQuickLook(GenericCompositor):
    """The SAR QuickLook composite."""

    def __call__(self, projectables, *args, **kwargs):
        """Create the SAR QuickLook composite."""
        mhh, mhv = _square_root_channels(*projectables)

        blue = mhv / mhh
        blue.attrs = combine_metadata(mhh, mhv)

        return super(SARQuickLook, self).__call__((mhh, mhv, blue), *args, **kwargs)
