"""Composite classes for spectral adjustments."""

import logging
import warnings

import numpy as np

from satpy.composites.core import GenericCompositor
from satpy.dataset import combine_metadata

LOG = logging.getLogger(__name__)


class SpectralBlender(GenericCompositor):
    """Construct new channel by blending contributions from a set of channels.

    This class can be used to compute weighted average of different channels.
    Primarily it's used to correct the green band of AHI and FCI in order to
    allow for proper true color imagery.

    Below is an example used to generate a corrected green channel for AHI using a weighted average from
    three channels, with 63% contribution from the native green channel (B02), 29% from the red channel (B03)
    and 8% from the near-infrared channel (B04)::

      corrected_green:
        compositor: !!python/name:satpy.composites.spectral.SpectralBlender
        fractions: [0.63, 0.29, 0.08]
        prerequisites:
          - name: B02
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B03
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B04
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    Other examples can be found in the``ahi.yaml`` composite file in the satpy distribution.
    """

    def __init__(self, *args, fractions=(), **kwargs):
        """Set default keyword argument values."""
        self.fractions = fractions
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Blend channels in projectables using the weights in self.fractions."""
        if len(self.fractions) != len(projectables):
            raise ValueError("fractions and projectables must have the same length.")

        projectables = self.match_data_arrays(projectables)
        new_channel = sum(fraction * value for fraction, value in zip(self.fractions, projectables))
        new_channel.attrs = combine_metadata(*projectables)
        return super().__call__((new_channel,), **attrs)


class HybridGreen(SpectralBlender):
    """Corrector of the FCI or AHI green band.

    The green band in FCI and AHI (and other bands centered at 0.51 microns) deliberately
    misses the chlorophyll spectral reflectance local maximum at 0.55 microns
    in order to focus on aerosol and ash rather than on vegetation. This
    affects true colour RGBs, because vegetation looks brown rather than green
    and barren surface types typically gets a reddish hue.

    To correct for this the hybrid green approach proposed by Miller et al. (2016, :doi:`10.1175/BAMS-D-15-00154.2`)
    is used. The basic idea is to include some contribution also from the 0.86 micron
    channel, which is known for its sensitivity to vegetation. The formula used for this is::

      hybrid_green = (1 - F) * R(0.51) + F * R(0.86)

    where F is a constant value, that is set to 0.15 by default in Satpy.

    For example, the HybridGreen compositor can be used as follows to construct a hybrid green channel for
    AHI, with 15% contibution from the near-infrared 0.85 µm band (B04) and the remaining 85% from the native
    green 0.51 µm band (B02)::

      hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.HybridGreen
        fraction: 0.15
        prerequisites:
          - name: B02
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: B04
            modifiers: [sunz_corrected, rayleigh_corrected]
        standard_name: toa_bidirectional_reflectance

    Other examples can be found in the ``ahi.yaml`` and ``ami.yaml`` composite
    files in the satpy distribution.
    """

    def __init__(self, *args, fraction=0.15, **kwargs):
        """Set default keyword argument values."""
        fractions = (1 - fraction, fraction)
        super().__init__(fractions=fractions, *args, **kwargs)


class NDVIHybridGreen(SpectralBlender):
    """Construct a NDVI-weighted hybrid green channel.

    This green band correction method follows a similar approach as the HybridGreen compositor, but uses a dynamic blend
    factor `f` that depends on the pixel-level Normalized Difference Vegetation Index (NDVI). The NIR contribution
    decreases with increasing NDVI, following a third-order polynomial derived from one year of collocated FCI and OLCI
    observations. Pixels with high NDVI (>= 0.9) have an NIR contribution of approximately 4%, while pixels with low
    NDVI (<= 0.1) have an NIR contribution of approximately 28%. See Strandgren et al. (2026, in prep.) for details on
    the NDVI-based hybrid green correction method.

    As an example, a new green channel using e.g. FCI data can be defined like this::

      ndvi_hybrid_green:
        compositor: !!python/name:satpy.composites.spectral.NDVIHybridGreen
        prerequisites:
          - name: vis_05
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_06
            modifiers: [sunz_corrected, rayleigh_corrected]
          - name: vis_08
            modifiers: [sunz_corrected ]
        standard_name: toa_bidirectional_reflectance


    """

    def __init__(self, *args, **kwargs):
        """Initialize class and set the NDVI limits and regression coefficients for the correction.

        Also issue warning if any deprecated arguments from earlier implementation of the correction
        are used.
        """
        deprecated_args = {"ndvi_min", "ndvi_max", "limits", "strength"}
        for name in deprecated_args & kwargs.keys():
            warnings.warn(
                f"'{name}' has been deprecated for the NDVIHybridGreen Compositor and will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            kwargs.pop(name)

        self.ndvi_min = 0.1
        self.ndvi_max = 0.9
        self.poly_coefs = (-1.0349, 2.0206, -1.3701, 0.3940)
        super().__init__(*args, **kwargs)

    def __call__(self, projectables, optional_datasets=None, **attrs):
        """Construct the NDVI hybrid green channel."""
        LOG.info("Applying NDVI hybrid green correction.")
        projectables = self.match_data_arrays(projectables)

        ndvi = (projectables[2] - projectables[1]) / (projectables[2] + projectables[1])

        ndvi = ndvi.clip(self.ndvi_min, self.ndvi_max)
        # dask nan_to_num does not accept kwargs (see https://github.com/dask/dask/issues/12350)
        # Second argument maps to `copy` kwarg, third maps to `nan` kwarg.
        # Copy should remain `True` as dask operations require copies to be made
        ndvi.data = np.nan_to_num(ndvi.data, True, self.ndvi_min)

        # Compute pixel-level NIR blend fractions from NDVI using third-order polynomial
        coef_3, coef_2, coef_1, coef_0 = self.poly_coefs
        fraction = ((coef_3 * ndvi + coef_2) * ndvi + coef_1) * ndvi + coef_0
        fraction = fraction.clip(0.0, 1.0)

        # Prepare input as required by parent class (SpectralBlender)
        self.fractions = (1 - fraction, fraction)

        return super().__call__([projectables[0], projectables[2]], **attrs)


# TODO: Turn this into a weighted RGB compositor
class NaturalEnh(GenericCompositor):
    """Enhanced version of natural color composite by Simon Proud.

    Args:
        ch16_w (float): weight for red channel (1.6 um). Default: 1.3
        ch08_w (float): weight for green channel (0.8 um). Default: 2.5
        ch06_w (float): weight for blue channel (0.6 um). Default: 2.2

    """

    def __init__(self, name, ch16_w=1.3, ch08_w=2.5, ch06_w=2.2,
                 *args, **kwargs):
        """Initialize the class."""
        self.ch06_w = ch06_w
        self.ch08_w = ch08_w
        self.ch16_w = ch16_w
        super(NaturalEnh, self).__init__(name, *args, **kwargs)

    def __call__(self, projectables, *args, **kwargs):
        """Generate the composite."""
        projectables = self.match_data_arrays(projectables)
        ch16 = projectables[0]
        ch08 = projectables[1]
        ch06 = projectables[2]

        ch1 = self.ch16_w * ch16 + self.ch08_w * ch08 + self.ch06_w * ch06
        ch1.attrs = ch16.attrs
        ch2 = ch08
        ch3 = ch06

        return super(NaturalEnh, self).__call__((ch1, ch2, ch3),
                                                *args, **kwargs)
