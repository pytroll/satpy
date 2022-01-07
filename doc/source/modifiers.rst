=========
Modifiers
=========

Modifiers are filters applied to datasets.  They take at least one input
(a dataset) and have exactly one output (the same dataset, modified).  They
can take additional input datasets or parameters.

Modifiers are defined in composites files in ``etc/composites`` within
``$SATPY_CONFIG_PATH``.

The instruction to use a certain modifier can be contained in a composite
definition or in a reader definition.  If it is defined in a reader,
it is always applied.  If it is defined in a composite definition,
it is applied upon constructing the composite.

When using built-in composites, Satpy users do not need to understand
the mechanics of modifiers, as they are applied automatically.
The :doc:`composites` documentation contains information on how to apply
modifiers when creating new composites.

Commonly used modifiers are listed in the table below.  Further details
on those modifiers can be found in the linked API documentation.

.. list-table:: Commonly used modifiers
    :header-rows: 1

    * - Label
      - Class
      - Description
    * - ``sunz_corrected``
      - :class:`~satpy.modifiers.geometry.SunZenithCorrector`
      - Modifies solar channels for the solar zenith angle to provide
        smoother images.
    * - ``effective_solar_pathlength_corrected``
      - :class:`~satpy.modifiers.geometry.EffectiveSolarPathLengthCorrector`
      - Modifies solar channels for atmospheric path length of solar radiation.
    * - ``nir_reflectance``
      - :class:`~satpy.modifiers.spectral.NIRReflectance`
      - Calculates reflective part of channels at the edge of solar and
        terrestrial radiation (3.7 µm or 3.9 µm).
    * - ``nir_emissive``
      - :class:`~satpy.modifiers.spectral.NIREmissivePartFromReflectance`
      - Calculates emissive part of channels at the edge of solar and terrestrial
        radiation (3.7 µm or 3.9 µm)
    * - ``rayleigh_corrected``
      - :class:`~satpy.modifiers.atmosphere.PSPRayleighReflectance`
      - Modifies solar channels to filter out the visual impact of rayleigh
        scattering.

A complete list can be found in the ``etc/composites`` source code and
in the :mod:`~satpy.modifiers` module documentation.

Parallax correction
-------------------

Since early 2022, Satpy has included a
modifier for parallax correction, implemented in the
:class:`~satpy.modifiers.parallax.ParallaxCorrectionModifier` class.
This modifier is important for some applications, but not applied
by default to any Satpy datasets or composites, because it can be
applied to any input dataset and used with any source of (cloud top)
height.  Therefore, users wishing to apply the parallax correction
semi-automagically have to define their own modifier and then apply
that modifier for their composites or datasets.  An example is included
with the :class:`~satpy.modifiers.parallax.ParallaxCorrectionModifier`
API documentation.
