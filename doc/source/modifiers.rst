Modifiers
=========

Modifiers are filters applied to datasets prior to computing composites.
They take at least one input (a dataset) and have exactly one output
(the same dataset, modified). They can take additional input datasets
or parameters.

Modifiers are defined in composites files in ``etc/composites`` within
``$SATPY_CONFIG_PATH``.

The instruction to use a certain modifier can be contained in a composite
definition or in a reader definition. If it is defined in a composite
definition, it is applied upon constructing the composite.

When using built-in composites, Satpy users do not need to understand
the mechanics of modifiers, as they are applied automatically.
The :doc:`composites` documentation contains information on how to apply
modifiers when creating new composites.

Some readers read data where certain modifiers are already applied. Here,
the reader definition will refer to the Satpy modifier. This marking
adds the modifier to the metadata to prevent it from being applied again
upon composite calculation.

Commonly used modifiers are listed in the table below. Further details
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

A complete list can be found in the `etc/composites
<https://github.com/pytroll/satpy/tree/main/satpy/etc/composites>`_
source code and in the :mod:`~satpy.modifiers` module documentation.
