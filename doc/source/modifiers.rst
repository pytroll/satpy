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

Parallax correction
-------------------

.. warning::

    The Satpy parallax correction is experimental and subject to change.

Since version 0.37 (mid 2022), Satpy has included a
modifier for parallax correction, implemented in the
:class:`~satpy.modifiers.parallax.ParallaxCorrectionModifier` class.
This modifier is important for some applications, but not applied
by default to any Satpy datasets or composites, because it can be
applied to any input dataset and used with any source of (cloud top)
height.  Therefore, users wishing to apply the parallax correction
semi-automagically have to define their own modifier and then apply
that modifier for their datasets.  An example is included
with the :class:`~satpy.modifiers.parallax.ParallaxCorrectionModifier`
API documentation.  Note that Satpy cannot apply modifiers to
composites, so users wishing to apply parallax correction to a composite
will have to use a lower level API or duplicate an existing composite
recipe to use modified inputs.

The parallax correction is directly calculated from the cloud top height.
Information on satellite position is obtained from cloud top height
metadata.  If no orbital parameters are present in the cloud top height
metadata, Satpy will attempt to calculate orbital parameters from the
platform name and start time.  The backup calculation requires skyfield
and astropy to be installed.  If the metadata include neither orbital
parameters nor platform name and start time, parallax calculation will
fail.  Because the cloud top height metadata are used, it is essential
that the cloud top height data are derived from the same platform as
the measurements to be corrected are taken by.

The parallax error moves clouds away from the observer.  Therefore, the
parallax correction shifts clouds in the direction of the observer.  The
space left behind by the cloud will be filled with fill values.  As the
cloud is shifted toward the observer, it may occupy less pixels than before,
because pixels closer to the observer have a smaller surface area.  It can
also be deformed (a "rectangular" cloud may get the shape of a parallelogram).

.. figure:: https://figshare.com/ndownloader/files/36422616/preview/36422616/preview.jpg
   :width: 512
   :height: 512
   :alt: Satellite image without parallax correction.

   SEVIRI view of southern Sweden, 2021-11-30 12:15Z, without parallax correction.
   This is the ``natural_color`` composite as built into Satpy.


.. figure:: https://figshare.com/ndownloader/files/36422613/preview/36422613/preview.jpg
   :width: 512
   :height: 512
   :alt: Satellite image with parallax correction.

   The same satellite view with parallax correction.  The most obvious change
   are the gaps left behind by the parallax correction, shown as black pixels.
   Otherwise it shows that clouds have "moved" south-south-west in the direction
   of the satellite.  To view the images side-by-side or alternating, look at
   `the figshare page <https://figshare.com/articles/figure/20211130121510-Meteosat-11-seviri-sswe-parallax_corrected_natural_color_jpg/20377203>`_

The utility function :func:`~satpy.modifiers.parallax.get_surface_parallax_displacement` allows to calculate the magnitude of the parallax error.  For a cloud with a cloud top height of 10 km:

.. figure:: https://figshare.com/ndownloader/files/36462435/preview/36462435/preview.jpg
   :width: 512
   :height: 512
   :alt: Figure showing magnitude of parallax effect.

   Magnitude of the parallax error for a fictitious cloud with a cloud top
   height of 10 km for the GOES-East (GOES-16) full disc.

The parallax correction is currently experimental and subject to change.
Although it is covered by tests, there may be cases that yield unexpected
or incorrect results.  It does not yet perform any checks that the
provided (cloud top) height covers the area of the dataset for which
the parallax correction shall be applied.

For more general background information and web routines related to the
parallax effect, see also `this collection at the CIMSS website <https://cimss.ssec.wisc.edu/goes/webapps/parallax/>_`.

.. versionadded:: 0.37
