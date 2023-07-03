==========
Composites
==========

Composites are defined as arrays of data that are created by processing and/or
combining one or multiple data arrays (prerequisites) together.

Composites are generated in satpy using Compositor classes. The attributes of the
resulting composites are usually a combination of the prerequisites' attributes and
the key/values of the DataID used to identify it.

Built-in Compositors
====================

.. py:currentmodule:: satpy.composites

There are many built-in compositors available in Satpy.
The majority use the :class:`GenericCompositor` base class
which handles various image modes (`L`, `LA`, `RGB`, and
`RGBA` at the moment) and updates attributes.

The below sections summarize the composites that come with Satpy and
show basic examples of creating and using them with an existing
:class:`~satpy.scene.Scene` object. It is recommended that any composites
that are used repeatedly be configured in YAML configuration files.
General-use compositor code dealing with visible or infrared satellite
data can be put in a configuration file called ``visir.yaml``. Composites
that are specific to an instrument can be placed in YAML config files named
accordingly (e.g., ``seviri.yaml`` or ``viirs.yaml``). See the
`satpy repository <https://github.com/pytroll/satpy/tree/main/satpy/etc/composites>`_
for more examples.

GenericCompositor
-----------------

:class:`GenericCompositor` class can be used to create basic single
channel and RGB composites. For example, building an overview composite
can be done manually within Python code with::

    >>> from satpy.composites import GenericCompositor
    >>> compositor = GenericCompositor("overview")
    >>> composite = compositor([local_scene[0.6],
    ...                         local_scene[0.8],
    ...                         local_scene[10.8]])

One important thing to notice is that there is an internal difference
between a composite and an image. A composite is defined as a special
dataset which may have several bands (like `R`, `G` and `B`  bands). However,
the data isn't stretched, or clipped or gamma filtered until an image
is generated.  To get an image out of the above composite::

    >>> from satpy.writers import to_image
    >>> img = to_image(composite)
    >>> img.invert([False, False, True])
    >>> img.stretch("linear")
    >>> img.gamma(1.7)
    >>> img.show()

This part is called `enhancement`, and is covered in more detail in
:doc:`enhancements`.

Single channel composites can also be generated with the
:class:`GenericCompositor`, but in some cases, the
:class:`SingleBandCompositor` may be more appropriate.  For example,
the :class:`GenericCompositor` removes attributes such as ``units``
because they are typically not meaningful for an RGB image.  Such attributes
are retained in the :class:`SingleBandCompositor`.

DifferenceCompositor
--------------------

:class:`DifferenceCompositor` calculates a difference of two datasets::

    >>> from satpy.composites import DifferenceCompositor
    >>> compositor = DifferenceCompositor("diffcomp")
    >>> composite = compositor([local_scene[10.8], local_scene[12.0]])

FillingCompositor
-----------------

:class:`FillingCompositor`:: fills the missing values in three datasets
with the values of another dataset:::

    >>> from satpy.composites import FillingCompositor
    >>> compositor = FillingCompositor("fillcomp")
    >>> filler = local_scene[0.6]
    >>> data_with_holes_1 = local_scene['ch_a']
    >>> data_with_holes_2 = local_scene['ch_b']
    >>> data_with_holes_3 = local_scene['ch_c']
    >>> composite = compositor([filler, data_with_holes_1, data_with_holes_2,
    ...                         data_with_holes_3])

PaletteCompositor
------------------

:class:`PaletteCompositor` creates a color version of a single channel
categorical dataset using a colormap::

    >>> from satpy.composites import PaletteCompositor
    >>> compositor = PaletteCompositor("palcomp")
    >>> composite = compositor([local_scene['cma'], local_scene['cma_pal']])

The palette should have a single entry for all the (possible) values
in the dataset mapping the value to an RGB triplet.  Typically the
palette comes with the categorical (e.g. cloud mask) product that is
being visualized.

.. deprecated:: 0.40

   Composites produced with :class:`PaletteCompositor` will result in
   an image with mode RGB when enhanced.  To produce an image with mode P, use
   the :class:`SingleBandCompositor` with an associated
   :func:`~satpy.enhancements.palettize` enhancement and pass ``keep_palette=True``
   to :meth:`~satpy.Scene.save_datasets`.  If the colormap is sourced from
   the same dataset as the dataset to be palettized, it must be contained
   in the auxiliary datasets.

   Since Satpy 0.40, all built-in composites that used
   :class:`PaletteCompositor` have been migrated to use
   :class:`SingleBandCompositor` instead.  This has no impact on resulting
   images unless ``keep_palette=True`` is passed to
   :meth:`~satpy.Scene.save_datasets`, but the loaded composite now has only
   one band (previously three).

DayNightCompositor
------------------

:class:`DayNightCompositor` merges two different composites.  The
first composite will be placed on the day-side of the scene, and the
second one on the night side.  The transition from day to night is
done by calculating solar zenith angle (SZA) weighed average of the
two composites.  The SZA can optionally be given as third dataset, and
if not given, the angles will be calculated.  Four arguments are used
to generate the image (default values shown in the example below).
They can be defined when initializing the compositor::

 - lim_low (float): lower limit of Sun zenith angle for the
                    blending of the given channels
 - lim_high (float): upper limit of Sun zenith angle for the
                     blending of the given channels
                     Together with `lim_low` they define the width
                     of the blending zone
 - day_night (string): "day_night" means both day and night portions will be kept
                       "day_only" means only day portion will be kept
                       "night_only" means only night portion will be kept
 - include_alpha (bool): This only affects the "day only" or "night only" result.
                         True means an alpha band will be added to the output image for transparency.
                         False means the output is a single-band image with undesired pixels being masked out
                         (replaced with NaNs).

Usage (with default values)::

    >>> from satpy.composites import DayNightCompositor
    >>> compositor = DayNightCompositor("dnc", lim_low=85., lim_high=88., day_night="day_night")
    >>> composite = compositor([local_scene['true_color'],
    ...                         local_scene['night_fog']])

As above, with `day_night` flag it is also available to use only
a day product or only a night product and mask out (make transparent)
the opposite portion of the image (night or day). The example below
provides only a day product with night portion masked-out::

    >>> from satpy.composites import DayNightCompositor
    >>> compositor = DayNightCompositor("dnc", lim_low=85., lim_high=88., day_night="day_only")
    >>> composite = compositor([local_scene['true_color'])

By default, the image under `day_only` or `night_only` flag will come out
with an alpha band to display its transparency. It could be changed by
setting `include_alpha` to False if there's no need for that alpha band.
In such cases, it is recommended to use it together with `fill_value=0`
when saving to geotiff to get a single-band image with black background.
In the case below, the image shows its day portion and day/night
transition with night portion blacked-out instead of transparent::

    >>> from satpy.composites import DayNightCompositor
    >>> compositor = DayNightCompositor("dnc", lim_low=85., lim_high=88., day_night="day_only", include_alpha=False)
    >>> composite = compositor([local_scene['true_color'])

RealisticColors
---------------

:class:`RealisticColors` compositor is a special compositor that is
used to create realistic near-true-color composite from MSG/SEVIRI
data::

    >>> from satpy.composites import RealisticColors
    >>> compositor = RealisticColors("realcols", lim_low=85., lim_high=95.)
    >>> composite = compositor([local_scene['VIS006'],
    ...                         local_scene['VIS008'],
    ...                         local_scene['HRV']])

CloudCompositor
---------------

:class:`CloudCompositor` can be used to threshold the data so that
"only" clouds are visible.  These composites can be used as an overlay
on top of e.g. static terrain images to show a rough idea where there
are clouds.  The data are thresholded using three variables::

 - `transition_min`: values below or equal to this are clouds -> opaque white
 - `transition_max`: values above this are cloud free -> transparent
 - `transition_gamma`: gamma correction applied to clarify the clouds

Usage (with default values)::

    >>> from satpy.composites import CloudCompositor
    >>> compositor = CloudCompositor("clouds", transition_min=258.15,
    ...                              transition_max=298.15,
    ...                              transition_gamma=3.0)
    >>> composite = compositor([local_scene[10.8]])

Support for using this compositor for VIS data, where the values for
high/thick clouds tend to be in reverse order to brightness
temperatures, is to be added.

RatioSharpenedRGB
-----------------

:class:`RatioSharpenedRGB`

SelfSharpenedRGB
----------------

:class:`SelfSharpenedRGB` sharpens the RGB with ratio of a band with a
strided version of itself.

LuminanceSharpeningCompositor
-----------------------------

:class:`LuminanceSharpeningCompositor` replaces the luminance from an
RGB composite with luminance created from reflectance data.  If the
resolutions of the reflectance data _and_ of the target area
definition are higher than the base RGB, more details can be
retrieved.  This compositor can be useful also with matching
resolutions, e.g. to highlight shadowing at cloudtops in colorized
infrared composite.

    >>> from satpy.composites import LuminanceSharpeningCompositor
    >>> compositor = LuminanceSharpeningCompositor("vis_sharpened_ir")
    >>> vis_data = local_scene['HRV']
    >>> colorized_ir_clouds = local_scene['colorized_ir_clouds']
    >>> composite = compositor([vis_data, colorized_ir_clouds])

SandwichCompositor
------------------

Similar to :class:`LuminanceSharpeningCompositor`,
:class:`SandwichCompositor` uses reflectance data to bring out more
details out of infrared or low-resolution composites.
:class:`SandwichCompositor` multiplies the RGB channels with (scaled)
reflectance.

    >>> from satpy.composites import SandwichCompositor
    >>> compositor = SandwichCompositor("ir_sandwich")
    >>> vis_data = local_scene['HRV']
    >>> colorized_ir_clouds = local_scene['colorized_ir_clouds']
    >>> composite = compositor([vis_data, colorized_ir_clouds])

StaticImageCompositor
---------------------

    :class:`StaticImageCompositor` can be used to read an image from disk
    and used just like satellite data, including resampling and using as a
    part of other composites.

    >>> from satpy.composites import StaticImageCompositor
    >>> compositor = StaticImageCompositor("static_image", filename="image.tif")
    >>> composite = compositor()

BackgroundCompositor
--------------------

    :class:`BackgroundCompositor` can be used to stack two composites
    together.  If the composites don't have `alpha` channels, the
    `background` is used where `foreground` has no data.  If `foreground`
    has alpha channel, the `alpha` values are used to weight when blending
    the two composites.

    >>> from satpy import Scene
    >>> from satpy.composites import BackgroundCompositor
    >>> compositor = BackgroundCompositor()
    >>> clouds = local_scene['ir_cloud_day']
    >>> background = local_scene['overview']
    >>> composite = compositor([clouds, background])

CategoricalDataCompositor
-------------------------

:class:`CategoricalDataCompositor` can be used to recategorize categorical data. This is for example useful to
combine comparable categories into a common category. The category remapping from `data` to `composite` is done
using a look-up-table (`lut`)::

    composite = [[lut[data[0,0]], lut[data[0,1]], lut[data[0,Nj]]],
                 [[lut[data[1,0]], lut[data[1,1]], lut[data[1,Nj]],
                 [[lut[data[Ni,0]], lut[data[Ni,1]], lut[data[Ni,Nj]]]

Hence, `lut` must have a length that is greater than the maximum value in `data` in orer to avoid an `IndexError`.
Below is an example on how to create a binary clear-sky/cloud mask from a pseodu cloud type product with six
categories representing clear sky (cat1/cat5), cloudy features (cat2-cat4) and missing/undefined data (cat0)::

    >>> cloud_type = local_scene['cloud_type']  # 0 - cat0, 1 - cat1, 2 - cat2, 3 - cat3, 4 - cat4, 5 - cat5,
    # categories: 0    1  2  3  4  5
    >>> lut = [np.nan, 0, 1, 1, 1, 0]
    >>> compositor = CategoricalDataCompositor('binary_cloud_mask', lut=lut)
    >>> composite = compositor([cloud_type])  # 0 - cat1/cat5, 1 - cat2/cat3/cat4, nan - cat0


Creating composite configuration files
======================================

To save the custom composite, follow the :ref:`component_configuration`
documentation. Once your component configuration directory is created
you can create your custom composite YAML configuration files.
Compositors that can be used for multiple instruments can be placed in the
generic ``$SATPY_CONFIG_PATH/composites/visir.yaml`` file. Composites that
are specific to one sensor should be placed in
``$SATPY_CONFIG_PATH/composites/<sensor>.yaml``. Custom enhancements for your new
composites can be stored in ``$SATPY_CONFIG_PATH/enhancements/generic.yaml`` or
``$SATPY_CONFIG_PATH/enhancements/<sensor>.yaml``.

With that, you should be able to load your new composite directly. Example
configuration files can be found in the satpy repository as well as a few
simple examples below.

Simple RGB composite
--------------------

This is the overview composite shown in the first code example above
using :class:`GenericCompositor`::

    sensor_name: visir

    composites:
      overview:
        compositor: !!python/name:satpy.composites.GenericCompositor
        prerequisites:
        - 0.6
        - 0.8
        - 10.8
        standard_name: overview

For an instrument specific version (here MSG/SEVIRI), we should use
the channel _names_ instead of wavelengths.  Note also that the
sensor_name is now combination of visir and seviri, which means that
it extends the generic visir composites::

    sensor_name: visir/seviri

    composites:

      overview:
        compositor: !!python/name:satpy.composites.GenericCompositor
        prerequisites:
        - VIS006
        - VIS008
        - IR_108
        standard_name: overview

In the following examples only the composite receipes are shown, and
the header information (sensor_name, composites) and intendation needs
to be added.

Using modifiers
---------------

In many cases the basic datasets that go into the composite need to be
adjusted, e.g. for Solar zenith angle normalization.  These modifiers
can be applied in the following way::

      overview:
        compositor: !!python/name:satpy.composites.GenericCompositor
        prerequisites:
        - name: VIS006
          modifiers: [sunz_corrected]
        - name: VIS008
          modifiers: [sunz_corrected]
        - IR_108
        standard_name: overview

Here we see two changes:

1. channels with modifiers need to have either `name` or `wavelength`
   added in front of the channel name or wavelength, respectively
2. a list of modifiers attached to the dictionary defining the channel

The modifier above is a built-in that normalizes the Solar zenith
angle to Sun being directly at the zenith.

More examples can be found in Satpy source code directory
`satpy/etc/composites <https://github.com/pytroll/satpy/tree/main/satpy/etc/composites>`_.

See the :doc:`modifiers` documentation for more information on
available built-in modifiers.

Using other composites
----------------------

Often it is handy to use other composites as a part of the composite.
In this example we have one composite that relies on solar channels on
the day side, and another for the night side::

    natural_with_night_fog:
      compositor: !!python/name:satpy.composites.DayNightCompositor
      prerequisites:
        - natural_color
        - night_fog
      standard_name: natural_with_night_fog

This compositor has three additional keyword arguments that can be
defined (shown with the default values, thus identical result as
above)::

    natural_with_night_fog:
      compositor: !!python/name:satpy.composites.DayNightCompositor
      prerequisites:
        - natural_color
        - night_fog
      lim_low: 85.0
      lim_high: 88.0
      day_night: "day_night"
      standard_name: natural_with_night_fog

Defining other composites in-line
---------------------------------

It is also possible to define sub-composites in-line.  This example is
the built-in airmass composite::

    airmass:
      compositor: !!python/name:satpy.composites.GenericCompositor
      prerequisites:
      - compositor: !!python/name:satpy.composites.DifferenceCompositor
        prerequisites:
        - wavelength: 6.2
        - wavelength: 7.3
      - compositor: !!python/name:satpy.composites.DifferenceCompositor
        prerequisites:
          - wavelength: 9.7
          - wavelength: 10.8
      - wavelength: 6.2
      standard_name: airmass

Using a pre-made image as a background
--------------------------------------

Below is an example composite config using
:class:`StaticImageCompositor`, :class:`DayNightCompositor`,
:class:`CloudCompositor` and :class:`BackgroundCompositor` to show how
to create a composite with a blended day/night imagery as background
for clouds.  As the images are in PNG format, and thus not
georeferenced, the name of the area definition for the background
images are given.  When using GeoTIFF images the `area` parameter can
be left out.

.. note::

    The background blending uses the current time if there is no
    timestamps in the image filenames.

::

    clouds_with_background:
      compositor: !!python/name:satpy.composites.BackgroundCompositor
      standard_name: clouds_with_background
      prerequisites:
        - ir_cloud_day
        - compositor: !!python/name:satpy.composites.DayNightCompositor
          prerequisites:
            - static_day
            - static_night

    static_day:
      compositor: !!python/name:satpy.composites.StaticImageCompositor
      standard_name: static_day
      filename: /path/to/day_image.png
      area: euro4

    static_night:
      compositor: !!python/name:satpy.composites.StaticImageCompositor
      standard_name: static_night
      filename: /path/to/night_image.png
      area: euro4

To ensure that the images aren't auto-stretched and possibly altered,
the following should be added to enhancement config (assuming 8-bit
image) for both of the static images::

    static_day:
      standard_name: static_day
      operations:
      - name: stretch
        method: !!python/name:satpy.enhancements.stretch
        kwargs:
          stretch: crude
          min_stretch: [0, 0, 0]
          max_stretch: [255, 255, 255]

.. _enhancing-the-images:

Enhancing the images
====================

.. todo::

    Explain how composite names, composite standard_name, enhancement
    names, and enhancement standard_name are related to each other

    Explain what happens when no enhancement is configured for a
    product (= use the default enhancement).

    Explain that the methods are often just a wrapper for XRImage
    methods, but can also be something completely custom.

    List and explain in detail the built-in enhancements:

    - stretch
    - gamma
    - invert
    - cira_stretch
    - lookup
    - colorize
    - palettize
    - three_d_effect
    - btemp_threshold

.. todo::

    Should this be in another file/page?

After the composite is defined and created, it needs to be converted
to an image.  To do this, it is necessary to describe how the data
values are mapped to values stored in the image format.  This
procedure is called ``stretching``, and in Satpy it is implemented by
``enhancements``.

The first step is to convert the composite to an
:class:`~trollimage.xrimage.XRImage` object::

    >>> from satpy.writers import to_image
    >>> img = to_image(composite)

Now it is possible to apply enhancements available in the class::

    >>> img.invert([False, False, True])
    >>> img.stretch("linear")
    >>> img.gamma(1.7)

And finally either show or save the image::

    >>> img.show()
    >>> img.save('image.tif')

As pointed out in the composite section, it is better to define
frequently used enhancements in configuration files under
``$SATPY_CONFIG_PATH/enhancements/``.  The enhancements can either be in
``generic.yaml`` or instrument-specific file (e.g., ``seviri.yaml``).

The above enhancement can be written (with the headers necessary for
the file) as::

  enhancements:
    overview:
      standard_name: overview
      operations:
        - name: inverse
          method: !!python/name:satpy.enhancements.invert
          args: [False, False, True]
        - name: stretch
          method: !!python/name:satpy.enhancements.stretch
          kwargs:
            stretch: linear
        - name: gamma
          method: !!python/name:satpy.enhancements.gamma
          kwargs:
            gamma: [1.7, 1.7, 1.7]

.. warning::
   If you define a composite with no matching enhancement, Satpy will by
   default apply the :func:`~trollimage.xrimage.XRImage.stretch_linear` enhancement with
   cutoffs of 0.5% and 99.5%.  If you want no enhancement at all (maybe you
   are enhancing a composite based on :class:`DayNightCompositor` where
   the components have their own enhancements defined), you need to define
   an enhancement that does nothing::

      enhancements:
        day_x:
          standard_name: day_x
          operations: []

   It is recommended to define an enhancement even if you intend to use
   the default, in case the default should change in future versions of
   Satpy.

More examples can be found in Satpy source code directory
``satpy/etc/enhancements/generic.yaml``.

See the :doc:`enhancements` documentation for more information on
available built-in enhancements.

.. include:: modifiers.rst
