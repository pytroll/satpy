============
Enhancements
============

Built-in enhancement methods
============================

stretch
-------

The most basic operation is to stretch the image so that the data fits
to the output format.  There are many different ways to stretch the
data, which are configured in YAML files by defining them in a `kwargs`
dictionary.  The default, if nothing else is defined, is to apply a
linear stretch.  For more details, see below.

linear
******

As the name suggests, ``linear`` stretch converts the input values to
output values in a linear fashion.  By default, 5% of the data is cut
on both ends of the scale, but these can be overridden with
``cutoffs=(0.005, 0.005)`` argument::

    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs:
        stretch: linear
        cutoffs: (0.003, 0.005)

.. note::

    This enhancement is currently not optimized for dask because it requires
    getting minimum/maximum information for the entire data array.

crude
*****

The ``crude`` stretching is used to limit the input values to a certain
range by clipping the data. This is followed by a linear stretch with
no cutoffs specified (see above). Example::

    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs:
        stretch: crude
        min_stretch: [0, 0, 0]
        max_stretch: [100, 100, 100]

It is worth noting that this stretch can also be used to _invert_ the
data by giving larger values to the min_stretch than to max_stretch.

histogram
*********

gamma
-----

The ``gamma`` enhancement applies gamma correction to the composite::

    - name: gamma
      method: !!python/name:satpy.enhancements.gamma
      kwargs:
        stretch: crude
        kwargs: {gamma: 1.6}

invert
------

The ``invert`` inverts the color scale of the image.  That is, black
becomes white, and vice-versa.  For an RGB composite that needs the
``B`` channel to be inverted::

    - name: invert
      method: !!python/name:satpy.enhancements.invert
      args:
      - [false, false, true]


crefl_scaling
-------------

cira_stretch
------------

lookup
------

colorize
--------

The ``colorize`` enhancement adds colors to an black-and-white
composite.  The composite is assumed not to be discrete-valued, and
the colors will be interpolated from the given colormap.  The
enhancement can either use ``trollimage`` colormaps::

      - name: colorize
        method: !!python/name:satpy.enhancements.colorize
        kwargs:
          palettes:
            - {colors: spectral, min_value: 203.15, max_value: 243.149999}
            - {colors: greys, min_value: 243.15, max_value: 303.15}

or the color look-up tables can be saved in Numpy ``.npy`` files::

      - name: colorize
        method: !!python/name:satpy.enhancements.colorize
        kwargs:
          palettes:
            - {filename: /path/to/colormap_1.npy, min_value: 203.15, max_value: 243.149999}
            - {filename: /path/to/colormap_2.npy, min_value: 243.15, max_value: 303.15}

The colormaps in the ``.npy`` files should have a shape of ``(N, 3)``,
where ``N`` is the number of pre-defined colors, and the other
dimension is for the ``R``, ``G`` and ``B`` values from the interval
``[0, 255]``.

As the examples show, the colorization can have several different
colormaps with their own value ranges.  The data are stretched
piece-vise and the colors applied individually.


palettize
---------

The ``palettize`` is similar to ``colorize``, but works on discrete
values and doesn't interpolate the colors.  The colormap should have
``RGB`` representation for each of the value in the input data.


three_d_effect
--------------

The ``three_d_effect`` enhancement adds an 3D look to an image by
convolving with a 3x3 kernel.  User can adjust the strength of the
effect by determining the weight (default: 1.0).  Example::

    - name: 3d_effect
      method: !!python/name:satpy.enhancements.three_d_effect
      kwargs:
        weight: 1.0


btemp_threshold
---------------
