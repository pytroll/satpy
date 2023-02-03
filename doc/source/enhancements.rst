============
Enhancements
============

Built-in enhancement methods
============================

stretch
-------

The most basic operation is to stretch the image so that the data fits to
the output format.  There are many different ways to stretch the data,
which are configured by giving them in `kwargs` dictionary, like in the
example above.  The default, if nothing else is defined, is to apply
a linear stretch.  For more details, see
:ref:`enhancing the images <enhancing-the-images>`.

linear
******

As the name suggests, linear stretch converts the input values to
output values in a linear fashion.  By default, 5% of the data is cut
on both ends of the scale, but these can be overridden with
``cutoffs=(0.005, 0.005)`` argument::

    - name: stretch
      method: !!python/name:satpy.enhancements.stretch
      kwargs:
        stretch: linear
        cutoffs: [0.003, 0.005]

.. note::

    This enhancement is currently not optimized for dask because it requires
    getting minimum/maximum information for the entire data array.

crude
*****

The crude stretching is used to limit the input values to a certain
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

invert
------

crefl_scaling
-------------

Deprecated. Use 'piecewise_linear_stretch' instead.

piecewise_linear_stretch
------------------------

Use :func:`numpy.interp` to linearly interpolate data to a new range. See
:func:`satpy.enhancements.piecewise_linear_stretch` for more information and examples.

cira_stretch
------------

Logarithmic stretch based on a cira recipe.

reinhard_to_srgb
----------------

Stretch method based on the Reinhard algorithm, using luminance.

The function includes conversion to sRGB colorspace.

    Reinhard, Erik & Stark, Michael & Shirley, Peter & Ferwerda, James. (2002).
    Photographic Tone Reproduction For Digital Images. ACM Transactions on Graphics.
    :doi: `21. 10.1145/566654.566575`

lookup
------

colorize
--------


The colorize enhancement can be used to map scaled/calibrated physical values
to colors. One or several `standard Trollimage color maps`_ may be used as in
the example here::

    - name: colorize
      method: !!python/name:satpy.enhancements.colorize
      kwargs:
          palettes:
            - {colors: spectral, min_value: 193.15, max_value: 253.149999}
            - {colors: greys, min_value: 253.15, max_value: 303.15}

It is also possible to provide your own custom defined color mapping by
specifying a list of RGB values and the corresponding min and max values
between which to apply the colors. This is for instance a common use case for
Sea Surface Temperature (SST) imagery, as in this example with the EUMETSAT
Ocean and Sea Ice SAF (OSISAF) GHRSST product::

    - name: osisaf_sst
      method: !!python/name:satpy.enhancements.colorize
      kwargs:
          palettes:
            - colors: [
              [255, 0, 255],
              [195, 0, 129],
              [129, 0, 47],
              [195, 0, 0],
              [255, 0, 0],
              [236, 43, 0],
              [217, 86, 0],
              [200, 128, 0],
              [211, 154, 13],
              [222, 180, 26],
              [233, 206, 39],
              [244, 232, 52],
              [255.99609375, 255.99609375, 63.22265625],
              [203.125, 255.99609375, 52.734375],
              [136.71875, 255.99609375, 27.34375],
              [0, 255.99609375, 0],
              [0, 207.47265625, 0],
              [0, 158.94921875, 0],
              [0, 110.42578125, 0],
              [0, 82.8203125, 63.99609375],
              [0, 55.21484375, 127.9921875],
              [0, 27.609375, 191.98828125],
              [0, 0, 255.99609375],
              [100.390625, 100.390625, 255.99609375],
              [150.5859375, 150.5859375, 255.99609375]]
              min_value: 296.55
              max_value: 273.55

The RGB color values will be interpolated to give a smooth result. This is
contrary to using the palettize enhancement.

If the source dataset already defines a palette, this can be applied directly.
This requires that the palette is listed as an auxiliary variable and loaded
as such by the reader.  To apply such a palette directly, pass the ``dataset``
keyword.  For example::

    - name: colorize
      method: !!python/name:satpy.enhancements.colorize
      kwargs:
        palettes:
          - dataset: ctth_alti_pal
            color_scale: 255

.. warning::
   If the source data have a valid range defined, one should **not** define
   ``min_value`` and ``max_value`` in the enhancement configuration!  If
   those are defined and differ from the values in the valid range, the
   colors will be wrong.

The above examples are just three different ways to apply colors to images with
Satpy. There is a wealth of other options for how to declare a colormap, please
see :func:`~satpy.enhancements.create_colormap` for more inspiration.

.. _`standard Trollimage color maps`: https://trollimage.readthedocs.io/en/latest/colormap.html#default-colormaps


palettize
---------

three_d_effect
--------------

The `three_d_effect` enhancement adds an 3D look to an image by
convolving with a 3x3 kernel.  User can adjust the strength of the
effect by determining the weight (default: 1.0).  Example::

    - name: 3d_effect
      method: !!python/name:satpy.enhancements.three_d_effect
      kwargs:
        weight: 1.0


btemp_threshold
---------------
