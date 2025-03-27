Enhancements
============

Enhancements are Satpy's way of preparing data to be saved in an output
image format. An enhancement function typically stretches (a.k.a. scales)
data between a set of limits either linearly, on a log scale, or some other
way in order to more easily understand the data. Enhancements are typically
applied automatically as part of the :doc:`writing <writing>` process for
image-like outputs. Note that not all writers apply enhancements if they
expect to save the "raw" data. Enhancements can also be applied manually.
For more information see the :ref:`manual_enhancements` section.

Matching Enhancements
---------------------

TODO

Configuring Enhancements
------------------------

Writing Enhancement Functions
-----------------------------

TODO

::

    Result is 0-1

Debugging Enhancement Configuration
-----------------------------------

Enhancement configuration can be customized in user-defined
:ref:`enhancement configuration files <component_configuration>`.
Sometimes

Built-in enhancement methods
----------------------------

stretch
^^^^^^^

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
^^^^^

invert
^^^^^^

piecewise_linear_stretch
^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`numpy.interp` to linearly interpolate data to a new range. See
:func:`satpy.enhancements.piecewise_linear_stretch` for more information and examples.

cira_stretch
^^^^^^^^^^^^

Logarithmic stretch based on a cira recipe.

reinhard_to_srgb
^^^^^^^^^^^^^^^^

Stretch method based on the Reinhard algorithm, using luminance.

The function includes conversion to sRGB colorspace.

    Reinhard, Erik & Stark, Michael & Shirley, Peter & Ferwerda, James. (2002).
    Photographic Tone Reproduction For Digital Images. ACM Transactions on Graphics.
    :doi: `21. 10.1145/566654.566575`

lookup
^^^^^^

colorize
^^^^^^^^


The colorize enhancement can be used to map scaled/calibrated physical values
to colors. One or several `standard Trollimage color maps`_ may be used as in
the example here::

    - name: colorize
      method: !!python/name:satpy.enhancements.colorize
      kwargs:
          palettes:
            - {colors: spectral, min_value: 193.15, max_value: 253.149999}
            - {colors: greys, min_value: 253.15, max_value: 303.15}

In addition, it is also possible to add a linear alpha channel to the colormap, as in the
following example::

    - name: colorize
      method: !!python/name:satpy.enhancements.colorize
      kwargs:
        palettes:
        - {colors: ylorrd, min_alpha: 100, max_alpha: 255}

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
^^^^^^^^^

three_d_effect
^^^^^^^^^^^^^^

The `three_d_effect` enhancement adds an 3D look to an image by
convolving with a 3x3 kernel.  User can adjust the strength of the
effect by determining the weight (default: 1.0).  Example::

    - name: 3d_effect
      method: !!python/name:satpy.enhancements.three_d_effect
      kwargs:
        weight: 1.0


btemp_threshold
^^^^^^^^^^^^^^^

TODO

.. _manual_enhancements:

Running Enhancements Manually
-----------------------------

Enhancements are typically run automatically when
a :doc:`Writer <writing>` is preparing data to be saved to an image-like
format. There are some occassions where you may want to enhance data
outside of the writing process (ex. preparing data for plotting).
There are two ways of doing this (see below).

Get Enhanced Image
^^^^^^^^^^^^^^^^^^

Assuming you have a :class:`~satpy.scene.Scene` object named ``scn`` with
loaded data, you can run the :func:`~satpy.writers.get_enhanced_image`
function. This function will convert the provided :class:`xarray.DataArray`
into a :class:`~trollimage.xrimage.XRImage` object with YAML configured
enhancments applied.

.. code-block:: python

   from satpy.writers import get_enhanced_image

   scn = Scene(...)
   scn.load([...])

Call Enhancement Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

To not use the YAML configuration files, you can also run the individual
enhancement operations manually. First, the DataArray must be converted
to an :class:`~trollimage.xrimage.XRImage` object using
:func:`~satpy.writers.to_image`.

.. code-block:: python

   from satpy.writers import to_image
   img = to_image(composite)

Note this function is different than the ``get_enhanced_image`` function
used in the previous section as ``to_image`` does not apply any configured
enhancements.

Now it is possible to apply enhancements available in the ``XRImage`` class:

.. code-block:: python

   img.invert([False, False, True])
   img.stretch("linear")
   img.gamma(1.7)

Or more complex enhancement functions in Satpy (described above):

.. code-block::

   from satpy.enhancements import
   img = three_d_effect(img)

.. note::

   At the time of writing Satpy's enhancement functions modify the image
   object and the DataArray underneath inplace. So although the ``img =``
   is unnecessary it is recommended for future compatibility if this changes.

Finally, the :class:`~trollimage.xrimageXRImage` class supports showing an
image in your system's image viewer:

.. code-block:: python

   img.show()

Or in various types of image formats:

.. code-block:: python

   img.save('image.tif')

Note that showing the image requires computing the underlying dask arrays
and loading the entire image into memory before it can be shown. This may
be slow and use up all of your memory. Similarly and similar to the writers
in Satpy, saving using the ``.save`` method requires computing the underlying
dask arrays as the image is saved to disk. If you use Satpy's writers, the
``.show()`` method, and the ``.save()`` method, each one will compute the
dask arrays separately from the beginning; computations are not shared.
See :ref:`scene_multiple_saves` for combining multiple Satpy writers into
a single dask computation.
