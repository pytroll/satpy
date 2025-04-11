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

Configuring Enhancements
------------------------

Satpy enhancements are configured in YAML files similar to other Satpy
components. They live in a ``$SATPY_CONFIG_PATH/enhancements/`` directory.
This base directory can be customized and include user-defined directories
as well. See :ref:`component_configuration` for more information.

Enhancements can be defined in a ``generic.yaml`` file that is always loaded
for all data or in an instrument-specific file (e.g. ``seviri.yaml``)
corresponding to the ``.attrs["sensor"]`` metadata of the ``DataArray`` being
processed. Generic enhancements are loaded first followed by sensor-specific
enhancement files.

Enhancement YAML Format
^^^^^^^^^^^^^^^^^^^^^^^

The enhancement YAML format starts with an ``enhancements:`` name followed
a series of enhancement "sections". An example file might look like:

.. code-block:: yaml

   enhancements:
     default:
       operations:
       - name: stretch
         method: !!python/name:satpy.enhancements.stretch
         kwargs: {stretch: linear}
     reflectance_default:
       standard_name: toa_bidirectional_reflectance
       operations:
       - name: linear_stretch
         method: !!python/name:satpy.enhancements.stretch
         kwargs: {stretch: 'crude', min_stretch: 0.0, max_stretch: 100.}
       - name: gamma
         method: !!python/name:satpy.enhancements.gamma
         kwargs: {gamma: 1.5}
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

The enhancement section titles, in this case "default", "reflectance_default",
and "overview", have no effect on the behavior of the enhancement and only
serve as a unique identifier for the enhancement section. A section title
must be unique or else one section will overwrite another. Order of the
sections in a YAML file also has no effect.

Inside an enhancement section are a series of enhancement "operations" which
define, in order, the enhancement functions that should be applied to data
matching this enhancement section. A function can be anything importable by
Python (installed in your python environment or on your PYTHONPATH). Arguments
and keyword arguments can be passed via ``args`` and ``kwargs``. The ``name``
of the operation is primarily used for logging and does not need to be anything
specific.

The last part of the enhancement section are the matching terms to determine
what data arrays should use this enhancement. See the below section for how
enhancements are chosen for particular dataset.

Real world examples can be found in Satpy source code directory such as
``satpy/etc/enhancements/generic.yaml``.

Matching Enhancements
---------------------

Choosing what enhancement should be applied to a particular ``DataArray``
is based on the the metadata of that ``DataArray``'s ``.attrs``. Mapping
this metadata to a configured section is done by a type of "decision tree"
that looks at specific metadata keys one at a time. In particular, the current
implementation depends on the following keys:

1. ``name``
2. ``reader``
3. ``platform_name``
4. ``sensor``
5. ``standard_name``
6. ``units``

For low-level implementation details see the
:class:`~satpy.writers.EnhancementDecisionTree` class.

The example YAML in the above section specified one of these keys,
``standard_name``.
One or more of these keys can be specified in a enhancement section,
but the section will only be used if all of those specified keys' values
match the metadata in the ``DataArray`` being processed. Additionally,
if a higher priority key (earlier in the above ordered list) matches then
that section will be used over one with lower priority keys matching.
Put another way, once a match is found for a higher priority key, matching
continues with other keys. Sections that don't define the higher priority
key are then ignored even if they have more matching keys.
See the below examples for a description of these cases.

Note that if two or more sections define the same exact set of matching
key-value pairs
only one of them will be available. Between configuration files the one applied
last will be available (ex. sensor-specific configuration files). In a single file
the section that will be available is undefined and dependent on YAML file loading
and python dictionary ordering.

Examples
^^^^^^^^

.. code-block:: yaml

   enhancements:
     default:
       operations: []
     abi_c01:
       name: C01
       operations: []
     abi_cmip_c01:
       name: C01
       reader: abi_l2_nc
       operations: []
     reflectance_default:
       standard_name: toa_bidirectional_reflectance
       operations: []

To avoid confusion the above sections all have an empty list of operations
to be applied. In a real world situation these would typically all have their
own differing set of operations.

**Example 1**

If this configuration is used for a
``DataArray`` with ``.attrs`` containing:

.. code-block:: python

   {
       "name": "C01",
       "reader": "abi_l1b",
       "standard_name": "toa_bidirectional_reflectance",
       ...
   }

Then it will match the "abi_c01" section because "name" matches and
it is the highest priority match key. The "abi_cmip_c01" section would
also match by "name", but the "reader" key does not match ("abi_l1b").
No other section is defined with a matching "name" and are therefore
not considered.

**Example 2**

Alternatively, if the ``DataArray`` was for a different channel like "C02",
but all other metadata the same then the "reflectance_default" section would
be used. No other section matches by "name" or any other key.

**Example 3**

If the ``DataArray`` was for a completely different channel from
the "abi_l2_nc" reader with the following metadata metadata:

.. code-block:: python

   {
       "name": "C14",
       "reader": "abi_l2_nc",
       "standard_name": "toa_brightness_temperature",
       ...
   }

Then the "default" section would be used. No "name" matches. The "reader"
matches in the "abi_cmip_c01" section, but the "name" does not so it
is ignored. The "standard_name" does not match in "reflectance_default".
The only other section left is the "default" section which has no
match keys and is treated as an overall wildcard section.

**Example 4**

Similar to example 1, if the reader of the DataArray was changed to "abi_l2_nc"
then the "abi_cmip_c01" section would be used.

The defined "name" in the "abi_cmip_c01" section is important as if we changed
the YAML to look like this:

.. code-block:: yaml

   enhancements:
     default:
       operations: []
     abi_c01:
       name: C01
       operations: []
     abi_cmip_c01:
       reader: abi_l2_nc
       operations: []
     reflectance_default:
       standard_name: toa_bidirectional_reflectance
       operations: []

That is, remove the "name" from "abi_cmip_c01", then this DataArray from
the "abi_l2_nc" reader would use the "abi_c01" section instead. This is due
to the higher priority "name" key matching first.

Writing Enhancement Functions
-----------------------------

As mentioend above, any importable function can be specified in the YAML
configuration file. The function should expect at least one argument which
is the :class:`~trollimage.xrimage.XRImage` object to be enhanced. Additional
arguments and keyword arguments can be specified and must be passed from the
YAML configuration. Enhancement functions must produce arrays in the range
0 to 1 for floating data or as integer data. Integer data types are typically
reserved for pre-enhanced images and category products.

At the time of writing enhancement functions must modify the ``DataArray``'s
dask array via ``.data`` directly (inplace). This is accessed from the
``XRImage`` object as ``img.data.data = new_dask_array``. In the future
functions may be expected to return a new copy of the ``XRImage`` so it is
recommended to at least return the original ``img`` object that was
passed to your function.

See the :mod:`satpy.enhancements` module for existing enhancement functions
and useful decorator helpers for managing dask arrays, alpha bands, or
splitting RGBs by band.

Debugging Enhancement Configuration
-----------------------------------

If you've configured your custom enhancement in YAML and Satpy's debug
logging shows you that your custom YAML files are being loaded, but your
enhancement is still not being used when you expect it, there are a couple
debug options.

You can turn on TRACE level logs which in addition to producing a lot
more log messages for other parts of Satpy, will produce information
about how a particular enhancement section was matched. You can turn
on trace log messages with:

.. code-block:: python

   from satpy.utils import trace_on
   trace_on()

   ... normal Satpy code ...

You should then see log messages like the following::

    TRACE    : Checking 'name' level for 'cloud_type': True
    TRACE    :   Checking 'reader' level for 'abi_l1b': False
    TRACE    :   Checking 'reader' level for <wildcard>: False
    TRACE    : Checking 'name' level for <wildcard>: True
    TRACE    :   Checking 'reader' level for 'abi_l1b': False
    TRACE    :   Checking 'reader' level for <wildcard>: True
    TRACE    :     Checking 'platform_name' level for 'GOES-16': False
    TRACE    :     Checking 'platform_name' level for <wildcard>: True
    TRACE    :       Checking 'sensor' level for 'abi': True
    TRACE    :         Checking 'standard_name' level for 'cloud_type': True
    TRACE    :           Match key 'units' not in query dict
    TRACE    :           Checking 'units' level for <wildcard>: True
    TRACE    :             Found match!
    TRACE    :             | sensor=abi
    TRACE    :             | standard_name=cloud_type

Additionally, you can directly load the :class:`~satpy.writers.Enhancer`
object used by Satpy and print the entire "tree" and attempt to follow the
path to match your particular DataArray's metadata:

.. code-block:: python

   from satpy.writers import Enhancer
   enh = Enhancer()
   # NOTE: This is not loading sensor-specific enhancement configs
   # You would need `enh.add_sensor_enhancements(["abi"])`
   enh.enhancement_tree.print_tree()

This would produce (long) output similar to::

    name=<wildcard>
      reader=<wildcard>
        platform_name=<wildcard>
          sensor=<wildcard>
            standard_name=<wildcard>
              units=<wildcard>
                | <global wildcard match>
            standard_name=toa_bidirectional_reflectance
              units=<wildcard>
                | standard_name=toa_bidirectional_reflectance
            standard_name=surface_bidirectional_reflectance
              units=<wildcard>
                | standard_name=surface_bidirectional_reflectance
            standard_name=true_color
              units=<wildcard>
                | standard_name=true_color
      reader=clavrx
        platform_name=<wildcard>
          sensor=<wildcard>
            standard_name=cloud_mask
              units=<wildcard>
                | reader=clavrx
                | standard_name=cloud_mask
    name=true_color_crefl
      reader=<wildcard>
        platform_name=<wildcard>
          sensor=<wildcard>
            standard_name=true_color
              units=<wildcard>
                | name=true_color_crefl
                | standard_name=true_color

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

If you have a :class:`~satpy.scene.Scene` object named ``scn`` with
loaded data, you can run the :func:`~satpy.writers.get_enhanced_image`
function. This function will convert the provided :class:`xarray.DataArray`
into a :class:`~trollimage.xrimage.XRImage` object with YAML configured
enhancements applied. The enhanced DataArray can then be access via the
``.data`` property of the ``XRImage``.

.. code-block:: python

   from satpy.writers import get_enhanced_image

   scn = Scene(...)
   scn.load(["my_dataset"])

   img = get_enhanced_image(scn["my_dataset"])
   enh_data_arr = img.data

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
