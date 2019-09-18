=======
Writers
=======

Satpy makes it possible to save datasets in multiple formats. For details
on additional arguments and features available for a specific Writer see
the table below. Most use cases will want to save datasets using the
:meth:`~satpy.scene.Scene.save_datasets` method::

    >>> scn.save_datasets(writer='simple_image')

The ``writer`` parameter defaults to using the ``geotiff`` writer.
One common parameter across almost all Writers is ``filename`` and
``base_dir`` to help automate saving files with custom filenames::

    >>> scn.save_datasets(
    ...     filename='{name}_{start_time:%Y%m%d_%H%M%S}.tif',
    ...     base_dir='/tmp/my_ouput_dir')

.. versionchanged:: 0.10

    The `file_pattern` keyword argument was renamed to `filename` to match
    the `save_dataset` method's keyword argument.

.. _writer_table:

.. list-table:: Satpy Writers
    :header-rows: 1

    * - Description
      - Writer name
      - Status
      - Examples
    * - GeoTIFF
      - :class:`geotiff <satpy.writers.geotiff.GeoTIFFWriter>`
      - Nominal
      -
    * - Simple Image (PNG, JPEG, etc)
      - :class:`simple_image <satpy.writers.simple_image.PillowWriter>`
      - Nominal
      -
    * - NinJo TIFF (using ``pyninjotiff`` package)
      - :class:`ninjotiff <satpy.writers.ninjotiff.NinjoTIFFWriter>`
      - Nominal
      -
    * - NetCDF (Standard CF)
      - :class:`cf <satpy.writers.cf_writer.CFWriter>`
      - Pre-alpha
      - :mod:`Usage example <satpy.writers.cf_writer>`
    * - AWIPS II Tiled SCMI NetCDF4
      - :class:`scmi <satpy.writers.scmi.SCMIWriter>`
      - Beta
      -

Available Writers
=================

To get a list of available writers use the `available_writers` function::

    >>> from satpy import available_writers
    >>> available_writers()


Colorizing and Palettizing using user-supplied colormaps
========================================================

.. note::

    In the future this functionality will be added to the ``Scene`` object.

It is possible to create single channel "composites" that are then colorized
using users' own colormaps.  The colormaps are Numpy arrays with shape
(num, 3), see the example below how to create the mapping file(s).

This example creates a 2-color colormap, and we interpolate the colors between
the defined temperature ranges.  Beyond those limits the image clipped to
the specified colors.

    >>> import numpy as np
    >>> from satpy.composites import BWCompositor
    >>> from satpy.enhancements import colorize
    >>> from satpy.writers import to_image
    >>> arr = np.array([[0, 0, 0], [255, 255, 255]])
    >>> np.save("/tmp/binary_colormap.npy", arr)
    >>> compositor = BWCompositor("test", standard_name="colorized_ir_clouds")
    >>> composite = compositor((local_scene[10.8], ))
    >>> img = to_image(composite)
    >>> kwargs = {"palettes": [{"filename": "/tmp/binary_colormap.npy",
    ...           "min_value": 223.15, "max_value": 303.15}]}
    >>> colorize(img, **kwargs)
    >>> img.show()

Similarly it is possible to use discrete values without color interpolation
using `palettize()` instead of `colorize()`.

You can define several colormaps and ranges in the `palettes` list and they
are merged together.  See trollimage_ documentation for more information how
colormaps and color ranges are merged.

The above example can be used in enhancements YAML config like this:

.. code-block:: yaml

  hot_or_cold:
    standard_name: hot_or_cold
    operations:
      - name: colorize
        method: &colorizefun !!python/name:satpy.enhancements.colorize ''
        kwargs:
          palettes:
            - {filename: /tmp/binary_colormap.npy, min_value: 223.15, max_value: 303.15}


.. _trollimage: http://trollimage.readthedocs.io/en/latest/

Saving multiple Scenes in one go
================================

As mentioned earlier, it is possible to save `Scene` datasets directly
using :meth:`~satpy.scene.Scene.save_datasets` method.  However,
sometimes it is beneficial to collect more `Scene`\ s together and process
and save them all at once.

::

    >>> from satpy.writers import compute_writer_results
    >>> res1 = scn.save_datasets(filename="/tmp/{name}.png",
    ...                          writer='simple_image',
    ...                          compute=False)
    >>> res2 = scn.save_datasets(filename="/tmp/{name}.tif",
    ...                          writer='geotiff',
    ...                          compute=False)
    >>> results = [res1, res2]
    >>> compute_writer_results(results)
