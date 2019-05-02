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
    * - GeoTIFF
      - :class:`geotiff <satpy.writers.geotiff.GeoTIFFWriter>`
      - Nominal
    * - Simple Image (PNG, JPEG, etc)
      - :class:`simple_image <satpy.writers.simple_image.PillowWriter>`
      - Nominal
    * - NinJo TIFF (using ``pyninjotiff`` package)
      - :class:`ninjotiff <satpy.writers.ninjotiff.NinjoTIFFWriter>`
      - Nominal
    * - NetCDF (Standard CF)
      - :class:`cf <satpy.writers.cf_writer.CFWriter>`
      - Pre-alpha
    * - AWIPS II Tiled SCMI NetCDF4
      - :class:`scmi <satpy.writers.scmi.SCMIWriter>`
      - Beta

Available Writers
=================

To get a list of available writers use the `available_writers` function::

    >>> from satpy import available_writers
    >>> available_writers()


Examples
========

CF-Writer
---------

The CF writer saves datasets in a Scene as `CF-compliant`_ netCDF file. Here is an example with MSG SEVIRI data in HRIT
format:

    >>> from satpy import Scene
    >>> import glob
    >>> filenames = glob.glob('data/H*201903011200*')
    >>> scn = Scene(filenames=filenames, reader='seviri_l1b_hrit')
    >>> scn.load(['VIS006', 'IR_108'])
    >>> scn.save_datasets(writer='cf', datasets=['VIS006', 'IR_108'], filename='seviri_test.nc',
                          exclude_attrs=['raw_metadata'])

You can select the netCDF backend using the ``engine`` keyword argument. Default is ``h5netcdf``, an alternative could
be, for example, ``netCDF4``.

In the above example, raw metadata from the HRIT files has been excluded. If you want all attributes to be included,
just remove the ``exclude_attrs`` keyword argument. By default, dict-type dataset attributes, such as the raw metadata,
are encoded as a string using json. Thus, you can use json to decode them afterwards:

    >>> import xarray as xr
    >>> import json
    >>> # Save scene to nc-file
    >>> scn.save_datasets(writer='cf', datasets=['VIS006', 'IR_108'], filename='seviri_test.nc')
    >>> # Now read data from the nc-file
    >>> ds = xr.open_dataset('seviri_test.nc')
    >>> raw_mda = json.loads(ds['IR_108'].attrs['raw_metadata'])
    >>> print(raw_mda['RadiometricProcessing']['Level15ImageCalibration']['CalSlope'])
    [0.020865   0.0278287  0.0232411  0.00365867 0.00831811 0.03862197
     0.12674432 0.10396091 0.20503568 0.22231115 0.1576069  0.0352385]


Alternatively it is possible to flatten dict-type attributes by setting ``flatten_attrs=True``. This is more human
readable as it will create a separate nc-attribute for each item in every dictionary. Keys oare concatenated with
underscore separators. The `CalSlope` attribute can then be accessed as follows:

    >>> scn.save_datasets(writer='cf', datasets=['VIS006', 'IR_108'], filename='seviri_test.nc',
                          flatten_attrs=True)
    >>> ds = xr.open_dataset('seviri_test.nc')
    >>> print(ds['IR_108'].attrs['raw_metadata_RadiometricProcessing_Level15ImageCalibration_CalSlope'])
    [0.020865   0.0278287  0.0232411  0.00365867 0.00831811 0.03862197
     0.12674432 0.10396091 0.20503568 0.22231115 0.1576069  0.0352385]

This is what the corresponding ``ncdump`` output would look like in this case:

.. code-block:: none

    $ ncdump -h test_seviri.nc
    ...
    IR_108:raw_metadata_RadiometricProcessing_Level15ImageCalibration_CalOffset = -1.064, ...;
    IR_108:raw_metadata_RadiometricProcessing_Level15ImageCalibration_CalSlope = 0.021, ...;
    IR_108:raw_metadata_RadiometricProcessing_MPEFCalFeedback_AbsCalCoeff = 0.021, ...;
    ...

.. _CF-compliant: http://cfconventions.org/

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
