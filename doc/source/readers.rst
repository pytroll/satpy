=======
Readers
=======

.. todo::

    How to read cloud products from NWCSAF software. (separate document?)

Satpy supports reading and loading data from many input file formats and
schemes. The :class:`~satpy.scene.Scene` object provides a simple interface
around all the complexity of these various formats through its ``load``
method. The following sections describe the different way data can be loaded,
requested, or added to a Scene object.

Available Readers
=================

To get a list of available readers use the `available_readers` function. By default,
it returns the names of available readers. To return additional reader information 
use `available_readers(as_dict=True)`::

    >>> from satpy import available_readers
    >>> available_readers()

Filter loaded files
===================

Coming soon...

Load data
=========

Datasets in Satpy are identified by certain pieces of metadata set during
data loading. These include `name`, `wavelength`, `calibration`,
`resolution`, `polarization`, and `modifiers`. Normally, once a ``Scene``
is created requesting datasets by `name` or `wavelength` is all that is
needed::

    >>> from satpy import Scene
    >>> scn = Scene(reader="seviri_l1b_hrit", filenames=filenames)
    >>> scn.load([0.6, 0.8, 10.8])
    >>> scn.load(['IR_120', 'IR_134'])

However, in many cases datasets are available in multiple spatial resolutions,
multiple calibrations (``brightness_temperature``, ``reflectance``,
``radiance``, etc),
multiple polarizations, or have corrections or other modifiers already applied
to them. By default Satpy will provide the version of the dataset with the
highest resolution and the highest level of calibration (brightness
temperature or reflectance over radiance). It is also possible to request one
of these exact versions of a dataset by using the
:class:`~satpy.dataset.DataQuery` class::

    >>> from satpy import DataQuery
    >>> my_channel_id = DataQuery(name='IR_016', calibration='radiance')
    >>> scn.load([my_channel_id])
    >>> print(scn['IR_016'])

Or request multiple datasets at a specific calibration, resolution, or
polarization::

    >>> scn.load([0.6, 0.8], resolution=1000)

Or multiple calibrations::

    >>> scn.load([0.6, 10.8], calibration=['brightness_temperature', 'radiance'])

In the above case Satpy will load whatever dataset is available and matches
the specified parameters. So the above ``load`` call would load the ``0.6``
(a visible/reflectance band) radiance data and ``10.8`` (an IR band)
brightness temperature data.

For geostationary satellites that have the individual channel data
separated to several files (segments) the missing segments are padded
by default to full disk area.  This is made to simplify caching of
resampling look-up tables (see :doc:`resample` for more information).
To disable this, the user can pass ``pad_data`` keyword argument when
loading datasets::

    >>> scn.load([0.6, 10.8], pad_data=False)

For geostationary products, where the imagery is stored in the files in a flipped orientation
(e.g. MSG SEVIRI L1.5 data which is flipped upside-down and left-right), the keyword argument
``upper_right_corner`` can be passed into the load call to automatically flip the datasets to the
wished orientation. Accepted argument values are ``'NE'``, ``'NW'``, ``'SE'``, ``'SW'``,
and ``'native'``.
By default, no flipping is applied (corresponding to ``upper_right_corner='native'``) and
the data is delivered in the original format. To get the data in the common upright orientation,
load the datasets using e.g.::

    >>> scn.load(['VIS008'], upper_right_corner='NE')

.. note::

    If a dataset could not be loaded there is no exception raised. You must
    check the
    :meth:`scn.missing_datasets <satpy.scene.Scene.missing_datasets>`
    property for any ``DataID`` that could not be loaded.

To find out what datasets are available from a reader from the files that were
provided to the ``Scene`` use
:meth:`~satpy.scene.Scene.available_dataset_ids`::

    >>> scn.available_dataset_ids()

Or :meth:`~satpy.scene.Scene.available_dataset_names` for just the string
names of Datasets::

    >>> scn.available_dataset_names()

Search for local files
======================

Satpy provides a utility
:func:`~satpy.readers.find_files_and_readers` for searching for files in
a base directory matching various search parameters. This function discovers
files based on filename patterns. It returns a dictionary mapping reader name
to a list of filenames supported. This dictionary can be passed directly to
the :class:`~satpy.scene.Scene` initialization.

::

    >>> from satpy import find_files_and_readers, Scene
    >>> from datetime import datetime
    >>> my_files = find_files_and_readers(base_dir='/data/viirs_sdrs',
    ...                                   reader='viirs_sdr',
    ...                                   start_time=datetime(2017, 5, 1, 18, 1, 0),
    ...                                   end_time=datetime(2017, 5, 1, 18, 30, 0))
    >>> scn = Scene(filenames=my_files)

See the :func:`~satpy.readers.find_files_and_readers` documentation for
more information on the possible parameters.

Metadata
========

.. _dataset_metadata:

The datasets held by a scene also provide vital metadata such as dataset name, units, observation
time etc. The following attributes are standardized across all readers:

* ``name``, and other identifying metadata keys: See :doc:`dev_guide/satpy_internals`.
* ``start_time``: Left boundary of the time interval covered by the dataset.
* ``end_time``: Right boundary of the time interval covered by the dataset.
* ``area``: :class:`~pyresample.geometry.AreaDefinition` or
  :class:`~pyresample.geometry.SwathDefinition` if data is geolocated. Areas are used for gridded
  projected data and Swaths when data must be described by individual longitude/latitude
  coordinates. See the Coordinates section below.
* ``reader``: The name of the Satpy reader that produced the dataset.
* ``orbital_parameters``: Dictionary of orbital parameters describing the satellite's position.

  * For *geostationary* satellites it is described using the following scalar attributes:

    * ``satellite_actual_longitude/latitude/altitude``: Current position of the satellite at the
      time of observation in geodetic coordinates (i.e. altitude is relative and normal to the
      surface of the ellipsoid).
    * ``satellite_nominal_longitude/latitude/altitude``: Center of the station keeping box (a
      confined area in which the satellite is actively maintained in using maneuvres). Inbetween
      major maneuvres, when the satellite is permanently moved, the nominal position is constant.
    * ``nadir_longitude/latitude``: Intersection of the instrument's Nadir with the surface of the
      earth. May differ from the actual satellite position, if the instrument is pointing slightly
      off the axis (satellite, earth-center). If available, this should be used to compute viewing
      angles etc. Otherwise, use the actual satellite position.
    * ``projection_longitude/latitude/altitude``: Projection center of the re-projected data. This
      should be used to compute lat/lon coordinates. Note that the projection center can differ
      considerably from the actual satellite position. For example MSG-1 was at times positioned
      at 3.4 degrees west, while the image data was re-projected to 0 degrees.
    * [DEPRECATED] ``satellite_longitude/latitude/altitude``: Current position of the satellite at
      the time of observation in geodetic coordinates.

    .. note:: Longitudes and latitudes are given in degrees, altitude in meters. For use in
              pyorbital, the altitude has to be converted to kilometers, see for example
              :func:`pyorbital.orbital.get_observer_look`.

  * For *polar orbiting* satellites the readers usually provide coordinates and viewing angles of
    the swath as ancillary datasets. Additional metadata related to the satellite position include:

      * ``tle``: Two-Line Element (TLE) set used to compute the satellite's orbit

* ``raw_metadata``: Raw, unprocessed metadata from the reader.

Note that the above attributes are not necessarily available for each dataset.

Coordinates
===========

Each :class:`~xarray.DataArray` produced by Satpy has several Xarray
coordinate variables added to them.

* ``x`` and ``y``: Projection coordinates for gridded and projected data.
  By default `y` and `x` are the preferred **dimensions** for all 2D data, but
  these **coordinates** are only added for gridded (non-swath) data. For 1D
  data only the ``y`` dimension may be specified.
* ``crs``: A :class:`~pyproj.crs.CRS` object defined the Coordinate Reference
  System for the data. Requires pyproj 2.0 or later to be installed. This is
  stored as a scalar array by Xarray so it must be accessed by doing
  ``crs = my_data_arr.attrs['crs'].item()``. For swath data this defaults
  to a ``longlat`` CRS using the WGS84 datum.
* ``longitude``: Array of longitude coordinates for swath data.
* ``latitude``: Array of latitude coordinates for swath data.

Readers are free to define any coordinates in addition to the ones above that
are automatically added. Other possible coordinates you may see:

* ``acq_time``: Instrument data acquisition time per scan or row of data.

Adding a Reader to Satpy
========================

This is described in the developer guide, see :doc:`dev_guide/custom_reader`.

Implemented readers
===================

SEVIRI L1.5 data readers
------------------------

.. automodule:: satpy.readers.seviri_base
    :noindex:

SEVIRI HRIT format reader
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: satpy.readers.seviri_l1b_hrit
    :noindex:

SEVIRI Native format reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: satpy.readers.seviri_l1b_native
    :noindex:

SEVIRI netCDF format reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: satpy.readers.seviri_l1b_nc
    :noindex:


Other xRIT-based readers
------------------------

.. automodule:: satpy.readers.hrit_base
    :noindex:


JMA HRIT format reader
^^^^^^^^^^^^^^^^^^^^^^


.. automodule:: satpy.readers.hrit_jma
    :noindex:

GOES HRIT format reader
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: satpy.readers.goes_imager_hrit
    :noindex:

Electro-L HRIT format reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: satpy.readers.electrol_hrit
    :noindex:

hdf-eos based readers
---------------------

.. automodule:: satpy.readers.modis_l1b
    :noindex:

.. automodule:: satpy.readers.modis_l2
    :noindex:

satpy cf nc readers
---------------------

.. automodule:: satpy.readers.satpy_cf_nc
    :noindex:
