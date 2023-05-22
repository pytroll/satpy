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

For readers currently available in Satpy see :ref:`reader_table`.
Additionally to get a list of available readers you can use the `available_readers`
function. By default, it returns the names of available readers.
To return additional reader information use `available_readers(as_dict=True)`::

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

For geostationary products, where the imagery is stored in the files in an unconventional orientation
(e.g. MSG SEVIRI L1.5 data are stored with the southwest corner in the upper right), the keyword argument
``upper_right_corner`` can be passed into the load call to automatically flip the datasets to the
wished orientation. Accepted argument values are ``'NE'``, ``'NW'``, ``'SE'``, ``'SW'``,
and ``'native'``.
By default, no flipping is applied (corresponding to ``upper_right_corner='native'``) and
the data are delivered in the original format. To get the data in the common upright orientation,
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

Load remote data
================

Starting with Satpy version 0.25.1 with supported readers it is possible to
load data from remote file systems like ``s3fs`` or ``fsspec``.
For example:

::

    >>> from satpy import Scene
    >>> from satpy.readers import FSFile
    >>> import fsspec

    >>> filename = 'noaa-goes16/ABI-L1b-RadC/2019/001/17/*_G16_s20190011702186*'

    >>> the_files = fsspec.open_files("simplecache::s3://" + filename, s3={'anon': True})

    >>> fs_files = [FSFile(open_file) for open_file in the_files]

    >>> scn = Scene(filenames=fs_files, reader='abi_l1b')
    >>> scn.load(['true_color_raw'])

Check the list of :ref:`reader_table` to see which reader supports remote
files. For the usage of ``fsspec`` and advanced features like caching files
locally see the `fsspec Documentation <https://filesystem-spec.readthedocs.io/en/latest>`_ .


.. _search_for_files:

Search for local/remote files
=============================

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
more information on the possible parameters as well as for searching on
remote file systems.

.. _dataset_metadata:

Metadata
========

The datasets held by a scene also provide vital metadata such as dataset name, units, observation
time etc. The following attributes are standardized across all readers:

* ``name``, and other identifying metadata keys: See :doc:`dev_guide/satpy_internals`.
* ``start_time``: Left boundary of the time interval covered by the dataset.
  For more information see the :ref:`time_metadata` section below.
* ``end_time``: Right boundary of the time interval covered by the dataset.
  For more information see the :ref:`time_metadata` section below.
* ``area``: :class:`~pyresample.geometry.AreaDefinition` or
  :class:`~pyresample.geometry.SwathDefinition` if data is geolocated. Areas are used for gridded
  projected data and Swaths when data must be described by individual longitude/latitude
  coordinates. See the Coordinates section below.
* ``reader``: The name of the Satpy reader that produced the dataset.
* ``orbital_parameters``: Dictionary of orbital parameters describing the satellite's position.
  See the :ref:`orbital_parameters` section below for more information.
* ``time_parameters``: Dictionary of additional time parameters describing the
  time ranges related to the requests or schedules for when observations
  should happen and when they actually do. See :ref:`time_metadata` below for
  details.
* ``raw_metadata``: Raw, unprocessed metadata from the reader.

Note that the above attributes are not necessarily available for each dataset.

.. _time_metadata:

Time Metadata
-------------

In addition to the generic ``start_time`` and ``end_time`` pieces of metadata
there are other time fields that may be provided if the reader supports them.
These items are stored in a ``time_parameters`` sub-dictionary and they include
values like:

* ``observation_start_time``: The point in time when a sensor began recording
  for the current data.
* ``observation_end_time``: Same as ``observation_start_time``, but when data
  has stopped being recorded.
* ``nominal_start_time``: The "human friendly" time describing the start of
  the data observation interval or repeat cycle. This time is often on a round
  minute (seconds=0). Along with the nominal end time, these times define the
  regular interval of the data collection. For example, GOES-16 ABI full disk
  images are collected every 10 minutes (in the common configuration) so
  ``nominal_start_time`` and ``nominal_end_time`` would be 10 minutes apart
  regardless of when the instrument recorded data inside that interval.
  This time may also be referred to as the repeat cycle, repeat slot, or time
  slot.
* ``nominal_end_time``: Same as ``nominal_start_time``, but the end of the
  interval.

In general, ``start_time`` and ``end_time`` will be set to the "nominal"
time by the reader. This ensures that other Satpy components get a
consistent time for calculations (ex. generation of solar zenith angles)
and can be reused between bands.

See the :ref:`data_array_coordinates` section below for more information on
time information that may show up as a per-element/row "coordinate" on the
DataArray (ex. acquisition time) instead of as metadata.

.. _orbital_parameters:

Orbital Parameters
------------------

Orbital parameters describe the position of the satellite. As such they
typically come in a few "flavors" for the common types of orbits a satellite
may have.

For *geostationary* satellites it is described using the following scalar attributes:

  * ``satellite_actual_longitude/latitude/altitude``: Current position of the satellite at the
    time of observation in geodetic coordinates (i.e. altitude is relative and normal to the
    surface of the ellipsoid). The longitude and latitude are given in degrees, the altitude in meters.
  * ``satellite_nominal_longitude/latitude/altitude``: Center of the station keeping box (a
    confined area in which the satellite is actively maintained in using maneuvers). Inbetween
    major maneuvers, when the satellite is permanently moved, the nominal position is constant.
    The longitude and latitude are given in degrees, the altitude in meters.
  * ``nadir_longitude/latitude``: Intersection of the instrument's Nadir with the surface of the
    earth. May differ from the actual satellite position, if the instrument is pointing slightly
    off the axis (satellite, earth-center). If available, this should be used to compute viewing
    angles etc. Otherwise, use the actual satellite position. The values are given in degrees.
  * ``projection_longitude/latitude/altitude``: Projection center of the re-projected data. This
    should be used to compute lat/lon coordinates. Note that the projection center can differ
    considerably from the actual satellite position. For example MSG-1 was at times positioned
    at 3.4 degrees west, while the image data was re-projected to 0 degrees.
    The longitude and latitude are given in degrees, the altitude in meters.

    .. note:: For use in pyorbital, the altitude has to be converted to kilometers, see for example
              :func:`pyorbital.orbital.get_observer_look`.

For *polar orbiting* satellites the readers usually provide coordinates and viewing angles of
the swath as ancillary datasets. Additional metadata related to the satellite position includes:

  * ``tle``: Two-Line Element (TLE) set used to compute the satellite's orbit

.. _data_array_coordinates:

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

hdf5 based readers
------------------

.. automodule:: satpy.readers.agri_l1
    :noindex:

.. automodule:: satpy.readers.ghi_l1
    :noindex:

Arctica-M N1 HDF5 format reader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: satpy.readers.msu_gsa_l1b
    :noindex:
