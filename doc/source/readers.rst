=======
Readers
=======

.. todo::

    How to read cloud products from NWCSAF software. (separate document?)

SatPy supports reading and loading data from many input file formats and
schemes. The :class:`~satpy.scene.Scene` object provides a simple interface
around all the complexity of these various formats through its ``load``
method. The following sections describe the different way data can be loaded,
requested, or added to a Scene object.

Available Readers
=================

To get a list of available readers use the `available_readers` function::

    >>> from satpy import available_readers
    >>> available_readers()

Filter loaded files
===================

Coming soon...

Load data
=========

Datasets in SatPy are identified by certain pieces of metadata set during
data loading. These include `name`, `wavelength`, `calibration`,
`resolution`, `polarization`, and `modifiers`. Normally, once a ``Scene``
is created requesting datasets by `name` or `wavelength` is all that is
needed::

    >>> from satpy import Scene
    >>> scn = Scene(reader="hrit_msg", filenames=filenames)
    >>> scn.load([0.6, 0.8, 10.8])
    >>> scn.load(['IR_120', 'IR_134'])

However, in many cases datasets are available in multiple spatial resolutions,
multiple calibrations (``brightness_temperature``, ``reflectance``,
``radiance``, etc),
multiple polarizations, or have corrections or other modifiers already applied
to them. By default SatPy will provide the version of the dataset with the
highest resolution and the highest level of calibration (brightness
temperature or reflectance over radiance). It is also possible to request one
of these exact versions of a dataset by using the
:class:`~satpy.dataset.DatasetID` class::

    >>> from satpy import DatasetID
    >>> my_channel_id = DatasetID(name='IR_016', calibration='radiance')
    >>> scn.load([my_channel_id])
    >>> print(scn['IR_016'])

Or request multiple datasets at a specific calibration, resolution, or
polarization::

    >>> scn.load([0.6, 0.8], resolution=1000)

Or multiple calibrations::

    >>> scn.load([0.6, 10.8], calibrations=['brightness_temperature', 'radiance'])

In the above case SatPy will load whatever dataset is available and matches
the specified parameters. So the above ``load`` call would load the ``0.6``
(a visible/reflectance band) radiance data and ``10.8`` (an IR band)
brightness temperature data.

.. note::

    If a dataset could not be loaded there is no exception raised. You must
    check the
    :meth:`scn.missing_datasets <satpy.scene.Scene.missing_datasets>`
    property for any ``DatasetID`` that could not be loaded.

To find out what datasets are available from a reader from the files that were
provided to the ``Scene`` use
:meth:`~satpy.scene.Scene.available_dataset_ids`::

    >>> scn.available_dataset_ids()

Or :meth:`~satpy.scene.Scene.available_dataset_names` for just the string
names of Datasets::

    >>> scn.available_dataset_names()

Search for local files
======================

SatPy provides a utility
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

Adding a Reader to SatPy
========================

Coming soon...
