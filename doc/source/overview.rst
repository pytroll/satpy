========
Overview
========

Satpy is designed to provide easy access to common operations for processing
meteorological remote sensing data. Any details needed to perform these
operations are configured internally to Satpy meaning users should not have to
worry about *how* something is done, only ask for what they want. Most of the
features provided by Satpy can be configured by keyword arguments (see the
:doc:`API Documentation <api/satpy>` or other specific section for more details).
For more complex customizations or added features Satpy uses a set of
configuration files that can be modified by the user. The various components
and concepts of Satpy are described below. The :doc:`quickstart` guide also
provides simple example code for the available features of Satpy.

Scene
=====

Satpy provides most of its functionality through the
:class:`~satpy.scene.Scene` class. This acts as a container for the datasets
being operated on and provides methods for acting on those datasets. It
attempts to reduce the amount of low-level knowledge needed by the user while
still providing a pythonic interface to the functionality underneath.

A Scene object represents a single geographic region of data, typically at a
single continuous time range. It is possible to combine Scenes to
form a Scene with multiple regions or multiple time observations, but
it is not guaranteed that all functionality works in these situations.

DataArrays
==========

Satpy's lower-level container for data is the
:class:`xarray.DataArray`. For historical reasons DataArrays are often
referred to as "Datasets" in Satpy. These objects act similar to normal
numpy arrays, but add additional metadata and attributes for describing the
data. Metadata is stored in a ``.attrs`` dictionary and named dimensions can
be accessed in a ``.dims`` attribute, along with other attributes.
In most use cases these objects can be operated on like normal NumPy arrays
with special care taken to make sure the metadata dictionary contains
expected values. See the XArray documentation for more info on handling
:class:`xarray.DataArray` objects.

Additionally, Satpy uses a special form of DataArrays where data is stored
in :class:`dask.array.Array` objects which allows Satpy to perform
multi-threaded lazy operations vastly improving the performance of processing.
For help on developing with dask and xarray see
:doc:`dev_guide/xarray_migration` or the documentation for the specific
project.

To uniquely identify ``DataArray`` objects Satpy uses `DataID`. A
``DataID`` consists of various pieces of available metadata. This usually
includes `name` and `wavelength` as identifying metadata, but can also include
`resolution`, `calibration`, `polarization`, and additional `modifiers`
to further distinguish one dataset from another. For more information on `DataID`
objects, have a look a :doc:`dev_guide/satpy_internals`.

.. warning::

    XArray includes other object types called "Datasets". These are different
    from the "Datasets" mentioned in Satpy.


Reading
=======

One of the biggest advantages of using Satpy is the large number of input
file formats that it can read. It encapsulates this functionality into
individual :doc:`readers`. Satpy Readers handle all of the complexity of
reading whatever format they represent. Meteorological Satellite file formats
can be extremely complex and formats are rarely reused across satellites
or instruments. No matter the format, Satpy's Reader interface is meant to
provide a consistent data loading interface while still providing flexibility
to add new complex file formats.

Compositing
===========

Many users of satellite imagery combine multiple sensor channels to bring
out certain features of the data. This includes using one dataset to enhance
another, combining 3 or more datasets in to an RGB image, or any other
combination of datasets. Satpy comes with a lot of common composite
combinations built-in and allows the user to request them like any other
dataset. Satpy also makes it possible to create your own custom composites
and have Satpy treat them like any other dataset. See :doc:`composites`
for more information.

Resampling
==========

Satellite imagery data comes in two forms when it comes to geolocation,
native satellite swath coordinates and uniform gridded projection
coordinates. It is also common to see the channels from a single sensor
in multiple resolutions, making it complicated to combine or compare the
datasets. Many use cases of satellite data require the data to
be in a certain projection other than the native projection or to have
output imagery cover a specific area of interest. Satpy makes it easy to
resample datasets to allow for users to combine them or grid them to these
projections or areas of interest. Satpy uses the PyTroll `pyresample` package
to provide nearest neighbor, bilinear, or elliptical weighted averaging
resampling methods. See :doc:`resample` for more information.

Enhancements
============

When making images from satellite data the data has to be manipulated to be
compatible with the output image format and still look good to the human eye.
Satpy calls this functionality "enhancing" the data, also commonly called
scaling or stretching the data. This process can become complicated not just
because of how subjective the quality of an image can be, but also because
of historical expectations of forecasters and other users for how the data
should look. Satpy tries to hide the complexity of all the possible
enhancement methods from the user and just provide the best looking image
by default. Satpy still makes it possible to customize these procedures, but
in most cases it shouldn't be necessary. See the documentation on
:doc:`writers` for more information on what's possible for output formats
and enhancing images.

Writing
=======

Satpy is designed to make data loading, manipulating, and analysis easy.
However, the best way to get satellite imagery data out to as many users
as possible is to make it easy to save it in multiple formats. Satpy allows
users to save data in image formats like PNG or GeoTIFF as well as data file
formats like NetCDF. Each format's complexity is hidden behind the interface
of individual Writer objects and includes keyword arguments for accessing
specific format features like compression and output data type. See the
:doc:`writers` documentation for the available writers and how to use them.
