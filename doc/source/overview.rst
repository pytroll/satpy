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

Data chunks
-----------

The usage of dask as the foundation for Satpy's operation means that the
underlying data is chunked, that is, cut in smaller pieces that can then be
processed in parallel. Information on dask's chunking can be found in the
dask documentation here: https://docs.dask.org/en/stable/array-chunks.html
The size of these chunks can have a significant impact on the performance of
satpy, so to achieve best performance it can be necessary to adjust it.

Default chunk size used by Satpy can be configured by using the following
around your code:

.. code-block:: python

    with dask.config.set({"array.chunk-size": "32MiB"}):
      # your code here

Or by using:

.. code-block:: python

    dask.config.set({"array.chunk-size": "32MiB"})

at the top of your code.

There are other ways to set dask configuration items, including configuration
files or environment variables, see here:
https://docs.dask.org/en/stable/configuration.html

The value of the chunk-size can be given in different ways, see here:
https://docs.dask.org/en/stable/api.html#dask.utils.parse_bytes

The default value for this parameter is 128MiB, which can translate to chunk
sizes of 4096x4096 for 64-bit float arrays.

Note however that some reader might choose to use a liberal interpretation of
the chunk size which will not necessarily result in a square chunk, or even to
a chunk size of the exact requested size. The motivation behind this is that
data stored as stripes may load much faster if the horizontal striping is kept
as much as possible instead of cutting the data in square chunks. However,
the Satpy readers should respect the overall chunk size when it makes sense.

.. note::

    The legacy way of providing the chunks size in Satpy is the
    ``PYTROLL_CHUNK_SIZE`` environment variable. This is now pending deprecation,
    so an equivalent way to achieve the same result is by using the
    ``DASK_ARRAY__CHUNK_SIZE`` environment variable. The value to assign to the
    variable is the square of the legacy variable, multiplied by the size of array data type
    at hand, so for example, for 64-bits floats::

      export DASK_ARRAY__CHUNK_SIZE=134217728

    which is the same as::

      export DASK_ARRAY__CHUNK_SIZE="128MiB"

    is equivalent to the deprecated::

      export PYTROLL_CHUNK_SIZE=4096

Reading
=======

One of the biggest advantages of using Satpy is the large number of input
file formats that it can read. It encapsulates this functionality into
individual :doc:`reading`. Satpy Readers handle all of the complexity of
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
:doc:`writing` for more information on what's possible for output formats
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
:doc:`writing` documentation for the available writers and how to use them.
