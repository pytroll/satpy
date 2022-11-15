============================
Migrating to xarray and dask
============================

Many python developers dealing with meteorologic satellite data begin with
using NumPy arrays directly. This work usually involves masked arrays,
boolean masks, index arrays, and reshaping. Due to the libraries used by
Satpy these operations can't always be done in the same way. This guide acts
as a starting point for new Satpy developers in transitioning from NumPy's
array operations to Satpy's operations, although they are very similar.

To provide the most functionality for users,
Satpy uses the `xarray <http://xarray.pydata.org/en/stable/>`_ library's
:class:`~xarray.DataArray` object as the main representation for its data.
DataArray objects can also benefit from the
`dask <https://dask.pydata.org/en/latest/>`_ library. The combination of
these libraries allow Satpy to easily distribute operations over multiple
workers, lazy evaluate operations, and keep track additional metadata and
coordinate information.

XArray
------

.. code-block:: python

    import xarray as xr

:class:`XArray's DataArray <xarray.DataArray>` is now the standard data
structure for arrays in satpy. They allow the array to define dimensions,
coordinates, and attributes (that we use for metadata).

To create such an array, you can do for example

.. code-block:: python

    my_dataarray = xr.DataArray(my_data, dims=['y', 'x'],
                                coords={'x': np.arange(...)},
                                attrs={'sensor': 'olci'})

where ``my_data`` can be a regular numpy array, a numpy memmap, or, if you
want to keep things lazy, a dask array (more on dask later). Satpy uses dask
arrays with all of its DataArrays.

Dimensions
**********

In satpy, the dimensions of the arrays should include:

- `x` for the x or column or pixel dimension
- `y` for the y or row or line dimension
- `bands` for composites
- `time` can also be provided, but we have limited support for it at the
  moment. Use metadata for common cases (`start_time`, `end_time`)

Dimensions are accessible through
:attr:`my_dataarray.dims <xarray.DataArray.dims>`. To get the size of a
given dimension, use :attr:`~xarray.DataArray.sizes`:

.. code-block:: python

    my_dataarray.sizes['x']

Coordinates
***********

Coordinates can be defined for those dimensions when it makes sense:

- `x` and `y`: Usually defined when the data's area is an
  :class:`~pyresample.geometry.AreaDefinition`, and they contain
  the projection coordinates in x and y.
- `bands`: Contain the letter of the color they represent, eg
  ``['R', 'G', 'B']`` for an RGB composite.

This allows then to select for example a single band like this:

.. code-block:: python

    red = my_composite.sel(bands='R')

or even multiple bands:

.. code-block:: python

    red_and_blue = my_composite.sel(bands=['R', 'B'])

To access the coordinates of the data array, use the following syntax:

.. code-block:: python

    x_coords = my_dataarray['x']
    my_dataarray['y'] = np.arange(...)

Most of the time, satpy will fill the coordinates for you, so you just need to provide the dimension names.

Attributes
**********

To save metadata, we use the :attr:`~xarray.DataArray.attrs` dictionary.

.. code-block:: python

    my_dataarray.attrs['platform_name'] = 'Sentinel-3A'

Some metadata that should always be present in our dataarrays:

- ``area`` the area of the dataset. This should be handled in the reader.
- ``start_time``, ``end_time``
- ``sensor``

Operations on DataArrays
************************

DataArrays work with regular arithmetic operation as one would expect of eg
numpy arrays, with the exception that using an operator on two DataArrays
requires both arrays to share the same dimensions, and coordinates if those
are defined.

For mathematical functions like cos or log, you can use numpy functions
directly and they will return a DataArray object:

.. code-block:: python

    import numpy as np
    cos_zen = np.cos(zen_xarray)

Masking data
************

In DataArrays, masked data is represented with NaN values. Hence the default
type is ``float64``, but ``float32`` works also in this case. XArray can't
handle masked data for integer data, but in satpy we try to use the special
``_FillValue`` attribute (in ``.attrs``) to handle this case. If you come
across a case where this isn't handled properly, contact us.

Masking data from a condition can be done with:

.. code-block:: python

    result = my_dataarray.where(my_dataarray > 5)

Result is then analogous to my_dataarray, with values lower or equal to 5 replaced by NaNs.

Further reading
***************

http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html#xarray.DataArray

Dask
----

.. code-block:: python

    import dask.array as da

The data part of the DataArrays we use in satpy are mostly dask Arrays. That allows lazy and chunked operations for efficient processing.

Creation
********

From a numpy array
++++++++++++++++++

To create a dask array from a numpy array, one can call the
:func:`~dask.array.from_array` function:

.. code-block:: python

    darr = da.from_array(my_numpy_array, chunks=4096)

The *chunks* keyword tells dask the size of a chunk of data. If the numpy
array is 3-dimensional, the chunk size provide above means that one chunk
will be 4096x4096x4096 elements. To prevent this, one can provide a tuple:

.. code-block:: python

    darr = da.from_array(my_numpy_array, chunks=(4096, 1024, 2))

meaning a chunk will be 4096x1024x2 elements in size.

Even more detailed sizes for the chunks can be provided if needed, see the
:doc:`dask documentation <dask:array-chunks>`.

From memmaps or other lazy objects
++++++++++++++++++++++++++++++++++

To avoid loading the data into memory when creating a dask array, other kinds
of arrays can be passed to :func:`~dask.array.from_array`. For example, a
numpy memmap allows dask to know where the data is, and will only be loaded
when the actual values need to be computed. Another example is a hdf5
variable read with h5py.

Procedural generation of data
+++++++++++++++++++++++++++++

Some procedural generation function are available in dask, eg
:func:`~dask.array.meshgrid`, :func:`~dask.array.arange`, or
:func:`random.random <dask.array.random.random>`.

From XArray to Dask and back
****************************

Certain operations are easiest to perform on dask arrays by themselves,
especially when certain functions are only available from the dask library.
In these cases you can operate on the dask array beneath the DataArray and
create a new DataArray when done. Note dask arrays do not support in-place
operations. In-place operations on xarray DataArrays will reassign the dask
array automatically.

.. code-block:: python

    dask_arr = my_dataarray.data
    dask_arr = dask_arr + 1
    # ... other non-xarray operations ...
    new_dataarr = xr.DataArray(dask_arr, dims=my_dataarray.dims, attrs=my_dataarray.attrs.copy())

Or if the operation should be assigned back to the original DataArray (if and
only if the data is the same size):

.. code-block:: python

    my_dataarray.data = dask_arr


Operations and how to get actual results
****************************************

Regular arithmetic operations are provided, and generate another dask array.

    >>> arr1 = da.random.uniform(0, 1000, size=(1000, 1000), chunks=100)
    >>> arr2 = da.random.uniform(0, 1000, size=(1000, 1000), chunks=100)
    >>> arr1 + arr2
    dask.array<add, shape=(1000, 1000), dtype=float64, chunksize=(100, 100)>

In order to compute the actual data during testing, use the
:func:`~dask.compute` method.
In normal Satpy operations you will want the data to be evaluated as late as
possible to improve performance so `compute` should only be used when needed.

    >>> (arr1 + arr2).compute()
    array([[  898.08811639,  1236.96107629,  1154.40255292, ...,
             1537.50752674,  1563.89278664,   433.92598566],
           [ 1657.43843608,  1063.82390257,  1265.08687916, ...,
             1103.90421234,  1721.73564104,  1276.5424228 ],
           [ 1620.11393216,   212.45816261,   771.99348555, ...,
             1675.6561068 ,   585.89123159,   935.04366354],
           ...,
           [ 1533.93265862,  1103.33725432,   191.30794159, ...,
              520.00434673,   426.49238283,  1090.61323471],
           [  816.6108554 ,  1526.36292498,   412.91953023, ...,
              982.71285721,   699.087645  ,  1511.67447362],
           [ 1354.6127365 ,  1671.24591983,  1144.64848757, ...,
             1247.37586051,  1656.50487092,   978.28184726]])

Dask also provides `cos`, `log` and other mathematical function, that you
can use with :func:`da.cos <dask.array.cos>` and
:func:`da.log <dask.array.log>`. However, since satpy uses xarrays as
standard data structure, prefer the xarray functions when possible (they call
in turn the dask counterparts when possible).

Wrapping non-dask friendly functions
************************************

Some operations are not supported by dask yet or are difficult to convert to
take full advantage of dask's multithreaded operations. In these cases you
can wrap a function to run on an entire dask array when it is being computed
and pass on the result. Note that this requires fully computing all of the
dask inputs to the function and are passed as a numpy array or in the case
of an XArray DataArray they will be a DataArray with a numpy array
underneath. You should *NOT* use dask functions inside the delayed function.


.. code-block:: python

    import dask
    import dask.array as da

    def _complex_operation(my_arr1, my_arr2):
        return my_arr1 + my_arr2

    delayed_result = dask.delayed(_complex_operation)(my_dask_arr1, my_dask_arr2)
    # to create a dask array to use in the future
    my_new_arr = da.from_delayed(delayed_result, dtype=my_dask_arr1.dtype, shape=my_dask_arr1.shape)

Dask Delayed objects can also be computed ``delayed_result.compute()`` if
the array is not needed or if the function doesn't return an array.

http://dask.pydata.org/en/latest/array-api.html#dask.array.from_delayed

Map dask blocks to non-dask friendly functions
**********************************************

If the complicated operation you need to perform can be vectorized and does
not need the entire data array to do its operations you can use
:func:`da.map_blocks <dask.array.core.map_blocks>` to get better performance
than creating a delayed function. Similar to delayed functions the inputs to
the function are fully computed DataArrays or numpy arrays, but only the
individual chunks of the dask array at a time. Note that ``map_blocks`` must
be provided dask arrays and won't function properly on XArray DataArrays.
It is recommended that the function object passed to ``map_blocks`` **not**
be an internal function (a function defined inside another function) or it
may be unserializable and can cause issues in some environments.

.. code-block:: python

    my_new_arr = da.map_blocks(_complex_operation, my_dask_arr1, my_dask_arr2, dtype=my_dask_arr1.dtype)

Helpful functions
*****************

- :func:`~dask.array.core.map_blocks`
- :func:`~dask.array.map_overlap`
- :func:`~dask.array.core.atop`
- :func:`~dask.array.store`
- :func:`~dask.array.tokenize`
- :func:`~dask.compute`
- :doc:`dask:delayed`
- :func:`~dask.array.rechunk`
- :attr:`~dask.array.Array.vindex`
