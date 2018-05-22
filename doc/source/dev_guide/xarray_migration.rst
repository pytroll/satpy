============================
Migrating to xarray and dask
============================

Many python developers dealing with meteorologic satellite data begin with
using NumPy arrays directly. This work usually involves masked arrays,
boolean masks, index arrays, and reshaping. Due to the libraries used by
SatPy these operations can't always be done in the same way. This guide acts
as a starting point for new SatPy developers in transitioning from NumPy's
array operations to SatPy's operations, although they are very similar.

To provide the most functionality for users,
SatPy uses the `xarray <http://xarray.pydata.org/en/stable/>`_ library's
:class:`~xarray.DataArray` object as the main representation for its data.
DataArray objects can also benefit from the
`dask <https://dask.pydata.org/en/latest/>`_ library. The combination of
these libraries allow SatPy to easily distribute operations over multiple
workers, lazy evaluate operations, and keep track additional metadata and
coordinate information.

Lazy Operations
===============

.. todo::

    Mention no inplace operations, compute multiple things at a time, etc.

Indexing
========


Masks and fill values
=====================


Chunks
======


