"""
h5netcdf compatibility fixes used by the UVNS reader.

These fixes address backend interoperability issues
encountered when loading NetCDF/HDF5 content through
xarray's h5netcdf backend.

Current fixes:

* HDF5 variable-length UTF-8 string datasets

The fixes are datatype-driven and are not tied to
specific products or variable names.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_h5netcdf_compatibility_fixes():
    """Install all h5netcdf compatibility fixes."""

    _apply_vlen_string_patch()


def _apply_vlen_string_patch():
    """Patch xarray's h5netcdf backend to safely handle VLEN strings."""

    from xarray.backends.h5netcdf_ import (
        H5NetCDFStore,
        H5NetCDFArrayWrapper,
        _read_attributes,
    )

    if getattr(H5NetCDFStore, "_uvns_vlen_patch", False):
        return

    logger.info(
        "Installing h5netcdf VLEN string compatibility patch"
    )

    original = H5NetCDFStore.open_store_variable

    def open_store_variable(self, name, var):
        """Open an h5netcdf variable as an xarray Variable.

        Background
        ----------
        Certain products contain HDF5 variable-length UTF-8 string
        datasets. These appear in h5netcdf as object-backed datasets
        and can be identified via::

            h5py.check_string_dtype(var._h5ds.dtype)

        Investigation showed that xarray's standard
        ``H5NetCDFStore.open_store_variable`` implementation may
        query filter and storage metadata including::

            var.fletcher32
            var.shuffle
            var.compression
            var.compression_opts

        For some VLEN string datasets these metadata accesses can
        trigger a segmentation fault in the h5py/h5netcdf/HDF5 stack
        before an xarray Variable is constructed.

        The underlying dataset itself is valid and can be loaded
        correctly. The failure occurs while inspecting storage
        metadata that is unnecessary for reading string values.

        Implementation
        --------------
        VLEN string datasets are detected using HDF5 dtype metadata.

        For these datasets the normal encoding-construction path is
        bypassed and a minimal Variable is built directly.

        Lazy loading is preserved through::

            indexing.LazilyIndexedArray(
                H5NetCDFArrayWrapper(name, self)
            )

        No data is eagerly read.

        Scope
        -----
        This workaround applies to any HDF5 VLEN string dataset and
        is not tied to specific UVNS products.
        """

        from xarray.core import indexing
        from xarray.core.variable import Variable

        root_h5py = var._root._h5py

        string_info = root_h5py.check_string_dtype(
            var._h5ds.dtype
        )

        is_vlen_string = (
            string_info is not None
            and string_info.length is None
        )

        if is_vlen_string:

            dimensions = var.dimensions

            data = indexing.LazilyIndexedArray(
                H5NetCDFArrayWrapper(
                    name,
                    self,
                )
            )

            attrs = _read_attributes(var)

            #
            # Avoid querying storage/filter metadata
            # such as compression, shuffle or
            # fletcher32 information.
            #
            encoding = {
                "chunksizes": None,
                "source": self._filename,
                "original_shape": data.shape,
                "dtype": str,
            }

            return Variable(
                dimensions,
                data,
                attrs,
                encoding,
            )

        return original(
            self,
            name,
            var,
        )

    H5NetCDFStore.open_store_variable = open_store_variable
    H5NetCDFStore._uvns_vlen_patch = True
