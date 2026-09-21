# satpy/readers/_h5netcdf_vlen_patch.py
2

def apply_h5netcdf_vlen_patch():
    """Patch xarray's h5netcdf backend to handle VLEN strings."""

    from xarray.backends.h5netcdf_ import (
        H5NetCDFStore,
        H5NetCDFArrayWrapper,
        _read_attributes,
    )
    from xarray.core import indexing
    from xarray.core.variable import Variable

    if getattr(H5NetCDFStore, "_uvns_vlen_patch", False):
        return

    original = H5NetCDFStore.open_store_variable

    def open_store_variable(self, name, var):
        """Open an h5netcdf variable as an xarray Variable.

        This method applies a special case for HDF5 variable-length (VLEN)
        string datasets.

        Background
        ----------
        Some products contain metadata variables stored as HDF5 VLEN UTF-8
        strings, for example::

            status/instrument/instrument_mode

        These appear in h5netcdf as::

            dtype=object

        and can be identified using::

            h5py.check_string_dtype(var._h5ds.dtype)

        During investigation it was found that xarray's normal
        ``H5NetCDFStore.open_store_variable`` implementation performs a
        number of storage/filter metadata queries, including::

            var.fletcher32
            var.shuffle
            var.compression
            var.compression_opts

        For VLEN string datasets these metadata accesses can trigger a
        segmentation fault in the h5py/h5netcdf/HDF5 stack before the
        Variable is constructed.

        The underlying dataset itself is valid and can be represented
        correctly. The failure occurs only while interrogating storage
        metadata that is not required for reading the string values.

        Implementation
        --------------
        VLEN string datasets are detected using h5py's dtype inspection
        utilities. For such datasets we bypass the normal metadata
        collection path and directly construct an xarray Variable.

        The standard lazy-loading behaviour is preserved because the data
        is still wrapped using::

            indexing.LazilyIndexedArray(
                H5NetCDFArrayWrapper(name, self)
            )

        No data is eagerly read from disk.

        Scope
        -----
        This workaround is datatype-driven and applies to any HDF5 VLEN
        string dataset. It is not tied to a specific product or variable
        name.
        """
        root_h5py = var._root._h5py

        string_info = root_h5py.check_string_dtype(var._h5ds.dtype)

        is_vlen_string = (
            string_info is not None
            and string_info.length is None
        )

        if is_vlen_string:
            dimensions = var.dimensions

            # Preserve lazy loading by using the standard xarray wrapper.
            data = indexing.LazilyIndexedArray(
                H5NetCDFArrayWrapper(name, self)
            )

            attrs = _read_attributes(var)

            # Minimal encoding avoids querying filter/layout metadata
            # (fletcher32, shuffle, compression, etc.) which is the path
            # known to fail for VLEN string datasets.
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

        return original(self, name, var)

    H5NetCDFStore.open_store_variable = open_store_variable
    H5NetCDFStore._uvns_vlen_patch = True
