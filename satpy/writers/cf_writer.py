# Copyright (c) 2017-2019 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Writer for netCDF4/CF.

Example usage
-------------

The CF writer saves datasets in a Scene as `CF-compliant`_ netCDF file. Here is an example with MSG SEVIRI data in HRIT
format:

    >>> from satpy import Scene
    >>> import glob
    >>> filenames = glob.glob('data/H*201903011200*')
    >>> scn = Scene(filenames=filenames, reader='seviri_l1b_hrit')
    >>> scn.load(['VIS006', 'IR_108'])
    >>> scn.save_datasets(writer='cf', datasets=['VIS006', 'IR_108'], filename='seviri_test.nc',
                          exclude_attrs=['raw_metadata'])

* You can select the netCDF backend using the ``engine`` keyword argument. If `None` if follows
  :meth:`~xarray.Dataset.to_netcdf` engine choices with a preference for 'netcdf4'.
* For datasets with area definition you can exclude lat/lon coordinates by setting ``include_lonlats=False``.
  If the area has a projected CRS, units are assumed to be in metre.  If the
  area has a geographic CRS, units are assumed to be in degrees.  The writer
  does not verify that the CRS is supported by the CF conventions.  One
  commonly used projected CRS not supported by the CF conventions is the
  equirectangular projection, such as EPSG 4087.
* By default non-dimensional coordinates (such as scanline timestamps) are prefixed with the corresponding
  dataset name. This is because they are likely to be different for each dataset. If a non-dimensional
  coordinate is identical for all datasets, the prefix can be removed by setting ``pretty=True``.
* Some dataset names start with a digit, like AVHRR channels 1, 2, 3a, 3b, 4 and 5. This doesn't comply with CF
  https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch02s03.html. These channels are prefixed
  with ``"CHANNEL_"`` by default. This can be controlled with the variable `numeric_name_prefix` to `save_datasets`.
  Setting it to `None` or `''` will skip the prefixing.

Grouping
~~~~~~~~

All datasets to be saved must have the same projection coordinates ``x`` and ``y``. If a scene holds datasets with
different grids, the CF compliant workaround is to save the datasets to separate files. Alternatively, you can save
datasets with common grids in separate netCDF groups as follows:

    >>> scn.load(['VIS006', 'IR_108', 'HRV'])
    >>> scn.save_datasets(writer='cf', datasets=['VIS006', 'IR_108', 'HRV'],
                          filename='seviri_test.nc', exclude_attrs=['raw_metadata'],
                          groups={'visir': ['VIS006', 'IR_108'], 'hrv': ['HRV']})

Note that the resulting file will not be fully CF compliant.


Dataset Encoding
~~~~~~~~~~~~~~~~

Dataset encoding can be specified in two ways:

1) Via the ``encoding`` keyword argument of ``save_datasets``:

    >>> my_encoding = {
    ...    'my_dataset_1': {
    ...        'compression': 'zlib',
    ...        'complevel': 9,
    ...        'scale_factor': 0.01,
    ...        'add_offset': 100,
    ...        'dtype': np.int16
    ...     },
    ...    'my_dataset_2': {
    ...        'compression': None,
    ...        'dtype': np.float64
    ...     }
    ... }
    >>> scn.save_datasets(writer='cf', filename='encoding_test.nc', encoding=my_encoding)


2) Via the ``encoding`` attribute of the datasets in a scene. For example

    >>> scn['my_dataset'].encoding = {'compression': 'zlib'}
    >>> scn.save_datasets(writer='cf', filename='encoding_test.nc')

See the `xarray encoding documentation`_ for all encoding options.

.. note::

    Chunk-based compression can be specified with the ``compression`` keyword
    since

        .. code::

            netCDF4-1.6.0
            libnetcdf-4.9.0
            xarray-2022.12.0

    The ``zlib`` keyword is deprecated. Make sure that the versions of
    these modules are all above or all below that reference. Otherwise,
    compression might fail or be ignored silently.


Attribute Encoding
~~~~~~~~~~~~~~~~~~

In the above examples, raw metadata from the HRIT files have been excluded. If you want all attributes to be included,
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
readable as it will create a separate nc-attribute for each item in every dictionary. Keys are concatenated with
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
.. _xarray encoding documentation:
    http://xarray.pydata.org/en/stable/user-guide/io.html?highlight=encoding#writing-encoded-data
"""
import copy
import logging
import warnings

import numpy as np
import xarray as xr
from packaging.version import Version

from satpy.cf.coords import EPOCH  # noqa: F401 (for backward compatibility)
from satpy.writers import Writer

logger = logging.getLogger(__name__)

# Check availability of either netCDF4 or h5netcdf package
try:
    import netCDF4
except ImportError:
    netCDF4 = None

try:
    import h5netcdf
except ImportError:
    h5netcdf = None

# Ensure that either netCDF4 or h5netcdf is available to avoid silent failure
if netCDF4 is None and h5netcdf is None:
    raise ImportError("Ensure that the netCDF4 or h5netcdf package is installed.")


CF_VERSION = "CF-1.7"


# Numpy datatypes compatible with all netCDF4 backends. ``np.str_`` is
# excluded because h5py (and thus h5netcdf) has problems with unicode, see
# https://github.com/h5py/h5py/issues/624."""
NC4_DTYPES = [np.dtype("int8"), np.dtype("uint8"),
              np.dtype("int16"), np.dtype("uint16"),
              np.dtype("int32"), np.dtype("uint32"),
              np.dtype("int64"), np.dtype("uint64"),
              np.dtype("float32"), np.dtype("float64"),
              np.bytes_]

# Unsigned and int64 isn't CF 1.7 compatible
# Note: Unsigned and int64 are CF 1.9 compatible
CF_DTYPES = [np.dtype("int8"),
             np.dtype("int16"),
             np.dtype("int32"),
             np.dtype("float32"),
             np.dtype("float64"),
             np.bytes_]


def _sanitize_writer_kwargs(writer_kwargs):
    """Remove satpy-specific kwargs."""
    writer_kwargs = copy.deepcopy(writer_kwargs)
    satpy_kwargs = ["overlay", "decorate", "config_files"]
    for kwarg in satpy_kwargs:
        writer_kwargs.pop(kwarg, None)
    return writer_kwargs


def _initialize_root_netcdf(filename, engine, header_attrs, to_netcdf_kwargs):
    """Initialize root empty netCDF."""
    root = xr.Dataset({}, attrs=header_attrs)
    init_nc_kwargs = to_netcdf_kwargs.copy()
    init_nc_kwargs.pop("encoding", None)  # No variables to be encoded at this point
    init_nc_kwargs.pop("unlimited_dims", None)
    written = [root.to_netcdf(filename, engine=engine, mode="w", **init_nc_kwargs)]
    return written


class CFWriter(Writer):
    """Writer producing NetCDF/CF compatible datasets."""

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Save the *dataset* to a given *filename*."""
        return self.save_datasets([dataset], filename, **kwargs)

    def save_datasets(self, datasets, filename=None, groups=None, header_attrs=None, engine=None, epoch=None,  # noqa: D417
                      flatten_attrs=False, exclude_attrs=None, include_lonlats=True, pretty=False,
                      include_orig_name=True, numeric_name_prefix="CHANNEL_", **to_netcdf_kwargs):
        """Save the given datasets in one netCDF file.

        Note that all datasets (if grouping: in one group) must have the same projection coordinates.

        Args:
            datasets (list): List of xr.DataArray to be saved.
            filename (str): Output file.
            groups (dict): Group datasets according to the given assignment:
                `{'group_name': ['dataset1', 'dataset2', ...]}`.
                The group name `None` corresponds to the root of the file, i.e., no group will be created.
                Warning: The results will not be fully CF compliant!
            header_attrs: Global attributes to be included.
            engine (str, optional): Module to be used for writing netCDF files. Follows xarray's
                :meth:`~xarray.Dataset.to_netcdf` engine choices with a preference for 'netcdf4'.
            epoch (str, optional): Reference time for encoding of time coordinates.
                If None, the default reference time is defined using `from satpy.cf.coords import EPOCH`.
            flatten_attrs (bool, optional): If True, flatten dict-type attributes.
            exclude_attrs (list, optional): List of dataset attributes to be excluded.
            include_lonlats (bool, optional): Always include latitude and longitude coordinates,
                even for datasets with area definition.
            pretty (bool, optional): Don't modify coordinate names, if possible.
                Makes the file prettier, but possibly less consistent.
            include_orig_name (bool, optional): Include the original dataset name as a variable
                attribute in the final netCDF.
            numeric_name_prefix (str, optional): Prefix to add to each variable with a name starting with a digit.
                Use '' or None to leave this out.
        """
        from satpy.cf.datasets import collect_cf_datasets
        from satpy.cf.encoding import update_encoding

        logger.info("Saving datasets to NetCDF4/CF.")
        _check_backend_versions()

        # Define netCDF filename if not provided
        # - It infers the name from the first DataArray
        filename = filename or self.get_filename(**datasets[0].attrs)

        # Collect xr.Dataset for each group
        grouped_datasets, header_attrs = collect_cf_datasets(list_dataarrays=datasets,  # list of xr.DataArray
                                                             header_attrs=header_attrs,
                                                             exclude_attrs=exclude_attrs,
                                                             flatten_attrs=flatten_attrs,
                                                             pretty=pretty,
                                                             include_lonlats=include_lonlats,
                                                             epoch=epoch,
                                                             include_orig_name=include_orig_name,
                                                             numeric_name_prefix=numeric_name_prefix,
                                                             groups=groups,
                                                             )
        # Remove satpy-specific kwargs
        # - This kwargs can contain encoding dictionary
        to_netcdf_kwargs = _sanitize_writer_kwargs(to_netcdf_kwargs)

        # If writing grouped netCDF, create an empty "root" netCDF file
        # - Add the global attributes
        # - All groups will be appended in the for loop below
        if groups is not None:
            written = _initialize_root_netcdf(filename=filename,
                                              engine=engine,
                                              header_attrs=header_attrs,
                                              to_netcdf_kwargs=to_netcdf_kwargs)
            mode = "a"
        else:
            mode = "w"
            written = []

        # Write the netCDF
        # - If grouped netCDF, it appends to the root file
        # - If single netCDF, it write directly
        for group_name, ds in grouped_datasets.items():
            encoding, other_to_netcdf_kwargs = update_encoding(ds,
                                                               to_engine_kwargs=to_netcdf_kwargs,
                                                               numeric_name_prefix=numeric_name_prefix)
            res = ds.to_netcdf(filename,
                               engine=engine,
                               group=group_name,
                               mode=mode,
                               encoding=encoding,
                               **other_to_netcdf_kwargs)
            written.append(res)
        return written

    @staticmethod
    def da2cf(dataarray, epoch=None, flatten_attrs=False, exclude_attrs=None,
              include_orig_name=True, numeric_name_prefix="CHANNEL_"):
        """Convert the dataarray to something cf-compatible.

        Args:
            dataarray (xr.DataArray):
                The data array to be converted.
            epoch (str):
                Reference time for encoding of time coordinates.
                If None, the default reference time is defined using `from satpy.cf.coords import EPOCH`
            flatten_attrs (bool):
                If True, flatten dict-type attributes.
            exclude_attrs (list):
                List of dataset attributes to be excluded.
            include_orig_name (bool):
                Include the original dataset name in the netcdf variable attributes.
            numeric_name_prefix (str):
                Prepend dataset name with this if starting with a digit.
        """
        from satpy.cf.data_array import make_cf_data_array
        warnings.warn("CFWriter.da2cf is deprecated."
                      "Use satpy.cf.dataarray.make_cf_data_array instead.",
                      DeprecationWarning, stacklevel=3)
        return make_cf_data_array(dataarray=dataarray,
                                  epoch=epoch,
                                  flatten_attrs=flatten_attrs,
                                  exclude_attrs=exclude_attrs,
                                  include_orig_name=include_orig_name,
                                  numeric_name_prefix=numeric_name_prefix)

    @staticmethod
    def update_encoding(dataset, to_netcdf_kwargs):
        """Update encoding info (deprecated)."""
        from satpy.cf.encoding import update_encoding

        warnings.warn("CFWriter.update_encoding is deprecated. "
                      "Use satpy.cf.encoding.update_encoding instead.",
                      DeprecationWarning, stacklevel=3)
        return update_encoding(dataset, to_netcdf_kwargs)


# --------------------------------------------------------------------------.
# NetCDF version


def _check_backend_versions():
    """Issue warning if backend versions do not match."""
    if not _backend_versions_match():
        warnings.warn(
            "Backend version mismatch. Compression might fail or be ignored "
            "silently. Recommended: All versions below or above "
            "netCDF4-1.6.0/libnetcdf-4.9.0/xarray-2022.12.0.",
            stacklevel=3
        )


def _backend_versions_match():
    versions = _get_backend_versions()
    reference = {
        "netCDF4": Version("1.6.0"),
        "libnetcdf": Version("4.9.0"),
        "xarray": Version("2022.12.0")
    }
    is_newer = [
        versions[module] >= reference[module]
        for module in versions
    ]
    all_newer = all(is_newer)
    all_older = not any(is_newer)
    return all_newer or all_older


def _get_backend_versions():
    import netCDF4
    return {
        "netCDF4": Version(netCDF4.__version__),
        "libnetcdf": Version(netCDF4.__netcdf4libversion__),
        "xarray": Version(xr.__version__)
    }
