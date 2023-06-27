#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from collections import defaultdict

import numpy as np
import xarray as xr
from packaging.version import Version

from satpy.writers import Writer
from satpy.writers.cf.coords_attrs import add_xy_coords_attrs

logger = logging.getLogger(__name__)

EPOCH = u"seconds since 1970-01-01 00:00:00"

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
    raise ImportError('Ensure that the netCDF4 or h5netcdf package is installed.')

# Numpy datatypes compatible with all netCDF4 backends. ``np.unicode_`` is
# excluded because h5py (and thus h5netcdf) has problems with unicode, see
# https://github.com/h5py/h5py/issues/624."""
NC4_DTYPES = [np.dtype('int8'), np.dtype('uint8'),
              np.dtype('int16'), np.dtype('uint16'),
              np.dtype('int32'), np.dtype('uint32'),
              np.dtype('int64'), np.dtype('uint64'),
              np.dtype('float32'), np.dtype('float64'),
              np.string_]

# Unsigned and int64 isn't CF 1.7 compatible
# Note: Unsigned and int64 are CF 1.9 compatible
CF_DTYPES = [np.dtype('int8'),
             np.dtype('int16'),
             np.dtype('int32'),
             np.dtype('float32'),
             np.dtype('float64'),
             np.string_]

CF_VERSION = 'CF-1.7'


def get_extra_ds(dataarray, keys=None):
    """Get the ancillary_variables DataArrays associated to a dataset."""
    ds_collection = {}
    # Retrieve ancillary variable datarrays
    for ancillary_dataarray in dataarray.attrs.get('ancillary_variables', []):
        ancillary_variable = ancillary_dataarray.name
        if keys and ancillary_variable not in keys:
            keys.append(ancillary_variable)
            ds_collection.update(get_extra_ds(ancillary_dataarray, keys=keys))
    # Add input dataarray
    ds_collection[dataarray.attrs['name']] = dataarray
    return ds_collection


# ###--------------------------------------------------------------------------.
# ### CF-conversion


def _handle_dataarray_name(original_name, numeric_name_prefix):
    if original_name[0].isdigit():
        if numeric_name_prefix:
            new_name = numeric_name_prefix + original_name
        else:
            warnings.warn(
                f'Invalid NetCDF dataset name: {original_name} starts with a digit.',
                stacklevel=5
            )
            new_name = original_name  # occurs when numeric_name_prefix = '', None or False
    else:
        new_name = original_name
    return original_name, new_name


def _preprocess_dataarray_name(dataarray, numeric_name_prefix, include_orig_name):
    """Change the DataArray name by prepending numeric_name_prefix if the name is a digit."""
    original_name = None
    dataarray = dataarray.copy()
    if 'name' in dataarray.attrs:
        original_name = dataarray.attrs.pop('name')
        original_name, new_name = _handle_dataarray_name(original_name, numeric_name_prefix)
        dataarray = dataarray.rename(new_name)

    if include_orig_name and numeric_name_prefix and original_name and original_name != new_name:
        dataarray.attrs['original_name'] = original_name

    return dataarray


def _get_groups(groups, list_datarrays):
    """Return a dictionary with the list of xr.DataArray associated to each group.

    If no groups (groups=None), return all DataArray attached to a single None key.
    Else, collect the DataArrays associated to each group.
    """
    if groups is None:
        grouped_dataarrays = {None: list_datarrays}
    else:
        grouped_dataarrays = defaultdict(list)
        for datarray in list_datarrays:
            for group_name, group_members in groups.items():
                if datarray.attrs['name'] in group_members:
                    grouped_dataarrays[group_name].append(datarray)
                    break
    return grouped_dataarrays


def make_cf_dataarray(dataarray,
                      epoch=EPOCH,
                      flatten_attrs=False,
                      exclude_attrs=None,
                      include_orig_name=True,
                      numeric_name_prefix='CHANNEL_'):
    """Make the xr.DataArray CF-compliant.

    Parameters
    ----------
    dataarray : xr.DataArray
        The data array to be made CF-compliant.
    epoch : str, optional
        Reference time for encoding of time coordinates.
    flatten_attrs : bool, optional
        If True, flatten dict-type attributes.
        The default is False.
    exclude_attrs : list, optional
        List of dataset attributes to be excluded.
        The default is None.
    include_orig_name : bool, optional
        Include the original dataset name in the netcdf variable attributes.
        The default is True.
    numeric_name_prefix : TYPE, optional
        Prepend dataset name with this if starting with a digit.
        The default is ``"CHANNEL_"``.

    Returns
    -------
    new_data : xr.DataArray
        CF-compliant xr.DataArray.

    """
    from satpy.writers.cf.attrs import preprocess_datarray_attrs
    from satpy.writers.cf.time import _process_time_coord

    dataarray = _preprocess_dataarray_name(dataarray=dataarray,
                                           numeric_name_prefix=numeric_name_prefix,
                                           include_orig_name=include_orig_name)
    dataarray = preprocess_datarray_attrs(dataarray=dataarray,
                                          flatten_attrs=flatten_attrs,
                                          exclude_attrs=exclude_attrs)
    dataarray = add_xy_coords_attrs(dataarray)
    dataarray = _process_time_coord(dataarray, epoch=epoch)
    return dataarray


def _collect_cf_dataset(list_dataarrays,
                        epoch=EPOCH,
                        flatten_attrs=False,
                        exclude_attrs=None,
                        include_lonlats=True,
                        pretty=False,
                        include_orig_name=True,
                        numeric_name_prefix='CHANNEL_'):
    """Process a list of xr.DataArray and return a dictionary with CF-compliant xr.Dataset.

    Parameters
    ----------
    list_dataarrays : list
        List of DataArrays to make CF compliant and merge into a xr.Dataset.
    epoch : str
        Reference time for encoding the time coordinates (if available).
        Example format: "seconds since 1970-01-01 00:00:00".
        If None, the default reference time is retrieved using `from satpy.cf_writer import EPOCH`
    flatten_attrs : bool, optional
        If True, flatten dict-type attributes.
    exclude_attrs : list, optional
        List of xr.DataArray attribute names to be excluded.
    include_lonlats : bool, optional
        If True, it includes 'latitude' and 'longitude' coordinates also for satpy scene defined on an AreaDefinition.
        If the 'area' attribute is a SwathDefinition, it always include latitude and longitude coordinates.
    pretty : bool, optional
        Don't modify coordinate names, if possible. Makes the file prettier, but possibly less consistent.
    include_orig_name : bool, optional
        Include the original dataset name as a variable attribute in the xr.Dataset.
    numeric_name_prefix : str, optional
        Prefix to add the each variable with name starting with a digit.
        Use '' or None to leave this out.

    Returns
    -------
    ds : xr.Dataset
        A partially CF-compliant xr.Dataset
    """
    from satpy.writers.cf.area import (
        area2cf,
        assert_xy_unique,
        has_projection_coords,
        link_coords,
        make_alt_coords_unique,
    )

    # Create dictionary of input datarrays
    # --> Since keys=None, it doesn't never retrieve ancillary variables !!!
    ds_collection = {}
    for dataarray in list_dataarrays:
        ds_collection.update(get_extra_ds(dataarray))

    # Check if one DataArray in the collection has 'longitude' or 'latitude'
    got_lonlats = has_projection_coords(ds_collection)

    # Sort dictionary by keys name
    ds_collection = dict(sorted(ds_collection.items()))

    dict_dataarrays = {}
    for dataarray in ds_collection.values():
        dataarray_type = dataarray.dtype
        if dataarray_type not in CF_DTYPES:
            warnings.warn(
                f'dtype {dataarray_type} not compatible with {CF_VERSION}.',
                stacklevel=3
            )
        # Deep copy the datarray since adding/modifying attributes and coordinates
        dataarray = dataarray.copy(deep=True)

        # Add CF-compliant area information from the pyresample area
        # - If include_lonlats=True, add latitude and longitude coordinates
        # - Add grid_mapping attribute to the DataArray
        # - Return the CRS DataArray as first list element
        # - Return the CF-compliant input DataArray as second list element
        try:
            list_new_dataarrays = area2cf(dataarray,
                                          include_lonlats=include_lonlats,
                                          got_lonlats=got_lonlats)
        except KeyError:
            list_new_dataarrays = [dataarray]

        # Ensure each DataArray is CF-compliant
        # --> NOTE: Here the CRS DataArray is repeatedly overwrited
        # --> NOTE: If the input list_dataarrays have different pyresample areas with the same name
        #           area information can be lost here !!!
        for new_dataarray in list_new_dataarrays:
            new_dataarray = make_cf_dataarray(new_dataarray,
                                              epoch=epoch,
                                              flatten_attrs=flatten_attrs,
                                              exclude_attrs=exclude_attrs,
                                              include_orig_name=include_orig_name,
                                              numeric_name_prefix=numeric_name_prefix)
            dict_dataarrays[new_dataarray.name] = new_dataarray

    # Check all DataArray have same size
    assert_xy_unique(dict_dataarrays)

    # Deal with the 'coordinates' attributes indicating lat/lon coords
    # NOTE: this currently is dropped by default !!!
    link_coords(dict_dataarrays)

    # Ensure non-dimensional coordinates to be unique across DataArrays
    # --> If not unique, prepend the DataArray name to the coordinate
    # --> If unique, does not prepend the DataArray name only if pretty=True
    # --> 'longitude' and 'latitude' coordinates are not prepended
    dict_dataarrays = make_alt_coords_unique(dict_dataarrays, pretty=pretty)

    # Create a xr.Dataset
    ds = xr.Dataset(dict_dataarrays)
    return ds


def collect_cf_datasets(list_dataarrays,
                        header_attrs=None,
                        exclude_attrs=None,
                        flatten_attrs=False,
                        pretty=True,
                        include_lonlats=True,
                        epoch=EPOCH,
                        include_orig_name=True,
                        numeric_name_prefix='CHANNEL_',
                        groups=None):
    """Process a list of xr.DataArray and return a dictionary with CF-compliant xr.Datasets.

    If the xr.DataArrays does not share the same dimensions, it creates a collection
    of xr.Datasets sharing the same dimensions.

    Parameters
    ----------
    list_dataarrays (list):
        List of DataArrays to make CF compliant and merge into groups of xr.Datasets.
    header_attrs: (dict):
        Global attributes of the output xr.Dataset.
    epoch (str):
        Reference time for encoding the time coordinates (if available).
        Example format: "seconds since 1970-01-01 00:00:00".
        If None, the default reference time is retrieved using `from satpy.cf_writer import EPOCH`
    flatten_attrs (bool):
        If True, flatten dict-type attributes.
    exclude_attrs (list):
        List of xr.DataArray attribute names to be excluded.
    include_lonlats (bool):
        If True, it includes 'latitude' and 'longitude' coordinates also for satpy scene defined on an AreaDefinition.
        If the 'area' attribute is a SwathDefinition, it always include latitude and longitude coordinates.
    pretty (bool):
        Don't modify coordinate names, if possible. Makes the file prettier, but possibly less consistent.
    include_orig_name (bool).
        Include the original dataset name as a variable attribute in the xr.Dataset.
    numeric_name_prefix (str):
        Prefix to add the each variable with name starting with a digit.
        Use '' or None to leave this out.
    groups (dict):
        Group datasets according to the given assignment:

            `{'<group_name>': ['dataset_name1', 'dataset_name2', ...]}`

        It is used to create grouped netCDFs using the CF_Writer.
        If None (the default), no groups will be created.

    Returns
    -------
    grouped_datasets : dict
        A dictionary of CF-compliant xr.Dataset: {group_name: xr.Dataset}
    header_attrs : dict
        Global attributes to be attached to the xr.Dataset / netCDF4.
    """
    from satpy.writers.cf.attrs import preprocess_header_attrs
    from satpy.writers.cf.time import add_time_bounds_dimension

    if not list_dataarrays:
        raise RuntimeError("None of the requested datasets have been "
                           "generated or could not be loaded. Requested "
                           "composite inputs may need to have matching "
                           "dimensions (eg. through resampling).")

    header_attrs = preprocess_header_attrs(header_attrs=header_attrs,
                                           flatten_attrs=flatten_attrs)

    # Retrieve groups
    # - If groups is None: {None: list_dataarrays}
    # - if groups not None: {group_name: [xr.DataArray, xr.DataArray ,..], ...}
    # Note: if all dataset names are wrong, behave like groups = None !
    grouped_dataarrays = _get_groups(groups, list_dataarrays)
    is_grouped = len(grouped_dataarrays) >= 2

    # If not grouped, add CF conventions.
    # - If 'Conventions' key already present, do not overwrite !
    if "Conventions" not in header_attrs and not is_grouped:
        header_attrs['Conventions'] = CF_VERSION

    # Create dictionary of group xr.Datasets
    # --> If no groups (groups=None) --> group_name=None
    grouped_datasets = {}
    for group_name, group_dataarrays in grouped_dataarrays.items():
        ds = _collect_cf_dataset(
            list_dataarrays=group_dataarrays,
            epoch=epoch,
            flatten_attrs=flatten_attrs,
            exclude_attrs=exclude_attrs,
            include_lonlats=include_lonlats,
            pretty=pretty,
            include_orig_name=include_orig_name,
            numeric_name_prefix=numeric_name_prefix)

        if not is_grouped:
            ds.attrs = header_attrs

        if 'time' in ds:
            ds = add_time_bounds_dimension(ds, time="time")

        grouped_datasets[group_name] = ds
    return grouped_datasets, header_attrs


def _sanitize_writer_kwargs(writer_kwargs):
    """Remove satpy-specific kwargs."""
    writer_kwargs = copy.deepcopy(writer_kwargs)
    satpy_kwargs = ['overlay', 'decorate', 'config_files']
    for kwarg in satpy_kwargs:
        writer_kwargs.pop(kwarg, None)
    return writer_kwargs


def _initialize_root_netcdf(filename, engine, header_attrs, to_netcdf_kwargs):
    """Initialize root empty netCDF."""
    root = xr.Dataset({}, attrs=header_attrs)
    init_nc_kwargs = to_netcdf_kwargs.copy()
    init_nc_kwargs.pop('encoding', None)  # No variables to be encoded at this point
    init_nc_kwargs.pop('unlimited_dims', None)
    written = [root.to_netcdf(filename, engine=engine, mode='w', **init_nc_kwargs)]
    return written


class CFWriter(Writer):
    """Writer producing NetCDF/CF compatible datasets."""

    @staticmethod
    def da2cf(dataarray, epoch=EPOCH, flatten_attrs=False, exclude_attrs=None,
              include_orig_name=True, numeric_name_prefix='CHANNEL_'):
        """Convert the dataarray to something cf-compatible.

        Args:
            dataarray (xr.DataArray):
                The data array to be converted
            epoch (str):
                Reference time for encoding of time coordinates
            flatten_attrs (bool):
                If True, flatten dict-type attributes
            exclude_attrs (list):
                List of dataset attributes to be excluded
            include_orig_name (bool):
                Include the original dataset name in the netcdf variable attributes
            numeric_name_prefix (str):
                Prepend dataset name with this if starting with a digit
        """
        warnings.warn('CFWriter.da2cf is deprecated.'
                      'Use satpy.writers.cf_writer.make_cf_dataarray instead.',
                      DeprecationWarning, stacklevel=3)
        return make_cf_dataarray(dataarray=dataarray,
                                 epoch=epoch,
                                 flatten_attrs=flatten_attrs,
                                 exclude_attrs=exclude_attrs,
                                 include_orig_name=include_orig_name,
                                 numeric_name_prefix=numeric_name_prefix)

    @staticmethod
    def update_encoding(dataset, to_netcdf_kwargs):
        """Update encoding info (deprecated)."""
        from satpy.writers.cf.encoding import update_encoding

        warnings.warn('CFWriter.update_encoding is deprecated. '
                      'Use satpy.writers.cf.encoding.update_encoding instead.',
                      DeprecationWarning, stacklevel=3)
        return update_encoding(dataset, to_netcdf_kwargs)

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Save the *dataset* to a given *filename*."""
        return self.save_datasets([dataset], filename, **kwargs)

    def save_datasets(self, datasets, filename=None, groups=None, header_attrs=None, engine=None, epoch=EPOCH,
                      flatten_attrs=False, exclude_attrs=None, include_lonlats=True, pretty=False,
                      include_orig_name=True, numeric_name_prefix='CHANNEL_', **to_netcdf_kwargs):
        """Save the given datasets in one netCDF file.

        Note that all datasets (if grouping: in one group) must have the same projection coordinates.

        Args:
            datasets (list):
                List of xr.DataArray to be saved.
            filename (str):
                Output file
            groups (dict):
                Group datasets according to the given assignment: `{'group_name': ['dataset1', 'dataset2', ...]}`.
                Group name `None` corresponds to the root of the file, i.e. no group will be created.
                Warning: The results will not be fully CF compliant!
            header_attrs:
                Global attributes to be included.
            engine (str):
                Module to be used for writing netCDF files. Follows xarray's
                :meth:`~xarray.Dataset.to_netcdf` engine choices with a
                preference for 'netcdf4'.
            epoch (str):
                Reference time for encoding of time coordinates.
            flatten_attrs (bool):
                If True, flatten dict-type attributes.
            exclude_attrs (list):
                List of dataset attributes to be excluded.
            include_lonlats (bool):
                Always include latitude and longitude coordinates, even for datasets with area definition.
            pretty (bool):
                Don't modify coordinate names, if possible. Makes the file prettier, but possibly less consistent.
            include_orig_name (bool).
                Include the original dataset name as a variable attribute in the final netCDF.
            numeric_name_prefix (str):
                Prefix to add the each variable with name starting with a digit. Use '' or None to leave this out.

        """
        from satpy.writers.cf.encoding import update_encoding

        logger.info('Saving datasets to NetCDF4/CF.')
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
                                                               to_netcdf_kwargs=to_netcdf_kwargs,
                                                               numeric_name_prefix=numeric_name_prefix)
            res = ds.to_netcdf(filename,
                               engine=engine,
                               group=group_name,
                               mode=mode,
                               encoding=encoding,
                               **other_to_netcdf_kwargs)
            written.append(res)
        return written

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
