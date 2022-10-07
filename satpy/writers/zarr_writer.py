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
"""Writer for Zarr.

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
* By default non-dimensional coordinates (such as scanline timestamps) are prefixed with the corresponding
  dataset name. This is because they are likely to be different for each dataset. If a non-dimensional
  coordinate is identical for all datasets, the prefix can be removed by setting ``pretty=True``.
* Some dataset names start with a digit, like AVHRR channels 1, 2, 3a, 3b, 4 and 5. This doesn't comply with CF
  https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/ch02s03.html. These channels are prefixed
  with `CHANNEL_` by default. This can be controlled with the variable `numeric_name_prefix` to `save_datasets`.
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
    ...        'zlib': True,
    ...        'complevel': 9,
    ...        'scale_factor': 0.01,
    ...        'add_offset': 100,
    ...        'dtype': np.int16
    ...     },
    ...    'my_dataset_2': {
    ...        'zlib': False
    ...     }
    ... }
    >>> scn.save_datasets(writer='cf', filename='encoding_test.nc', encoding=my_encoding)


2) Via the ``encoding`` attribute of the datasets in a scene. For example

    >>> scn['my_dataset'].encoding = {'zlib': False}
    >>> scn.save_datasets(writer='cf', filename='encoding_test.nc')

See the `xarray encoding documentation`_ for all encoding options.


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

from satpy.writers import Writer
from satpy.writers.utils import flatten_dict

from .cf_writer import (
    _get_compression,
    _get_groups,
    _handle_dataarray_name,
    _set_history,
    area2cf,
    assert_xy_unique,
    encode_attrs_nc,
    get_extra_ds,
    has_projection_coords,
    link_coords,
    make_alt_coords_unique,
    make_time_bounds,
    update_encoding,
)

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
CF_DTYPES = [np.dtype('int8'),
             np.dtype('int16'),
             np.dtype('int32'),
             np.dtype('float32'),
             np.dtype('float64'),
             np.string_]

CF_VERSION = 'CF-1.7'


class ZarrWriter(Writer):
    """Writer producing CF compatible Zarr storage."""

    @staticmethod
    def da2cf(dataarray, epoch=EPOCH, flatten_attrs=False, exclude_attrs=None, compression=None,
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
        if exclude_attrs is None:
            exclude_attrs = []

        original_name = None
        new_data = dataarray.copy()
        if 'name' in new_data.attrs:
            name = new_data.attrs.pop('name')
            original_name, name = _handle_dataarray_name(name, numeric_name_prefix)
            new_data = new_data.rename(name)

        ZarrWriter._remove_satpy_attributes(new_data)

        # Remove area as well as user-defined attributes
        for key in ['area'] + exclude_attrs:
            new_data.attrs.pop(key, None)

        anc = [ds.attrs['name']
               for ds in new_data.attrs.get('ancillary_variables', [])]
        if anc:
            new_data.attrs['ancillary_variables'] = ' '.join(anc)
        # TODO: make this a grid mapping or lon/lats
        # new_data.attrs['area'] = str(new_data.attrs.get('area'))
        ZarrWriter._cleanup_attrs(new_data)

        if compression is not None:
            new_data.encoding.update(compression)

        new_data = ZarrWriter._encode_time(new_data, epoch)
        new_data = ZarrWriter._encode_coords(new_data)

        if 'long_name' not in new_data.attrs and 'standard_name' not in new_data.attrs:
            new_data.attrs['long_name'] = new_data.name
        if 'prerequisites' in new_data.attrs:
            new_data.attrs['prerequisites'] = [np.string_(str(prereq)) for prereq in new_data.attrs['prerequisites']]

        if include_orig_name and numeric_name_prefix and original_name and original_name != name:
            new_data.attrs['original_name'] = original_name

        # Flatten dict-type attributes, if desired
        if flatten_attrs:
            new_data.attrs = flatten_dict(new_data.attrs)

        # Encode attributes to netcdf-compatible datatype
        new_data.attrs = encode_attrs_nc(new_data.attrs)

        return new_data

    @staticmethod
    def _cleanup_attrs(new_data):
        for key, val in new_data.attrs.copy().items():
            if val is None:
                new_data.attrs.pop(key)
            if key == 'ancillary_variables' and val == []:
                new_data.attrs.pop(key)

    @staticmethod
    def _encode_coords(new_data):
        if 'x' in new_data.coords:
            new_data['x'].attrs['standard_name'] = 'projection_x_coordinate'
            new_data['x'].attrs['units'] = 'm'
        if 'y' in new_data.coords:
            new_data['y'].attrs['standard_name'] = 'projection_y_coordinate'
            new_data['y'].attrs['units'] = 'm'
        if 'crs' in new_data.coords:
            new_data = new_data.drop_vars('crs')
        return new_data

    @staticmethod
    def _encode_time(new_data, epoch):
        if 'time' in new_data.coords:
            new_data['time'].encoding['units'] = epoch
            new_data['time'].attrs['standard_name'] = 'time'
            new_data['time'].attrs.pop('bounds', None)
            new_data = ZarrWriter._add_time_dimension(new_data)
        return new_data

    @staticmethod
    def _add_time_dimension(new_data):
        if 'time' not in new_data.dims and new_data["time"].size not in new_data.shape:
            new_data = new_data.expand_dims('time')
        return new_data

    @staticmethod
    def _remove_satpy_attributes(new_data):
        # Remove _satpy* attributes
        satpy_attrs = [key for key in new_data.attrs if key.startswith('_satpy')]
        for satpy_attr in satpy_attrs:
            new_data.attrs.pop(satpy_attr)
        new_data.attrs.pop('_last_resampler', None)

    @staticmethod
    def update_encoding(dataset, to_netcdf_kwargs):
        """Update encoding info (deprecated)."""
        warnings.warn('ZarrWriter.update_encoding is deprecated. '
                      'Use satpy.writers.cf_writer.update_encoding instead.',
                      DeprecationWarning)
        return update_encoding(dataset, to_netcdf_kwargs)

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Save the *dataset* to a given *filename*."""
        return self.save_datasets([dataset], filename, **kwargs)

    def _collect_datasets(self, datasets, epoch=EPOCH, flatten_attrs=False, exclude_attrs=None, include_lonlats=True,
                          pretty=False, compression=None, include_orig_name=True, numeric_name_prefix='CHANNEL_'):
        """Collect and prepare datasets to be written."""
        ds_collection = {}
        for ds in datasets:
            ds_collection.update(get_extra_ds(ds))
        got_lonlats = has_projection_coords(ds_collection)
        datas = {}
        start_times = []
        end_times = []
        # sort by name, but don't use the name
        for _, ds in sorted(ds_collection.items()):
            if ds.dtype not in CF_DTYPES:
                warnings.warn('Dtype {} not compatible with {}.'.format(str(ds.dtype), CF_VERSION))
            # we may be adding attributes, coordinates, or modifying the
            # structure of attributes
            ds = ds.copy(deep=True)
            try:
                new_datasets = area2cf(ds, strict=include_lonlats, got_lonlats=got_lonlats)
            except KeyError:
                new_datasets = [ds]
            for new_ds in new_datasets:
                start_times.append(new_ds.attrs.get("start_time", None))
                end_times.append(new_ds.attrs.get("end_time", None))
                new_var = self.da2cf(new_ds, epoch=epoch, flatten_attrs=flatten_attrs,
                                     exclude_attrs=exclude_attrs, compression=compression,
                                     include_orig_name=include_orig_name,
                                     numeric_name_prefix=numeric_name_prefix)
                datas[new_var.name] = new_var

        # Check and prepare coordinates
        assert_xy_unique(datas)
        link_coords(datas)
        datas = make_alt_coords_unique(datas, pretty=pretty)

        return datas, start_times, end_times

    def save_datasets(self, datasets, filename=None, groups=None, header_attrs=None, engine=None, epoch=EPOCH,
                      flatten_attrs=False, exclude_attrs=None, include_lonlats=True, pretty=False,
                      compression=None, include_orig_name=True, numeric_name_prefix='CHANNEL_', **to_netcdf_kwargs):
        """Save the given datasets in one netCDF file.

        Note that all datasets (if grouping: in one group) must have the same projection coordinates.

        Args:
            datasets (list):
                Datasets to be saved
            filename (str):
                Output file
            groups (dict):
                Group datasets according to the given assignment: `{'group_name': ['dataset1', 'dataset2', ...]}`.
                Group name `None` corresponds to the root of the file, i.e. no group will be created. Warning: The
                results will not be fully CF compliant!
            header_attrs:
                Global attributes to be included
            engine (str):
                Module to be used for writing netCDF files. Follows xarray's
                :meth:`~xarray.Dataset.to_netcdf` engine choices with a
                preference for 'netcdf4'.
            epoch (str):
                Reference time for encoding of time coordinates
            flatten_attrs (bool):
                If True, flatten dict-type attributes
            exclude_attrs (list):
                List of dataset attributes to be excluded
            include_lonlats (bool):
                Always include latitude and longitude coordinates, even for datasets with area definition
            pretty (bool):
                Don't modify coordinate names, if possible. Makes the file prettier, but possibly less consistent.
            compression (dict):
                Compression to use on the datasets before saving, for example {'zlib': True, 'complevel': 9}.
                This is in turn passed the xarray's `to_netcdf` method:
                http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html for more possibilities.
                (This parameter is now being deprecated, please use the DataArrays's `encoding` from now on.)
            include_orig_name (bool).
                Include the original dataset name as an varaibel attribute in the final netcdf
            numeric_name_prefix (str):
                Prefix to add the each variable with name starting with a digit. Use '' or None to leave this out.

        """
        logger.info('Saving datasets to NetCDF4/CF.')
        compression = _get_compression(compression)

        # Write global attributes to file root (creates the file)
        filename = filename or self.get_filename(**datasets[0].attrs)

        root = xr.Dataset({}, attrs={})
        if header_attrs is not None:
            if flatten_attrs:
                header_attrs = flatten_dict(header_attrs)
            root.attrs = encode_attrs_nc(header_attrs)

        _set_history(root)

        # Remove satpy-specific kwargs
        to_netcdf_kwargs = copy.deepcopy(to_netcdf_kwargs)  # may contain dictionaries (encoding)
        satpy_kwargs = ['overlay', 'decorate', 'config_files']
        for kwarg in satpy_kwargs:
            to_netcdf_kwargs.pop(kwarg, None)

        init_nc_kwargs = to_netcdf_kwargs.copy()
        init_nc_kwargs.pop('encoding', None)  # No variables to be encoded at this point
        init_nc_kwargs.pop('unlimited_dims', None)

        groups_ = _get_groups(groups, datasets, root)

        written = [root.to_zarr(filename, mode='w', **init_nc_kwargs)]

        # Write datasets to groups (appending to the file; group=None means no group)
        for group_name, group_datasets in groups_.items():
            # XXX: Should we combine the info of all datasets?
            datas, start_times, end_times = self._collect_datasets(
                group_datasets, epoch=epoch, flatten_attrs=flatten_attrs, exclude_attrs=exclude_attrs,
                include_lonlats=include_lonlats, pretty=pretty, compression=compression,
                include_orig_name=include_orig_name, numeric_name_prefix=numeric_name_prefix)
            dataset = xr.Dataset(datas)
            if 'time' in dataset:
                dataset['time_bnds'] = make_time_bounds(start_times,
                                                        end_times)
                dataset['time'].attrs['bounds'] = "time_bnds"
                dataset['time'].attrs['standard_name'] = "time"
            else:
                grp_str = ' of group {}'.format(group_name) if group_name is not None else ''
                logger.warning('No time dimension in datasets{}, skipping time bounds creation.'.format(grp_str))

            encoding, other_to_netcdf_kwargs = update_encoding(dataset, to_netcdf_kwargs, numeric_name_prefix)
            # zarr encoding does not allow chunksizes
            for k, v in encoding.items():
                nv = v
                if "chunksizes" in nv:
                    del nv["chunksizes"]
                encoding[k] = nv
            res = dataset.to_zarr(filename, group=group_name, mode='a', encoding=encoding,
                                  **other_to_netcdf_kwargs)
            written.append(res)

        return written
