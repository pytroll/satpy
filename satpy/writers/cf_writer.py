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
import json
import logging
import warnings
from collections import OrderedDict, defaultdict
from datetime import datetime

import numpy as np
import xarray as xr
from dask.base import tokenize
from packaging.version import Version
from pyresample.geometry import AreaDefinition, SwathDefinition
from xarray.coding.times import CFDatetimeCoder

from satpy.writers import Writer
from satpy.writers.cf.coords_attrs import add_xy_coords_attrs
from satpy.writers.utils import flatten_dict

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

# Numpy datatypes compatible with all netCDF4 backends. ``np.str_`` is
# excluded because h5py (and thus h5netcdf) has problems with unicode, see
# https://github.com/h5py/h5py/issues/624."""
NC4_DTYPES = [np.dtype('int8'), np.dtype('uint8'),
              np.dtype('int16'), np.dtype('uint16'),
              np.dtype('int32'), np.dtype('uint32'),
              np.dtype('int64'), np.dtype('uint64'),
              np.dtype('float32'), np.dtype('float64'),
              np.bytes_]

# Unsigned and int64 isn't CF 1.7 compatible
# Note: Unsigned and int64 are CF 1.9 compatible
CF_DTYPES = [np.dtype('int8'),
             np.dtype('int16'),
             np.dtype('int32'),
             np.dtype('float32'),
             np.dtype('float64'),
             np.bytes_]

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
# ### CF-Area


def add_lonlat_coords(dataarray):
    """Add 'longitude' and 'latitude' coordinates to DataArray."""
    dataarray = dataarray.copy()
    area = dataarray.attrs['area']
    ignore_dims = {dim: 0 for dim in dataarray.dims if dim not in ['x', 'y']}
    chunks = getattr(dataarray.isel(**ignore_dims), 'chunks', None)
    lons, lats = area.get_lonlats(chunks=chunks)
    dataarray['longitude'] = xr.DataArray(lons, dims=['y', 'x'],
                                          attrs={'name': "longitude",
                                                 'standard_name': "longitude",
                                                 'units': 'degrees_east'},
                                          name='longitude')
    dataarray['latitude'] = xr.DataArray(lats, dims=['y', 'x'],
                                         attrs={'name': "latitude",
                                                'standard_name': "latitude",
                                                'units': 'degrees_north'},
                                         name='latitude')
    return dataarray


def _create_grid_mapping(area):
    """Create the grid mapping instance for `area`."""
    import pyproj

    if Version(pyproj.__version__) < Version('2.4.1'):
        # technically 2.2, but important bug fixes in 2.4.1
        raise ImportError("'cf' writer requires pyproj 2.4.1 or greater")
    # let pyproj do the heavily lifting (pyproj 2.0+ required)
    grid_mapping = area.crs.to_cf()
    return area.area_id, grid_mapping


def _add_grid_mapping(dataarray):
    """Convert an area to at CF grid mapping."""
    dataarray = dataarray.copy()
    area = dataarray.attrs['area']
    gmapping_var_name, attrs = _create_grid_mapping(area)
    dataarray.attrs['grid_mapping'] = gmapping_var_name
    return dataarray, xr.DataArray(0, attrs=attrs, name=gmapping_var_name)


def area2cf(dataarray, include_lonlats=False, got_lonlats=False):
    """Convert an area to at CF grid mapping or lon and lats."""
    res = []
    if not got_lonlats and (isinstance(dataarray.attrs['area'], SwathDefinition) or include_lonlats):
        dataarray = add_lonlat_coords(dataarray)
    if isinstance(dataarray.attrs['area'], AreaDefinition):
        dataarray, gmapping = _add_grid_mapping(dataarray)
        res.append(gmapping)
    res.append(dataarray)
    return res


def is_lon_or_lat_dataarray(dataarray):
    """Check if the DataArray represents the latitude or longitude coordinate."""
    if 'standard_name' in dataarray.attrs and dataarray.attrs['standard_name'] in ['longitude', 'latitude']:
        return True
    return False


def has_projection_coords(ds_collection):
    """Check if DataArray collection has a "longitude" or "latitude" DataArray."""
    for dataarray in ds_collection.values():
        if is_lon_or_lat_dataarray(dataarray):
            return True
    return False


def make_alt_coords_unique(datas, pretty=False):
    """Make non-dimensional coordinates unique among all datasets.

    Non-dimensional (or alternative) coordinates, such as scanline timestamps,
    may occur in multiple datasets with the same name and dimension
    but different values.

    In order to avoid conflicts, prepend the dataset name to the coordinate name.
    If a non-dimensional coordinate is unique among all datasets and ``pretty=True``,
    its name will not be modified.

    Since all datasets must have the same projection coordinates,
    this is not applied to latitude and longitude.

    Args:
        datas (dict):
            Dictionary of (dataset name, dataset)
        pretty (bool):
            Don't modify coordinate names, if possible. Makes the file prettier, but possibly less consistent.

    Returns:
        Dictionary holding the updated datasets

    """
    # Determine which non-dimensional coordinates are unique
    tokens = defaultdict(set)
    for dataset in datas.values():
        for coord_name in dataset.coords:
            if not is_lon_or_lat_dataarray(dataset[coord_name]) and coord_name not in dataset.dims:
                tokens[coord_name].add(tokenize(dataset[coord_name].data))
    coords_unique = dict([(coord_name, len(tokens) == 1) for coord_name, tokens in tokens.items()])

    # Prepend dataset name, if not unique or no pretty-format desired
    new_datas = datas.copy()
    for coord_name, unique in coords_unique.items():
        if not pretty or not unique:
            if pretty:
                warnings.warn(
                    'Cannot pretty-format "{}" coordinates because they are '
                    'not identical among the given datasets'.format(coord_name),
                    stacklevel=2
                )
            for ds_name, dataset in datas.items():
                if coord_name in dataset.coords:
                    rename = {coord_name: '{}_{}'.format(ds_name, coord_name)}
                    new_datas[ds_name] = new_datas[ds_name].rename(rename)

    return new_datas


def assert_xy_unique(datas):
    """Check that all datasets share the same projection coordinates x/y."""
    unique_x = set()
    unique_y = set()
    for dataset in datas.values():
        if 'y' in dataset.dims:
            token_y = tokenize(dataset['y'].data)
            unique_y.add(token_y)
        if 'x' in dataset.dims:
            token_x = tokenize(dataset['x'].data)
            unique_x.add(token_x)
    if len(unique_x) > 1 or len(unique_y) > 1:
        raise ValueError('Datasets to be saved in one file (or one group) must have identical projection coordinates. '
                         'Please group them by area or save them in separate files.')


def link_coords(datas):
    """Link dataarrays and coordinates.

    If the `coordinates` attribute of a data array links to other dataarrays in the scene, for example
    `coordinates='lon lat'`, add them as coordinates to the data array and drop that attribute. In the final call to
    `xr.Dataset.to_netcdf()` all coordinate relations will be resolved and the `coordinates` attributes be set
    automatically.

    """
    for da_name, data in datas.items():
        declared_coordinates = data.attrs.get('coordinates', [])
        if isinstance(declared_coordinates, str):
            declared_coordinates = declared_coordinates.split(' ')
        for coord in declared_coordinates:
            if coord not in data.coords:
                try:
                    dimensions_not_in_data = list(set(datas[coord].dims) - set(data.dims))
                    data[coord] = datas[coord].squeeze(dimensions_not_in_data, drop=True)
                except KeyError:
                    warnings.warn(
                        'Coordinate "{}" referenced by dataarray {} does not '
                        'exist, dropping reference.'.format(coord, da_name),
                        stacklevel=2
                    )
                    continue

        # Drop 'coordinates' attribute in any case to avoid conflicts in xr.Dataset.to_netcdf()
        data.attrs.pop('coordinates', None)


# ###--------------------------------------------------------------------------.
# ### CF-Time
def add_time_bounds_dimension(ds, time="time"):
    """Add time bound dimension to xr.Dataset."""
    start_times = []
    end_times = []
    for _var_name, data_array in ds.items():
        start_times.append(data_array.attrs.get("start_time", None))
        end_times.append(data_array.attrs.get("end_time", None))

    start_time = min(start_time for start_time in start_times
                     if start_time is not None)
    end_time = min(end_time for end_time in end_times
                   if end_time is not None)
    ds['time_bnds'] = xr.DataArray([[np.datetime64(start_time),
                                     np.datetime64(end_time)]],
                                   dims=['time', 'bnds_1d'])
    ds[time].attrs['bounds'] = "time_bnds"
    ds[time].attrs['standard_name'] = "time"
    return ds


def _process_time_coord(dataarray, epoch):
    """Process the 'time' coordinate, if existing.

    If expand the DataArray with a time dimension if does not yet exists.

    The function assumes

        - that x and y dimensions have at least shape > 1
        - the time coordinate has size 1

    """
    if 'time' in dataarray.coords:
        dataarray['time'].encoding['units'] = epoch
        dataarray['time'].attrs['standard_name'] = 'time'
        dataarray['time'].attrs.pop('bounds', None)

        if 'time' not in dataarray.dims and dataarray["time"].size not in dataarray.shape:
            dataarray = dataarray.expand_dims('time')

    return dataarray


# --------------------------------------------------------------------------.
# ### Attributes


class AttributeEncoder(json.JSONEncoder):
    """JSON encoder for dataset attributes."""

    def default(self, obj):
        """Return a json-serializable object for *obj*.

        In order to facilitate decoding, elements in dictionaries, lists/tuples and multi-dimensional arrays are
        encoded recursively.
        """
        if isinstance(obj, dict):
            serialized = {}
            for key, val in obj.items():
                serialized[key] = self.default(val)
            return serialized
        elif isinstance(obj, (list, tuple, np.ndarray)):
            return [self.default(item) for item in obj]
        return self._encode(obj)

    def _encode(self, obj):
        """Encode the given object as a json-serializable datatype."""
        if isinstance(obj, (bool, np.bool_)):
            # Bool has to be checked first, because it is a subclass of int
            return str(obj).lower()
        elif isinstance(obj, (int, float, str)):
            return obj
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.void):
            return tuple(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return str(obj)


def _encode_nc(obj):
    """Try to encode `obj` as a netcdf compatible datatype which most closely resembles the object's nature.

    Raises:
        ValueError if no such datatype could be found

    """
    if isinstance(obj, int) and not isinstance(obj, (bool, np.bool_)):
        return obj
    elif isinstance(obj, (float, str, np.integer, np.floating)):
        return obj
    elif isinstance(obj, np.ndarray):
        # Only plain 1-d arrays are supported. Skip record arrays and multi-dimensional arrays.
        is_plain_1d = not obj.dtype.fields and len(obj.shape) <= 1
        if is_plain_1d:
            if obj.dtype in NC4_DTYPES:
                return obj
            elif obj.dtype == np.bool_:
                # Boolean arrays are not supported, convert to array of strings.
                return [s.lower() for s in obj.astype(str)]
            return obj.tolist()

    raise ValueError('Unable to encode')


def encode_nc(obj):
    """Encode the given object as a netcdf compatible datatype."""
    try:
        return obj.to_cf()
    except AttributeError:
        return _encode_python_objects(obj)


def _encode_python_objects(obj):
    """Try to find the datatype which most closely resembles the object's nature.

    If on failure, encode as a string. Plain lists are encoded recursively.
    """
    if isinstance(obj, (list, tuple)) and all([not isinstance(item, (list, tuple)) for item in obj]):
        return [encode_nc(item) for item in obj]
    try:
        dump = _encode_nc(obj)
    except ValueError:
        try:
            # Decode byte-strings
            decoded = obj.decode()
        except AttributeError:
            decoded = obj
        dump = json.dumps(decoded, cls=AttributeEncoder).strip('"')
    return dump


def encode_attrs_nc(attrs):
    """Encode dataset attributes in a netcdf compatible datatype.

    Args:
        attrs (dict):
            Attributes to be encoded
    Returns:
        dict: Encoded (and sorted) attributes

    """
    encoded_attrs = []
    for key, val in sorted(attrs.items()):
        if val is not None:
            encoded_attrs.append((key, encode_nc(val)))
    return OrderedDict(encoded_attrs)


def _add_ancillary_variables_attrs(dataarray):
    """Replace ancillary_variables DataArray with a list of their name."""
    list_ancillary_variable_names = [da_ancillary.attrs['name']
                                     for da_ancillary in dataarray.attrs.get('ancillary_variables', [])]
    if list_ancillary_variable_names:
        dataarray.attrs['ancillary_variables'] = ' '.join(list_ancillary_variable_names)
    else:
        dataarray.attrs.pop("ancillary_variables", None)
    return dataarray


def _drop_exclude_attrs(dataarray, exclude_attrs):
    """Remove user-specified list of attributes."""
    if exclude_attrs is None:
        exclude_attrs = []
    for key in exclude_attrs:
        dataarray.attrs.pop(key, None)
    return dataarray


def _remove_satpy_attrs(new_data):
    """Remove _satpy attribute."""
    satpy_attrs = [key for key in new_data.attrs if key.startswith('_satpy')]
    for satpy_attr in satpy_attrs:
        new_data.attrs.pop(satpy_attr)
    new_data.attrs.pop('_last_resampler', None)
    return new_data


def _format_prerequisites_attrs(dataarray):
    """Reformat prerequisites attribute value to string."""
    if 'prerequisites' in dataarray.attrs:
        dataarray.attrs['prerequisites'] = [np.bytes_(str(prereq)) for prereq in dataarray.attrs['prerequisites']]
    return dataarray


def _remove_none_attrs(dataarray):
    """Remove attribute keys with None value."""
    for key, val in dataarray.attrs.copy().items():
        if val is None:
            dataarray.attrs.pop(key)
    return dataarray


def preprocess_datarray_attrs(dataarray, flatten_attrs, exclude_attrs):
    """Preprocess DataArray attributes to be written into CF-compliant netCDF/Zarr."""
    dataarray = _remove_satpy_attrs(dataarray)
    dataarray = _add_ancillary_variables_attrs(dataarray)
    dataarray = _drop_exclude_attrs(dataarray, exclude_attrs)
    dataarray = _format_prerequisites_attrs(dataarray)
    dataarray = _remove_none_attrs(dataarray)
    _ = dataarray.attrs.pop("area", None)

    if 'long_name' not in dataarray.attrs and 'standard_name' not in dataarray.attrs:
        dataarray.attrs['long_name'] = dataarray.name

    if flatten_attrs:
        dataarray.attrs = flatten_dict(dataarray.attrs)

    dataarray.attrs = encode_attrs_nc(dataarray.attrs)

    return dataarray


def preprocess_header_attrs(header_attrs, flatten_attrs=False):
    """Prepare file header attributes."""
    if header_attrs is not None:
        if flatten_attrs:
            header_attrs = flatten_dict(header_attrs)
        header_attrs = encode_attrs_nc(header_attrs)  # OrderedDict
    else:
        header_attrs = {}
    header_attrs = _add_history(header_attrs)
    return header_attrs


# ###--------------------------------------------------------------------------.
# ### netCDF encodings


def _set_default_chunks(encoding, dataset):
    """Update encoding to preserve current dask chunks.

    Existing user-defined chunks take precedence.
    """
    for var_name, variable in dataset.variables.items():
        if variable.chunks:
            chunks = tuple(
                np.stack([variable.data.chunksize,
                          variable.shape]).min(axis=0)
            )  # Chunksize may not exceed shape
            encoding.setdefault(var_name, {})
            encoding[var_name].setdefault('chunksizes', chunks)
    return encoding


def _set_default_fill_value(encoding, dataset):
    """Set default fill values.

    Avoid _FillValue attribute being added to coordinate variables
    (https://github.com/pydata/xarray/issues/1865).
    """
    coord_vars = []
    for data_array in dataset.values():
        coord_vars.extend(set(data_array.dims).intersection(data_array.coords))
    for coord_var in coord_vars:
        encoding.setdefault(coord_var, {})
        encoding[coord_var].update({'_FillValue': None})
    return encoding


def _set_default_time_encoding(encoding, dataset):
    """Set default time encoding.

    Make sure time coordinates and bounds have the same units.
    Default is xarray's CF datetime encoding, which can be overridden
    by user-defined encoding.
    """
    if 'time' in dataset:
        try:
            dtnp64 = dataset['time'].data[0]
        except IndexError:
            dtnp64 = dataset['time'].data

        default = CFDatetimeCoder().encode(xr.DataArray(dtnp64))
        time_enc = {'units': default.attrs['units'], 'calendar': default.attrs['calendar']}
        time_enc.update(encoding.get('time', {}))
        bounds_enc = {'units': time_enc['units'],
                      'calendar': time_enc['calendar'],
                      '_FillValue': None}
        encoding['time'] = time_enc
        encoding['time_bnds'] = bounds_enc  # FUTURE: Not required anymore with xarray-0.14+
    return encoding


def _update_encoding_dataset_names(encoding, dataset, numeric_name_prefix):
    """Ensure variable names of the encoding dictionary account for numeric_name_prefix.

    A lot of channel names in satpy starts with a digit.
    When preparing CF-compliant datasets, these channels are prefixed with numeric_name_prefix.

    If variables names in the encoding dictionary are numeric digits, their name is prefixed
    with numeric_name_prefix
    """
    for var_name in list(dataset.variables):
        if not numeric_name_prefix or not var_name.startswith(numeric_name_prefix):
            continue
        orig_var_name = var_name.replace(numeric_name_prefix, '')
        if orig_var_name in encoding:
            encoding[var_name] = encoding.pop(orig_var_name)
    return encoding


def update_encoding(dataset, to_netcdf_kwargs, numeric_name_prefix='CHANNEL_'):
    """Update encoding.

    Preserve dask chunks, avoid fill values in coordinate variables and make sure that
    time & time bounds have the same units.
    """
    other_to_netcdf_kwargs = to_netcdf_kwargs.copy()
    encoding = other_to_netcdf_kwargs.pop('encoding', {}).copy()
    encoding = _update_encoding_dataset_names(encoding, dataset, numeric_name_prefix)
    encoding = _set_default_chunks(encoding, dataset)
    encoding = _set_default_fill_value(encoding, dataset)
    encoding = _set_default_time_encoding(encoding, dataset)
    return encoding, other_to_netcdf_kwargs


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


def _add_history(attrs):
    """Add 'history' attribute to dictionary."""
    _history_create = 'Created by pytroll/satpy on {}'.format(datetime.utcnow())
    if 'history' in attrs:
        if isinstance(attrs['history'], list):
            attrs['history'] = ''.join(attrs['history'])
        attrs['history'] += '\n' + _history_create
    else:
        attrs['history'] = _history_create
    return attrs


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
        warnings.warn('CFWriter.update_encoding is deprecated. '
                      'Use satpy.writers.cf_writer.update_encoding instead.',
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
