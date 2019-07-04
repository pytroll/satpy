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

* You can select the netCDF backend using the ``engine`` keyword argument. Default is ``h5netcdf``.
* For datasets with area definition you can exclude lat/lon coordinates by setting ``include_lonlats=False``.
* By default the dataset name is prepended to non-dimensional coordinates such as scanline timestamps. This ensures
  maximum consistency, i.e. the netCDF variable names are independent of the number/set of datasets to be written.
  If a non-dimensional coordinate is identical for


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
"""

from collections import OrderedDict, defaultdict
import logging
from datetime import datetime
import json
import warnings

from dask.base import tokenize
import xarray as xr
import numpy as np

from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy.writers import Writer
from satpy.writers.utils import flatten_dict


logger = logging.getLogger(__name__)

EPOCH = u"seconds since 1970-01-01 00:00:00"

NC4_DTYPES = [np.dtype('int8'), np.dtype('uint8'),
              np.dtype('int16'), np.dtype('uint16'),
              np.dtype('int32'), np.dtype('uint32'),
              np.dtype('int64'), np.dtype('uint64'),
              np.dtype('float32'), np.dtype('float64'),
              np.string_]
"""Numpy datatypes compatible with all netCDF4 backends. ``np.unicode_`` is excluded because h5py (and thus h5netcdf)
has problems with unicode, see https://github.com/h5py/h5py/issues/624."""


def omerc2cf(area):
    """Return the cf grid mapping for the omerc projection."""
    proj_dict = area.proj_dict

    args = dict(azimuth_of_central_line=proj_dict.get('alpha'),
                latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lonc'),
                grid_mapping_name='oblique_mercator',
                reference_ellipsoid_name=proj_dict.get('ellps', 'WGS84'),
                prime_meridian_name=proj_dict.get('pm', 'Greenwich'),
                horizontal_datum_name=proj_dict.get('datum', 'unknown'),
                geographic_crs_name='unknown',
                false_easting=0.,
                false_northing=0.
                )
    if "no_rot" in proj_dict:
        args['no_rotation'] = 1
    if "gamma" in proj_dict:
        args['gamma'] = proj_dict['gamma']
    return args


def geos2cf(area):
    """Return the cf grid mapping for the geos projection."""
    proj_dict = area.proj_dict
    args = dict(perspective_point_height=proj_dict.get('h'),
                latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lon_0'),
                grid_mapping_name='geostationary',
                semi_major_axis=proj_dict.get('a'),
                semi_minor_axis=proj_dict.get('b'),
                sweep_axis=proj_dict.get('sweep'),
                )
    return args


def laea2cf(area):
    """Return the cf grid mapping for the laea projection."""
    proj_dict = area.proj_dict
    args = dict(latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lon_0'),
                grid_mapping_name='lambert_azimuthal_equal_area',
                )
    return args


mappings = {'omerc': omerc2cf,
            'laea': laea2cf,
            'geos': geos2cf}


def create_grid_mapping(area):
    """Create the grid mapping instance for `area`."""
    try:
        grid_mapping = mappings[area.proj_dict['proj']](area)
        grid_mapping['name'] = area.proj_dict['proj']
    except KeyError:
        warnings.warn('The projection "{}" is either not CF compliant or not implemented yet. '
                      'Using the proj4 string instead.'.format(area.proj_str))
        grid_mapping = {'name': 'proj4', 'proj4': area.proj_str}

    return grid_mapping


def get_extra_ds(dataset):
    """Get the extra datasets associated to *dataset*."""
    ds_collection = {}
    for ds in dataset.attrs.get('ancillary_variables', []):
        ds_collection.update(get_extra_ds(ds))
    ds_collection[dataset.attrs['name']] = dataset

    return ds_collection


def area2lonlat(dataarray):
    """Convert an area to longitudes and latitudes."""
    area = dataarray.attrs['area']
    lons, lats = area.get_lonlats_dask()
    lons = xr.DataArray(lons, dims=['y', 'x'],
                        attrs={'name': "longitude",
                               'standard_name': "longitude",
                               'units': 'degrees_east'},
                        name='longitude')
    lats = xr.DataArray(lats, dims=['y', 'x'],
                        attrs={'name': "latitude",
                               'standard_name': "latitude",
                               'units': 'degrees_north'},
                        name='latitude')
    dataarray['longitude'] = lons
    dataarray['latitude'] = lats
    return [dataarray]


def area2gridmapping(dataarray):
    """Convert an area to at CF grid mapping."""
    area = dataarray.attrs['area']
    attrs = create_grid_mapping(area)
    if attrs is not None and 'name' in attrs.keys() and attrs['name'] != "proj4":
        dataarray.attrs['grid_mapping'] = attrs['name']
        name = attrs['name']
    else:
        # Handle the case when the projection cannot be converted to a standard CF representation or this has not
        # been implemented yet.
        dataarray.attrs['grid_proj4'] = area.proj4_string
        name = "proj4"
    return [dataarray, xr.DataArray(0, attrs=attrs, name=name)]


def area2cf(dataarray, strict=False):
    """Convert an area to at CF grid mapping or lon and lats."""
    res = []
    dataarray = dataarray.copy(deep=True)
    if isinstance(dataarray.attrs['area'], SwathDefinition) or strict:
        res = area2lonlat(dataarray)
    if isinstance(dataarray.attrs['area'], AreaDefinition):
        res.extend(area2gridmapping(dataarray))

    res.append(dataarray)
    return res


def make_time_bounds(dataarray, start_times, end_times):
    """Create time bounds for the current *dataarray*."""
    import numpy as np
    start_time = min(start_time for start_time in start_times
                     if start_time is not None)
    end_time = min(end_time for end_time in end_times
                   if end_time is not None)
    try:
        dtnp64 = dataarray['time'].data[0]
    except IndexError:
        dtnp64 = dataarray['time'].data
    time_bnds = [(np.datetime64(start_time) - dtnp64),
                 (np.datetime64(end_time) - dtnp64)]
    data = xr.DataArray(np.array(time_bnds)[None, :] / np.timedelta64(1, 's'),
                        dims=['time', 'bnds_1d'])
    data.encoding['_FillValue'] = None
    return data


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
    """Link datasets and coordinates.

    If the `coordinates` attribute of a data array links to other datasets in the scene, for example
    `coordinates='lon lat'`, add them as coordinates to the data array and drop that attribute. In the final call to
    `xr.Dataset.to_netcdf()` all coordinate relations will be resolved and the `coordinates` attributes be set
    automatically.

    """
    for ds_name, dataset in datas.items():
        coords = dataset.attrs.get('coordinates', [])
        if isinstance(coords, str):
            coords = coords.split(' ')
        for coord in coords:
            if coord not in dataset.coords:
                try:
                    dataset[coord] = datas[coord]
                except KeyError:
                    warnings.warn('Coordinate "{}" referenced by dataset {} does not exist, dropping reference.'.format(
                        coord, ds_name))
                    continue

        # Drop 'coordinates' attribute in any case to avoid conflicts in xr.Dataset.to_netcdf()
        dataset.attrs.pop('coordinates', None)


def make_alt_coords_unique(datas, pretty=False):
    """Make non-dimensional coordinates unique among all datasets.

    Non-dimensional (or alternative) coordinates, such as scanline timestamps, may occur in multiple datasets with
    the same name and dimension but different values. In order to avoid conflicts, prepend the dataset name to the
    coordinate name. If a non-dimensional coordinate is unique among all datasets and ``pretty=True``, its name will not
    be modified.

    Since all datasets must have the same projection coordinates, this is not applied to latitude and longitude.

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
            if coord_name.lower() not in ('latitude', 'longitude', 'lat', 'lon') and coord_name not in dataset.dims:
                tokens[coord_name].add(tokenize(dataset[coord_name].data))
    coords_unique = dict([(coord_name, len(tokens) == 1) for coord_name, tokens in tokens.items()])

    # Prepend dataset name, if not unique or no pretty-format desired
    new_datas = datas.copy()
    for coord_name, unique in coords_unique.items():
        if not pretty or not unique:
            if pretty:
                warnings.warn('Cannot pretty-format "{}" coordinates because they are not unique among the '
                              'given datasets'.format(coord_name))
            for ds_name, dataset in datas.items():
                if coord_name in dataset.coords:
                    rename = {coord_name: '{}_{}'.format(ds_name, coord_name)}
                    new_datas[ds_name] = new_datas[ds_name].rename(rename)

    return new_datas


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
        """Encode the given object as a json-serializable datatype.

        Use the netcdf encoder as it covers most of the datatypes appearing in dataset attributes. If that fails,
        return the string representation of the object.
        """
        try:
            return _encode_nc(obj)
        except ValueError:
            return str(obj)


def _encode_nc(obj):
    """Try to encode `obj` as a netcdf compatible datatype which most closely resembles the object's nature.

    Raises:
        ValueError if no such datatype could be found

    """
    if isinstance(obj, (bool, np.bool_)):
        # Bool has to be checked first, because it is a subclass of int
        return str(obj)
    elif isinstance(obj, (int, float, str)):
        return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.void):
        return tuple(obj)
    elif isinstance(obj, np.ndarray):
        if not obj.dtype.fields and obj.dtype == np.bool_:
            # Convert array of booleans to array of strings
            obj = obj.astype(str)

        # Multi-dimensional nc attributes are not supported, so we have to skip record arrays and multi-dimensional
        # arrays here
        is_plain_1d = not obj.dtype.fields and len(obj.shape) <= 1
        if is_plain_1d:
            if obj.dtype in NC4_DTYPES:
                return obj
            return obj.tolist()

    raise ValueError('Unable to encode')


def encode_nc(obj):
    """Encode the given object as a netcdf compatible datatype.

    Try to find the datatype which most closely resembles the object's nature. If that fails, encode as a string.
    Plain lists are encoded recursively.
    """
    if isinstance(obj, (list, tuple)) and all([not isinstance(item, (list, tuple)) for item in obj]):
        return [encode_nc(item) for item in obj]
    try:
        return _encode_nc(obj)
    except ValueError:
        try:
            # Decode byte-strings
            decoded = obj.decode()
        except AttributeError:
            decoded = obj
        return json.dumps(decoded, cls=AttributeEncoder).strip('"')


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
        encoded_attrs.append((key, encode_nc(val)))
    return OrderedDict(encoded_attrs)


class CFWriter(Writer):
    """Writer producing NetCDF/CF compatible datasets."""

    @staticmethod
    def da2cf(dataarray, epoch=EPOCH, flatten_attrs=False, exclude_attrs=None, compression=None):
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
        """
        if exclude_attrs is None:
            exclude_attrs = []

        new_data = dataarray.copy()

        # Remove area as well as user-defined attributes
        for key in ['area'] + exclude_attrs:
            new_data.attrs.pop(key, None)

        anc = [ds.attrs['name']
               for ds in new_data.attrs.get('ancillary_variables', [])]
        if anc:
            new_data.attrs['ancillary_variables'] = ' '.join(anc)
        # TODO: make this a grid mapping or lon/lats
        # new_data.attrs['area'] = str(new_data.attrs.get('area'))
        for key, val in new_data.attrs.copy().items():
            if val is None:
                new_data.attrs.pop(key)
            if key == 'ancillary_variables' and val == []:
                new_data.attrs.pop(key)
        new_data.attrs.pop('_last_resampler', None)
        if compression is not None:
            new_data.encoding.update(compression)

        if 'time' in new_data.coords:
            new_data['time'].encoding['units'] = epoch
            new_data['time'].attrs['standard_name'] = 'time'
            new_data['time'].attrs.pop('bounds', None)
            if 'time' not in new_data.dims:
                new_data = new_data.expand_dims('time')

        if 'x' in new_data.coords:
            new_data['x'].attrs['standard_name'] = 'projection_x_coordinate'
            new_data['x'].attrs['units'] = 'm'

        if 'y' in new_data.coords:
            new_data['y'].attrs['standard_name'] = 'projection_y_coordinate'
            new_data['y'].attrs['units'] = 'm'

        if 'crs' in new_data.coords:
            new_data = new_data.drop('crs')

        new_data.attrs.setdefault('long_name', new_data.attrs.pop('name'))
        if 'prerequisites' in new_data.attrs:
            new_data.attrs['prerequisites'] = [np.string_(str(prereq)) for prereq in new_data.attrs['prerequisites']]

        # Flatten dict-type attributes, if desired
        if flatten_attrs:
            new_data.attrs = flatten_dict(new_data.attrs)

        # Encode attributes to netcdf-compatible datatype
        new_data.attrs = encode_attrs_nc(new_data.attrs)

        return new_data

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Save the *dataset* to a given *filename*."""
        return self.save_datasets([dataset], filename, **kwargs)

    def _collect_datasets(self, datasets, epoch=EPOCH, flatten_attrs=False, exclude_attrs=None, include_lonlats=True,
                          pretty=False, compression=None):
        """Collect and prepare datasets to be written."""
        ds_collection = {}
        for ds in datasets:
            ds_collection.update(get_extra_ds(ds))

        datas = {}
        start_times = []
        end_times = []
        for ds in ds_collection.values():
            try:
                new_datasets = area2cf(ds, strict=include_lonlats)
            except KeyError:
                new_datasets = [ds.copy(deep=True)]
            for new_ds in new_datasets:
                start_times.append(new_ds.attrs.get("start_time", None))
                end_times.append(new_ds.attrs.get("end_time", None))
                datas[new_ds.attrs['name']] = self.da2cf(new_ds, epoch=epoch, flatten_attrs=flatten_attrs,
                                                         exclude_attrs=exclude_attrs, compression=compression)

        # Check and prepare coordinates
        assert_xy_unique(datas)
        link_coords(datas)
        datas = make_alt_coords_unique(datas, pretty=pretty)

        return datas, start_times, end_times

    def update_encoding(self, datasets, to_netcdf_kwargs):
        """Update encoding.

        Avoid _FillValue attribute being added to coordinate variables (https://github.com/pydata/xarray/issues/1865).
        """
        other_to_netcdf_kwargs = to_netcdf_kwargs.copy()
        encoding = other_to_netcdf_kwargs.pop('encoding', {}).copy()
        coord_vars = []
        for data_array in datasets:
            coord_vars.extend(set(data_array.dims).intersection(data_array.coords))
        for coord_var in coord_vars:
            encoding.setdefault(coord_var, {})
            encoding[coord_var].update({'_FillValue': None})

        return encoding, other_to_netcdf_kwargs

    def save_datasets(self, datasets, filename=None, groups=None, header_attrs=None, engine=None, epoch=EPOCH,
                      flatten_attrs=False, exclude_attrs=None, include_lonlats=True, pretty=False,
                      compression=None, **to_netcdf_kwargs):
        """Save the given datasets in one netCDF file.

        Note that all datasets (if grouping: in one group) must have the same projection coordinates.

        Args:
            datasets (list):
                Names of datasets to be saved
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
        """
        logger.info('Saving datasets to NetCDF4/CF.')

        if groups is None:
            # Write all datasets to the file root without creating a group
            groups_ = {None: datasets}
        else:
            # User specified a group assignment using dataset names. Collect the corresponding datasets.
            groups_ = defaultdict(list)
            for dataset in datasets:
                for group_name, group_members in groups.items():
                    if dataset.attrs['name'] in group_members:
                        groups_[group_name].append(dataset)
                        break

        if compression is None:
            compression = {'zlib': True}

        # Write global attributes to file root (creates the file)
        filename = filename or self.get_filename(**datasets[0].attrs)

        root = xr.Dataset({}, attrs={'history': 'Created by pytroll/satpy on {}'.format(datetime.utcnow())})
        if header_attrs is not None:
            root.attrs.update({k: v for k, v in header_attrs.items() if v})
        if groups is None:
            # Groups are not CF-1.7 compliant
            root.attrs['Conventions'] = 'CF-1.7'

        # Remove satpy-specific kwargs
        satpy_kwargs = ['overlay', 'decorate', 'config_files']
        for kwarg in satpy_kwargs:
            to_netcdf_kwargs.pop(kwarg, None)

        init_nc_kwargs = to_netcdf_kwargs.copy()
        init_nc_kwargs.pop('encoding', None)  # No variables to be encoded at this point
        written = [root.to_netcdf(filename, engine=engine, mode='w', **init_nc_kwargs)]

        # Write datasets to groups (appending to the file; group=None means no group)
        for group_name, group_datasets in groups_.items():
            # XXX: Should we combine the info of all datasets?
            datas, start_times, end_times = self._collect_datasets(
                group_datasets, epoch=epoch, flatten_attrs=flatten_attrs, exclude_attrs=exclude_attrs,
                include_lonlats=include_lonlats, pretty=pretty, compression=compression)
            dataset = xr.Dataset(datas)
            try:
                dataset['time_bnds'] = make_time_bounds(dataset,
                                                        start_times,
                                                        end_times)
                dataset['time'].attrs['bounds'] = "time_bnds"
                dataset['time'].attrs['standard_name'] = "time"
            except KeyError:
                grp_str = ' of group {}'.format(group_name) if group_name is not None else ''
                logger.warning('No time dimension in datasets{}, skipping time bounds creation.'.format(grp_str))

            encoding, other_to_netcdf_kwargs = self.update_encoding(datasets, to_netcdf_kwargs)
            res = dataset.to_netcdf(filename, engine=engine, group=group_name, mode='a', encoding=encoding,
                                    **other_to_netcdf_kwargs)
            written.append(res)
        return written
