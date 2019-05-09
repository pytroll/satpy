#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015.

# Author(s):

#   David Hoese <david.hoese@ssec.wisc.edu>
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of satpy.

# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

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

You can select the netCDF backend using the ``engine`` keyword argument. Default is ``h5netcdf``, an alternative could
be, for example, ``netCDF4``.

In the above example, raw metadata from the HRIT files has been excluded. If you want all attributes to be included,
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
readable as it will create a separate nc-attribute for each item in every dictionary. Keys oare concatenated with
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

from collections import OrderedDict
import logging
from datetime import datetime
import json
import warnings

import xarray as xr
import numpy as np

from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy.writers import Writer
from satpy.writers.utils import flatten_dict


logger = logging.getLogger(__name__)

EPOCH = u"seconds since 1970-01-01 00:00:00"


def omerc2cf(area):
    """Return the cf grid mapping for the omerc projection."""
    proj_dict = area.proj_dict

    args = dict(azimuth_of_central_line=proj_dict.get('alpha'),
                latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lonc'),
                grid_mapping_name='oblique_mercator',
                reference_ellipsoid_name=proj_dict.get('ellps', 'WGS84'),
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
    dataarray.attrs['coordinates'] = 'longitude latitude'
    return [dataarray, lons, lats]


def area2gridmapping(dataarray):
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
    res = []
    dataarray = dataarray.copy(deep=True)
    if isinstance(dataarray.attrs['area'], SwathDefinition) or strict:
        res = area2lonlat(dataarray)
    if isinstance(dataarray.attrs['area'], AreaDefinition):
        res.extend(area2gridmapping(dataarray))

    res.append(dataarray)
    return res


def make_time_bounds(dataarray, start_times, end_times):
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
    return xr.DataArray(np.array(time_bnds) / np.timedelta64(1, 's'),
                        dims=['time_bnds'], coords={'time_bnds': [0, 1]})


def make_coords_unique(datas):
    """Make sure non-dimensional (or alternative) coordinates have unique names by prepending the dataset name

    In principle this would only be required if multiple datasets had alternative coordinates with the same name but
    different values. But to ensure consistency we always prepend the dataset name. Otherwise the name of the
    alternative coordinates in the nc file would depend on the number/set of datasets written to it.

    Args:
        datas (dict): Dictionary of (dataset name, dataset)

    Returns:
        Dictionary holding the updated datasets
    """
    # Collect all alternative coordinates
    alt_coords = [c for data_array in datas.values() for c in data_array.coords if c not in data_array.dims]

    # Prepend dataset name
    new_datas = {}
    for ds_name, data_array in datas.items():
        rename = {}
        for coord_name in alt_coords:
            if coord_name in data_array.coords:
                rename[coord_name] = '{}_{}'.format(ds_name, coord_name)
        if rename:
            data_array = data_array.rename(rename)
        new_datas[ds_name] = data_array

    return new_datas


class AttributeEncoder(json.JSONEncoder):
    """JSON encoder for dataset attributes"""
    def default(self, obj):
        """Returns a json-serializable object for 'obj'

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
        return the string representation of the object."""
        try:
            return _encode_nc(obj)
        except ValueError:
            return str(obj)


def _encode_nc(obj):
    """Encode an arbitrary object in a netcdf compatible datatype

    Raises:
        ValueError if no netcdf compatible datatype could be found
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
        if not len(obj.dtype) and obj.dtype == np.bool_:
            # Convert array of booleans to array of strings
            obj = obj.astype(str)
        if not len(obj.dtype) and len(obj.shape) <= 1:
            # Multi-dimensional nc attributes are not supported, so we have to skip record arrays and multi-dimensional
            # arrays here
            return obj.tolist()

    raise ValueError('Unable to encode')


def encode_nc(obj):
    """Encode an arbitrary object in a netcdf compatible datatype

    Try to find the best matching datatype. If that fails, encode as a string. Plain lists are encoded recursively.
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
    """Encode dataset attributes in a netcdf compatible datatype

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
    def da2cf(dataarray, epoch=EPOCH, flatten_attrs=False, exclude_attrs=None):
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

        # Remove the area as well as user-defined attributes
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
        new_data.attrs.pop('_last_resampler', None)

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

    def _collect_datasets(self, datasets, epoch=EPOCH, flatten_attrs=False, exclude_attrs=None, latlon=False, **kwargs):
        ds_collection = {}
        for ds in datasets:
            ds_collection.update(get_extra_ds(ds))

        datas = {}
        start_times = []
        end_times = []
        for ds in ds_collection.values():
            try:
                new_datasets = area2cf(ds, strict=latlon)
            except KeyError:
                new_datasets = [ds.copy(deep=True)]
            for new_ds in new_datasets:
                start_times.append(new_ds.attrs.get("start_time", None))
                end_times.append(new_ds.attrs.get("end_time", None))
                datas[new_ds.attrs['name']] = self.da2cf(new_ds, epoch=epoch, flatten_attrs=flatten_attrs,
                                                         exclude_attrs=exclude_attrs)
        datas = make_coords_unique(datas)

        return datas, start_times, end_times

    def save_datasets(self, datasets, filename=None, **kwargs):
        """Save all datasets to one or more files."""
        logger.info('Saving datasets to NetCDF4/CF.')
        # XXX: Should we combine the info of all datasets?
        filename = filename or self.get_filename(**datasets[0].attrs)
        datas, start_times, end_times = self._collect_datasets(datasets, **kwargs)

        dataset = xr.Dataset(datas)
        try:
            dataset['time_bnds'] = make_time_bounds(dataset,
                                                    start_times,
                                                    end_times)
            dataset['time'].attrs['bounds'] = "time_bnds"
        except KeyError:
            logger.warning('No time dimension in datasets, skipping time bounds creation.')

        header_attrs = kwargs.pop('header_attrs', None)

        if header_attrs is not None:
            dataset.attrs.update({k: v for k, v in header_attrs.items() if v})

        dataset.attrs['history'] = ("Created by pytroll/satpy on " +
                                    str(datetime.utcnow()))
        dataset.attrs['conventions'] = 'CF-1.7'
        engine = kwargs.pop("engine", 'h5netcdf')
        for key in list(kwargs.keys()):
            if key not in ['mode', 'format', 'group', 'encoding', 'unlimited_dims', 'compute']:
                kwargs.pop(key, None)
        return dataset.to_netcdf(filename, engine=engine, **kwargs)
