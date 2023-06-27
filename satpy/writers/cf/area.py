#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2023 Satpy developers
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
"""CF processing of pyresample area information."""
import logging
import warnings
from collections import defaultdict

import xarray as xr
from dask.base import tokenize
from packaging.version import Version
from pyresample.geometry import AreaDefinition, SwathDefinition

logger = logging.getLogger(__name__)


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
