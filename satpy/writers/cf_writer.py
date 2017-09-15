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
"""Writer for netCDF4/CF."""

import logging
from datetime import datetime

import cf
import numpy as np
from satpy.writers import Writer

LOG = logging.getLogger(__name__)


def omerc2cf(proj_dict):
    """Return the cf grid mapping for the omerc projection."""
    grid_mapping_name = 'oblique_mercator'

    if "no_rot" in proj_dict:
        no_rotation = " "
    else:
        no_rotation = None

    args = dict(azimuth_of_central_line=proj_dict.get('alpha'),
                latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lonc'),
                # longitude_of_projection_origin=0.,
                no_rotation=no_rotation,
                # reference_ellipsoid_name=proj_dict.get('ellps'),
                semi_major_axis=6378137.0,
                semi_minor_axis=6356752.3142,
                false_easting=0.,
                false_northing=0.,
                crtype='grid_mapping',
                coords=['projection_x_coordinate', 'projection_y_coordinate'])
    return cf.CoordinateReference(grid_mapping_name, **args)


def geos2cf(proj_dict):
    """Return the cf grid mapping for the geos projection."""
    grid_mapping_name = 'vertical_perspective'

    args = dict(perspective_point_height=proj_dict.get('h'),
                latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lon_0'),
                semi_major_axis=proj_dict.get('a'),
                semi_minor_axis=proj_dict.get('b'),
                sweep_axis=proj_dict.get('sweep'),
                crtype='grid_mapping',
                coords=['projection_x_coordinate', 'projection_y_coordinate'])
    return cf.CoordinateReference(grid_mapping_name, **args)


def laea2cf(proj_dict):
    """Return the cf grid mapping for the laea projection."""
    grid_mapping_name = 'lambert_azimuthal_equal_area'

    args = dict(latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lon_0'),
                crtype='grid_mapping',
                coords=['projection_x_coordinate', 'projection_y_coordinate'])
    return cf.CoordinateReference(grid_mapping_name, **args)


mappings = {'omerc': omerc2cf,
            'laea': laea2cf,
            'geos': geos2cf}


def create_grid_mapping(area):
    """Create the grid mapping instance for `area`."""

    try:
        grid_mapping = mappings[area.proj_dict['proj']](area.proj_dict)
    except KeyError:
        raise NotImplementedError

    return grid_mapping


def auxiliary_coordinates_from_area(area):
    """Create auxiliary coordinates for `area` if possible."""
    try:
        # Create a longitude auxiliary coordinate
        lons, lats = area.get_lonlats()
        lat = cf.AuxiliaryCoordinate(data=cf.Data(lats,
                                                  'degrees_north'))
        lat.standard_name = 'latitude'

        # Create a latitude auxiliary coordinate
        lon = cf.AuxiliaryCoordinate(data=cf.Data(lons,
                                                  'degrees_east'))
        lon.standard_name = 'longitude'
        return [lat, lon]
    except AttributeError:
        LOG.info('No longitude and latitude data to save.')


def grid_mapping_from_area(area):
    """Get the grid mapping for `area` if possible."""
    try:
        return create_grid_mapping(area)

    except (AttributeError, NotImplementedError):
        LOG.info('No grid mapping to save.')


def create_time_coordinate_with_bounds(start_time, end_time, properties):
    """Create a time coordinate from start and end times."""
    middle_time = cf.dt((end_time - start_time) / 2 + start_time)
    start_time = cf.dt(start_time)
    end_time = cf.dt(end_time)
    bounds = cf.CoordinateBounds(data=cf.Data([start_time, end_time],
                                              cf.Units('days since 1970-1-1')))
    properties.update(dict(standard_name='time'))
    return cf.DimensionCoordinate(properties=properties,
                                  data=cf.Data(middle_time,
                                               cf.Units('days since 1970-1-1')),
                                  bounds=bounds)


def get_data_array_coords(data_array):
    """Get the coordinates for `data_array`."""
    coords = []
    transdims = {'x': 'grid_longitude',
                 'y': 'grid_latitude'}
    for name in data_array.dims:
        if name not in data_array.coords:
            dimc = cf.DimensionCoordinate(data=cf.Data(range(data_array[name].size),
                                                       cf.Units('Degree')))
            dimc.standard_name = transdims[name]
            coords.append(dimc)
        elif name == 'time' and len(data_array.coords[name]) == 1:
            coords.append(create_time_coordinate_with_bounds(
                data_array.attrs['start_time'],
                data_array.attrs['end_time'],
                data_array.coords[name].attrs.copy()))
        else:
            coord = data_array.coords[name]
            dimc = cf.DimensionCoordinate(properties=coord.attrs.copy(),
                                          data=cf.Data(coord.values,
                                                       cf.Units(
                                                           coord.attrs['units'])),
                                          )
            dimc.standard_name = coord.name
            coords.append(dimc)
    return coords


def create_data_domain(data_array):
    """Create a cf domain from `data_array`."""
    area = data_array.attrs.get('area')
    y, x =get_data_array_coords(data_array)
    lat, lon =auxiliary_coordinates_from_area(area)
    ref=grid_mapping_from_area(area)
    return(y, x, lat, lon, ref) 

def cf_field_from_data_array(data_array, **extra_properties):
    """Get the cf Field object corresponding to `data_array`."""
    wanted_keys = set(['standard_name', 'long_name', 'valid_range'])
    properties = {k: data_array.attrs[k]
                  for k in wanted_keys & set(data_array.attrs.keys())}
    properties.update(extra_properties)
    Y, X, lat, lon, ref = create_data_domain(data_array)
    new_field = cf.Field(properties=properties)
    new_field.insert_dim(X)
    new_field.insert_dim(Y)
    new_field.insert_aux(lat, axes=['Y', 'X'])
    new_field.insert_aux(lon, axes=['X', 'Y'])
    new_field.insert_ref(ref)
    new_field.insert_data(data=cf.Data(data_array.values,
                                       data_array.attrs['units']),
                          axes=['Y', 'X'])

    return new_field


class CFWriter(Writer):
    """Writer producing NetCDF/CF compatible datasets."""

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Saves the *dataset* to a given *filename*."""
        return self.save_datasets([dataset], filename, **kwargs)

    def save_datasets(self, datasets, filename, **kwargs):
        """Save all datasets to one or more files."""
        LOG.info('Saving datasets to NetCDF4/CF.')
        fields = []

        history = ("Created by pytroll/satpy on " + str(datetime.utcnow()))
        conventions = 'CF-1.7'

        ds_to_save = [dataset for dataset in datasets]
        for ds in ds_to_save:
            for anc_var in ds.attrs.get('ancillary_variables', []):
                # if anc_var not in ds_to_save:
                ds_to_save.append(anc_var)

        fields = [cf_field_from_data_array(data_array,
                                           history=history,
                                           Conventions=conventions)
                  for data_array in ds_to_save]

        flist = cf.FieldList(fields)

        cf.write(flist, filename, fmt='NETCDF4', compress=6)
