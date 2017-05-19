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

from satpy.writers import Writer

LOG = logging.getLogger(__name__)


def omerc2cf(proj_dict):
    """Return the cf grid mapping for the omerc projection."""
    grid_mapping_name = 'oblique_mercator'

    args = dict(azimuth_of_central_line=proj_dict.get('alpha'),
                latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lonc'),
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
            'laea': laea2cf}


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
        lat = cf.AuxiliaryCoordinate(data=cf.Data(area.lats,
                                                  'degrees_north'))
        lat.standard_name = 'latitude'

        # Create a latitude auxiliary coordinate
        lon = cf.AuxiliaryCoordinate(data=cf.Data(area.lons,
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
    for name, coord in data_array.coords.items():
        if name == 'time' and len(coord) == 1:
            coords.append(create_time_coordinate_with_bounds(
                data_array.attrs['start_time'],
                data_array.attrs['end_time'],
                coord.attrs.copy()))
        else:
            dimc = cf.DimensionCoordinate(properties=coord.attrs.copy(),
                                          data=cf.Data(coord.values,
                                                       cf.Units(
                                                           coord.attrs['units'])),
                                          )
            dimc.id = coord.name
            coords.append(dimc)
    return coords


def create_data_domain(data_array):
    """Create a cf domain from `data_array`."""
    area = data_array.attrs.get('area')
    return cf.Domain(dim=get_data_array_coords(data_array),
                     aux=auxiliary_coordinates_from_area(area),
                     ref=grid_mapping_from_area(area))


def cf_field_from_data_array(data_array, **extra_properties):
    """Get the cf Field object corresponding to `data_array`."""
    wanted_keys = set(['standard_name', 'long_name'])
    properties = {k: data_array.attrs[k]
                  for k in wanted_keys & set(data_array.attrs.keys())}
    properties.update(extra_properties)

    new_field = cf.Field(properties=properties,
                         data=cf.Data(data_array.values,
                                      data_array.attrs['units']),
                         domain=create_data_domain(data_array))

    new_field.valid_range = data_array.attrs['valid_range']

    return new_field


class CFWriter(Writer):

    def save_datasets(self, datasets, filename, **kwargs):
        """Save all datasets to one or more files."""
        LOG.info('Saving datasets to NetCDF4/CF.')
        fields = []

        history = ("Created by pytroll/satpy on " + str(datetime.utcnow()))
        conventions = 'CF-1.7'

        fields = [cf_field_from_data_array(data_array,
                                           history=history,
                                           Conventions=conventions)
                  for data_array in datasets]

        flist = cf.FieldList(fields)

        cf.write(flist, filename, fmt='NETCDF4', compress=6)
