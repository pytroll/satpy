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


class CFWriter(Writer):

    def save_datasets(self, datasets, filename, **kwargs):
        """Save all datasets to one or more files."""
        LOG.info('Saving datasets to NetCDF4/CF.')
        fields = []

        history = ("Created by pytroll/satpy on " +
                   str(datetime.utcnow()))
        conventions = 'CF-1.7'
        for dataset in datasets:

            area = dataset.attrs.get('area')
            add_time = False

            try:
                # Create a longitude auxiliary coordinate
                lat = cf.AuxiliaryCoordinate(data=cf.Data(area.lats,
                                                          'degrees_north'))
                lat.standard_name = 'latitude'

                # Create a latitude auxiliary coordinate
                lon = cf.AuxiliaryCoordinate(data=cf.Data(area.lons,
                                                          'degrees_east'))
                lon.standard_name = 'longitude'
                aux = [lat, lon]
                add_time = True
            except AttributeError:
                LOG.info('No longitude and latitude data to save.')
                aux = None

            try:
                grid_mapping = create_grid_mapping(area)

            except (AttributeError, NotImplementedError):
                LOG.info('No grid mapping to save.')
                grid_mapping = None

            coords = []
            for name, coord in dataset.coords.items():
                properties = coord.attrs.copy()
                # properties.update(dict(history=history,
                #                        Conventions=conventions))
                if name == 'time' and len(coord) == 1:
                    start_time = cf.dt(dataset.attrs['start_time'])
                    end_time = cf.dt(dataset.attrs['end_time'])
                    middle_time = cf.dt((dataset.attrs['end_time'] -
                                         dataset.attrs['start_time']) / 2 +
                                        dataset.attrs['start_time'])
                    bounds = cf.CoordinateBounds(
                        data=cf.Data([start_time, end_time],
                                     cf.Units('days since 1970-1-1')))

                    properties.update(dict(standard_name='time'))
                    coords.append(cf.DimensionCoordinate(properties=properties,
                                                         data=cf.Data(middle_time,
                                                                      cf.Units('days since 1970-1-1')),
                                                         bounds=bounds))
                else:
                    dimc = cf.DimensionCoordinate(properties=properties,
                                                  data=cf.Data(coord.values,
                                                               cf.Units(
                                                                   coord.attrs['units'])),
                                                  #attributes={id: coord.name}
                                                  )
                    dimc.id = coord.name
                    coords.append(dimc)

            domain = cf.Domain(dim=coords,
                               aux=aux,
                               ref=grid_mapping)

            data = cf.Data(dataset.values, dataset.attrs['units'])

            wanted_keys = ['standard_name', 'long_name']
            properties = {k: dataset.attrs[k]
                          for k in set(wanted_keys) & set(dataset.attrs.keys())}
            properties.update(history=history,
                              Conventions=conventions)
            new_field = cf.Field(properties=properties,
                                 data=data,
                                 domain=domain)

            new_field.valid_range = dataset.attrs['valid_range']
            fields.append(new_field)

        flist = cf.FieldList(fields)

        cf.write(flist, filename, fmt='NETCDF4', compress=6)
