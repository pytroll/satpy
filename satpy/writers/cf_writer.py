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
    grid_mapping_name = 'geostationary'

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


class CFWriter(Writer):

    def save_datasets(self, datasets, filename, **kwargs):
        """Save all datasets to one or more files."""
        LOG.info('Saving datasets to NetCDF4/CF.')
        fields = []
        shapes = {}
        for dataset in datasets:
            if dataset.shape in shapes:
                domain = shapes[dataset.shape]
            else:
                lines, pixels = dataset.shape

                area = dataset.info.get('area')
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
                    units = area.proj_dict.get('units', 'm')

                    line_coord = cf.DimensionCoordinate(
                        data=cf.Data(area.proj_y_coords, units))
                    line_coord.standard_name = "projection_y_coordinate"
                    pixel_coord = cf.DimensionCoordinate(
                        data=cf.Data(area.proj_x_coords, units))
                    pixel_coord.standard_name = "projection_x_coordinate"
                    add_time = True

                except (AttributeError, NotImplementedError):
                    LOG.info('No grid mapping to save.')
                    grid_mapping = None
                    line_coord = cf.DimensionCoordinate(
                        data=cf.Data(np.arange(lines), '1'))
                    line_coord.standard_name = "line"
                    pixel_coord = cf.DimensionCoordinate(
                        data=cf.Data(np.arange(pixels), '1'))
                    pixel_coord.standard_name = "pixel"

                start_time = cf.dt(dataset.info['start_time'])
                end_time = cf.dt(dataset.info['end_time'])
                middle_time = cf.dt((dataset.info['end_time'] -
                                     dataset.info['start_time']) / 2 +
                                    dataset.info['start_time'])
                # import ipdb
                # ipdb.set_trace()
                if add_time:
                    info = dataset.info
                    dataset = dataset[np.newaxis, :, :]
                    dataset.info = info

                    bounds = cf.CoordinateBounds(
                        data=cf.Data([start_time, end_time],
                                     cf.Units('days since 1970-1-1')))
                    time_coord = cf.DimensionCoordinate(properties=dict(standard_name='time'),
                                                        data=cf.Data(middle_time,
                                                                     cf.Units('days since 1970-1-1')),
                                                        bounds=bounds)
                    coords = [time_coord, line_coord, pixel_coord]
                else:
                    coords = [line_coord, pixel_coord]

                domain = cf.Domain(dim=coords,
                                   aux=aux,
                                   ref=grid_mapping)
                shapes[dataset.shape] = domain
            data = cf.Data(dataset, dataset.info.get('units', 'm'))

            # import ipdb
            # ipdb.set_trace()

            wanted_keys = ['standard_name', 'long_name']
            properties = {k: dataset.info[k]
                          for k in set(wanted_keys) & set(dataset.info.keys())}
            new_field = cf.Field(properties=properties,
                                 data=data,
                                 domain=domain)

            new_field._FillValue = dataset.fill_value
            try:
                new_field.valid_range = dataset.info['valid_range']
            except KeyError:
                new_field.valid_range = new_field.min(), new_field.max()
            new_field.Conventions = 'CF-1.7'
            fields.append(new_field)

        fields[0].history = ("Created by pytroll/satpy on " +
                             str(datetime.utcnow()))

        flist = cf.FieldList(fields)

        cf.write(flist, filename, fmt='NETCDF4', compress=6)
