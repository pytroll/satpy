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

import xarray as xr

from pyresample.geometry import AreaDefinition, SwathDefinition
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
                )
    return args


def geos2cf(proj_dict):
    """Return the cf grid mapping for the geos projection."""
    grid_mapping_name = 'geostationary'

    args = dict(perspective_point_height=proj_dict.get('h'),
                latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lon_0'),
                semi_major_axis=proj_dict.get('a'),
                semi_minor_axis=proj_dict.get('b'),
                sweep_axis=proj_dict.get('sweep'),
                )
    return args


def laea2cf(proj_dict):
    """Return the cf grid mapping for the laea projection."""
    grid_mapping_name = 'lambert_azimuthal_equal_area'

    args = dict(latitude_of_projection_origin=proj_dict.get('lat_0'),
                longitude_of_projection_origin=proj_dict.get('lon_0'),
                )
    return args


mappings = {'omerc': omerc2cf,
            'laea': laea2cf,
            'geos': geos2cf}


def create_grid_mapping(area):
    """Create the grid mapping instance for `area`."""
    try:
        grid_mapping = mappings[area.proj_dict['proj']](area.proj_dict)
        grid_mapping['name'] = area.proj_dict['proj']
    except KeyError:
        raise NotImplementedError

    return grid_mapping


def get_extra_ds(dataset):
    ds_collection = {}
    for ds in dataset.attrs.get('ancillary_variables', []):
        ds_collection.update(get_extra_ds(ds))
    ds_collection[dataset.attrs['name']] = dataset

    return ds_collection


def area2lonlat(dataarray):
    area = dataarray.attrs['area']
    lonlats = area.get_lonlats_dask(blocksize=1000)
    lons = xr.DataArray(lonlats[:, :, 0], dims=['y', 'x'],
                        attrs={'name': "longitude",
                               'standard_name': "longitude",
                               'units': 'degrees_east'},
                        name='longitude')
    lats = xr.DataArray(lonlats[:, :, 1], dims=['y', 'x'],
                        attrs={'name': "latitude",
                               'standard_name': "latitude",
                               'units': 'degrees_north'},
                        name='latitude')
    dataarray.attrs['coordinates'] = 'longitude latitude'

    return [dataarray, lons, lats]


def area2gridmapping(dataarray):
    area = dataarray.attrs['area']
    attrs = create_grid_mapping(area)
    name = attrs['name']
    dataarray.attrs['grid_mapping'] = name
    return [dataarray, xr.DataArray([], attrs=attrs, name=name)]


def area2cf(dataarray, strict=False):
    res = []
    if isinstance(dataarray.attrs['area'], SwathDefinition) or strict:
        res = area2lonlat(dataarray)
    if isinstance(dataarray.attrs['area'], AreaDefinition):
        res.extend(area2gridmapping(dataarray))

    dataarray.attrs.pop('area')
    res.append(dataarray)
    return res


class CFWriter(Writer):
    """Writer producing NetCDF/CF compatible datasets."""

    @staticmethod
    def da2cf(dataarray):
        """Convert the dataarray to something cf-compatible"""
        new_data = dataarray.copy()
        # TODO: make these boundaries of the time dimension
        new_data.attrs.pop('start_time', None)
        new_data.attrs.pop('end_time', None)

        anc = [ds.attrs['name']
               for ds in new_data.attrs.get('ancillary_variables', [])]
        if anc:
            new_data.attrs['ancillary_variables'] = ' '.join(anc)
        # TODO: make this a grid mapping or lon/lats
        # new_data.attrs['area'] = str(new_data.attrs.get('area'))
        for key, val in new_data.attrs.copy().items():
            if val is None:
                new_data.attrs.pop(key)

        new_data.attrs.setdefault('long_name', new_data.attrs.pop('name'))
        return new_data

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Saves the *dataset* to a given *filename*."""
        # self.da2cf(dataset).to_netcdf(filename)
        return self.save_datasets([dataset], filename, **kwargs)

    def save_datasets(self, datasets, filename, **kwargs):
        """Save all datasets to one or more files."""
        LOG.info('Saving datasets to NetCDF4/CF.')

        ds_collection = {}
        for ds in datasets:
            ds_collection.update(get_extra_ds(ds))

        datas = {}
        for ds in ds_collection.values():
            try:
                new_datasets = area2cf(ds)
            except KeyError:
                new_datasets = [ds]
            for new_ds in new_datasets:
                datas[new_ds.attrs['name']] = self.da2cf(new_ds)

        dataset = xr.Dataset(datas)
        dataset.attrs['history'] = ("Created by pytroll/satpy on " +
                                    str(datetime.utcnow()))
        dataset.attrs['conventions'] = 'CF-1.7'
        dataset.to_netcdf(filename)
