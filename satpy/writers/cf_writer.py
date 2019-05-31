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
import numpy as np

from pyresample.geometry import AreaDefinition, SwathDefinition
from satpy.writers import Writer

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
        raise NotImplementedError

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
    name = attrs['name']
    dataarray.attrs['grid_mapping'] = name
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


class CFWriter(Writer):
    """Writer producing NetCDF/CF compatible datasets."""

    @staticmethod
    def da2cf(dataarray, epoch=EPOCH):
        """Convert the dataarray to something cf-compatible."""
        new_data = dataarray.copy()

        # Remove the area
        new_data.attrs.pop('area', None)

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
        return new_data

    def save_dataset(self, dataset, filename=None, fill_value=None, **kwargs):
        """Save the *dataset* to a given *filename*."""
        return self.save_datasets([dataset], filename, **kwargs)

    def _collect_datasets(self, datasets, kwargs):
        ds_collection = {}
        for ds in datasets:
            ds_collection.update(get_extra_ds(ds))

        datas = {}
        start_times = []
        end_times = []
        for ds in ds_collection.values():
            try:
                new_datasets = area2cf(ds)
            except KeyError:
                new_datasets = [ds.copy(deep=True)]
            for new_ds in new_datasets:
                start_times.append(new_ds.attrs.pop("start_time", None))
                end_times.append(new_ds.attrs.pop("end_time", None))
                datas[new_ds.attrs['name']] = self.da2cf(new_ds,
                                                         kwargs.get('epoch',
                                                                    EPOCH))
        return datas, start_times, end_times

    def save_datasets(self, datasets, filename=None, **kwargs):
        """Save all datasets to one or more files."""
        logger.info('Saving datasets to NetCDF4/CF.')
        # XXX: Should we combine the info of all datasets?
        filename = filename or self.get_filename(**datasets[0].attrs)
        datas, start_times, end_times = self._collect_datasets(datasets, kwargs)

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
