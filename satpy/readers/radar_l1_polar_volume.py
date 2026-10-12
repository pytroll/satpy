#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Reader for radar data in Odim format."""

import h5py
from satpy.readers.file_handlers import BaseFileHandler
import xarray as xr
import dask.array as da
import numpy as np
import pyproj


class OdimH5PolarFileHandler(BaseFileHandler):
    """File handler for Odim hdf5 polar files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Set up the file handler."""
        super(OdimH5PolarFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.h5file = None
        self.longitude = None
        self.latitude = None

    def get_dataset(self, dataset_id, dataset_info):
        """Get the dataset."""
        if self.h5file is None:
            self.h5file = h5py.File(self.filename)
        if dataset_id.name not in ['longitude', 'latitude']:
            elevation_angle = self.h5file['dataset5']['where'].attrs['elangle']
            if elevation_angle != 0.5:
                return
            dataset = xr.DataArray(da.from_array(self.h5file['dataset5'][dataset_info['h5_key']]['data']),
                                   dims=['y', 'x'])
            scale = self.h5file['dataset5'][dataset_info['h5_key']]['what'].attrs['gain']
            offset = self.h5file['dataset5'][dataset_info['h5_key']]['what'].attrs['offset']
            unit = self.h5file['dataset5'][dataset_info['h5_key']]['what'].attrs['quantity']
            nodata = self.h5file['dataset5'][dataset_info['h5_key']]['what'].attrs['nodata']
            dataset = dataset.where(dataset != nodata)
            dataset = dataset.where(dataset != 0)
            dataset = dataset * scale + offset
            dataset.attrs.update(dataset_info)
            dataset.attrs['units'] = unit
        else:
            if self.longitude is None or self.latitude is None:
                radar_lon = self.h5file['where'].attrs['lon']
                radar_lat = self.h5file['where'].attrs['lat']
                # radar_height = self.h5file['where'].attrs['height']
                start_angle = self.h5file['dataset5']['how'].attrs['startazA']
                # stop_angle = self.h5file['dataset5']['how'].attrs['stopazA']
                start_range = self.h5file['dataset5']['where'].attrs['rstart']
                range_increment = self.h5file['dataset5']['where'].attrs['rscale']
                elevation_angle = self.h5file['dataset5']['where'].attrs['elangle']
                if elevation_angle != 0.5:
                    return

                nbins = self.h5file['dataset5']['where'].attrs['nbins']
                # define set of angles and radii values
                theta = np.deg2rad(start_angle)  # radians
                radii = np.linspace(start_range, range_increment * (nbins - 1), nbins) + range_increment / 2  # meters
                radii, theta = np.meshgrid(radii, theta)
                y = radii * np.cos(theta)
                x = radii * np.sin(theta)

                # create projection from proj string
                crs = pyproj.CRS.from_string('+proj=gnom +lon_0={} +lat_0={}'.format(radar_lon, radar_lat))
                proj = pyproj.Proj(crs)

                self.longitude, self.latitude = proj(x, y, inverse=True)

            if dataset_id.name == 'longitude':
                dataset = xr.DataArray(da.from_array(self.longitude), dims=['y', 'x'])
            elif dataset_id.name == 'latitude':
                dataset = xr.DataArray(da.from_array(self.latitude), dims=['y', 'x'])
            dataset.attrs.update(dataset_info)
        return dataset
