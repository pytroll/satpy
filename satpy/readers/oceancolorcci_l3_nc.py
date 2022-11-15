#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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

"""Reader for files produced by ESA's Ocean Color CCI project.

This reader currently supports the lat/lon gridded products and does not yet support the
products on a sinusoidal grid. The products on each of the composite periods (1, 5 and 8 day plus monthly)
are supported and both the merged product files (OC_PRODUCTS) and single product (RRS, CHLOR_A, IOP, K_490) are
supported.
"""
import logging
from datetime import datetime

import dask.array as da
import numpy as np
from pyresample import geometry

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class OCCCIFileHandler(NetCDF4FileHandler):
    """File handler for Ocean Color CCI netCDF files."""

    @staticmethod
    def _parse_datetime(datestr):
        """Parse datetime."""
        return datetime.strptime(datestr, "%Y%m%d%H%MZ")

    @property
    def start_time(self):
        """Get the start time."""
        return self._parse_datetime(self['/attr/time_coverage_start'])

    @property
    def end_time(self):
        """Get the end time."""
        return self._parse_datetime(self['/attr/time_coverage_end'])

    @property
    def composite_period(self):
        """Determine composite period from filename information."""
        comp1 = self.filename_info['composite_period_1']
        comp2 = self.filename_info['composite_period_2']
        if comp2 == 'MONTHLY' and comp1 == "1M":
            return 'monthly'
        elif comp1 == '1D':
            return 'daily'
        elif comp1 == '5D':
            return '5-day'
        elif comp1 == '8D':
            return '8-day'
        else:
            raise ValueError(f"Unknown data compositing period: {comp1}_{comp2}")

    def _update_attrs(self, dataset, dataset_info):
        """Update dataset attributes."""
        dataset.attrs.update(self[dataset_info['nc_key']].attrs)
        dataset.attrs.update(dataset_info)
        dataset.attrs['sensor'] = 'merged'
        dataset.attrs['composite_period'] = self.composite_period
        # remove attributes from original file which don't apply anymore
        dataset.attrs.pop("nc_key")

    def get_dataset(self, dataset_id, ds_info):
        """Get dataset."""
        dataset = da.squeeze(self[ds_info['nc_key']])
        if '_FillValue' in dataset.attrs:
            dataset.data = da.where(dataset.data == dataset.attrs['_FillValue'], np.nan, dataset.data)
        self._update_attrs(dataset, ds_info)
        return dataset

    def get_area_def(self, dsid):
        """Get the area definition based on information in file.

        There is no area definition in the file itself, so we have to compute it
        from the metadata, which specifies the area extent and pixel resolution.
        """
        proj_param = 'EPSG:4326'

        lon_res = float(self['/attr/geospatial_lon_resolution'])
        lat_res = float(self['/attr/geospatial_lat_resolution'])

        min_lon = self['/attr/geospatial_lon_min']
        max_lon = self['/attr/geospatial_lon_max']
        min_lat = self['/attr/geospatial_lat_min']
        max_lat = self['/attr/geospatial_lat_max']

        area_extent = (min_lon, min_lat, max_lon, max_lat)
        lon_size = np.round((max_lon - min_lon) / lon_res).astype(int)
        lat_size = np.round((max_lat - min_lat) / lat_res).astype(int)

        area = geometry.AreaDefinition('gridded_occci',
                                       'Full globe gridded area',
                                       'longlat',
                                       proj_param,
                                       lon_size,
                                       lat_size,
                                       area_extent)
        return area
