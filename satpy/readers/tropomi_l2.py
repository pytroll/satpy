#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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

"""Interface to TROPOMI L2 Reader.

The TROPOspheric Monitoring Instrument (TROPOMI) is the satellite instrument
on board the Copernicus Sentinel-5 Precursor satellite. It measures key
atmospheric trace gasses, such as ozone, nitrogen oxides, sulfur dioxide,
carbon monoxide, methane, and formaldehyde.

Level 2 data products are available via the Copernicus Open Access Hub.
For more information visit the following URL:
http://www.tropomi.eu/data-products/level-2-products

"""

from satpy.readers.netcdf_utils import NetCDF4FileHandler, netCDF4
import logging
import numpy as np
import xarray as xr
import dask.array as da
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)


class TROPOMIL2FileHandler(NetCDF4FileHandler):
    """File handler for TROPOMI L2 netCDF files."""

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info.get('end_time', self.start_time)

    @property
    def platform_shortname(self):
        """Get platform shortname."""
        return self.filename_info['platform_shortname']

    @property
    def sensor(self):
        """Get sensor."""
        res = self['/attr/sensor']
        if isinstance(res, np.ndarray):
            return str(res.astype(str)).lower()
        return res.lower()

    @property
    def sensor_names(self):
        """Get sensor set."""
        return {self.sensor}

    def available_datasets(self, configured_datasets=None):
        """Automatically determine datasets provided by this file."""
        logger.debug("Available_datasets begin...")

        # Determine shape of the geolocation data (lat/lon)
        lat_shape = None
        for var_name, _val in self.file_content.items():
            # Could probably avoid this hardcoding, will think on it
            if (var_name == 'PRODUCT/latitude'):
                lat_shape = self[var_name + "/shape"]
                break

        handled_variables = set()

        # update previously configured datasets
        logger.debug("Starting previously configured variables loop...")
        # if bounds exists, we can assemble them later
        bounds_exist = 'latitude_bounds' in self and 'longitude_bounds' in self
        for is_avail, ds_info in (configured_datasets or []):

            # some other file handler knows how to load this
            if is_avail is not None:
                yield is_avail, ds_info

            var_name = ds_info.get('file_key', ds_info['name'])
            # logger.debug("Evaluating previously configured variable: %s", var_name)
            matches = self.file_type_matches(ds_info['file_type'])
            # we can confidently say that we can provide this dataset and can
            # provide more info
            assembled = var_name in ['assembled_lat_bounds', 'assembled_lon_bounds']
            if (matches and var_name in self) or (assembled and bounds_exist):
                logger.debug("Handling previously configured variable: %s", var_name)
                if not assembled:
                    # Because assembled variables and bounds use the same file_key,
                    # we need to omit file_key once.
                    handled_variables.add(var_name)
                new_info = ds_info.copy()  # don't mess up the above yielded
                yield True, new_info
            elif is_avail is None:
                # if we didn't know how to handle this dataset and no one else did
                # then we should keep it going down the chain
                yield is_avail, ds_info

        # This is where we dynamically add new datasets
        # We will sift through all groups and variables, looking for data matching
        # the geolocation bounds

        # Iterate over dataset contents
        for var_name, val in self.file_content.items():
            # Only evaluate variables
            if isinstance(val, netCDF4.Variable):
                logger.debug("Evaluating new variable: %s", var_name)
                var_shape = self[var_name + "/shape"]
                logger.debug("Dims:{}".format(var_shape))
                if (lat_shape == var_shape[:len(lat_shape)]):
                    logger.debug("Found valid additional dataset: %s", var_name)
                    # Skip anything we have already configured
                    if (var_name in handled_variables):
                        logger.debug("Already handled, skipping: %s", var_name)
                        continue
                    handled_variables.add(var_name)
                    last_index_separator = var_name.rindex('/')
                    last_index_separator = last_index_separator + 1
                    var_name_no_path = var_name[last_index_separator:]
                    logger.debug("Using short name of: %s", var_name_no_path)
                    # Create new ds_info object
                    if var_name_no_path in ['latitude_bounds', 'longitude_bounds']:
                        coordinates = []
                    else:
                        coordinates = ['longitude', 'latitude']
                    new_info = {
                        'name': var_name_no_path,
                        'file_key': var_name,
                        'coordinates': coordinates,
                        'file_type': self.filetype_info['file_type'],
                        'resolution': None,
                    }
                    yield True, new_info

    def get_metadata(self, data, ds_info):
        """Get metadata."""
        metadata = {}
        metadata.update(data.attrs)
        metadata.update(ds_info)
        metadata.update({
            'platform_shortname': self.platform_shortname,
            'sensor': self.sensor,
            'start_time': self.start_time,
            'end_time': self.end_time,
        })

        return metadata

    def _rename_dims(self, data_arr):
        """Normalize dimension names with the rest of Satpy."""
        dims_dict = {}
        if 'ground_pixel' in data_arr.dims:
            dims_dict['ground_pixel'] = 'x'
        if 'scanline' in data_arr.dims:
            dims_dict['scanline'] = 'y'
        return data_arr.rename(dims_dict)

    def prepare_geo(self, bounds_data):
        """Prepare lat/lon bounds for pcolormesh.

        lat/lon bounds are ordered in the following way::

            3----2
            |    |
            0----1

        Extend longitudes and latitudes with one element to support
        "pcolormesh"::

            (X[i+1, j], Y[i+1, j])         (X[i+1, j+1], Y[i+1, j+1])
                                  +--------+
                                  | C[i,j] |
                                  +--------+
                 (X[i, j], Y[i, j])        (X[i, j+1], Y[i, j+1])

        """
        # Create the left array
        left = np.vstack([bounds_data[:, :, 0], bounds_data[-1:, :, 3]])
        # Create the right array
        right = np.vstack([bounds_data[:, -1:, 1], bounds_data[-1:, -1:, 2]])
        # Stack horizontally
        dest = np.hstack([left, right])
        # Convert to DataArray
        dask_dest = da.from_array(dest, chunks=CHUNK_SIZE)
        dest = xr.DataArray(dask_dest,
                            dims=('y_bounds', 'x_bounds'),
                            attrs=bounds_data.attrs
                            )
        return dest

    def get_dataset(self, ds_id, ds_info):
        """Get dataset."""
        logger.debug("Getting data for: %s", ds_id.name)
        file_key = ds_info.get('file_key', ds_id.name)
        data = self[file_key]
        data.attrs = self.get_metadata(data, ds_info)
        fill_value = data.attrs.get('_FillValue', np.float32(np.nan))
        data = data.squeeze()

        # preserve integer data types if possible
        if np.issubdtype(data.dtype, np.integer):
            new_fill = fill_value
        else:
            new_fill = np.float32(np.nan)
            data.attrs.pop('_FillValue', None)
        good_mask = data != fill_value

        scale_factor = data.attrs.get('scale_factor')
        add_offset = data.attrs.get('add_offset')
        if scale_factor is not None:
            data = data * scale_factor + add_offset

        data = data.where(good_mask, new_fill)
        data = self._rename_dims(data)

        # drop coords whose units are not meters
        drop_list = ['y', 'x', 'layer', 'vertices']
        coords_exist = [coord for coord in drop_list if coord in data.coords]
        if coords_exist:
            data = data.drop_vars(coords_exist)

        if ds_id.name in ['assembled_lat_bounds', 'assembled_lon_bounds']:
            data = self.prepare_geo(data)
        return data
