#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""Reader for the FCI L2 products in NetCDF4 format."""

import logging
import numpy as np
import xarray as xr

from datetime import datetime, timedelta

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers._geos_area import get_area_definition, make_ext
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)

PRODUCT_DATA_DURATION_MINUTES = 20


class FciL2NCFileHandler(BaseFileHandler):
    """Reader class for FCI L2 products in NetCDF4 format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Open the NetCDF file with xarray and prepare for dataset reading."""
        super().__init__(filename, filename_info, filetype_info)

        # Use xarray's default netcdf4 engine to open the file
        self.nc = xr.open_dataset(
            self.filename,
            decode_cf=True,
            mask_and_scale=True,
            chunks={
                'number_of_columns': CHUNK_SIZE,
                'number_of_rows': CHUNK_SIZE
            }
        )

        # Read metadata which are common to all datasets
        self.nlines = self.nc['y'].size
        self.ncols = self.nc['x'].size
        self._projection = self.nc['mtg_geos_projection']

        # Compute the area definition
        self._area_def = self._compute_area_def()

    @property
    def start_time(self):
        """Get observation start time."""
        try:
            start_time = datetime.strptime(self.nc.attrs['time_coverage_start'], '%Y%m%d%H%M%S')
        except (ValueError, KeyError):
            # TODO if the sensing_start_time_utc attribute is not valid, uses a hardcoded value
            logger.warning("Start time cannot be obtained from file content, using default value instead")
            start_time = datetime.strptime('20200101120000', '%Y%m%d%H%M%S')
        return start_time

    @property
    def end_time(self):
        """Get observation end time."""
        try:
            end_time = datetime.strptime(self.nc.attrs['time_coverage_end'], '%Y%m%d%H%M%S')
        except (ValueError, KeyError):
            # TODO if the sensing_end_time_utc attribute is not valid, adds 20 minutes to the start time
            end_time = self.start_time + timedelta(minutes=PRODUCT_DATA_DURATION_MINUTES)
        return end_time

    @property
    def spacecraft_name(self):
        """Return spacecraft name."""
        try:
            return self.nc.attrs['platform']
        except KeyError:
            # TODO if the platform attribute is not valid, return a default value
            logger.warning("Spacecraft name cannot be obtained from file content, using default value instead")
            return 'DEFAULT_MTG'

    @property
    def sensor(self):
        """Return instrument."""
        try:
            return self.nc.attrs['data_source']
        except KeyError:
            # TODO if the data_source attribute is not valid, return a default value
            logger.warning("Sensor cannot be obtained from file content, using default value instead")
            return 'fci'

    @property
    def ssp_lon(self):
        """Return subsatellite point longitude."""
        try:
            return float(self._projection.attrs['longitude_of_projection_origin'])
        except KeyError:
            # TODO if the longitude_of_projection_origin attribute is not valid, return a default value
            logger.warning("ssp_lon cannot be obtained from file content, using default value instead")
            return 0.

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the file_key in dataset_info."""
        var_key = dataset_info['file_key']
        logger.debug('Reading in file to get dataset with key %s.', var_key)

        try:
            variable = self.nc[var_key]
        except KeyError:
            logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
            return None

        # TODO in some of the test files, invalid pixels contain the value defined as "fill_value" in the YAML file
        # instead of being masked directly in the netCDF variable.
        # therefore NaN is applied where such value is found or (0 if the array contains integer values)
        # the next 11 lines have to be removed once the product files are correctly configured
        try:
            mask_value = dataset_info['mask_value']
        except KeyError:
            mask_value = np.NaN
        try:
            fill_value = dataset_info['fill_value']
        except KeyError:
            fill_value = np.NaN
        float_variable = variable.where(variable != fill_value, mask_value).astype('float32', copy=False)
        float_variable.attrs = variable.attrs
        variable = float_variable

        # If the variable has 3 dimensions, select the required layer
        if variable.ndim == 3:
            layer = dataset_info.get('layer', 0)
            logger.debug('Selecting the layer %d.', layer)
            variable = variable.sel(maximum_number_of_layers=layer)

        # Rename the dimensions as required by Satpy
        variable = variable.rename({'number_of_rows': 'y', 'number_of_columns': 'x'})

        # Manage the attributes of the dataset
        variable.attrs.setdefault('units', None)

        variable.attrs.update(dataset_info)
        variable.attrs.update(self._get_global_attributes())

        return variable

    def _get_global_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets.

        Returns:
            dict: A dictionary of global attributes.
                filename: name of the product file
                start_time: sensing start time from best available source
                end_time: sensing end time from best available source
                spacecraft_name: name of the spacecraft
                ssp_lon: longitude of subsatellite point
                sensor: name of sensor
                creation_time: creation time of the product
                platform_name: name of the platform

        """
        attributes = {
            'filename': self.filename,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'spacecraft_name': self.spacecraft_name,
            'ssp_lon': self.ssp_lon,
            'sensor': self.sensor,
            'creation_time': self.filename_info['creation_time'],
            'platform_name': self.spacecraft_name,
        }
        return attributes

    def get_area_def(self, key):
        """Return the area definition (common to all data in product)."""
        return self._area_def

    def _compute_area_def(self):
        """Compute the area definition.

        Returns:
            AreaDefinition: A pyresample AreaDefinition object containing the area definition.

        """
        # Read the projection data from the mtg_geos_projection variable
        a = float(self._projection.attrs['semi_major_axis'])
        b = float(self._projection.attrs['semi_minor_axis'])
        h = float(self._projection.attrs['perspective_point_height'])

        # TODO sweep_angle_axis value not handled at the moment, therefore commented out
        # sweep_axis = self._projection.attrs['sweep_angle_axis']

        # Coordinates of the pixel in radians
        x = self.nc['x']
        y = self.nc['y']
        # TODO conversion to radians: offset and scale factor are missing from some test NetCDF file
        # TODO the next two lines should be removed when the offset and scale factor are correctly configured
        if not hasattr(x, 'standard_name'):
            x = np.radians(x * 0.003202134 - 8.914740401)
            y = np.radians(y * 0.003202134 - 8.914740401)

        # Convert to degrees as required by the make_ext function
        x_deg = np.degrees(x)
        y_deg = np.degrees(y)

        # Select the extreme points of the extension area
        x_l, x_r = x_deg.values[0], x_deg.values[-1]
        y_l, y_u = y_deg.values[0], y_deg.values[-1]

        # Compute the extension area in meters
        area_extent = make_ext(x_l, x_r, y_l, y_u, h)

        # Assemble the projection definition dictionary
        p_dict = {
            'nlines': self.nlines,
            'ncols': self.ncols,
            'ssp_lon': self.ssp_lon,
            'a': a,
            'b': b,
            'h': h,
            'a_name': 'FCI Area',                 # TODO to be confirmed
            'a_desc': 'Area for FCI instrument',  # TODO to be confirmed
            'p_id': 'geos'
        }

        # Compute the area definition
        area_def = get_area_definition(p_dict, area_extent)

        return area_def

    def __del__(self):
        """Close the NetCDF file that may still be open."""
        try:
            self.nc.close()
        except AttributeError:
            pass
