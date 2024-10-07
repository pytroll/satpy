#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Satpy developers
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
"""EUMETSAT EPS-SG Multi-view, Multi-channel, Multi-polarisation Imager (3MI) Level 1C products reader.

The ``3mi_l1c_nc`` reader reads EPS-SG 3MI L1C image data in netCDF format. The format is explained
in the `EPS-SG 3MI Level 1C Product Format Specification`_. Details of format and test data can be
found at:
https://user.eumetsat.int/resources/user-guides/metop-sg-3-mi-l1b-and-l1c-data-guide

This version is an initial draft trial version.

"""

import logging
from datetime import datetime
import xarray as xr
from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


class Thr3miL1cNCFileHandler(NetCDF4FileHandler):

    """Base reader class for 3MI products in netCDF format.

    Args:
        filename (str): File to read
        filename_info (dict): Dictionary with filename information
        filetype_info (dict): Dictionary with filetype information

    """

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Prepare the class for dataset reading."""
        super().__init__(filename, filename_info, filetype_info, auto_maskandscale=True)

    def _standardize_dims(self, variable):
        """Standardize dims to y, note only 1D data for 3MI"""

        # lat/lon dimensions
        if 'geo_reference_grid_cells' in variable.dims:
            variable = variable.rename({'geo_reference_grid_cells': 'y'})
        return variable

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using file_key in dataset_info."""
        var_key_xxx = dataset_info['file_key']
        var_key_overlap = dataset_info['file_key_overlap']
        view_key = 0
        try:
            number_overlaps = self[var_key_overlap]
        except KeyError:
            logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key_overlap)
            return None

        # Loop over the number of overlaps present in the granule
        for i_overlap in range(number_overlaps):
            # set file_key for current overlap
            str_overlap = '0' + '0' + str(i_overlap)
            var_key = var_key_xxx.replace('XXX', str_overlap)
            # Radiance data has multiple views and polarisations, geolocation not.
            # If radiance data then get the view and append the polarisation.
            if var_key[-9:] != 'longitude' and var_key[-8:] != 'latitude':
                view_key = dataset_info['view']
                var_key = var_key + dataset_info['polarization']

            logger.debug('Reading in file to get dataset with key %s.', var_key)
            try:
                variable = self[var_key]
            except KeyError:
                logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
                return None
            if i_overlap > 0:
                if var_key[-9:] != 'longitude' and var_key[-8:] != 'latitude':
                    variable = xr.concat([variable[:, view_key], variable_old[:, view_key]], dim="geo_reference_grid_cells")
                else:
                    variable = xr.concat([variable, variable_old], dim="geo_reference_grid_cells")
            variable_old = variable.copy(deep=True)
        # Manage the attributes of the dataset
        variable.attrs.setdefault('units', None)
        variable.attrs.update(dataset_info)
        variable.attrs.update(self._get_global_attributes())
        variable = self._standardize_dims(variable)
        return variable

    def _get_global_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets."""
        attributes = {
            'filename': self.filename,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'spacecraft_name': self.spacecraft_name,
            'ssp_lon': self.ssp_lon,
            'sensor': self.sensor,
            'filename_start_time': self.filename_info['sensing_start_time'],
            'filename_end_time': self.filename_info['sensing_end_time'],
            'platform_name': self.spacecraft_name,
        }

        # Add a "quality_group" item to the dictionary with all the variables and attributes
        # which are found in the 'quality' group of the 3MI product
        quality_group = self['quality']
        quality_dict = {}
        for key in quality_group:
            # Add the values (as Numpy array) of each variable in the group where possible
            try:
                quality_dict[key] = quality_group[key].values
            except ValueError:
                quality_dict[key] = None
        # Add the attributes of the quality group
        quality_dict.update(quality_group.attrs)

        attributes['quality_group'] = quality_dict

        return attributes

    @property
    def start_time(self):
        """Get observation start time."""
        try:
            start_time = datetime.strptime(self['/attr/sensing_start_time_utc'], '%Y%m%d%H%M%S.%f')
        except ValueError:
            start_time = datetime.strptime(self['/attr/sensing_start_time_utc'], '%Y-%m-%d %H:%M:%S.%f')
        return start_time

    @property
    def end_time(self):
        """Get observation end time."""
        try:
            end_time = datetime.strptime(self['/attr/sensing_end_time_utc'], '%Y%m%d%H%M%S.%f')
        except ValueError:
            end_time = datetime.strptime(self['/attr/sensing_end_time_utc'], '%Y-%m-%d %H:%M:%S.%f')
        return end_time

    @property
    def spacecraft_name(self):
        """Return spacecraft name."""
        return self['/attr/spacecraft']

    @property
    def sensor(self):
        """Return sensor."""
        return self['/attr/instrument']

    @property
    def ssp_lon(self):
        """Return subsatellite point longitude."""
        # This parameter is not applicable to 3MI
        return None