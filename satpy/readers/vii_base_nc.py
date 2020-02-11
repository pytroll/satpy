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

"""EUMETSAT EPS-SG Visible/Infrared Imager (VII) readers base class."""

import logging
import xarray as xr
import dask.array as da

from datetime import datetime

from satpy.readers.netcdf_utils import NetCDF4FileHandler
from satpy.readers.vii_utils import tie_points_interpolation
from satpy import CHUNK_SIZE

logger = logging.getLogger(__name__)


class ViiNCBaseFileHandler(NetCDF4FileHandler):
    """Base reader class for VII products in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info, orthorect=False):
        """Prepare the class for dataset reading."""
        super().__init__(filename, filename_info, filetype_info, auto_maskandscale=True)

        # Saves the orthorectification flag
        self.orthorect = orthorect and filetype_info.get('orthorect', True)

        # Saves the interpolation flag
        self.interpolate = filetype_info.get('interpolate', True)

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using file_key in dataset_info."""
        var_key = dataset_info['file_key']
        logger.debug('Reading in file to get dataset with key %s.', var_key)

        try:
            variable = self[var_key]
        except KeyError:
            logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
            return None

        # If the dataset is marked for interpolation, perform the interpolation from tie points to pixels
        if dataset_info.get('interpolate', False) and self.interpolate:
            variable = self._perform_interpolation(variable)

        # Perform the calibration if required
        if dataset_info['calibration'] is not None:
            variable = self._perform_calibration(variable, dataset_info)

        # Perform the orthorectification if required
        if self.orthorect:
            orthorect_data_name = dataset_info.get('orthorect_data', None)
            if orthorect_data_name is not None:
                variable = self._perform_orthorectification(variable, orthorect_data_name)

        # If the dataset contains a longitude, change it to the interval [0., 360.) as natively in the product
        # since the unwrapping performed during the interpolation might have created values outside this range
        if dataset_info.get('standard_name', None) == 'longitude':
            variable = self._perform_wrapping(variable)

        # Manages the dataset attributes
        variable.attrs['units'] = variable.attrs['units'] if 'units' in variable.attrs else None

        variable.attrs.update(dataset_info)
        global_attributes = self._get_global_attributes()
        variable.attrs.update(global_attributes)

        return variable

    def _perform_wrapping(self, variable):
        """Wrap the values to the interval [0, 360.).

        Args:
            variable: xarray DataArray containing the dataset to wrap.

        Returns:
            DataArray: array containing the wrapped values and all the original metadata

        """
        variable %= 360.
        new_variable = xr.DataArray(
            data=da.from_array(variable, CHUNK_SIZE),
            attrs=variable.attrs,
            name=variable.name,
            dims=variable.dims
        )
        return new_variable

    def _perform_interpolation(self, variable):
        """Perform the interpolation from tie points to pixel points.

        Args:
            variable: xarray DataArray containing the dataset to interpolate.

        Returns:
            DataArray: array containing the interpolate values, all the original metadata
                       and the updated dimension names.

        """
        interpolated_values = tie_points_interpolation(variable)
        new_variable = xr.DataArray(
            data=da.from_array(interpolated_values, CHUNK_SIZE),
            attrs=variable.attrs,
            name=variable.name,
            dims=('num_pixels', 'num_lines')
        )
        return new_variable

    def _perform_orthorectification(self, variable, orthorect_data_name):
        """Perform the orthorectification."""
        raise NotImplementedError

    def _perform_calibration(self, variable, dataset_info):
        """Perform the calibration."""
        raise NotImplementedError

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
        # which are found in the 'quality' group of the VII product
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
        return datetime.strptime(self['/attr/sensing_start_time_utc'], '%Y%m%d%H%M%S.%f')

    @property
    def end_time(self):
        """Get observation end time."""
        return datetime.strptime(self['/attr/sensing_end_time_utc'], '%Y%m%d%H%M%S.%f')

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
        # This parameter is not applicable to VII
        return None
