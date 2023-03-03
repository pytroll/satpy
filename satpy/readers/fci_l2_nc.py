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
from contextlib import suppress

import numpy as np
import xarray as xr
from pyresample import geometry

from satpy import CHUNK_SIZE
from satpy.readers._geos_area import get_geos_area_naming, make_ext
from satpy.readers.eum_base import get_service_mode
from satpy.readers.file_handlers import BaseFileHandler
from satpy.resample import get_area_def

logger = logging.getLogger(__name__)

SSP_DEFAULT = 0.0


class FciL2CommonFunctions(object):
    """Shared operations for file handlers."""

    @property
    def spacecraft_name(self):
        """Return spacecraft name."""
        return self.nc.attrs['platform']

    @property
    def sensor_name(self):
        """Return instrument name."""
        return self.nc.attrs['data_source']

    @property
    def ssp_lon(self):
        """Return longitude at subsatellite point."""
        try:
            return float(self.nc['mtg_geos_projection'].attrs['longitude_of_projection_origin'])
        except (KeyError, AttributeError):
            logger.warning(f"ssp_lon could not be obtained from file content, using default value "
                           f"of {SSP_DEFAULT} degrees east instead")
            return SSP_DEFAULT

    def _get_global_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets.

        Returns:
            dict: A dictionary of global attributes.
                filename: name of the product file
                spacecraft_name: name of the spacecraft
                ssp_lon: longitude of subsatellite point
                sensor: name of sensor
                platform_name: name of the platform

        """
        attributes = {
            'filename': self.filename,
            'spacecraft_name': self.spacecraft_name,
            'ssp_lon': self.ssp_lon,
            'sensor': self.sensor_name,
            'platform_name': self.spacecraft_name,
        }
        return attributes

    def _set_attributes(self, variable, dataset_info, segmented=False):
        """Set dataset attributes."""
        if segmented:
            xdim, ydim = "number_of_FoR_cols", "number_of_FoR_rows"
        else:
            xdim, ydim = "number_of_columns", "number_of_rows"

        if dataset_info['file_key'] not in ['product_quality', 'product_completeness', 'product_timeliness']:
            variable = variable.rename({ydim: 'y', xdim: 'x'})

        variable.attrs.setdefault('units', None)
        variable.attrs.update(dataset_info)
        variable.attrs.update(self._get_global_attributes())

        return variable

    def _slice_dataset(self, variable, dataset_info, dimensions):
        """Slice data if dimension layers have been provided in yaml-file."""
        slice_dict = {dim: dataset_info[dim_id] for (dim, dim_id) in dimensions.items()
                      if dim_id in dataset_info.keys() and dim in variable.dims}
        for dim, dim_ind in slice_dict.items():
            logger.debug(f"Extracting {dimensions[dim]}-index {dim_ind} from dimension {dim!r}.")
        variable = variable.sel(slice_dict)

        return variable

    @staticmethod
    def _mask_data(variable, fill_value):
        """Set fill_values, as defined in yaml-file, to NaN.

        Set data points in variable to NaN if they are equal to fill_value
        or any of the values in fill_value if fill_value is a list.
        """
        if not isinstance(fill_value, list):
            fill_value = [fill_value]

        for val in fill_value:
            variable = variable.where(variable != val).astype('float32')

        return variable

    def __del__(self):
        """Close the NetCDF file that may still be open."""
        with suppress(AttributeError, OSError):
            self.nc.close()


class FciL2NCFileHandler(FciL2CommonFunctions, BaseFileHandler):
    """Reader class for FCI L2 products in NetCDF4 format."""

    def __init__(self, filename, filename_info, filetype_info, with_area_definition=True):
        """Open the NetCDF file with xarray and prepare for dataset reading."""
        super().__init__(filename, filename_info, filetype_info)

        # Use xarray's default netcdf4 engine to open the fileq
        self.nc = xr.open_dataset(
            self.filename,
            decode_cf=True,
            mask_and_scale=True,
            chunks={
                'number_of_columns': CHUNK_SIZE,
                'number_of_rows': CHUNK_SIZE
            }
        )

        if with_area_definition is False:
            logger.info("Setting `with_area_defintion=False` has no effect on pixel-based products.")

        # Read metadata which are common to all datasets
        self.nlines = self.nc['y'].size
        self.ncols = self.nc['x'].size
        self._projection = self.nc['mtg_geos_projection']
        self.multi_dims = {'maximum_number_of_layers': 'layer', 'number_of_vis_channels': 'vis_channel_id'}

    def get_area_def(self, key):
        """Return the area definition."""
        try:
            return self._area_def
        except AttributeError:
            raise NotImplementedError

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the file_key in dataset_info."""
        var_key = dataset_info['file_key']
        par_name = dataset_info['name']
        logger.debug('Reading in file to get dataset with key %s.', var_key)

        try:
            variable = self.nc[var_key]
        except KeyError:
            logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
            return None

        # Compute the area definition
        if var_key not in ['product_quality', 'product_completeness', 'product_timeliness']:
            self._area_def = self._compute_area_def(dataset_id)

        if any(dim_id in dataset_info.keys() for dim_id in self.multi_dims.values()):
            variable = self._slice_dataset(variable, dataset_info, self.multi_dims)

        if par_name == 'retrieved_cloud_optical_thickness':
            variable = self.get_total_cot(variable)

        if dataset_info['file_type'] == 'nc_fci_test_clm':
            variable = self._decode_clm_test_data(variable, dataset_info)

        if 'fill_value' in dataset_info:
            variable = self._mask_data(variable, dataset_info['fill_value'])

        variable = self._set_attributes(variable, dataset_info)

        return variable

    @staticmethod
    def _decode_clm_test_data(variable, dataset_info):
        if dataset_info['file_key'] != 'cloud_mask_cmrt6_test_result':
            variable = variable.astype('uint32')
            variable.values = (variable.values >> dataset_info['extract_byte'] << 31 >> 31).astype('int8')

        return variable

    def _compute_area_def(self, dataset_id):
        """Compute the area definition.

        Returns:
            AreaDefinition: A pyresample AreaDefinition object containing the area definition.

        """
        area_extent = self._get_area_extent()
        area_naming, proj_dict = self._get_proj_area(dataset_id)
        area_def = geometry.AreaDefinition(
            area_naming['area_id'],
            area_naming['description'],
            "",
            proj_dict,
            self.ncols,
            self.nlines,
            area_extent)

        return area_def

    def _get_area_extent(self):
        """Calculate area extent of dataset."""
        # Load and convert x/y coordinates to degrees as required by the make_ext function
        x = self.nc['x']
        y = self.nc['y']
        x_deg = np.degrees(x)
        y_deg = np.degrees(y)

        # Select the extreme points and calcualte area extent (not: these refer to pixel center)
        ll_x, ur_x = -x_deg.values[0], -x_deg.values[-1]
        ll_y, ur_y = y_deg.values[-1], y_deg.values[0]
        h = float(self._projection.attrs['perspective_point_height'])
        area_extent_pixel_center = make_ext(ll_x, ur_x, ll_y, ur_y, h)

        # Shift area extent by half a pixel to get the area extent w.r.t. the dataset/pixel corners
        scale_factor = (x[1:]-x[0:-1]).values.mean()
        res = abs(scale_factor) * h
        area_extent = tuple(i + res/2 if i > 0 else i - res/2 for i in area_extent_pixel_center)

        return area_extent

    def _get_proj_area(self, dataset_id):
        """Extract projection and area information."""
        # Read the projection data from the mtg_geos_projection variable
        a = float(self._projection.attrs['semi_major_axis'])
        h = float(self._projection.attrs['perspective_point_height'])

        # Some L2PF test data files have a typo in the keyname for the inverse flattening parameter. Use a default value
        # as fallback until all L2PF test files are correctly formatted.
        rf = float(self._projection.attrs.get('inverse_flattening', 298.257223563))

        res = dataset_id.resolution

        area_naming_input_dict = {'platform_name': 'mtg',
                                  'instrument_name': 'fci',
                                  'resolution': res,
                                  }

        area_naming = get_geos_area_naming({**area_naming_input_dict,
                                            **get_service_mode('fci', self.ssp_lon)})

        proj_dict = {'a': a,
                     'lon_0': self.ssp_lon,
                     'h': h,
                     "rf": rf,
                     'proj': 'geos',
                     'units': 'm',
                     "sweep": 'y'}

        return area_naming, proj_dict

    @staticmethod
    def get_total_cot(variable):
        """Sum the cloud optical thickness from the two OCA layers.

        The optical thickness has to be transformed to linear space before adding the values from the two layers. The
        combined/total optical thickness is then transformed back to logarithmic space.
        """
        attrs = variable.attrs
        variable = 10 ** variable
        variable = variable.fillna(0.)
        variable = variable.sum(dim='maximum_number_of_layers', keep_attrs=True)
        variable = variable.where(variable != 0., np.nan)
        variable = np.log10(variable)
        variable.attrs = attrs

        return variable


class FciL2NCSegmentFileHandler(FciL2CommonFunctions, BaseFileHandler):
    """Reader class for FCI L2 Segmented products in NetCDF4 format."""

    def __init__(self, filename, filename_info, filetype_info, with_area_definition=False):
        """Open the NetCDF file with xarray and prepare for dataset reading."""
        super().__init__(filename, filename_info, filetype_info)
        # Use xarray's default netcdf4 engine to open the file
        self.nc = xr.open_dataset(
            self.filename,
            decode_cf=True,
            mask_and_scale=True,
            chunks={
                'number_of_FoR_cols': CHUNK_SIZE,
                'number_of_FoR_rows': CHUNK_SIZE
            }
        )

        # Read metadata which are common to all datasets
        self.nlines = self.nc['number_of_FoR_rows'].size
        self.ncols = self.nc['number_of_FoR_cols'].size
        self.with_adef = with_area_definition
        self.multi_dims = {
            'number_of_categories': 'category_id', 'number_of_channels': 'channel_id',
            'number_of_vis_channels': 'vis_channel_id', 'number_of_ir_channels': 'ir_channel_id',
            'number_test': 'test_id',
        }

    def get_area_def(self, key):
        """Return the area definition."""
        try:
            return self._area_def
        except AttributeError:
            raise NotImplementedError

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the file_key in dataset_info."""
        var_key = dataset_info['file_key']
        logger.debug('Reading in file to get dataset with key %s.', var_key)

        try:
            variable = self.nc[var_key]
        except KeyError:
            logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
            return None

        if any(dim_id in dataset_info.keys() for dim_id in self.multi_dims.values()):
            variable = self._slice_dataset(variable, dataset_info, self.multi_dims)

        if self.with_adef and var_key not in ['longitude', 'latitude',
                                              'product_quality', 'product_completeness', 'product_timeliness']:
            self._area_def = self._construct_area_def(dataset_id)

            # coordinates are not relevant when returning data with an AreaDefinition
            if 'coordinates' in dataset_info.keys():
                del dataset_info['coordinates']

        if 'fill_value' in dataset_info:
            variable = self._mask_data(variable, dataset_info['fill_value'])

        variable = self._set_attributes(variable, dataset_info, segmented=True)

        return variable

    def _construct_area_def(self, dataset_id):
        """Construct the area definition.

        Returns:
            AreaDefinition: A pyresample AreaDefinition object containing the area definition.

        """
        res = dataset_id.resolution

        area_naming_input_dict = {'platform_name': 'mtg',
                                  'instrument_name': 'fci',
                                  'resolution': res,
                                  }

        area_naming = get_geos_area_naming({**area_naming_input_dict,
                                            **get_service_mode('fci', self.ssp_lon)})

        # Construct area definition from standardized area definition.
        stand_area_def = get_area_def(area_naming['area_id'])

        if (stand_area_def.x_size != self.ncols) | (stand_area_def.y_size != self.nlines):
            raise NotImplementedError('Unrecognised AreaDefinition.')

        mod_area_extent = self._modify_area_extent(stand_area_def.area_extent)

        area_def = geometry.AreaDefinition(
            stand_area_def.area_id,
            stand_area_def.description,
            "",
            stand_area_def.proj_dict,
            stand_area_def.x_size,
            stand_area_def.y_size,
            mod_area_extent)

        return area_def

    @staticmethod
    def _modify_area_extent(stand_area_extent):
        """Modify area extent to macth satellite projection.

        Area extent has to be modified since the L2 products are stored with the south-east
        in the upper-right corner (as opposed to north-east in the standardized area definitions).
        """
        ll_x, ll_y, ur_x, ur_y = stand_area_extent
        ll_y *= -1.
        ur_y *= -1.
        area_extent = tuple([ll_x, ll_y, ur_x, ur_y])

        return area_extent
