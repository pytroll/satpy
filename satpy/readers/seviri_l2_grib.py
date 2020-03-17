#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2019-2020 Satpy developers
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

"""Reader for the SEVIRI L2 products in GRIB2 format.

References:
    FM 92 GRIB Edition 2
    https://www.wmo.int/pages/prog/www/WMOCodes/Guides/GRIB/GRIB2_062006.pdf

"""

import logging
import numpy as np
import xarray as xr
import dask.array as da

from datetime import timedelta

from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers._geos_area import get_area_definition
from satpy.readers.seviri_base import (calculate_area_extent,
                                       PLATFORM_DICT,
                                       REPEAT_CYCLE_DURATION)
from satpy import CHUNK_SIZE

try:
    import eccodes as ec
except ImportError:
    raise ImportError(
        "Missing eccodes-python and/or eccodes C-library installation. Use conda to install eccodes")


logger = logging.getLogger(__name__)


class SeviriL2GribFileHandler(BaseFileHandler):
    """Reader class for SEVIRI L2 products in GRIB format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Read the global attributes and prepare for dataset reading."""
        super().__init__(filename, filename_info, filetype_info)
        # Turn on support for multiple fields in single GRIB messages (required for SEVIRI L2 files)
        ec.codes_grib_multi_support_on()
        self._read_global_attributes()

    def _read_global_attributes(self):
        """Read the global product attributes from the first message.

        Read the information about the date and time of the data product,
        the projection and area definition and the number of messages.

        """
        with open(self.filename, 'rb') as fh:
            gid = ec.codes_grib_new_from_file(fh)

            if gid is None:
                # Could not obtain a valid message id: set attributes to None, number of messages to 0
                logger.warning("Could not obtain a valid message id in GRIB file")

                self._ssp_lon = None
                self._nrows = None
                self._ncols = None
                self._pdict, self._area_dict = None, None

                return

            # Read SSP and date/time
            self._ssp_lon = self._get_from_msg(gid, 'longitudeOfSubSatellitePointInDegrees')

            # Read number of points on the x and y axes
            self._nrows = self._get_from_msg(gid, 'Ny')
            self._ncols = self._get_from_msg(gid, 'Nx')

            # Creates the projection and area dictionaries
            self._pdict, self._area_dict = self._get_proj_area(gid)

            # Determine the number of messages in the product by iterating until an invalid id is obtained
            i = 1
            ec.codes_release(gid)
            while True:
                gid = ec.codes_grib_new_from_file(fh)
                if gid is None:
                    break
                ec.codes_release(gid)
                i = i+1

    @property
    def start_time(self):
        """Return the sensing start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Return the sensing end time."""
        return self.start_time + timedelta(minutes=REPEAT_CYCLE_DURATION)

    def get_area_def(self, dataset_id):
        """Return the area definition for a dataset."""
        # The area extension depends on the resolution of the dataset
        area_dict = self._area_dict.copy()
        area_dict['resolution'] = dataset_id.resolution
        area_extent = calculate_area_extent(area_dict)

        # Call the get_area_definition function to obtain the area
        area_def = get_area_definition(self._pdict, area_extent)

        return area_def

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the parameter_number key in dataset_info."""
        logger.debug('Reading in file to get dataset with parameter number %d.',
                     dataset_info['parameter_number'])

        xarr = None

        with open(self.filename, 'rb') as fh:
            # Iterate until a message containing the correct parameter number is found
            while True:
                gid = ec.codes_grib_new_from_file(fh)

                if gid is None:
                    # Could not obtain a valid message ID, break out of the loop
                    logger.warning("Could not find parameter_number %d in GRIB file, no valid Dataset created",
                                   dataset_info['parameter_number'])
                    break

                # Check if the parameter number in the GRIB message corresponds to the required key
                parameter_number = self._get_from_msg(gid, 'parameterNumber')
                if parameter_number != dataset_info['parameter_number']:
                    # The parameter number is not the correct one, skip to next message
                    ec.codes_release(gid)
                    continue

                # Read the missing value
                missing_value = self._get_from_msg(gid, 'missingValue')

                # Retrieve values and metadata from the GRIB message, masking the values equal to missing_value
                xarr = self._get_xarray_from_msg(gid)
                xarr.where(xarr.data == missing_value, np.NaN)

                ec.codes_release(gid)

                # Combine all metadata into the dataset attributes and break out of the loop
                xarr.attrs.update(dataset_info)
                xarr.attrs.update(self._get_global_attributes())
                break

        return xarr

    def _get_proj_area(self, gid):
        """Compute the dictionary with the projection and area definition from a GRIB message.

        Args:
            gid: The ID of the GRIB message.

        Returns:
            tuple: A tuple of two dictionaries for the projection and the area definition.
                pdict:
                    a: Earth major axis [m]
                    b: Earth minor axis [m]
                    h: Height over surface [m]
                    ssp_lon: longitude of subsatellite point [deg]
                    nlines: number of lines
                    ncols: number of columns
                    a_name: name of the area
                    a_desc: description of the area
                    p_id: id of the projection
                area_dict:
                    center_point: coordinate of the center point
                    north: coodinate of the north limit
                    east: coodinate of the east limit
                    west: coodinate of the west limit
                    south: coodinate of the south limit

        """
        # Read all projection and area parameters from the message
        earth_major_axis_in_meters = self._get_from_msg(gid, 'earthMajorAxis') * 1000.0   # [m]
        earth_minor_axis_in_meters = self._get_from_msg(gid, 'earthMinorAxis') * 1000.0   # [m]
        nr_in_radius_of_earth = self._get_from_msg(gid, 'NrInRadiusOfEarth')
        xp_in_grid_lengths = self._get_from_msg(gid, 'XpInGridLengths')
        h_in_meters = earth_major_axis_in_meters * (nr_in_radius_of_earth - 1.0)   # [m]

        # Create the dictionary with the projection data
        pdict = {
            'a': earth_major_axis_in_meters,
            'b': earth_minor_axis_in_meters,
            'h': h_in_meters,
            'ssp_lon': self._ssp_lon,
            'nlines': self._ncols,
            'ncols': self._nrows,
            'a_name': 'geos_seviri',
            'a_desc': 'Calculated area for SEVIRI L2 GRIB product',
            'p_id': 'geos',
        }

        # Compute the dictionary with the area extension
        area_dict = {
            'center_point': xp_in_grid_lengths + 0.5,
            'north': self._nrows,
            'east': 1,
            'west': self._ncols,
            'south': 1,
        }

        return pdict, area_dict

    def _get_xarray_from_msg(self, gid):
        """Read the values from the GRIB message and return a DataArray object.

        Args:
            gid: The ID of the GRIB message.

        Returns:
            DataArray: The array containing the retrieved values.

        """
        # Data from GRIB message are read into an Xarray...
        xarr = xr.DataArray(da.from_array(ec.codes_get_values(
            gid).reshape(self._nrows, self._ncols), CHUNK_SIZE), dims=('y', 'x'))

        return xarr

    def _get_global_attributes(self):
        """Create a dictionary of global attributes to be added to all datasets.

        Returns:
            dict: A dictionary of global attributes.
                ssp_lon: longitude of subsatellite point
                sensor: name of sensor
                platform_name: name of the platform

        """
        orbital_parameters = {
            'projection_longitude': self._ssp_lon
        }

        attributes = {
            'orbital_parameters': orbital_parameters,
            'sensor': 'seviri',
            'platform_name': PLATFORM_DICT[self.filename_info['spacecraft']]
        }
        return attributes

    def _get_from_msg(self, gid, key):
        """Get a value from the GRIB message based on the key, return None if missing.

        Args:
            gid: The ID of the GRIB message.
            key: The key of the required attribute.

        Returns:
            The retrieved attribute or None if the key is missing.

        """
        try:
            attr = ec.codes_get(gid, key)
        except ec.KeyValueNotFoundError:
            logger.warning("Key %s not found in GRIB message", key)
            attr = None
        return attr
