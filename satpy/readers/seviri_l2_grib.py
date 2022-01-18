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
    EUMETSAT Product Navigator
    https://navigator.eumetsat.int/
"""

import logging
from datetime import timedelta

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers._geos_area import get_area_definition, get_geos_area_naming
from satpy.readers.eum_base import get_service_mode
from satpy.readers.file_handlers import BaseFileHandler
from satpy.readers.seviri_base import PLATFORM_DICT, REPEAT_CYCLE_DURATION, calculate_area_extent

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
        self._area_dict['column_step'] = dataset_id.resolution
        self._area_dict['line_step'] = dataset_id.resolution

        area_extent = calculate_area_extent(self._area_dict)

        # Call the get_area_definition function to obtain the area
        area_def = get_area_definition(self._pdict, area_extent)

        return area_def

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using the parameter_number key in dataset_info.

        In a previous version of the reader, the attributes (nrows, ncols, ssp_lon) and projection information
        (pdict and area_dict) were computed while initializing the file handler. Also the code would break out from
        the While-loop below as soon as the correct parameter_number was found. This has now been revised becasue the
        reader would sometimes give corrupt information about the number of messages in the file and the dataset
        dimensions within a given message if the file was only partly read (not looping over all messages) in an earlier
        instance.
        """
        logger.debug('Reading in file to get dataset with parameter number %d.',
                     dataset_info['parameter_number'])

        xarr = None
        message_found = False
        with open(self.filename, 'rb') as fh:

            # Iterate over all messages and fetch data when the correct parameter number is found
            while True:
                gid = ec.codes_grib_new_from_file(fh)

                if gid is None:
                    if not message_found:
                        # Could not obtain a valid message ID from the grib file
                        logger.warning("Could not find parameter_number %d in GRIB file, no valid Dataset created",
                                       dataset_info['parameter_number'])
                    break

                # Check if the parameter number in the GRIB message corresponds to the required key
                parameter_number = self._get_from_msg(gid, 'parameterNumber')

                if parameter_number == dataset_info['parameter_number']:

                    self._res = dataset_id.resolution
                    self._read_attributes(gid)

                    # Read the missing value
                    missing_value = self._get_from_msg(gid, 'missingValue')

                    # Retrieve values and metadata from the GRIB message, masking the values equal to missing_value
                    xarr = self._get_xarray_from_msg(gid)

                    xarr.data = da.where(xarr.data == missing_value, np.nan, xarr.data)

                    ec.codes_release(gid)

                    # Combine all metadata into the dataset attributes and break out of the loop
                    xarr.attrs.update(dataset_info)
                    xarr.attrs.update(self._get_attributes())

                    message_found = True

                else:
                    # The parameter number is not the correct one, release gid and skip to next message
                    ec.codes_release(gid)

        return xarr

    def _read_attributes(self, gid):
        """Read the parameter attributes from the message and create the projection and area dictionaries."""
        # Read SSP and date/time
        self._ssp_lon = self._get_from_msg(gid, 'longitudeOfSubSatellitePointInDegrees')

        # Read number of points on the x and y axes
        self._nrows = self._get_from_msg(gid, 'Ny')
        self._ncols = self._get_from_msg(gid, 'Nx')

        # Creates the projection and area dictionaries
        self._pdict, self._area_dict = self._get_proj_area(gid)

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
        # Get name of area definition
        area_naming_input_dict = {'platform_name': 'msg',
                                  'instrument_name': 'seviri',
                                  'resolution': self._res,
                                  }

        area_naming = get_geos_area_naming({**area_naming_input_dict,
                                            **get_service_mode('seviri', self._ssp_lon)})

        # Read all projection and area parameters from the message
        earth_major_axis_in_meters = self._get_from_msg(gid, 'earthMajorAxis') * 1000.0  # [m]
        earth_minor_axis_in_meters = self._get_from_msg(gid, 'earthMinorAxis') * 1000.0  # [m]

        earth_major_axis_in_meters = self._scale_earth_axis(earth_major_axis_in_meters)
        earth_minor_axis_in_meters = self._scale_earth_axis(earth_minor_axis_in_meters)

        nr_in_radius_of_earth = self._get_from_msg(gid, 'NrInRadiusOfEarth')
        xp_in_grid_lengths = self._get_from_msg(gid, 'XpInGridLengths')
        h_in_meters = earth_major_axis_in_meters * (nr_in_radius_of_earth - 1.0)  # [m]

        # Create the dictionary with the projection data
        pdict = {
            'a': earth_major_axis_in_meters,
            'b': earth_minor_axis_in_meters,
            'h': h_in_meters,
            'ssp_lon': self._ssp_lon,
            'nlines': self._ncols,
            'ncols': self._nrows,
            'a_name': area_naming['area_id'],
            'a_desc': area_naming['description'],
            'p_id': "",
        }

        # Compute the dictionary with the area extension
        area_dict = {
            'center_point': xp_in_grid_lengths,
            'north': self._nrows,
            'east': 1,
            'west': self._ncols,
            'south': 1,
        }

        return pdict, area_dict

    @staticmethod
    def _scale_earth_axis(data):
        """Scale Earth axis data to make sure the value matched the expected unit [m].

        The earthMinorAxis value stored in the aerosol over sea product is scaled incorrectly by a factor of 1e8. This
        method provides a flexible temporarily workaraound by making sure that all earth axis values are scaled such
        that they are on the order of millions of meters as expected by the reader. As soon as the scaling issue has
        been resolved by EUMETSAT this workaround can be removed.

        """
        scale_factor = 10 ** np.ceil(np.log10(1e6/data))
        return data * scale_factor

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

    def _get_attributes(self):
        """Create a dictionary of  attributes to be added to the dataset.

        Returns:
            dict: A dictionary of parameter attributes.
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

    @staticmethod
    def _get_from_msg(gid, key):
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
