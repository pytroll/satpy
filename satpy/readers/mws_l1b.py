#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Adam.Dybbroe

# Author(s):

#   Adam.Dybbroe <a000680@c21856.ad.smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Reader for the EPS-SG Microwave Sounder (MWS) level-1b data.

Documentation: https://www.eumetsat.int/media/44139
"""

import logging

import numpy as np
from netCDF4 import default_fillvals

from .netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


# dict containing all available auxiliary data parameters to be read using the index map. Keys are the
# parameter name and values are the paths to the variable inside the netcdf

AUX_DATA = {
    'subsatellite_latitude': 'state/platform/subsatellite_latitude',
    'subsatellite_longitude': 'state/platform/subsatellite_longitude',
    'platform_altitude': 'state/platform/platform_altitude',
    'subsolar_latitude': 'state/celestial/subsolar_latitude',
    'subsolar_longitude': 'state/celestial/subsolar_longitude',
    'earth_sun_distance': 'state/celestial/earth_sun_distance',
    'sun_satellite_distance': 'state/celestial/sun_satellite_distance',
    'time': 'time',
    'swath_number': 'data/swath_number',
    'swath_direction': 'data/swath_direction',
    'mws_lat': 'data/navigation/mws_lat',
    'mws_lon': 'data/navigation/mws_lon',
}

MWS_CHANNEL_NAMES_TO_NUMBER = {'1': 1, '2': 2, '3': 3, '4': 4,
                               '5': 5, '6': 6, '7': 7, '8': 8}

MWS_CHANNEL_NAMES = ['1', '2', '3', '4', '5', '6']


def get_channel_index_from_name(chname):
    """Get the MWS channel index from the channel name."""
    chindex = MWS_CHANNEL_NAMES_TO_NUMBER.get(chname, 0) - 1
    if 0 <= chindex < 24:
        return chindex
    raise AttributeError("Channel name %s not supported: " % chname)


def _get_aux_data_name_from_dsname(dsname):
    aux_data_name = [key for key in AUX_DATA.keys() if key in dsname]
    if len(aux_data_name) > 0:
        return aux_data_name[0]

    return None


class MWSL1BFile(NetCDF4FileHandler):
    """Class implementing the EPS-SG-A1 MWS L1b Filehandler.

    This class implements the European Polar System Second Generation (EPS-SG)
    Microwave Sounder (MWS) Level-1b NetCDF reader.  It is designed to be used
    through the :class:`~satpy.Scene` class using the :mod:`~satpy.Scene.load`
    method with the reader ``"mws_l1b_nc"``.

    """

    # FIXME!
    #
    # After launch: translate to Metop-X instead?
    _platform_name_translate = {
        "SGA1": "Metop-SG-A1",
        "SGA2": "Metop-SG-A2",
        "SGA3": "Metop-SG-A3"}

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize file handler."""
        super().__init__(filename, filename_info,
                         filetype_info,
                         cache_var_size=10000,
                         cache_handle=True)
        logger.debug('Reading: {}'.format(self.filename))
        logger.debug('Start: {}'.format(self.start_time))
        logger.debug('End: {}'.format(self.end_time))

        self._cache = {}

        self._channel_names = MWS_CHANNEL_NAMES

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info['start_time']

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info['end_time']

    def get_dataset(self, key, info=None):
        """Load a dataset."""
        logger.debug('Reading {} from {}'.format(key['name'], self.filename))

        if _get_aux_data_name_from_dsname(key['name']) is not None:
            return self._get_dataset_aux_data(key['name'], info=info)
        elif any(lb in key['name'] for lb in {"1", "2", "3", "4"}):
            return self._get_dataset_channel(key, info=info)
        else:
            raise ValueError("Unknown dataset key, not a channel, quality or auxiliary data: "
                             f"{key['name']:s}")

    def _standardize_dims(self, variable):
        """Standardize dims to y, x."""
        if 'n_scans' in variable.dims:
            variable = variable.rename({'n_fovs': 'x', 'n_scans': 'y'})
        if variable.dims[0] == 'x':
            variable = variable.transpose('y', 'x')
        return variable

    def _get_dataset_channel(self, key, info=None):
        """Load dataset corresponding to channel measurement.

        Load a dataset when the key refers to a measurand, whether uncalibrated
        (counts) or calibrated in terms of brightness temperature, radiance, or
        reflectance.
        """
        # Get the dataset
        # Get metadata for given dataset
        grp_pth = 'data/calibration/mws_toa_brightness_temperature'
        channel_index = get_channel_index_from_name(key['name'])

        data = self[grp_pth][:, :, channel_index]
        attrs = data.attrs.copy()

        fv = attrs.pop(
            "FillValue",
            default_fillvals.get(data.dtype.str[1:], np.nan))
        vr = attrs.get("valid_range", [-np.inf, np.inf])
        if key['calibration'] == "counts":
            attrs["_FillValue"] = fv
            nfv = fv
        else:
            nfv = np.nan
        data = data.where(data >= vr[0], nfv)
        data = data.where(data <= vr[1], nfv)

        # Manage the attributes of the dataset
        data.attrs.setdefault('units', None)
        data.attrs.update(info)
        # variable.attrs.update(self._get_global_attributes()) # FIXME! See VII reader

        return data

    def _get_dataset_aux_data(self, dsname, info=None):
        """Get the auxiliary data arrays using the index map."""
        # Geolocation:
        if dsname in ['mws_lat', 'mws_lon']:
            var_key = AUX_DATA.get(dsname)
        else:
            raise NotImplementedError("Only lons and lats supported - no other auxillary data yet...")

        try:
            variable = self[var_key]
        except KeyError:
            logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
            return None

        # Scale the data:
        variable.data = variable.data * variable.attrs['scale_factor'] + variable.attrs['add_offset']

        # Manage the attributes of the dataset
        variable.attrs.setdefault('units', None)
        variable.attrs.update(info)
        # variable.attrs.update(self._get_global_attributes()) # FIXME! See VII reader
        return variable
