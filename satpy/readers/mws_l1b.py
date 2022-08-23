# -*- coding: utf-8 -*-

# Copyright (c) 2022 Pytroll Developers

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
from datetime import datetime

import dask.array as da
import numpy as np
from netCDF4 import default_fillvals

from .netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


# dict containing all available auxiliary data parameters to be read using the index map. Keys are the
# parameter name and values are the paths to the variable inside the netcdf

AUX_DATA = {
    'scantime_utc': 'data/navigation/mws_scantime_utc',
    'solar_azimuth': 'data/navigation/mws_solar_azimuth_angle',
    'solar_zenith': 'data/navigation/mws_solar_zenith_angle',
    'satellite_azimuth': 'data/navigation/mws_satellite_azimuth_angle',
    'satellite_zenith': 'data/navigation/mws_satellite_zenith_angle',
    'surface_type': 'data/navigation/mws_surface_type',
    'terrain_elevation': 'data/navigation/mws_terrain_elevation',
    'mws_lat': 'data/navigation/mws_lat',
    'mws_lon': 'data/navigation/mws_lon',
}

MWS_CHANNEL_NAMES_TO_NUMBER = {'1': 1, '2': 2, '3': 3, '4': 4,
                               '5': 5, '6': 6, '7': 7, '8': 8,
                               '9': 9, '10': 10, '11': 11, '12': 12,
                               '13': 13, '14': 14, '15': 15, '16': 16,
                               '17': 17, '18': 18, '19': 19, '20': 20,
                               '21': 21, '22': 22, '23': 23, '24': 24}

MWS_CHANNEL_NAMES = list(MWS_CHANNEL_NAMES_TO_NUMBER.keys())
MWS_CHANNELS = set(MWS_CHANNEL_NAMES_TO_NUMBER.keys())


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
        return datetime.strptime(self['/attr/sensing_start_time_utc'],
                                 '%Y-%m-%d %H:%M:%S.%f')

    @property
    def end_time(self):
        """Get end time."""
        return datetime.strptime(self['/attr/sensing_end_time_utc'],
                                 '%Y-%m-%d %H:%M:%S.%f')

    @property
    def sensor(self):
        """Get the sensor name."""
        return self['/attr/instrument']

    @property
    def platform_name(self):
        """Get the platform name."""
        return self._platform_name_translate.get(self['/attr/spacecraft'])

    @property
    def sub_satellite_longitude_start(self):
        """Get the longitude of sub-satellite point at start of the product."""
        return self['status/satellite/subsat_longitude_start'].data.item()

    @property
    def sub_satellite_latitude_start(self):
        """Get the latitude of sub-satellite point at start of the product."""
        return self['status/satellite/subsat_latitude_start'].data.item()

    @property
    def sub_satellite_longitude_end(self):
        """Get the longitude of sub-satellite point at end of the product."""
        return self['status/satellite/subsat_longitude_end'].data.item()

    @property
    def sub_satellite_latitude_end(self):
        """Get the latitude of sub-satellite point at end of the product."""
        return self['status/satellite/subsat_latitude_end'].data.item()

    def get_dataset(self, dataset_id, info=None):
        """Load a dataset."""
        logger.debug('Reading {} from {}'.format(dataset_id['name'], self.filename))

        if _get_aux_data_name_from_dsname(dataset_id['name']) is not None:
            return self._get_dataset_aux_data(dataset_id['name'], info=info)
        elif any(lb in dataset_id['name'] for lb in MWS_CHANNELS):
            return self._get_dataset_channel(dataset_id, info=info)
        else:
            raise ValueError("Unknown dataset key, not a channel, quality or auxiliary data: "
                             f"{dataset_id['name']:s}")

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

        i = getattr(data, 'attrs', {})
        i.update(info)
        i.update({
            "platform_name": self.platform_name,
            "sensor": self.sensor,
            "orbital_parameters": {'sub_satellite_latitude_start': self.sub_satellite_latitude_start,
                                   'sub_satellite_longitude_start': self.sub_satellite_longitude_start,
                                   'sub_satellite_latitude_end': self.sub_satellite_latitude_end,
                                   'sub_satellite_longitude_end': self.sub_satellite_longitude_end},
        })
        i.update(key.to_dict())
        data.attrs.update(i)

        return data

    def _get_dataset_aux_data(self, dsname, info=None):
        """Get the auxiliary data arrays using the index map."""
        # Geolocation and navigation data:
        if dsname in ['mws_lat', 'mws_lon',
                      'solar_azimuth', 'solar_zenith',
                      'satellite_azimuth', 'satellite_zenith',
                      'surface_type', 'terrain_elevation']:
            var_key = AUX_DATA.get(dsname)
        else:
            raise NotImplementedError("Dataset %s not supported..." % dsname)

        try:
            variable = self[var_key]
        except KeyError:
            logger.warning("Could not find key %s in NetCDF file, no valid Dataset created", var_key)
            return None

        # Scale the data:
        missing_value = variable.attrs['missing_value']
        if 'scale_factor' in variable.attrs and 'add_offset' in variable.attrs:
            variable.data = da.where(variable.data == missing_value, np.nan,
                                     variable.data * variable.attrs['scale_factor'] + variable.attrs['add_offset'])

        # Manage the attributes of the dataset
        variable.attrs.setdefault('units', None)
        if info:
            variable.attrs.update(info)
        return variable
