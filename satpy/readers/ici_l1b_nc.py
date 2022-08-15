#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Satpy developers
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
"""EUMETSAT EPS-SG Ice Cloud Imager (ICI) Level 1B products reader.

The format is explained in the
`EPS-SG ICI Level 1B Product Format Specification V3A`_.

This version is applicable for the ici test data released in Jan 2021.

.. _EPS-SG ICI Level 1B Product Format Specification V3A: https://www.eumetsat.int/media/47582

"""

import logging
from datetime import datetime
from enum import Enum
from functools import cached_property

import dask.array as da
import numpy as np
import xarray as xr
from geotiepoints.geointerpolator import GeoInterpolator

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


# PLANCK COEFFICIENTS FOR CALIBRATION AS DEFINED BY EUMETSAT
C1 = 1.191042e-5  # [mW/(sr·m2·cm-4)]
C2 = 1.4387752  # [K·cm]
# MEAN EARTH RADIUS AS DEFINED BY IUGG
MEAN_EARTH_RADIUS = 6371008.7714  # [m]


class InterpolationType(Enum):
    """Enum for interpolation types."""

    LONLAT = 0
    SOLAR_ANGLES = 1
    OBSERVATION_ANGLES = 2


class IciL1bNCFileHandler(NetCDF4FileHandler):
    """Reader class for ICI L1B products in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Read the calibration data and prepare the class for dataset reading."""  # noqa: E501
        super().__init__(
            filename, filename_info, filetype_info, auto_maskandscale=True,
        )
        # Read the variables which are required for the calibration
        measurement = 'data/measurement_data'
        self._bt_conversion_a = self[f'{measurement}/bt_conversion_a'].values
        self._bt_conversion_b = self[f'{measurement}/bt_conversion_b'].values
        self._channel_cw = self[f'{measurement}/centre_wavenumber'].values
        self._n_samples = self[measurement].n_samples.size
        self._filetype_info = filetype_info
        self.orthorect = filetype_info.get('orthorect', True)

    @property
    def start_time(self):
        """Get observation start time."""
        try:
            start_time = datetime.strptime(
                self['/attr/sensing_start_time_utc'],
                '%Y%m%d%H%M%S.%f',
            )
        except ValueError:
            start_time = datetime.strptime(
                self['/attr/sensing_start_time_utc'],
                '%Y-%m-%d %H:%M:%S.%f',
            )
        return start_time

    @property
    def end_time(self):
        """Get observation end time."""
        try:
            end_time = datetime.strptime(
                self['/attr/sensing_end_time_utc'],
                '%Y%m%d%H%M%S.%f',
            )
        except ValueError:
            end_time = datetime.strptime(
                self['/attr/sensing_end_time_utc'],
                '%Y-%m-%d %H:%M:%S.%f',
            )
        return end_time

    @property
    def platform_name(self):
        """Return platform name."""
        return self['/attr/spacecraft']

    @property
    def sensor(self):
        """Return sensor."""
        return self['/attr/instrument']

    @property
    def ssp_lon(self):
        """Return subsatellite point longitude."""
        # This parameter is not applicable to ICI?
        return None

    @property
    def observation_azimuth(self):
        """Get observation azimuth angles."""
        observation_azimuth, _ = self.observation_azimuth_and_zenith
        return observation_azimuth

    @property
    def observation_zenith(self):
        """Get observation zenith angles."""
        _, observation_zenith = self.observation_azimuth_and_zenith
        return observation_zenith

    @property
    def solar_azimuth(self):
        """Get solar azimuth angles."""
        solar_azimuth, _ = self.solar_azimuth_and_zenith
        return solar_azimuth

    @property
    def solar_zenith(self):
        """Get solar zenith angles."""
        _, solar_zenith = self.solar_azimuth_and_zenith
        return solar_zenith

    @property
    def longitude(self):
        """Get longitude coordinates."""
        longitude, _ = self.longitude_and_latitude
        return longitude

    @property
    def latitude(self):
        """Get latitude coordinates."""
        _, latitude = self.longitude_and_latitude
        return latitude

    @cached_property
    def observation_azimuth_and_zenith(self):
        """Get observation azimuth and zenith angles."""
        return self._interpolate(InterpolationType.OBSERVATION_ANGLES)

    @cached_property
    def solar_azimuth_and_zenith(self):
        """Get solar azimuth and zenith angles."""
        return self._interpolate(InterpolationType.SOLAR_ANGLES)

    @cached_property
    def longitude_and_latitude(self):
        """Get longitude and latitude coordinates."""
        return self._interpolate(InterpolationType.LONLAT)

    @staticmethod
    def _interpolate_geo(
        longitude,
        latitude,
        n_samples,
    ):
        """
        Perform the interpolation of geographic coordinates from tie points to pixel points.

        Args:
            longitude: xarray DataArray containing the longitude dataset to
                interpolate.
            latitude: xarray DataArray containing the longitude dataset to
                interpolate.
            n_samples: int describing number of samples per scan to interpolate
                onto.

        Returns:
            tuple of arrays containing the interpolate values, all the original
                metadata and the updated dimension names.

        """
        third_dim_name = longitude.dims[2]
        horns = longitude[third_dim_name]
        n_scan = longitude.n_scan
        n_subs = longitude.n_subs
        lons = da.zeros((n_scan.size, n_samples, horns.size))
        lats = da.zeros((n_scan.size, n_samples, horns.size))
        n_subs = np.linspace(0, n_samples - 1, n_subs.size).astype(int)
        for horn in horns.values:
            satint = GeoInterpolator(
                (longitude.values[:, :, horn], latitude.values[:, :, horn]),
                (n_scan.values, n_subs),
                (n_scan.values, np.arange(n_samples)),
            )
            lons_horn, lats_horn = satint.interpolate()
            lons[:, :, horn] = lons_horn
            lats[:, :, horn] = lats_horn
        dims = ['y', 'x', third_dim_name]
        lon = xr.DataArray(
            lons,
            attrs=longitude.attrs,
            dims=dims,
            coords={third_dim_name: horns},
        )
        lat = xr.DataArray(
            lats,
            attrs=latitude.attrs,
            dims=dims,
            coords={third_dim_name: horns},
        )
        return lon, lat

    def _interpolate_viewing_angle(
        self,
        azimuth,
        zenith,
        n_samples,
    ):
        """
        Perform the interpolation of angular coordinates from tie points to pixel points.

        Args:
            azimuth: xarray DataArray containing the azimuth angle dataset to
                interpolate.
            zenith: xarray DataArray containing the zenith angle dataset to
                interpolate.
            n_samples: int describing number of samples per scan to interpolate
                onto.

        Returns:
            tuple of arrays containing the interpolate values, all the original
                metadata and the updated dimension names.

        """
        # interpolate onto spherical coords system with origin at equator
        azimuth, zenith = self._interpolate_geo(azimuth, 90. - zenith, n_samples)
        # transform back such that the origin is at the north pole
        zenith = 90. - zenith
        return azimuth, zenith

    def _interpolate(
        self,
        interpolation_type,
    ):
        """Interpolate from tie points to pixel points."""
        try:
            if interpolation_type is InterpolationType.SOLAR_ANGLES:
                var_key1 = self.filetype_info['solar_azimuth']
                var_key2 = self.filetype_info['solar_zenith']
                interp_method = self._interpolate_viewing_angle
            elif interpolation_type is InterpolationType.OBSERVATION_ANGLES:
                var_key1 = self.filetype_info['observation_azimuth']
                var_key2 = self.filetype_info['observation_zenith']
                interp_method = self._interpolate_viewing_angle
            else:
                var_key1 = self.filetype_info['longitude']
                var_key2 = self.filetype_info['latitude']
                interp_method = self._interpolate_geo
            return interp_method(
                self[var_key1],
                self[var_key2],
                self._n_samples,
            )
        except KeyError:
            logger.warning(f'Datasets for {interpolation_type.name} interpolation not correctly defined in YAML file')  # noqa: E501
        return None, None

    @staticmethod
    def _calibrate_bt(radiance, cw, a, b):
        """Perform the calibration to brightness temperature.

        Args:
            radiance: xarray DataArray or numpy ndarray containing the
                radiance values.
            cw: center wavenumber [cm-1].
            a: temperature coefficient [-].
            b: temperature coefficient [K].

        Returns:
            DataArray: array containing the calibrated brightness
                temperature values.

        """
        return b + (a * C2 * cw / np.log(1 + C1 * cw ** 3 / radiance))

    def _calibrate(self, variable, dataset_info):
        """Perform the calibration.

        Args:
            variable: xarray DataArray containing the dataset to calibrate.
            dataset_info: dictionary of information about the dataset.

        Returns:
            DataArray: array containing the calibrated values and all the
                original metadata.

        """
        calibration_name = dataset_info['calibration']
        if calibration_name == 'brightness_temperature':
            chan_index = dataset_info['chan_index']
            cw = self._channel_cw[chan_index]
            a = self._bt_conversion_a[chan_index]
            b = self._bt_conversion_b[chan_index]
            calibrated_variable = self._calibrate_bt(variable, cw, a, b)
            calibrated_variable.attrs = variable.attrs
        elif calibration_name == 'radiance':
            calibrated_variable = variable
        else:
            raise ValueError("Unknown calibration %s for dataset %s" % (calibration_name, dataset_info['name']))  # noqa: E501

        return calibrated_variable

    def _orthorectify(self, variable, orthorect_data_name):
        """Perform the orthorectification.

        Args:
            variable: xarray DataArray containing the dataset to correct for
                orthorectification.
            orthorect_data_name: name of the orthorectification correction data
                in the product.

        Returns:
            DataArray: array containing the corrected values and all the
                original metadata.

        """
        try:
            # Convert the orthorectification delta values from meters to
            # degrees based on the simplified formula using mean Earth radius
            orthorect_data = self[orthorect_data_name]
            dim = self._get_third_dimension_name(orthorect_data)
            orthorect_data = orthorect_data.sel({dim: variable[dim]})
            variable += np.degrees(orthorect_data.values / MEAN_EARTH_RADIUS)
        except KeyError:
            logger.warning('Required dataset %s for orthorectification not available, skipping', orthorect_data_name)  # noqa: E501
        return variable

    @staticmethod
    def _standardize_dims(variable):
        """Standardize dims to y, x."""
        if 'n_scan' in variable.dims:
            variable = variable.rename({'n_scan': 'y'})
        if 'n_samples' in variable.dims:
            variable = variable.rename({'n_samples': 'x'})
        if variable.dims[0] == 'x':
            variable = variable.transpose('y', 'x')
        return variable

    def _filter_variable(self, variable, dataset_info):
        """Filter variable in the third dimension."""
        dim = self._get_third_dimension_name(variable)
        if dim is not None and dim in dataset_info:
            variable = variable.sel({dim: dataset_info[dim]})
        return variable

    @staticmethod
    def _drop_coords(variable):
        """Drop coords that are not in dims."""
        for coord in variable.coords:
            if coord not in variable.dims:
                variable = variable.drop_vars(coord)
        return variable

    @staticmethod
    def _get_third_dimension_name(variable):
        """Get name of the third dimension of the variable."""
        dims = variable.dims
        if len(dims) < 3:
            return None
        return dims[2]

    def _fetch_variable(self, var_key):
        """Fetch variable."""
        if var_key in [
            'longitude',
            'latitude',
            'observation_zenith',
            'observation_azimuth',
            'solar_zenith',
            'solar_azimuth',
        ] and getattr(self, var_key) is not None:
            variable = getattr(self, var_key).copy()
        else:
            variable = self[var_key]
        return variable

    def get_dataset(self, dataset_id, dataset_info):
        """Get dataset using file_key in dataset_info."""
        var_key = dataset_info['file_key']
        logger.debug(f'Reading in file to get dataset with key {var_key}.')
        try:
            variable = self._fetch_variable(var_key)
        except KeyError:
            logger.warning(f'Could not find key {var_key} in NetCDF file, no valid Dataset created')  # noqa: E501
            return None
        variable = self._filter_variable(variable, dataset_info)
        if dataset_info.get('calibration') is not None:
            variable = self._calibrate(variable, dataset_info)
        if self.orthorect:
            orthorect_data_name = dataset_info.get('orthorect_data', None)
            if orthorect_data_name is not None:
                variable = self._orthorectify(variable, orthorect_data_name)
        variable = self._manage_attributes(variable, dataset_info)
        variable = self._drop_coords(variable)
        variable = self._standardize_dims(variable)
        return variable

    def _manage_attributes(self, variable, dataset_info):
        """Manage attributes of the dataset."""
        variable.attrs.setdefault('units', None)
        variable.attrs.update(dataset_info)
        variable.attrs.update(self._get_global_attributes())
        return variable

    def _get_global_attributes(self):
        """Create a dictionary of global attributes."""
        return {
            'filename': self.filename,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'spacecraft_name': self.platform_name,
            'ssp_lon': self.ssp_lon,
            'sensor': self.sensor,
            'filename_start_time': self.filename_info['sensing_start_time'],
            'filename_end_time': self.filename_info['sensing_end_time'],
            'platform_name': self.platform_name,
            'quality_group': self._get_quality_attributes(),
        }

    def _get_quality_attributes(self):
        """Get quality attributes."""
        quality_group = self['quality']
        quality_dict = {}
        for key in quality_group:
            # Add the values (as Numpy array) of each variable in the group
            # where possible
            try:
                quality_dict[key] = quality_group[key].values
            except ValueError:
                quality_dict[key] = None
        # Add the attributes of the quality group
        quality_dict.update(quality_group.attrs)
        return quality_dict
