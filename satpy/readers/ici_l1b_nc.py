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
"""EUMETSAT EPS-SG Ice Cloud Imager (ICI) Level 1B products reader.

The format is explained in the
`EPS-SG ICI Level 1B Product Format Specification V3A`_.

This version is applicable for the ici test data released in Jan 2021.

.. _EPS-SG ICI Level 1B Product Format Specification V3A: https://www.eumetsat.int/media/47582

"""

import logging
from datetime import datetime
from functools import cached_property

import numpy as np
import xarray as xr
from geotiepoints.geointerpolator import GeoInterpolator, lonlat2xyz, xyz2lonlat

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)


# PLANCK COEFFICIENTS FOR CALIBRATION AS DEFINED BY EUMETSAT
C1 = 1.191042e-5  # [mW/(sr·m2·cm-4)]
C2 = 1.4387752  # [K·cm]
# MEAN EARTH RADIUS AS DEFINED BY IUGG
MEAN_EARTH_RADIUS = 6371008.7714  # [m]


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

    @cached_property
    def observation_azimuth_and_zenith(self):
        """Get observation azimuth and zenith angles."""
        try:
            return self._perform_viewing_angle_interpolation(
                self[self.filetype_info['cached_observation_azimuth']],
                self[self.filetype_info['cached_observation_zenith']],
                self._n_samples,
            )
        except KeyError:
            logger.warning("Cached observation zenith and/or azimuth datasets are not correctly defined in YAML file")  # noqa: E501
        return None, None

    @cached_property
    def solar_azimuth_and_zenith(self):
        """Get solar azimuth and zenith angles."""
        try:
            return self._perform_viewing_angle_interpolation(
                self[self.filetype_info['cached_solar_azimuth']],
                self[self.filetype_info['cached_solar_zenith']],
                self._n_samples,
            )
        except KeyError:
            logger.warning("Cached solar zenith and/or azimuth datasets are not correctly defined in YAML file")  # noqa: E501
        return None, None

    @cached_property
    def longitude_and_latitude(self):
        """Get longitude and latitude coordinates."""
        try:
            return self._perform_geo_interpolation(
                self[self.filetype_info['cached_longitude']],
                self[self.filetype_info['cached_latitude']],
                self._n_samples,
            )
        except KeyError:
            logger.warning("Cached longitude and/or latitude datasets are not correctly defined in YAML file")  # noqa: E501
        return None, None

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
        # This parameter is not applicable to ICI?
        return None

    @property
    def cached_observation_azimuth(self):
        """Get observation azimuth angles."""
        observation_azimuth, _ = self.observation_azimuth_and_zenith
        return observation_azimuth

    @property
    def cached_observation_zenith(self):
        """Get observation zenith angles."""
        _, observation_zenith = self.observation_azimuth_and_zenith
        return observation_zenith

    @property
    def cached_solar_azimuth(self):
        """Get solar azimuth angles."""
        solar_azimuth, _ = self.solar_azimuth_and_zenith
        return solar_azimuth

    @property
    def cached_solar_zenith(self):
        """Get solar zenith angles."""
        _, solar_zenith = self.solar_azimuth_and_zenith
        return solar_zenith

    @property
    def cached_longitude(self):
        """Get longitude coordinates."""
        longitude, _ = self.longitude_and_latitude
        return longitude

    @property
    def cached_latitude(self):
        """Get latitude coordinates."""
        _, latitude = self.longitude_and_latitude
        return latitude

    @staticmethod
    def _calibrate_bt(radiance, cw, a, b):
        """Perform the calibration to brightness temperature.

        Args:
            radiance: numpy ndarray containing the radiance values.
            cw: center wavenumber [cm-1].
            a: temperature coefficient [-].
            b: temperature coefficient [K].

        Returns:
            numpy ndarray: array containing the calibrated brightness
                temperature values.

        """
        return b + (a * C2 * cw / np.log(1 + C1 * cw ** 3 / radiance))

    @staticmethod
    def _perform_geo_interpolation(
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
        horns = longitude.n_horns
        n_scan = longitude.n_scan
        n_subs = longitude.n_subs
        lons = np.zeros((n_scan.size, n_samples, horns.size))
        lats = np.zeros((n_scan.size, n_samples, horns.size))
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
        lon = xr.DataArray(
            lons,
            attrs=longitude.attrs,
            dims=['y', 'x', 'n_horns'],
            coords={"n_horns": horns},
        )
        lat = xr.DataArray(
            lats,
            attrs=latitude.attrs,
            dims=['y', 'x', 'n_horns'],
            coords={"n_horns": horns},
        )
        return lon, lat

    def _perform_viewing_angle_interpolation(
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
        # transform to coordinates where origin is at the equator
        x, y, z = lonlat2xyz(azimuth, 90. - zenith)
        lon, lat = xyz2lonlat(x, y, z)
        lon, lat = self._perform_geo_interpolation(
            lon,
            xr.DataArray(lat),
            n_samples,
        )
        x, y, z = lonlat2xyz(lon, lat)
        # transform from cartesian to spherical coords following Eumetsat spec
        aa = np.degrees(np.arctan2(y, x))
        za = np.degrees(np.arctan2(np.sqrt(x ** 2 + y ** 2), z))
        aa.attrs = azimuth.attrs
        za.attrs = zenith.attrs
        return aa, za

    def _perform_calibration(self, variable, dataset_info):
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

    def _perform_orthorectification(self, variable, orthorect_data_name):
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
            orthorect_data = self[orthorect_data_name].sel(
                {"n_horns": variable.n_horns}
            )
            variable += np.degrees(orthorect_data.values / MEAN_EARTH_RADIUS)
        except KeyError:
            logger.warning('Required dataset %s for orthorectification not available, skipping', orthorect_data_name)  # noqa: E501
        return variable

    def _standardize_dims(self, variable):
        """Standardize dims to y, x."""
        if 'n_scan' in variable.dims and 'n_samples' in variable.dims:
            variable = variable.rename({'n_samples': 'x', 'n_scan': 'y'})
        if variable.dims[0] == 'x':
            variable = variable.transpose('y', 'x')
        return variable

    def _filter_variable(self, variable, dataset_info):
        """Select desired data."""
        for dim in ["n_183", "n_243", "n_325", "n_448", "n_664"]:
            if dim in dataset_info and dim in variable.dims:
                variable = variable.sel({dim: dataset_info[dim]})
                break
        dim = 'n_horns'
        if dim in dataset_info and dim in variable.dims:
            variable = variable.sel({dim: dataset_info[dim]})
        return variable

    def _drop_coords(self, variable, coords):
        if coords in variable.coords:
            variable = variable.drop_vars(coords)
        return variable

    def _fetch_variable(self, var_key):
        """Fetch variable."""
        if var_key in [
            'cached_longitude',
            'cached_latitude',
            'cached_observation_zenith',
            'cached_observation_azimuth',
            'cached_solar_zenith',
            'cached_solar_azimuth',
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
        # Perform the calibration if required
        if dataset_info.get('calibration') is not None:
            variable = self._perform_calibration(variable, dataset_info)
        # Perform the orthorectification if required
        if self.orthorect:
            orthorect_data_name = dataset_info.get('orthorect_data', None)
            if orthorect_data_name is not None:
                variable = self._perform_orthorectification(
                    variable,
                    orthorect_data_name,
                )
        # Manage the attributes of the dataset
        variable.attrs.setdefault('units', None)
        variable.attrs.update(dataset_info)
        variable.attrs.update(self._get_global_attributes())
        variable = self._drop_coords(variable, 'n_horns')
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

        # Add a "quality_group" item to the dictionary with all the variables
        # and attributes which are found in the 'quality' group of the product
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
        attributes['quality_group'] = quality_dict
        return attributes
