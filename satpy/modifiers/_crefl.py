#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Classes related to the CREFL (corrected reflectance) modifier."""

import logging
import warnings

import numpy as np
import xarray as xr
from dask import array as da
from satpy.aux_download import DataDownloadMixin, retrieve
from satpy.modifiers import ModifierBase
from satpy.utils import get_satpos

LOG = logging.getLogger(__name__)


class ReflectanceCorrector(ModifierBase, DataDownloadMixin):
    """Corrected Reflectance (crefl) modifier.

    Uses a python rewrite of the C CREFL code written for VIIRS and MODIS.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the compositor with values from the user or from the configuration file.

        If `dem_filename` can't be found or opened then correction is done
        assuming TOA or sealevel options.

        Args:
            dem_filename (str): DEPRECATED
            url (str): URL or local path to the Digital Elevation Model (DEM)
                HDF4 file. If unset (None or empty string), then elevation
                is assumed to be 0 everywhere.
            known_hash (str): Optional SHA256 checksum to verify the download
                of ``url``.
            dem_sds (str): Name of the variable in the elevation file to load.

        """
        dem_filename = kwargs.pop("dem_filename", None)
        if dem_filename is not None:
            warnings.warn("'dem_filename' for 'ReflectanceCorrector' is "
                          "deprecated. Use 'url' instead.", DeprecationWarning)

        self.dem_sds = kwargs.pop("dem_sds", "averaged elevation")
        self.url = kwargs.pop('url', None)
        self.known_hash = kwargs.pop('known_hash', None)
        super(ReflectanceCorrector, self).__init__(*args, **kwargs)
        self.dem_cache_key = None
        if self.url:
            reg_files = self.register_data_files([{
                'url': self.url, 'known_hash': self.known_hash}
            ])
            self.dem_cache_key = reg_files[0]

    def __call__(self, datasets, optional_datasets, **info):
        """Create modified DataArray object by applying the crefl algorithm."""
        from satpy.composites.crefl_utils import run_crefl, get_coefficients

        datasets = self._get_data_and_angles(datasets, optional_datasets)
        refl_data, sensor_aa, sensor_za, solar_aa, solar_za = datasets
        avg_elevation = self._get_average_elevation()
        is_percent = refl_data.attrs["units"] == "%"
        coefficients = get_coefficients(refl_data.attrs["sensor"],
                                        refl_data.attrs["wavelength"],
                                        refl_data.attrs["resolution"])
        use_abi = refl_data.attrs['sensor'] == 'abi'
        lons, lats = refl_data.attrs['area'].get_lonlats(chunks=refl_data.chunks)
        results = run_crefl(refl_data,
                            coefficients,
                            lons,
                            lats,
                            sensor_aa,
                            sensor_za,
                            solar_aa,
                            solar_za,
                            avg_elevation=avg_elevation,
                            percent=is_percent,
                            use_abi=use_abi)
        info.update(refl_data.attrs)
        info["rayleigh_corrected"] = True
        factor = 100. if is_percent else 1.
        results = results * factor
        results.attrs = info
        self.apply_modifier_info(refl_data, results)
        return results

    def _get_average_elevation(self):
        if self.dem_cache_key is None:
            return

        LOG.debug("Loading CREFL averaged elevation information from: %s",
                  self.dem_cache_key)
        local_filename = retrieve(self.dem_cache_key)
        from netCDF4 import Dataset as NCDataset
        # HDF4 file, NetCDF library needs to be compiled with HDF4 support
        nc = NCDataset(local_filename, "r")
        # average elevation is stored as a 16-bit signed integer but with
        # scale factor 1 and offset 0, convert it to float here
        avg_elevation = nc.variables[self.dem_sds][:].astype(np.float64)
        if isinstance(avg_elevation, np.ma.MaskedArray):
            avg_elevation = avg_elevation.filled(np.nan)
        return avg_elevation

    def _get_data_and_angles(self, datasets, optional_datasets):
        all_datasets = datasets + optional_datasets
        if len(all_datasets) == 1:
            vis = self.match_data_arrays(datasets)[0]
            sensor_aa, sensor_za, solar_aa, solar_za = self.get_angles(vis)
        elif len(all_datasets) == 5:
            vis, sensor_aa, sensor_za, solar_aa, solar_za = self.match_data_arrays(
                datasets + optional_datasets)
            # get the dask array underneath
            sensor_aa = sensor_aa.data
            sensor_za = sensor_za.data
            solar_aa = solar_aa.data
            solar_za = solar_za.data
        else:
            raise ValueError("Not sure how to handle provided dependencies. "
                             "Either all 4 angles must be provided or none of "
                             "of them.")

        # angles must be xarrays
        sensor_aa = xr.DataArray(sensor_aa, dims=['y', 'x'])
        sensor_za = xr.DataArray(sensor_za, dims=['y', 'x'])
        solar_aa = xr.DataArray(solar_aa, dims=['y', 'x'])
        solar_za = xr.DataArray(solar_za, dims=['y', 'x'])
        refl_data = datasets[0]
        return refl_data, sensor_aa, sensor_za, solar_aa, solar_za

    def get_angles(self, vis):
        """Get sun and satellite angles to use in crefl calculations."""
        from pyorbital.astronomy import get_alt_az, sun_zenith_angle
        from pyorbital.orbital import get_observer_look
        lons, lats = vis.attrs['area'].get_lonlats(chunks=vis.data.chunks)
        lons = da.where(lons >= 1e30, np.nan, lons)
        lats = da.where(lats >= 1e30, np.nan, lats)
        suna = get_alt_az(vis.attrs['start_time'], lons, lats)[1]
        suna = np.rad2deg(suna)
        sunz = sun_zenith_angle(vis.attrs['start_time'], lons, lats)
        sat_lon, sat_lat, sat_alt = get_satpos(vis)
        sata, satel = get_observer_look(
            sat_lon,
            sat_lat,
            sat_alt / 1000.0,  # km
            vis.attrs['start_time'],
            lons, lats, 0)
        satz = 90 - satel
        return sata, satz, suna, sunz
