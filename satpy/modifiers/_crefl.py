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

from satpy.aux_download import DataDownloadMixin, retrieve
from satpy.modifiers import ModifierBase
from satpy.modifiers.angles import get_angles

LOG = logging.getLogger(__name__)


class ReflectanceCorrector(ModifierBase, DataDownloadMixin):
    """Corrected Reflectance (crefl) modifier.

    Uses a python rewrite of the C CREFL code written for VIIRS and MODIS.
    """

    def __init__(self, *args, dem_filename=None, dem_sds="averaged elevation",
                 url=None, known_hash=None, **kwargs):
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
        if dem_filename is not None:
            warnings.warn("'dem_filename' for 'ReflectanceCorrector' is "
                          "deprecated. Use 'url' instead.", DeprecationWarning)

        super(ReflectanceCorrector, self).__init__(*args, **kwargs)
        self.dem_sds = dem_sds
        self.url = url
        self.known_hash = known_hash
        self.dem_cache_key = self._get_registered_dem_cache_key()

    def _get_registered_dem_cache_key(self):
        if not self.url:
            return
        reg_files = self.register_data_files([{
            'url': self.url, 'known_hash': self.known_hash}
        ])
        return reg_files[0]

    def __call__(self, datasets, optional_datasets, **info):
        """Create modified DataArray object by applying the crefl algorithm."""
        refl_data, angles = self._extract_angle_data_arrays(datasets, optional_datasets)
        results = self._call_crefl(refl_data, angles)
        info.update(refl_data.attrs)
        info["rayleigh_corrected"] = True
        results.attrs = info
        self.apply_modifier_info(refl_data, results)
        return results

    def _call_crefl(self, refl_data, angles):
        from satpy.modifiers._crefl_utils import run_crefl
        avg_elevation = self._get_average_elevation()
        results = run_crefl(refl_data,
                            *angles,
                            avg_elevation=avg_elevation,
                            )
        return results

    def _get_average_elevation(self):
        if self.dem_cache_key is None:
            return

        LOG.debug("Loading CREFL averaged elevation information from: %s",
                  self.dem_cache_key)
        local_filename = retrieve(self.dem_cache_key)
        avg_elevation = self._read_var_from_hdf4_file(local_filename, self.dem_sds).astype(np.float64)
        if isinstance(avg_elevation, np.ma.MaskedArray):
            avg_elevation = avg_elevation.filled(np.nan)
        return avg_elevation

    @staticmethod
    def _read_var_from_hdf4_file(local_filename, var_name):
        try:
            return ReflectanceCorrector._read_var_from_hdf4_file_pyhdf(local_filename, var_name)
        except (ImportError, OSError):
            return ReflectanceCorrector._read_var_from_hdf4_file_netcdf4(local_filename, var_name)

    @staticmethod
    def _read_var_from_hdf4_file_netcdf4(local_filename, var_name):
        from netCDF4 import Dataset as NCDataset

        # HDF4 file, NetCDF library needs to be compiled with HDF4 support
        nc = NCDataset(local_filename, "r")
        # average elevation is stored as a 16-bit signed integer but with
        # scale factor 1 and offset 0, convert it to float here
        return nc.variables[var_name][:]

    @staticmethod
    def _read_var_from_hdf4_file_pyhdf(local_filename, var_name):
        from pyhdf.SD import SD, SDC
        f = SD(local_filename, SDC.READ)
        var = f.select(var_name)
        data = var[:]
        fill = ReflectanceCorrector._read_fill_value_from_hdf4(var, data.dtype)
        return np.ma.MaskedArray(data, data == fill)

    @staticmethod
    def _read_fill_value_from_hdf4(var, dtype):
        from pyhdf.error import HDF4Error
        try:
            return var.getfillvalue()
        except HDF4Error:
            return np.iinfo(dtype).min

    def _extract_angle_data_arrays(self, datasets, optional_datasets):
        all_datasets = datasets + optional_datasets
        if len(all_datasets) == 1:
            vis = self.match_data_arrays(datasets)[0]
            return vis, get_angles(vis)
        if len(all_datasets) == 5:
            vis, *angles = self.match_data_arrays(
                datasets + optional_datasets)
            return vis, angles
        raise ValueError("Not sure how to handle provided dependencies. "
                         "Either all 4 angles must be provided or none of "
                         "of them.")
