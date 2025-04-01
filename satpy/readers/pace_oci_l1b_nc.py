#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Satpy developers
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
"""PACE OCI L1b reader.

This reader supports an optional argument to choose the "engine" for reading
the OCI netCDF4 files. By default, this reader uses the default xarray choice of
engine, as defined in the :func:`xarray.open_dataset` documentation`.

As an alternative, the user may wish to use the "h5netcdf" engine, but that is
not default as it typically prints many non-fatal but confusing error messages
to the terminal.
To choose between engines the user can  do as follows for the default::

    scn = Scene(filenames=my_files, reader="olci_l1b")

or as follows for the h5netcdf engine::

    scn = Scene(filenames=my_files,
                reader="olci_l1b", reader_kwargs={"engine": "h5netcdf"})

References:
    - :func:`xarray.open_dataset`

"""

import logging
from datetime import datetime

import numpy as np
import xarray as xr

from satpy._compat import cached_property
from satpy.readers import open_file_or_filename
from satpy.readers.file_handlers import BaseFileHandler
from satpy.utils import get_legacy_chunk_size

CHUNK_SIZE = get_legacy_chunk_size()

logger = logging.getLogger(__name__)


class NCOCIL1B(BaseFileHandler):
    """The OCI reader base."""

    rows_name = "scans"
    cols_name = "pixels"

    @staticmethod
    def _sort_wvls(inwvls, bw):
        """Compute the bandwidth for each hyperspectral channel."""
        wvls = np.dstack((inwvls, inwvls, inwvls)).squeeze()

        # If we have a float, use that as the bandwidth for all bands.
        if type(bw) is float:
            wvls[:, 0] = wvls[:, 0] - bw
            wvls[:, 2] = wvls[:, 2] + bw
        # But otherwise, use the array as per-band bandwidths.
        else:
            wvls[:, 0] = wvls[:, 0] - bw / 2.
            wvls[:, 2] = wvls[:, 2] + bw / 2.

        return wvls / 1000.

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Init the oci reader base."""
        super().__init__(filename, filename_info, filetype_info)
        self._engine = kwargs.get("engine", None)
        self.sensor = "oci"

        # Get the per-band solar irradiance and central wavelength values, these are split across
        # three variables, one each for the blue, red and SWIR bands.
        self.irradiance = {"blue": self.nc["sensor_band_parameters"]["blue_solar_irradiance"].values,
                           "red": self.nc["sensor_band_parameters"]["red_solar_irradiance"].values,
                           "swir": self.nc["sensor_band_parameters"]["SWIR_solar_irradiance"].values}

        swir_bw = self.nc["sensor_band_parameters"]["SWIR_bandpass"].values

        self.wvls = {"blue": self._sort_wvls(self.nc["sensor_band_parameters"]["blue_wavelength"].values, 0.5),
                     "red": self._sort_wvls(self.nc["sensor_band_parameters"]["red_wavelength"].values, 0.5),
                     "swir": self._sort_wvls(self.nc["sensor_band_parameters"]["SWIR_wavelength"].values, swir_bw)}

        # Spatial resolution is hardcoded, same for all datasets.
        self.resolution = 1200

    @cached_property
    def nc(self):
        """Open the nc xr dataset."""
        f_obj = open_file_or_filename(self.filename)
        dataset = xr.open_datatree(f_obj,
                                   decode_cf=True,
                                   mask_and_scale=True,
                                   engine=self._engine,
                                   chunks={self.cols_name: CHUNK_SIZE,
                                           self.rows_name: CHUNK_SIZE})
        return dataset

    def available_datasets(self, configured_datasets=None):
        """Form the names for the available datasets."""
        for bnd in ["blue", "red", "SWIR"]:
            shper = self.nc["observation_data"][f"rhot_{bnd}"].shape
            for i in range(0, shper[0]):
                yield True, self._retr_dsinfo_chans(bnd, i, {"cal": "refl", "dst": "chan", "vatype": "rhot"})
                yield True, self._retr_dsinfo_chans(bnd, i, {"cal": "radi", "dst": "chan", "vatype": "rhot"})
                yield True, self._retr_dsinfo_chans(bnd, i, {"cal": "", "dst": "qual", "vatype": "qual"})

        for is_avail, ds_info in (configured_datasets or []):
            yield True, ds_info

    def _retr_dsinfo_chans(self, band, i, idic):
        """Retrieve the ds info for a given channel."""
        ds_info = {"file_type": self.filetype_info["file_type"],
                   "resolution": self.resolution,
                   "name": f"{idic['dst']}_{band.lower()}_{self.wvls[band.lower()][i][1] * 1000:4.0f}".replace(" ", ""),
                   "wavelength": [self.wvls[band.lower()][i][0],
                                  self.wvls[band.lower()][i][1],
                                  self.wvls[band.lower()][i][2]],
                   "file_key": f"{idic['vatype']}_{band}",
                   "ds_key": band,
                   "idx": i,
                   "grp_key": "observation_data",
                   "coordinates": ("longitude", "latitude")}

        if idic["cal"] == "refl":
            ds_info["standard_name"] = "toa_bidirectional_reflectance"
            ds_info["units"] = "%"
            ds_info["calibration"] = "reflectance"
        elif idic["cal"] == "radi":
            ds_info["standard_name"] = "toa_outgoing_radiance_per_unit_wavelength"
            ds_info["units"] = "W m-2 sr-1 um-1"
            ds_info["calibration"] = "radiance"
        elif idic["dst"] == "qual":
            ds_info["standard_name"] = "quality_flags"
            ds_info["units"] = "1"
        return ds_info

    def _sort_cal(self, variable, caltype, info):
        """Apply calibration to the data, if needed, and update attrs."""
        if caltype == "reflectance":
            variable = variable * 100.
            variable.attrs["units"] = "%"
            variable.attrs["standard_name"] = "toa_bidirectional_reflectance"
        elif caltype == "radiance":
            sza = self.nc["geolocation_data/solar_zenith"]
            esd = self.nc.attrs["earth_sun_distance_correction"]
            irr = self.irradiance[info["ds_key"].lower()][info["idx"]]
            variable = (variable * irr * np.cos(np.radians(sza))) / (esd * np.pi)
            variable.attrs["units"] = "W m-2 sr-1 um-1"
            variable.attrs["standard_name"] = "toa_outgoing_radiance_per_unit_wavelength"

        return variable

    def get_dataset(self, key, info):
        """Load a dataset."""
        logger.debug("Reading %s.", key["name"])
        if "file_key" not in info:
            variable = self.nc[info["grp_key"]][key["name"]]
        else:
            if "idx" in info:
                variable = self.nc[info["grp_key"]][info["file_key"]][info["idx"]]
            else:
                variable = self.nc[info["grp_key"]][info["file_key"]]

        # Datatree messes with the coordinates, for lats + lons we need to remove the coords.
        if key["name"] in ["longitude", "latitude"]:
            variable = variable.reset_coords(names="longitude", drop=True)
            variable = variable.reset_coords(names="latitude", drop=True)
        variable.attrs["standard_name"] = info["standard_name"]
        if "calibration" in key:
            variable = self._sort_cal(variable, key["calibration"], info)

        return variable.rename({self.cols_name: "x", self.rows_name: "y"})

    @property
    def start_time(self):
        """Start time property."""
        st_dt = self.nc.attrs["time_coverage_start"]
        return datetime.strptime(st_dt, "%Y-%m-%dT%H:%M:%S.%fZ")

    @property
    def end_time(self):
        """End time property."""
        """Start time property."""
        st_dt = self.nc.attrs["time_coverage_end"]
        return datetime.strptime(st_dt, "%Y-%m-%dT%H:%M:%S.%fZ")
