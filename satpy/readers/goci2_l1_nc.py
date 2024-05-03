#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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
"""Reader for GK-2B GOCI-II L1 products from NOSC.

For more information about the data, see: <https://www.nosc.go.kr/eng/boardContents/actionBoardContentsCons0028.do>

The L1 data products from NOSC do not contain solar irradiance factors, which are necessary to transform
radiance to reflectance. The reader hardcodes these values based on calculation from `pyspectral`:
```
from pyspectral.utils import convert2wavenumber, get_central_wave
from pyspectral.rsr_reader import RelativeSpectralResponse
from pyspectral.solar import SolarIrradianceSpectrum

goci2 = RelativeSpectralResponse("GK-2B", "goci2")
rsr, info = convert2wavenumber(goci2.rsr)
solar_irr = SolarIrradianceSpectrum(dlambda=0.0005, wavespace="wavenumber")
for band in goci2.band_names:
    print(f"Solar Irradiance (GOCI2 band {band}) = {solar_irr.inband_solarirradiance(rsr[band]):12.6f}")
```

"""

import logging
from datetime import datetime

import xarray as xr

from satpy.readers.netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)

GOCI2_SOLAR_IRRAD = {"L380": 15.389094,
                     "L412": 29.085551,
                     "L443": 37.377115,
                     "L490": 46.492590,
                     "L510": 48.745104,
                     "L555": 57.122588,
                     "L620": 65.075606,
                     "L660": 67.138780,
                     "L680": 68.981090,
                     "L709": 69.870426,
                     "L745": 70.849054,
                     "L865": 72.470692
                     }

GOCI2_SOLAR_IRRAD = {"L380": 1061.509391,
                     "L412": 1710.525606,
                     "L443": 1899.063039,
                     "L490": 1931.795505,
                     "L510": 1871.312024,
                     "L555": 1853.857524,
                     "L620": 1693.385039,
                     "L660": 1541.451305,
                     "L680": 1491.561883,
                     "L709": 1389.725683,
                     "L745": 1274.906171,
                     "L865": 971.089088}

class GOCI2L1NCFileHandler(NetCDF4FileHandler):
    """File handler for GOCI-II L1 official data in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info, mask_zeros=True):
        """Initialize the reader."""
        super().__init__(filename, filename_info, filetype_info, auto_maskandscale=True)

        # By default, we mask nodata areas (zero values) near the edges of the extent
        self.mask_zeros = mask_zeros

        self.attrs = self["/attrs"]
        self.nc = self._merge_navigation_data(filetype_info["file_type"])

        # Read metadata which are common to all datasets
        self.nlines = self.nc.sizes["number_of_lines"]
        self.ncols = self.nc.sizes["number_of_columns"]

    def _merge_navigation_data(self, filetype):
        """Merge navigation data and geophysical data."""
        groups = ["geophysical_data", "navigation_data"]
        return xr.merge([self[group] for group in groups])

    @property
    def start_time(self):
        """Start timestamp of the dataset."""
        dt = self.attrs["observation_start_time"]
        return datetime.strptime(dt, "%Y%m%d_%H%M%S")

    @property
    def end_time(self):
        """End timestamp of the dataset."""
        dt = self.attrs["observation_end_time"]
        return datetime.strptime(dt, "%Y%m%d_%H%M%S")


    def _calibrate(self, data, bname):
        """Convert raw radiances into reflectance."""
        import numpy as np
        from pyorbital.astronomy import sun_earth_distance_correction

        esd = sun_earth_distance_correction(self.start_time)

        factor = np.pi * esd * esd / GOCI2_SOLAR_IRRAD[bname]

        res = data * np.float32(factor)

        # Convert from 0-1 range to 0-100
        res = 100 * res

        res.attrs = data.attrs

        res.attrs["units"] = "1"
        res.attrs["long_name"] = "Bidirectional Reflectance"
        res.attrs["standard_name"] = "toa_bidirectional_reflectance"

        return res

    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info["file_key"]
        logger.debug("Reading in get_dataset %s.", var)
        variable = self.nc[var]

        variable = variable.rename({"number_of_lines": "y", "number_of_columns": "x"})

        # Some products may miss lon/lat standard_name, use name as base name if it is not already present
        if variable.attrs.get("standard_name", None) is None:
            variable.attrs.update({"standard_name": variable.name})
        variable.attrs.update({"platform_name": self.attrs["platform"],
                               "sensor": "goci2"})

        # The data lists "0" as the valid minimum, but this is also used for fill values
        # at the edge of the image extent. If required, filter these.
        if self.mask_zeros:
            variable = variable.where(variable != 0)

        # If required, convert raw radiances to reflectance
        if "calibration" in key:
            if key["calibration"] == "reflectance":
                variable = self._calibrate(variable, info["name"])
            elif key["calibration"] != "radiance":
                raise ValueError(f"Calibration type {key["calibration"]} not supported.")

        variable.attrs.update(key.to_dict())

        variable.attrs["orbital_parameters"] = {
            "satellite_nominal_longitude": self.attrs["sub_longitude"],
            "satellite_nominal_latitude": 0.,
            "projection_longitude": self.attrs["longitude_of_projection_origin"],
            "projection_latitude": self.attrs["latitude_of_projection_origin"],
            "projection_altitude": self.attrs["perspective_point_height"]
        }
        return variable
