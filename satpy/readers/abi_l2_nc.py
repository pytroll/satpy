# Copyright (c) 2019-2023 Satpy developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Advance Baseline Imager NOAA Level 2+ products reader.

The files read by this reader are described in the official PUG document:
    https://www.goes-r.gov/products/docs/PUG-L2+-vol5.pdf

Data Quality Filtering
^^^^^^^^^^^^^^^^^^^^^^

Some variables can be filtered based on Data Quality Flags (DQF) in the
data files. The flag meanings to retain, where others are marked as invalid,
are specified with the file handler keyword argument ``filters=["good_quality_qf"]``.
These invalid values are marked by NaN in the returned data for floating point arrays.
The values in this list
must match the entries in the "flag_meanings" attribute of the "DQF"
variable in the file. By default no filtering is applied.

"""

import logging
import operator
from functools import reduce

import numpy as np
import xarray as xr

from satpy.readers.core.abi import NC_ABI_BASE

LOG = logging.getLogger(__name__)


class NC_ABI_L2(NC_ABI_BASE):
    """Reader class for NOAA ABI l2+ products in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info,
                 filters: list[str] | None = None, **kwargs):
        """Initialize file handler and store filter_sst state."""
        super().__init__(filename, filename_info, filetype_info, **kwargs)
        self.filters = filters or []

    def get_dataset(self, key, info):
        """Load a dataset."""
        var = info["file_key"]
        if self.filetype_info["file_type"] == "abi_l2_mcmip":
            var += "_" + key["name"]
        LOG.debug("Reading in get_dataset %s.", var)
        variable = self[var]
        variable.attrs.update(key.to_dict())
        self._update_data_arr_with_filename_attrs(variable)
        self._remove_problem_attrs(variable)
        variable = self._filter_dqf(variable)

        # convert to satpy standard units
        if variable.attrs["units"] == "1" and key.get("calibration") == "reflectance":
            variable *= 100.0
            variable.attrs["units"] = "%"

        return variable

    def _update_data_arr_with_filename_attrs(self, variable):
        _units = variable.attrs["units"] if "units" in variable.attrs else None
        variable.attrs.update({
            "platform_name": self.platform_name,
            "sensor": self.sensor,
            "units": _units,
            "orbital_parameters": {
                "satellite_nominal_latitude": float(self.nc["nominal_satellite_subpoint_lat"]),
                "satellite_nominal_longitude": float(self.nc["nominal_satellite_subpoint_lon"]),
                "satellite_nominal_altitude": float(self.nc["nominal_satellite_height"]) * 1000.,
            },
        })
        self._convert_flag_attrs(variable)

        # add in information from the filename that may be useful to the user
        for attr in ("scene_abbr", "scan_mode", "platform_shortname"):
            variable.attrs[attr] = self.filename_info.get(attr)

        # add in information hardcoded in the filetype YAML
        for attr in ("observation_type",):
            if attr in self.filetype_info:
                variable.attrs[attr] = self.filetype_info[attr]

        # copy global attributes to metadata
        for attr in ("scene_id", "orbital_slot", "instrument_ID", "production_site", "timeline_ID"):
            variable.attrs[attr] = self.nc.attrs.get(attr)

    @staticmethod
    def _convert_flag_attrs(variable: xr.DataArray) -> None:
        if "flag_meanings" in variable.attrs:
            variable.attrs["flag_meanings"] = variable.attrs["flag_meanings"].split(" ")
        if "flag_values" in variable.attrs:
            variable.attrs["flag_values"] = [int(val) for val in variable.attrs["flag_values"]]

    @staticmethod
    def _remove_problem_attrs(variable):
        # remove attributes that could be confusing later
        if not np.issubdtype(variable.dtype, np.integer):
            # integer fields keep the _FillValue
            variable.attrs.pop("_FillValue", None)
        variable.attrs.pop("scale_factor", None)
        variable.attrs.pop("add_offset", None)
        variable.attrs.pop("valid_range", None)
        variable.attrs.pop("_Unsigned", None)
        variable.attrs.pop("valid_range", None)
        variable.attrs.pop("ancillary_variables", None)  # Can't currently load DQF

    def _filter_dqf(self, variable: xr.DataArray) -> xr.DataArray:
        if "DQF" not in self:
            return variable
        dqf = self["DQF"]
        if "flag_meanings" not in dqf.attrs or "flag_values" not in dqf.attrs:
            return variable

        self._convert_flag_attrs(dqf)
        flag_dict = dict(zip(dqf.attrs["flag_meanings"], dqf.attrs["flag_values"], strict=True))
        values_to_keep = [flag_dict[flag_name] for flag_name in self.filters if flag_name in flag_dict]
        if not values_to_keep:
            # user didn't specify any filters for this variable
            return variable

        LOG.debug(f"Filtering {variable.attrs['name']} for DQF flags {values_to_keep!r}")
        good_mask = reduce(operator.or_, [dqf == flag_value for flag_value in values_to_keep])
        return variable.where(good_mask)

    def available_datasets(self, configured_datasets=None):
        """Add resolution to configured datasets."""
        for is_avail, ds_info in (configured_datasets or []):
            # some other file handler knows how to load this
            # don't override what they've done
            if is_avail is not None:
                yield is_avail, ds_info
            matches = self.file_type_matches(ds_info["file_type"])
            if matches:
                # we have this dataset
                resolution = self.spatial_resolution_to_number()
                new_info = ds_info.copy()
                new_info.setdefault("resolution", resolution)
                yield True, ds_info
            elif is_avail is None:
                # we don't know what to do with this
                # see if another future file handler does
                yield is_avail, ds_info
