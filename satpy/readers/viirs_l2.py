import logging
from datetime import datetime

import numpy as np

from satpy.readers.netcdf_utils import NetCDF4FileHandler
import xarray as xr

LOG = logging.getLogger(__name__)


class VIIRSL2FileHandler(NetCDF4FileHandler):
    def _parse_datetime(self, datestr):
        """Parse datetime."""
        return datetime.strptime(datestr, "%Y-%m-%dT%H:%M:%S.000Z")

    @property
    def start_time(self):
        """Get start time."""
        return self._parse_datetime(self["/attr/time_coverage_start"])

    @property
    def end_time(self):
        """Get end time."""
        return self._parse_datetime(self["/attr/time_coverage_end"])

    @property
    def start_orbit_number(self):
        """Get start orbit number."""
        try:
            return int(self["/attr/orbit_number"])
        except KeyError:
            return int(self["/attr/OrbitNumber"])

    @property
    def end_orbit_number(self):
        """Get end orbit number."""
        try:
            return int(self["/attr/orbit_number"])
        except KeyError:
            return int(self["/attr/OrbitNumber"])

    @property
    def platform_name(self):
        """Get platform name."""
        try:
            res = self.get("/attr/platform", self.filename_info["platform_shortname"])
        except KeyError:
            res = "Unknown"

        return {
            "JPSS-1": "NOAA-20",
            "NP": "Suomi-NPP",
            "J1": "NOAA-20",
            "J2": "NOAA-21",
            "JPSS-2": "NOAA-21",
        }.get(res, res)

    @property
    def sensor_name(self):
        """Get sensor name."""
        return self["/attr/instrument"].lower()

    def _get_dataset_file_units(self, dataset_id, ds_info, var_path):
        file_units = ds_info.get("units")
        if file_units is None:
            file_units = self.get(var_path + "/attr/units")
        if file_units == "none" or file_units == "None":
            file_units = "1"
        return file_units

    def _get_dataset_valid_range(self, dataset_id, ds_info, var_path):
        valid_min = self.get(var_path + "/attr/valid_min")
        valid_max = self.get(var_path + "/attr/valid_max")
        if not valid_min and not valid_max:
            valid_range = self.get(var_path + "/attr/valid_range")
            if valid_range is not None:
                valid_min = valid_range[0]
                valid_max = valid_range[1]
        scale_factor = self.get(var_path + "/attr/scale_factor")
        scale_offset = self.get(var_path + "/attr/add_offset")
        return valid_min, valid_max, scale_factor, scale_offset

    def get_metadata(self, dataset_id, ds_info):
        """Get metadata."""
        var_path = ds_info.get("file_key", ds_info["name"])
        file_units = self._get_dataset_file_units(dataset_id, ds_info, var_path)

        # Get extra metadata
        i = getattr(self[var_path], "attrs", {})
        i.update(ds_info)
        i.update(dataset_id.to_dict())
        i.update(
            {
                "file_units": file_units,
                "platform_name": self.platform_name,
                "sensor": self.sensor_name,
                "start_orbit": self.start_orbit_number,
                "end_orbit": self.end_orbit_number,
            }
        )
        i.update(dataset_id.to_dict())
        return i

    def adjust_scaling_factors(self, factors, file_units, output_units):
        """Adjust scaling factors."""
        if factors is None or factors[0] is None:
            factors = [1, 0]
        if file_units == output_units:
            LOG.debug("File units and output units are the same (%s)", file_units)
            return factors
        factors = np.array(factors)

        if file_units == "1" and output_units == "%":
            LOG.debug(
                "Adjusting scaling factors to convert '%s' to '%s'",
                file_units,
                output_units,
            )
            factors[::2] = np.where(factors[::2] != -999, factors[::2] * 100.0, -999)
            factors[1::2] = np.where(factors[1::2] != -999, factors[1::2] * 100.0, -999)
            return factors
        else:
            return factors

    def available_datasets(self, configured_datasets=None):
        """Generate dataset info and their availablity.

        See
        :meth:`satpy.readers.file_handlers.BaseFileHandler.available_datasets`
        for details.

        """
        for is_avail, ds_info in configured_datasets or []:
            if is_avail is not None:
                yield is_avail, ds_info
                continue
            ft_matches = self.file_type_matches(ds_info["file_type"])
            var_path = ds_info.get("file_key", ds_info["name"])
            is_in_file = var_path in self
            yield ft_matches and is_in_file, ds_info

    def get_dataset(self, ds_id: int, ds_info: str) -> xr.DataArray:
        var_path = ds_info.get("file_key", ds_info["name"])
        metadata = self.get_metadata(ds_id, ds_info)
        (
            valid_min,
            valid_max,
            scale_factor,
            scale_offset,
        ) = self._get_dataset_valid_range(ds_id, ds_info, var_path)
        data = self[var_path]
        data.attrs.update(metadata)
        if valid_min is not None and valid_max is not None:
            data = data.where((data >= valid_min) & (data <= valid_max))
        factors = (scale_factor, scale_offset)
        factors = self.adjust_scaling_factors(
            factors, metadata["file_units"], ds_info.get("units")
        )
        if factors[0] != 1 or factors[1] != 0:
            data *= factors[0]
            data += factors[1]
        # rename dimensions to correspond to satpy's 'y' and 'x' standard
        if "number_of_lines" in data.dims:
            data = data.rename({"number_of_lines": "y", "number_of_pixels": "x"})
        return data
