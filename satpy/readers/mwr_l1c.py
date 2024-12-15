# Copyright (c) 2023, 2024 Pytroll Developers

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
"""Reader for the Arctic Weather Satellite (AWS) Sounder level-1c data.

Sample data provided by ESA September 27, 2024.
"""

import logging

import xarray as xr

from .netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

AWS_CHANNEL_NAMES = list(str(i) for i in range(1, 20))


class AWSL1CFile(NetCDF4FileHandler):
    """Class implementing the AWS L1c Filehandler.

    This class implements the ESA Arctic Weather Satellite (AWS) Level-1b
    NetCDF reader. It is designed to be used through the :class:`~satpy.Scene`
    class using the :mod:`~satpy.Scene.load` method with the reader
    ``"aws_l1c_nc"``.

    """

    def __init__(self, filename, filename_info, filetype_info, auto_maskandscale=True):
        """Initialize the handler."""
        super().__init__(filename, filename_info, filetype_info,
                         cache_var_size=10000,
                         cache_handle=True)
        self.filename_info = filename_info

    @property
    def start_time(self):
        """Get the start time."""
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        """Get the end time."""
        return self.filename_info["end_time"]

    @property
    def sensor(self):
        """Get the sensor name."""
        return "MWR"

    @property
    def platform_name(self):
        """Get the platform name."""
        return self.filename_info["platform_name"]

    def get_dataset(self, dataset_id, dataset_info):
        """Get the data."""
        if dataset_id["name"] in AWS_CHANNEL_NAMES:
            data_array = self._get_channel_data(dataset_id, dataset_info)
        elif (dataset_id["name"] in ["longitude", "latitude",
                                     "solar_azimuth", "solar_zenith",
                                     "satellite_zenith", "satellite_azimuth"]):
            data_array = self._get_navigation_data(dataset_id, dataset_info)
        else:
            raise NotImplementedError(f"Dataset {dataset_id['name']} not available or not supported yet!")

        data_array = mask_and_scale(data_array)
        if dataset_id["name"] == "longitude":
            data_array = data_array.where(data_array <= 180, data_array - 360)

        data_array.attrs.update(dataset_info)

        data_array.attrs["platform_name"] = self.platform_name
        data_array.attrs["sensor"] = self.sensor
        return data_array

    def _get_channel_data(self, dataset_id, dataset_info):
        channel_data = self[dataset_info["file_key"]]
        channel_data.coords["n_channels"] = AWS_CHANNEL_NAMES
        channel_data = channel_data.rename({"n_fovs": "x", "n_scans": "y"})
        return channel_data.sel(n_channels=dataset_id["name"]).drop_vars("n_channels")

    def _get_navigation_data(self, dataset_id, dataset_info):
        geo_data = self[dataset_info["file_key"]]
        geo_data = geo_data.rename({"n_fovs": "x", "n_scans": "y"})
        return geo_data


def mask_and_scale(data_array):
    """Mask then scale the data array."""
    if "missing_value" in data_array.attrs:
        with xr.set_options(keep_attrs=True):
            data_array = data_array.where(data_array != data_array.attrs["missing_value"])
        data_array.attrs.pop("missing_value")
    if "valid_max" in data_array.attrs:
        with xr.set_options(keep_attrs=True):
            data_array = data_array.where(data_array <= data_array.attrs["valid_max"])
        data_array.attrs.pop("valid_max")
    if "valid_min" in data_array.attrs:
        with xr.set_options(keep_attrs=True):
            data_array = data_array.where(data_array >= data_array.attrs["valid_min"])
        data_array.attrs.pop("valid_min")
    if "scale_factor" in data_array.attrs and "add_offset" in data_array.attrs:
        with xr.set_options(keep_attrs=True):
            data_array = data_array * data_array.attrs["scale_factor"] + data_array.attrs["add_offset"]
        data_array.attrs.pop("scale_factor")
        data_array.attrs.pop("add_offset")
    return data_array
