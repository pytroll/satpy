# Copyright (c) 2023 Pytroll Developers

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
"""Reader for the Arctic Weather Satellite (AWS) Sounder level-1b data.

Test data provided by ESA August 23, 2023.
"""

import logging

import xarray as xr

from .netcdf_utils import NetCDF4FileHandler

logger = logging.getLogger(__name__)

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

AWS_CHANNEL_NAMES_TO_NUMBER = {"1": 1, "2": 2, "3": 3, "4": 4,
                               "5": 5, "6": 6, "7": 7, "8": 8,
                               "9": 9, "10": 10, "11": 11, "12": 12,
                               "13": 13, "14": 14, "15": 15, "16": 16,
                               "17": 17, "18": 18, "19": 19}

AWS_CHANNEL_NAMES = list(AWS_CHANNEL_NAMES_TO_NUMBER.keys())


class AWSL1BFile(NetCDF4FileHandler):
    """Class implementing the AWS L1b Filehandler.

    This class implements the ESA Arctic Weather Satellite (AWS) Level-1b
    NetCDF reader. It is designed to be used through the :class:`~satpy.Scene`
    class using the :mod:`~satpy.Scene.load` method with the reader
    ``"aws_l1b_nc"``.

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
        return self["/attr/instrument"]

    @property
    def platform_name(self):
        """Get the platform name."""
        return self.filename_info["platform_name"]

    @property
    def sub_satellite_longitude_start(self):
        """Get the longitude of sub-satellite point at start of the product."""
        return self["status/satellite/subsat_longitude_start"].data.item()

    @property
    def sub_satellite_latitude_start(self):
        """Get the latitude of sub-satellite point at start of the product."""
        return self["status/satellite/subsat_latitude_start"].data.item()

    @property
    def sub_satellite_longitude_end(self):
        """Get the longitude of sub-satellite point at end of the product."""
        return self["status/satellite/subsat_longitude_end"].data.item()

    @property
    def sub_satellite_latitude_end(self):
        """Get the latitude of sub-satellite point at end of the product."""
        return self["status/satellite/subsat_latitude_end"].data.item()

    def get_dataset(self, dataset_id, dataset_info):
        """Get the data."""
        if dataset_id["name"] in AWS_CHANNEL_NAMES:
            data_array = self._get_channel_data(dataset_id, dataset_info)
        elif (dataset_id["name"] in ["longitude", "latitude",
                                     "solar_azimuth", "solar_zenith",
                                     "satellite_zenith", "satellite_azimuth"]):
            data_array = self._get_navigation_data(dataset_id, dataset_info)
        else:
            raise NotImplementedError

        data_array = mask_and_scale(data_array)
        if dataset_id["name"] == "longitude":
            data_array = data_array.where(data_array <= 180, data_array - 360)

        data_array.attrs.update(dataset_info)

        data_array.attrs["orbital_parameters"] = {"sub_satellite_latitude_start": self.sub_satellite_latitude_start,
                                                  "sub_satellite_longitude_start": self.sub_satellite_longitude_start,
                                                  "sub_satellite_latitude_end": self.sub_satellite_latitude_end,
                                                  "sub_satellite_longitude_end": self.sub_satellite_longitude_end}

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
        geo_data.coords["n_geo_groups"] = ["1", "2", "3", "4"]
        geo_data = geo_data.rename({"n_fovs": "x", "n_scans": "y"})
        horn = dataset_id["horn"].name
        return geo_data.sel(n_geo_groups=horn).drop_vars("n_geo_groups")


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

#
#     def _get_quality_attributes(self):
#         """Get quality attributes."""
#         quality_group = self['quality']
#         quality_dict = {}
#         for key in quality_group:
#             # Add the values (as Numpy array) of each variable in the group
#             # where possible
#             try:
#                 quality_dict[key] = quality_group[key].values
#             except ValueError:
#                 quality_dict[key] = None
#
#         quality_dict.update(quality_group.attrs)
#         return quality_dict
