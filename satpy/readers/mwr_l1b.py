# Copyright (c) 2023 - 2025 Pytroll Developers

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
"""Reader for the level-1b data from the MWR sounder onboard AWS and EPS-STerna.

AWS = Arctic Weather Satellite. MWR = Microwave Radiometer.

AWS test data provided by ESA August 23, 2023.

Sample data for five orbits in September 2024 provided by ESA to the Science
Advisory Group for MWS and AWS, November 26, 2024.

Sample EPS-Sterna l1b format AWS data from 16 orbits the 9th of November 2024.

Continous feed (though restricted to the SAG members and selected European
users/evaluators) in the EUMETSAT Data Store of global AWS data from January
9th, 2025.

Example:
--------
Here is an example how to read the data in satpy:

.. code-block:: python

    from satpy import Scene
    from glob import glob

    filenames = glob("data/W_NO-KSAT-Tromso,SAT,AWS1-MWR-1B-RAD_C_OHB__*_G_O_20250110114708*.nc"
    scn = Scene(filenames=filenames, reader='aws1_mwr_l1b_nc')

    composites = ['mw183_humidity']
    dataset_names = composites + ['1']

    scn.load(dataset_names)
    print(scn['1'])
    scn.show('mw183_humidity')


As the file format for the EPS Sterna Level-1b is slightly different from the
ESA format, reading the EPS Sterna level-1b data uses a different reader, named
`eps_sterna_mwr_l1b_nc`. So, if specifying the reader name as in the above code
example, please provide the actual name for that data: eps_sterna_mwr_l1b_nc.


"""

import xarray as xr

from .netcdf_utils import NetCDF4FileHandler

MWR_CHANNEL_NAMES = [str(i) for i in range(1, 20)]

NAVIGATION_DATASET_NAMES = ["satellite_zenith_horn1",
                            "satellite_zenith_horn2",
                            "satellite_zenith_horn3",
                            "satellite_zenith_horn4",
                            "solar_azimuth_horn1",
                            "solar_azimuth_horn2",
                            "solar_azimuth_horn3",
                            "solar_azimuth_horn4",
                            "solar_zenith_horn1",
                            "solar_zenith_horn2",
                            "solar_zenith_horn3",
                            "solar_zenith_horn4",
                            "satellite_azimuth_horn1",
                            "satellite_azimuth_horn2",
                            "satellite_azimuth_horn3",
                            "satellite_azimuth_horn4",
                            "longitude",
                            "latitude"]

class AWS_EPS_Sterna_BaseFileHandler(NetCDF4FileHandler):
    """Base class implementing the AWS/EPS-Sterna MWR Level-1b&c Filehandlers."""

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
        # This should have been self["/attr/instrument"]
        # But the sensor name is currently incorrect in the ESA level-1b files
        return "mwr"

    @property
    def platform_name(self):
        """Get the platform name."""
        return self.filename_info["platform_name"]

    @property
    def orbit_start(self):
        """Get the orbit number for the start of data."""
        return int(self["/attr/orbit_start"])

    @property
    def orbit_end(self):
        """Get the orbit number for the end of data."""
        return int(self["/attr/orbit_end"])

    def get_dataset(self, dataset_id, dataset_info):
        """Get the data."""
        raise NotImplementedError("This is not implemented in the Base class.")

    def _get_channel_data(self, dataset_id, dataset_info):
        channel_data = self[dataset_info["file_key"]]
        channel_data.coords["n_channels"] = MWR_CHANNEL_NAMES
        channel_data = channel_data.rename({"n_fovs": "x", "n_scans": "y"})
        return channel_data.sel(n_channels=dataset_id["name"]).drop_vars("n_channels")



class AWS_EPS_Sterna_MWR_L1BFile(AWS_EPS_Sterna_BaseFileHandler):
    """Class implementing the AWS/EPS-Sterna MWR L1b Filehandler."""

    def __init__(self, filename, filename_info, filetype_info, auto_maskandscale=True):
        """Initialize the handler."""
        super().__init__(filename, filename_info, filetype_info, auto_maskandscale)
        self._feed_horn_group_name = filetype_info.get("feed_horn_group_name")

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
        if dataset_id["name"] in MWR_CHANNEL_NAMES:
            data_array = self._get_channel_data(dataset_id, dataset_info)
        elif dataset_id["name"] in NAVIGATION_DATASET_NAMES:
            data_array = self._get_navigation_data(dataset_id, dataset_info)
        else:
            raise NotImplementedError(f"Dataset {dataset_id['name']} not available or not supported yet!")

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
        data_array.attrs["orbit_number"] = self.orbit_start
        return data_array

    def _get_navigation_data(self, dataset_id, dataset_info):
        """Get the navigation (geolocation) data for one feed horn."""
        geo_data = self[dataset_info["file_key"]]
        geo_data.coords[self._feed_horn_group_name] = ["1", "2", "3", "4"]
        geo_data = geo_data.rename({"n_fovs": "x", "n_scans": "y"})
        horn = dataset_id["horn"].name
        _selection = {self._feed_horn_group_name: horn}
        return geo_data.sel(_selection).drop_vars(self._feed_horn_group_name)


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
