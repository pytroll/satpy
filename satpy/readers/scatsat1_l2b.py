# type: ignore
"""ScatSat-1 L2B Reader, distributed by Eumetsat in HDF5 format."""

import datetime as dt

import h5py

from satpy.dataset import Dataset
from satpy.readers.core.file_handlers import BaseFileHandler


class SCATSAT1L2BFileHandler(BaseFileHandler):
    """File handler for ScatSat level 2 files, as distributed by Eumetsat in HDF5 format."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the file handler."""
        super(SCATSAT1L2BFileHandler, self).__init__(filename, filename_info, filetype_info)
        self.h5f = h5py.File(self.filename, "r")
        h5data = self.h5f["science_data"]

        self.filename_info["start_time"] = dt.datetime.strptime(
            h5data.attrs["Range Beginning Date"], "%Y-%jT%H:%M:%S.%f")
        self.filename_info["end_time"] = dt.datetime.strptime(
            h5data.attrs["Range Ending Date"], "%Y-%jT%H:%M:%S.%f")

        self.lons = None
        self.lats = None

        self.wind_speed_scale = float(h5data.attrs["Wind Speed Selection Scale"])
        self.wind_direction_scale = float(h5data.attrs["Wind Direction Selection Scale"])
        self.latitude_scale = float(h5data.attrs["Latitude Scale"])
        self.longitude_scale = float(h5data.attrs["Longitude Scale"])

    def get_dataset(self, key, info):
        """Get the dataset."""
        h5data = self.h5f["science_data"]
        stdname = info.get("standard_name")

        if stdname in ["latitude", "longitude"]:

            if self.lons is None or self.lats is None:
                self.lons = h5data["Longitude"][:]*self.longitude_scale
                self.lats = h5data["Latitude"][:]*self.latitude_scale

            if info["standard_name"] == "longitude":
                return Dataset(self.lons, id=key, **info)
            else:
                return Dataset(self.lats, id=key, **info)

        if stdname in ["wind_speed"]:
            windspeed = h5data["Wind_speed_selection"][:, :] * self.wind_speed_scale
            return Dataset(windspeed, id=key, **info)

        if stdname in ["wind_direction"]:
            wind_direction = h5data["Wind_direction_selection"][:, :] * self.wind_direction_scale
            return Dataset(wind_direction, id=key, **info)
