"""Reader for ISCCP-NG L1G data (https://cimss.ssec.wisc.edu/isccp-ng/)."""

import logging

import numpy as np
import xarray as xr
from pyresample import geometry

from satpy.readers.core.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)


class IsccpngL1gFileHandler(BaseFileHandler):
    """Reader L1G ISCCP-NG data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(IsccpngL1gFileHandler, self).__init__(
            filename, filename_info, filetype_info)

        self._start_time = filename_info["start_time"]
        self._end_time = None
        self.sensor = "multiple_sensors"
        self.filename_info = filename_info

    def tile_geolocation(self, data, key):
        """Get geolocation on full swath."""
        if key in "latitude":
            return xr.DataArray(np.tile(data.values[:, np.newaxis], (1, 7200)), dims=["y", "x"], attrs=data.attrs)
        if key in "longitude":
            return xr.DataArray(np.tile(data.values, (3600, 1)), dims=["y", "x"], attrs=data.attrs)
        return data

    def get_best_layer_of_data(self, data):
        """Get the layer with best data (= layer 0). There are two more layers with additional data."""
        if len(data.dims) == 4:
            data = data[0, 0, :, :]
        return data.squeeze(drop=True)

    def get_area_def(self, dsid):
        """Get area definition."""
        proj_dict = {
            "proj": "latlong",
            "datum": "WGS84",
        }
        area = geometry.AreaDefinition(
            "lat lon grid",
            "name_of_proj",
            "id_of_proj",
            proj_dict,
            7200,
            3600,
            np.asarray([-180, -90, 180, 90])
        )
        return area

    def modify_dims_and_coords(self, data):
        """Remove coords and rename dims to x and y."""
        if len(data.dims) > 2:
            data = data.drop_vars("latitude")
            data = data.drop_vars("longitude")
            data = data.drop_vars("start_time")
            data = data.drop_vars("end_time")
            data = data.rename({"longitude": "x", "latitude": "y"})
        return data

    def set_time_attrs(self, data):
        """Set time from attributes."""
        if "start_time" in data.coords:
            data.attrs["start_time"] = data["start_time"].values[0]
            data.attrs["end_time"] = data["end_time"].values[0]
            self._end_time = data.attrs["end_time"]
            self._start_time = data.attrs["start_time"]

    def get_dataset(self, key, yaml_info):
        """Get dataset."""
        logger.debug("Getting data for: %s", yaml_info["name"])
        nc = xr.open_dataset(self.filename, chunks={"y": "auto", "x": 900})
        name = yaml_info.get("nc_store_name", yaml_info["name"])
        file_key = yaml_info.get("nc_key", name)
        data = nc[file_key]
        self.set_time_attrs(data)
        data = self.modify_dims_and_coords(data)
        data = self.get_best_layer_of_data(data)
        data = self.tile_geolocation(data, file_key)
        return data

    @property
    def start_time(self):
        """Get the start time."""
        return self._start_time

    @property
    def end_time(self):
        """Get the end time."""
        return self._end_time
