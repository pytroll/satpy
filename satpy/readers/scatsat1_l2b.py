# type: ignore
"""ScatSat-1 L2B Reader, distributed by Eumetsat in HDF5 format."""

import datetime as dt

import xarray as xr

from satpy.readers.core.hdf5 import HDF5FileHandler


class SCATSAT1L2BFileHandler(HDF5FileHandler):
    """File handler for ScatSat level 2 files, as distributed by Eumetsat in HDF5 format."""

    @property
    def start_time(self):
        """Time for first observation."""
        return dt.datetime.strptime(self["science_data/attr/Range Beginning Date"],
                                    "%Y-%jT%H:%M:%S.%f")

    @property
    def end_time(self):
        """Time for final observation."""
        return dt.datetime.strptime(self["science_data/attr/Range Ending Date"],
                                    "%Y-%jT%H:%M:%S.%f")

    @property
    def platform_name(self):
        """Get the Platform ShortName."""
        return self["science_data/attr/Satellite Name"]

    def get_dataset(self, ds_id, ds_info):
        """Get output data and metadata of specified dataset."""
        var_path = ds_info["file_key"]
        data = self[var_path]

        data = data.where(data != ds_info.get("fill_value", 65535))
        if "Longitude" in var_path:
            data = data * float(self["science_data/attr/Longitude Scale"])
            data = xr.where(data > 180, data - 360., data)
        elif "Latitude" in var_path:
            data = data * float(self["science_data/attr/Latitude Scale"])
        elif "Wind_speed_selection" in var_path:
            data = data * float(self["science_data/attr/Wind Speed Selection Scale"])
        elif "Wind_direction_selection" in var_path:
            data = data * float(self["science_data/attr/Wind Direction Selection Scale"])

        data.attrs.update({"platform_name": self.platform_name})
        data.attrs.update(ds_info)
        return data
