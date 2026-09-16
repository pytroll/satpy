"""Advanced Geostationary Radiation Imager reader for the Level_1 HDF format.

The files read by this reader are described in the official Real Time Data Service:

    http://fy4.nsmc.org.cn/data/en/data/realtime.html

"""

import logging

from satpy.readers.core.fy4 import FY4Base

logger = logging.getLogger(__name__)


class HDF_AGRI_L1(FY4Base):
    """AGRI l1 file handler."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init filehandler."""
        super(HDF_AGRI_L1, self).__init__(filename, filename_info, filetype_info)
        self.sensor = "AGRI"

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        ds_name = dataset_id["name"]
        logger.debug("Reading in get_dataset %s.", ds_name)
        file_key = ds_info.get("file_key", ds_name)
        if self.PLATFORM_ID == "FY-4B":
            if self.CHANS_ID in file_key:
                file_key = f"Data/{file_key}"
            elif self.SUN_ID in file_key or self.SAT_ID in file_key:
                file_key = f"Navigation/{file_key}"
        data = self.get(file_key)
        if data.ndim >= 2:
            data = data.rename({data.dims[-2]: "y", data.dims[-1]: "x"})
        data = self.calibrate(data, ds_info, ds_name, file_key)

        self.adjust_attrs(data, ds_info)

        return data

    def adjust_attrs(self, data, ds_info):
        """Adjust the attrs of the data."""
        satname = self.PLATFORM_NAMES.get(self["/attr/Satellite Name"], self["/attr/Satellite Name"])
        data.attrs.update({"platform_name": satname,
                           "sensor": self["/attr/Sensor Identification Code"].lower(),
                           "orbital_parameters": {
                               "satellite_nominal_latitude": self["/attr/NOMCenterLat"].item(),
                               "satellite_nominal_longitude": self["/attr/NOMCenterLon"].item(),
                               "satellite_nominal_altitude": self["/attr/NOMSatHeight"].item()}})
        data.attrs.update(ds_info)
        # remove attributes that could be confusing later
        data.attrs.pop("FillValue", None)
        data.attrs.pop("Intercept", None)
        data.attrs.pop("Slope", None)
