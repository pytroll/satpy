
"""Reader for GPM imerg data on half-hourly timesteps.

References:
   - The NASA IMERG ATBD:
     https://pmm.nasa.gov/sites/default/files/document_files/IMERG_ATBD_V06.pdf

"""

import datetime as dt
import logging

import dask.array as da
import h5py
import numpy as np
from pyresample.geometry import AreaDefinition

from satpy.readers.core.hdf5 import HDF5FileHandler

logger = logging.getLogger(__name__)


class Hdf5IMERG(HDF5FileHandler):
    """IMERG hdf5 reader."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init method."""
        super(Hdf5IMERG, self).__init__(filename, filename_info,
                                        filetype_info)
        self.finfo = filename_info
        self.cache = {}

    @property
    def start_time(self):
        """Find the start time from filename info."""
        return dt.datetime(self.finfo["date"].year,
                           self.finfo["date"].month,
                           self.finfo["date"].day,
                           self.finfo["start_time"].hour,
                           self.finfo["start_time"].minute,
                           self.finfo["start_time"].second)

    @property
    def end_time(self):
        """Find the end time from filename info."""
        return dt.datetime(self.finfo["date"].year,
                           self.finfo["date"].month,
                           self.finfo["date"].day,
                           self.finfo["end_time"].hour,
                           self.finfo["end_time"].minute,
                           self.finfo["end_time"].second)

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        file_key = ds_info.get("file_key", dataset_id["name"])
        dsname = "Grid/" + file_key
        data = self.get(dsname)
        data = data.squeeze().transpose()
        if data.ndim >= 2:
            data = data.rename({data.dims[-2]: "y", data.dims[-1]: "x"})
        data.data = da.flip(data.data, axis=0)

        fill = data.attrs["_FillValue"]
        data = data.where(data != fill)

        for key in list(data.attrs.keys()):
            val = data.attrs[key]
            if isinstance(val, h5py.h5r.Reference):
                del data.attrs[key]
            if isinstance(val, np.ndarray) and isinstance(val[0][0], h5py.h5r.Reference):
                del data.attrs[key]
        return data

    def get_area_def(self, dsid):
        """Create area definition from the gridded lat/lon values."""
        lats = self.__getitem__("Grid/lat").values
        lons = self.__getitem__("Grid/lon").values

        width = lons.shape[0]
        height = lats.shape[0]

        lower_left_x = lons[0]
        lower_left_y = lats[0]

        upper_right_x = lons[-1]
        upper_right_y = lats[-1]

        area_extent = (lower_left_x, lower_left_y, upper_right_x, upper_right_y)
        description = "IMERG GPM Equirectangular Projection"
        area_id = "imerg"
        proj_id = "equirectangular"
        proj_dict = {"proj": "longlat", "datum": "WGS84", "ellps": "WGS84", }
        area_def = AreaDefinition(area_id, description, proj_id, proj_dict, width, height, area_extent, )
        return area_def
