#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reader for AMSR2 L1B files in HDF5 format.
"""

import numpy as np

from satpy.projectable import Projectable
from satpy.readers.hdf5_utils import HDF5FileHandler


class AMSR2L1BFileHandler(HDF5FileHandler):

    @property
    def start_time(self):
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        return self.filename_info["start_time"]

    def get_lonlats(self, navid, nav_info, lon_out, lat_out):
        lon_key = nav_info["longitude_key"]
        lat_key = nav_info["latitude_key"]

        # FIXME: Lower frequency channels need CoRegistration parameters applied
        if self[lon_key].shape[1] > lon_out.shape[1]:
            lon_out.data[:] = self[lon_key][:, ::2]
            lat_out.data[:] = self[lat_key][:, ::2]
        else:
            lon_out.data[:] = self[lon_key]
            lat_out.data[:] = self[lat_key]

        fill_value = nav_info.get("fill_value", -9999.)
        lon_out.mask[:] = lon_out == fill_value
        lat_out.mask[:] = lat_out == fill_value

        return {}

    def get_metadata(self, m_key):
        raise NotImplementedError()

    def get_dataset(self, ds_key, ds_info, out=None):
        var_path = ds_info["file_key"]
        fill_value = ds_info.get("fill_value", 65535)

        data = self[var_path]
        dtype = ds_info.get("dtype", np.float32)
        mask = data == fill_value
        data = data.astype(dtype) * self[var_path + "/attr/SCALE FACTOR"]
        ds_info.update({
            "name": ds_key.name,
            "id": ds_key,
            "units": self[var_path + "/attr/UNIT"],
            "platform": self["/attr/PlatformShortName"].item(),
            "sensor": self["/attr/SensorShortName"].item(),
            "start_orbit": int(self["/attr/StartOrbitNumber"].item()),
            "end_orbit": int(self["/attr/StopOrbitNumber"].item()),
        })
        return Projectable(data, mask=mask, **ds_info)
