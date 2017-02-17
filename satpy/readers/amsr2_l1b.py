#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reader for AMSR2 L1B files in HDF5 format.
"""

import numpy as np

from satpy.dataset import Dataset
from satpy.readers.hdf5_utils import HDF5FileHandler


class AMSR2L1BFileHandler(HDF5FileHandler):

    @property
    def start_time(self):
        return self.filename_info["start_time"]

    @property
    def end_time(self):
        return self.filename_info["start_time"]

    def get_metadata(self, m_key):
        raise NotImplementedError()

    def get_dataset(self, ds_key, ds_info, out=None):
        var_path = ds_info["file_key"]
        fill_value = ds_info.get("fill_value", 65535)

        data = self[var_path]
        dtype = ds_info.get("dtype", np.float32)
        mask = data == fill_value
        data = data.astype(dtype) * self[var_path + "/attr/SCALE FACTOR"]

        if ((ds_info.get('standard_name') == "longitude" or
             ds_info.get('standard_name') == "latitude") and
                ds_key.resolution == 10000):
            # FIXME: Lower frequency channels need CoRegistration parameters
            # applied
            data = data[:, ::2]
            mask = mask[:, ::2]

        ds_info.update({
            "units": self[var_path + "/attr/UNIT"],
            "platform": self["/attr/PlatformShortName"].item(),
            "sensor": self["/attr/SensorShortName"].item(),
            "start_orbit": int(self["/attr/StartOrbitNumber"].item()),
            "end_orbit": int(self["/attr/StopOrbitNumber"].item()),
        })
        ds_info.update(ds_key.to_dict())
        return Dataset(data, mask=mask, **ds_info)
