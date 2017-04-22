#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reader for AMSR2 L1B files in HDF5 format.
"""

import numpy as np

from satpy.dataset import Dataset
from satpy.readers.hdf5_utils import HDF5FileHandler


class AMSR2L1BFileHandler(HDF5FileHandler):
    def get_metadata(self, m_key):
        raise NotImplementedError()

    def get_shape(self, ds_id, ds_info):
        """Get output shape of specified dataset"""
        var_path = ds_info['file_key']
        shape = self[var_path + '/shape']
        if ((ds_info.get('standard_name') == "longitude" or
                     ds_info.get('standard_name') == "latitude") and
                    ds_id.resolution == 10000):
            return shape[0], int(shape[1] / 2)
        return shape

    def get_dataset(self, ds_id, ds_info, out=None):
        """Get output data and metadata of specified dataset"""
        var_path = ds_info['file_key']
        fill_value = ds_info.get('fill_value', 65535)
        dtype = ds_info.get('dtype', np.float32)

        if out is None:
            shape = self.get_shape(ds_id, ds_info)
            out = np.ma.empty(shape, dtype=dtype)
            out.mask = np.zeros(shape, dtype=np.bool)

        data = self[var_path]
        if ((ds_info.get('standard_name') == "longitude" or
             ds_info.get('standard_name') == "latitude") and
                ds_id.resolution == 10000):
            # FIXME: Lower frequency channels need CoRegistration parameters
            # applied
            out.mask[:] = data[:, ::2] == fill_value
            out.data[:] = data[:, ::2].astype(dtype) * self[var_path + "/attr/SCALE FACTOR"]
        else:
            out.mask[:] = data == fill_value
            out.data[:] = data.astype(dtype) * self[var_path + "/attr/SCALE FACTOR"]

        i = getattr(out, 'info', {})
        i.update(ds_info)
        i.update({
            "units": self[var_path + "/attr/UNIT"],
            "platform": self["/attr/PlatformShortName"].item(),
            "sensor": self["/attr/SensorShortName"].item(),
            "start_orbit": int(self["/attr/StartOrbitNumber"].item()),
            "end_orbit": int(self["/attr/StopOrbitNumber"].item()),
        })
        i.update(ds_id.to_dict())
        cls = ds_info.pop("container", Dataset)
        return cls(out.data, mask=out.mask, copy=False, **i)
