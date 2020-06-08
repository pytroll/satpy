#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2016-2018 Satpy developers
#
# This file is part of satpy.
#
# satpy is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# satpy is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# satpy.  If not, see <http://www.gnu.org/licenses/>.
"""Reader for AMSR2 L1B files in HDF5 format.
"""

from satpy.readers.hdf5_utils import HDF5FileHandler


class AMSR2L1BFileHandler(HDF5FileHandler):
    def get_metadata(self, ds_id, ds_info):
        var_path = ds_info['file_key']
        info = getattr(self[var_path], 'attrs', {})
        info.update(ds_info)
        info.update({
            "shape": self.get_shape(ds_id, ds_info),
            "units": self[var_path + "/attr/UNIT"],
            "platform_name": self["/attr/PlatformShortName"],
            "sensor": self["/attr/SensorShortName"],
            "start_orbit": int(self["/attr/StartOrbitNumber"]),
            "end_orbit": int(self["/attr/StopOrbitNumber"]),
        })
        info.update(ds_id.to_dict())
        return info

    def get_shape(self, ds_id, ds_info):
        """Get output shape of specified dataset."""
        var_path = ds_info['file_key']
        shape = self[var_path + '/shape']
        if ((ds_info.get('standard_name') == "longitude" or ds_info.get('standard_name') == "latitude") and
                ds_id.resolution == 10000):
            return shape[0], int(shape[1] / 2)
        return shape

    def get_dataset(self, ds_id, ds_info):
        """Get output data and metadata of specified dataset."""
        var_path = ds_info['file_key']
        fill_value = ds_info.get('fill_value', 65535)
        metadata = self.get_metadata(ds_id, ds_info)

        data = self[var_path]
        if ((ds_info.get('standard_name') == "longitude" or
             ds_info.get('standard_name') == "latitude") and
                ds_id.resolution == 10000):
            # FIXME: Lower frequency channels need CoRegistration parameters applied
            data = data[:, ::2] * self[var_path + "/attr/SCALE FACTOR"]
        else:
            data = data * self[var_path + "/attr/SCALE FACTOR"]
        data = data.where(data != fill_value)
        data.attrs.update(metadata)
        return data
