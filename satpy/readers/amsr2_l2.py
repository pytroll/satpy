#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Satpy developers
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
"""Reader for AMSR2 L2 files in HDF5 format."""

from satpy.readers.amsr2_l1b import AMSR2L1BFileHandler


class AMSR2L2FileHandler(AMSR2L1BFileHandler):
    def mask_dataset(self, ds_info, data):
        """Mask data with the fill value"""
        fill_value = ds_info.get('fill_value', 65535)
        return data.where(data != fill_value)

    def scale_dataset(self, var_path, data):
        """scale data with the scale factor attribute"""
        return data * self[var_path + "/attr/SCALE FACTOR"]

    def get_dataset(self, ds_id, ds_info):
        """Get output data and metadata of specified dataset."""
        var_path = ds_info['file_key']

        data = self[var_path].squeeze()
        data = self.mask_dataset(ds_info, data)
        data = self.scale_dataset(var_path, data)

        if ds_info.get('name') == "ssw":
            data = data.rename({'dim_0': 'y', 'dim_1': 'x'})
        metadata = self.get_metadata(ds_id, ds_info)
        data.attrs.update(metadata)
        return data
