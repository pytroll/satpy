#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2024.
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
"""A reader for Level 1C data produced by the MSI instrument aboard EarthCARE."""
import logging

import numpy as np

from satpy.readers.hdf5_utils import HDF5FileHandler
from satpy.utils import get_legacy_chunk_size

LOG = logging.getLogger(__name__)
CHUNK_SIZE = get_legacy_chunk_size()


class MSIECL1CFileHandler(HDF5FileHandler):
    """File handler for MSI L1c H5 files."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init the file handler."""
        super(MSIECL1CFileHandler, self).__init__(filename,
                                                  filename_info,
                                                  filetype_info)

    @property
    def end_time(self):
        """Get end time."""
        return self.filename_info["end_time"]

    @property
    def start_time(self):
        """Get start time."""
        return self.filename_info["start_time"]

    def get_dataset(self, dataset_id, ds_info):
        """Load data variable and metadata and calibrate if needed."""
        file_key = ds_info.get("file_key", dataset_id["name"])
        data = self[file_key]

        # Band data is stored in a 3d array (Band x Along_Track x Across_Track).
        # This means we have to select a single 2d array for a given band,
        # and the correct index is given in the reader YAML.
        band_index = ds_info.get("band_index")
        if band_index is not None:
            data = data[band_index]

        # The dataset has incorrect units attribute (due to storing multiple types). Fix it here.
        data.attrs.update(ds_info)
        data.attrs.update({"units": ds_info.get("units")})
        # VIS/SWIR data can have radiance or reflectance calibration.
        if "calibration" in ds_info:
            cal_type = ds_info["calibration"].name
            data = self._calibrate(data, band_index, cal_type)

        # Rename dimensions, as some have incorrect names (notably the pixel value data).
        if "dim_1" in data.dims:
            data = data.rename({"dim_1": "y", "dim_2": "x"})

        # The dimension list is usually a reference to an H5 variable, which is problematic
        # when making a copy of the data. This sorts out the dimensions and sets them correctly
        # following the process done in the OMPS reader.
        if "DIMENSION_LIST" in data.attrs:
            data = self._fix_dims(data, file_key)

        return data

    def _fix_dims(self, data, file_key):
        """The pixel data has badly named coordinates, this fixes them."""
        data.attrs.pop("DIMENSION_LIST")
        dimensions = self.get_reference(file_key, "DIMENSION_LIST")
        dim_dict = {}
        # We have to loop over dimensions to match dim sizes as the pixel data is 3d rather than 2d.
        for i in range(0, len(data.dims)):
            c_dim = data.dims[i]
            for r_dim in dimensions:
                if data.shape[i] == r_dim[0].shape[0]:
                    dim_dict[c_dim] = r_dim[0]
        data.assign_coords(dim_dict)
        return data

    def _calibrate(self, chan_data, band_index, cal_type):
        """Calibrate the data."""
        if cal_type == "reflectance":
            sol_irrad = self["NonStandard/solar_irradiance"]
            chan_data.data = chan_data.data * 100. * np.pi / float(sol_irrad[band_index])
            return chan_data
        elif cal_type not in ["radiance", "brightness_temperature"]:
            raise ValueError(f"Unknown calibration type:{cal_type}")

        return chan_data
