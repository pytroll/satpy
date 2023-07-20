#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2022 Satpy developers
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
"""Reader for Radarsat 2 SGF SAR data.

Format description:
RADARSAT-2 PRODUCT FORMAT DEFINITION
RN-RP-51-2713
Issue 1/15:  OCT. 26, 2016
https://earth.esa.int/eogateway/documents/20142/0/Radarsat-2-Product-Format-Definition.pdf/1ca0cf1e-5a15-a29b-6187-9e5cb1650048
"""

import defusedxml.ElementTree as ElementTree
import numpy as np
import rioxarray

from satpy.readers.file_handlers import BaseFileHandler

from .sar_c_safe import change_quantity


class SARRS02GenericFileHandler(BaseFileHandler):
    """Base class for Radarsat 2 file handlers."""

    @property
    def start_time(self):
        """Start time for the data."""
        return self.filename_info["start_time"]


class SARRS02MeasurementFileHandler(SARRS02GenericFileHandler):
    """File handler for Radarsat 2 imagery files."""

    def __init__(self, filename, filename_info, filetype_info, cal_beta_fh, cal_sigma_fh, cal_gamma_fh):
        """Initialize the file handler."""
        super().__init__(filename, filename_info, filetype_info)
        self.gamma_fh = cal_gamma_fh
        self.sigma_fh = cal_sigma_fh
        self.beta_fh = cal_beta_fh
        self.polarization = self.filename_info["polarization"].lower()

    def get_dataset(self, dataid, ds_info=None):
        """Get dataset."""
        if self.polarization != dataid["polarization"]:
            return
        data = rioxarray.open_rasterio(self.filename, chunks=(1, "auto", "auto")).squeeze()
        if dataid.get("calibration") == "gamma":
            dn_squared = data.astype(float) ** 2
            data = dn_squared / self.gamma_fh.get_dataset()
        if dataid.get("calibration") == "sigma_nought":
            dn_squared = data.astype(float) ** 2
            data = dn_squared / self.sigma_fh.get_dataset()
        if dataid.get("calibration") == "beta_nought":
            dn_squared = data.astype(float) ** 2
            data = dn_squared / self.beta_fh.get_dataset()
        data = change_quantity(data, dataid.get("quantity"))
        return data


class SARRS02CalibrationFileHandler(SARRS02GenericFileHandler):
    """File handler for Radarsat 2 calibration look-up tables."""

    def get_dataset(self):
        """Get dataset."""
        root = ElementTree.parse(self.filename)
        res = root.find("gains").text.split()
        return np.array(res).astype(float)
