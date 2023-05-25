#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2023 Satpy developers
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
"""File handler for DSCOVR EPIC L1B data in hdf5 format.

The ``epic_l1b_h5`` reader reads and calibrates EPIC L1B image data in hdf5 format.

This reader supports all image and most ancillary datasets.
Once the reader is initialised:

`` scn = Scene([epic_filename], reader='epic_l1b_h5')``

Channels can be loaded with the 'B' prefix and their wavelength in nanometers:

``scn.load(['B317', 'B688'])``

while ancillary data can be loaded by its name:

``scn.load(['solar_zenith_angle'])``

Note that ancillary dataset names use common standards and not the dataset names in the file.
By default, channel data is loaded as calibrated reflectances, but counts data is also available.

"""

import logging
from datetime import datetime

import dask.array as da
import numpy as np

from satpy.readers.hdf5_utils import HDF5FileHandler

logger = logging.getLogger(__name__)

# Level 1b is given as counts. These factors convert to reflectance.
# Retrieved from: https://asdc.larc.nasa.gov/documents/dscovr/DSCOVR_EPIC_Calibration_Factors_V03.pdf
CALIB_COEFS = {'B317': 1.216e-4,
               'B325': 1.111e-4,
               'B340': 1.975e-5,
               'B388': 2.685e-5,
               'B443': 8.34e-6,
               'B551': 6.66e-6,
               'B680': 9.3e-6,
               'B688': 2.02e-5,
               'B764': 2.36e-5,
               'B780': 1.435e-5}


class DscovrEpicL1BH5FileHandler(HDF5FileHandler):
    """File handler for DSCOVR EPIC L1b data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Init filehandler."""
        super(DscovrEpicL1BH5FileHandler, self).__init__(filename, filename_info, filetype_info)

        self.sensor = 'epic'
        self.platform_name = 'dscovr'

    @property
    def start_time(self):
        """Get the start time."""
        start_time = datetime.strptime(self.file_content['/attr/begin_time'], '%Y-%m-%d %H:%M:%S')
        return start_time

    @property
    def end_time(self):
        """Get the end time."""
        end_time = datetime.strptime(self.file_content['/attr/end_time'], '%Y-%m-%d %H:%M:%S')
        return end_time

    @staticmethod
    def _mask_infinite(band):
        band.data = da.where(np.isfinite(band.data), band.data, np.nan)
        return band

    @staticmethod
    def calibrate(data, ds_name, calibration=None):
        """Convert counts input reflectance."""
        if calibration == "reflectance":
            return data * CALIB_COEFS[ds_name] * 100.
        return data

    def get_dataset(self, dataset_id, ds_info):
        """Load a dataset."""
        ds_name = dataset_id['name']

        logger.debug('Reading in get_dataset %s.', ds_name)
        file_key = ds_info.get('file_key', ds_name)

        band = self._mask_infinite(self.get(file_key))
        band = self.calibrate(band, ds_name, calibration=dataset_id.get('calibration'))
        band = self._update_metadata(band)

        return band

    def _update_metadata(self, band):
        band = band.rename({band.dims[0]: 'x', band.dims[1]: 'y'})
        band.attrs.update({'platform_name': self.platform_name, 'sensor': self.sensor})

        return band
