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
"""GCOM-C SGLI L1b reader.

GCOM-C has an imager instrument: SGLI
https://www.wmo-sat.info/oscar/instruments/view/505

Test data is available here:
https://suzaku.eorc.jaxa.jp/GCOM_C/data/product_std.html
The live data is available from here:
https://gportal.jaxa.jp/gpr/search?tab=1
And the format description is here:
https://gportal.jaxa.jp/gpr/assets/mng_upload/GCOM-C/SGLI_Level1_Product_Format_Description_en.pdf

"""

from satpy.readers.file_handlers import BaseFileHandler
from datetime import datetime
from satpy import CHUNK_SIZE
import xarray as xr
import dask.array as da
import h5py
import logging
import numpy as np

logger = logging.getLogger(__name__)

resolutions = {'Q': 250,
               'K': 1000,
               'L': 1000}


def interpolate(arr, sampling, full_shape):
    """Interpolate the angles and navigation."""
    # TODO: daskify this!
    # TODO: do it in cartesian coordinates ! pbs at date line and poles
    # possible
    tie_x = np.arange(0, arr.shape[0] * sampling, sampling)
    tie_y = np.arange(0, arr.shape[1] * sampling, sampling)
    full_x = np.arange(0, full_shape[0])
    full_y = np.arange(0, full_shape[1])

    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(
        tie_x, tie_y, arr)

    values = spl(full_x, full_y)

    return da.from_array(values, chunks=(CHUNK_SIZE, CHUNK_SIZE))


class HDF5SGLI(BaseFileHandler):
    """File handler for the SGLI l1b data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the filehandler."""
        super(HDF5SGLI, self).__init__(filename, filename_info, filetype_info)
        self.resolution = resolutions[self.filename_info['resolution']]
        self.fh = h5py.File(self.filename, 'r')

    @property
    def start_time(self):
        """Get the start time."""
        the_time = self.fh['Global_attributes'].attrs['Scene_start_time'].item()
        return datetime.strptime(the_time.decode('ascii'), '%Y%m%d %H:%M:%S.%f')

    @property
    def end_time(self):
        """Get the end time."""
        the_time = self.fh['Global_attributes'].attrs['Scene_end_time'].item()
        return datetime.strptime(the_time.decode('ascii'), '%Y%m%d %H:%M:%S.%f')

    def get_dataset(self, key, info):
        """Get the dataset."""
        if key.resolution != self.resolution:
            return

        h5dataset = self.fh[info['file_key']]
        resampling_interval = h5dataset.attrs.get('Resampling_interval', 1)
        if resampling_interval != 1:
            logger.debug('Interpolating %s.', key.name)
            full_shape = (self.fh['Image_data'].attrs['Number_of_lines'],
                          self.fh['Image_data'].attrs['Number_of_pixels'])
            dataset = interpolate(h5dataset, resampling_interval, full_shape)
        else:
            dataset = da.from_array(h5dataset[:].astype('<u2'), chunks=h5dataset.chunks)
        dataset = xr.DataArray(dataset, attrs=h5dataset.attrs, dims=['y', 'x'])
        dataset.attrs.update(info)
        with xr.set_options(keep_attrs=True):
            if 'Mask' in h5dataset.attrs:
                mask_value = h5dataset.attrs['Mask'].item()
                dataset = dataset & mask_value
            if 'Bit00(LSB)-13' in h5dataset.attrs:
                mask_info = h5dataset.attrs['Bit00(LSB)-13'].item()
                mask_vals = mask_info.split(b'\n')[1:]
                missing = int(mask_vals[0].split(b':')[0].strip())
                saturation = int(mask_vals[1].split(b':')[0].strip())
                dataset = dataset.where(dataset < min(missing, saturation))
            if 'Maximum_valid_DN' in h5dataset.attrs:
                # dataset = dataset.where(dataset <= h5dataset.attrs['Maximum_valid_DN'].item())
                pass
            if key.name.startswith('VN'):
                if key.calibration == 'counts':
                    pass
                if key.calibration == 'radiance':
                    dataset = dataset * h5dataset.attrs['Slope'] + h5dataset.attrs['Offset']
                if key.calibration == 'reflectance':
                    # dataset = dataset * h5dataset.attrs['Slope'] + h5dataset.attrs['Offset']
                    # dataset *= np.pi / h5dataset.attrs['Band_weighted_TOA_solar_irradiance'] * 100
                    # equivalent to the two lines above
                    dataset = (dataset * h5dataset.attrs['Slope_reflectance']
                               + h5dataset.attrs['Offset_reflectance']) * 100
            else:
                dataset = dataset * h5dataset.attrs['Slope'] + h5dataset.attrs['Offset']

        dataset.attrs['platform_name'] = 'GCOM-C1'
        return dataset
