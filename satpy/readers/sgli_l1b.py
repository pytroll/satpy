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

import logging
from datetime import datetime

import dask.array as da
import h5py
import numpy as np
import xarray as xr
from dask.array.core import normalize_chunks
from xarray import Dataset, Variable
from xarray.backends import BackendArray, BackendEntrypoint
from xarray.core import indexing

# from satpy import CHUNK_SIZE
from satpy.readers.file_handlers import BaseFileHandler

logger = logging.getLogger(__name__)

resolutions = {"Q": 250,
               "K": 1000,
               "L": 1000}


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

    return da.from_array(values, chunks=(1000, 1000))


class HDF5SGLI(BaseFileHandler):
    """File handler for the SGLI l1b data."""

    def __init__(self, filename, filename_info, filetype_info):
        """Initialize the filehandler."""
        super().__init__(filename, filename_info, filetype_info)
        self.resolution = resolutions[self.filename_info["resolution"]]
        self.h5file = h5py.File(self.filename, "r")

    @property
    def start_time(self):
        """Get the start time."""
        the_time = self.h5file["Global_attributes"].attrs["Scene_start_time"].item()
        return datetime.strptime(the_time.decode("ascii"), "%Y%m%d %H:%M:%S.%f")

    @property
    def end_time(self):
        """Get the end time."""
        the_time = self.h5file["Global_attributes"].attrs["Scene_end_time"].item()
        return datetime.strptime(the_time.decode("ascii"), "%Y%m%d %H:%M:%S.%f")

    def get_dataset(self, key, info):
        """Get the dataset."""
        if key["resolution"] != self.resolution:
            return

        # if key["polarization"] is not None:
        #     pols = {0: '0', -60: 'm60', 60: 'p60'}
        #     file_key = info['file_key'].format(pol=pols[key["polarization"]])
        # else:
        #     file_key = info['file_key']
        file_key = info["file_key"]
        h5dataset = self.h5file[file_key]

        # resampling_interval = h5dataset.attrs.get('Resampling_interval', 1)
        # if resampling_interval != 1:
        #     logger.debug('Interpolating %s.', key["name"])
        #     full_shape = (self.h5file['Image_data'].attrs['Number_of_lines'],
        #                   self.h5file['Image_data'].attrs['Number_of_pixels'])
        #     dataset = interpolate(h5dataset, resampling_interval, full_shape)
        # else:
        #     dataset = da.from_array(h5dataset[:].astype('<u2'), chunks=h5dataset.chunks)
        chunks = normalize_chunks(("auto", "auto"), h5dataset.shape, previous_chunks=h5dataset.chunks, dtype=np.float32)
        dataset = da.from_array(h5dataset, chunks=chunks)
        attrs = h5dataset.attrs

        dataset = xr.DataArray(dataset, attrs=attrs, dims=["y", "x"])
        with xr.set_options(keep_attrs=True):
            if key["name"][:2] in ["VN", "SW", "P1", "P2"]:
                dataset = self.get_visible_dataset(key, h5dataset, dataset)
            elif key["name"][:-2] in ["longitude", "latitude"]:
                resampling_interval = attrs["Resampling_interval"]
                if resampling_interval != 1:
                    new_lons, new_lats = self.interpolate_lons_lats(resampling_interval)
                    if key["name"][:-2] == "longitude":
                        dataset = new_lons
                    else:
                        dataset = new_lats
                    dataset = xr.DataArray(da.from_array(dataset, chunks="auto"), attrs=attrs)
                return dataset
            else:
                raise NotImplementedError()

        dataset.attrs["platform_name"] = "GCOM-C1"

        return dataset

    def interpolate_lons_lats(self, resampling_interval):
        lons = self.h5file["Geometry_data/Longitude"]
        lats = self.h5file["Geometry_data/Latitude"]

        full_shape = (self.h5file["Image_data"].attrs["Number_of_lines"],
                                  self.h5file["Image_data"].attrs["Number_of_pixels"])

        tie_x = np.arange(0, lons.shape[0] * resampling_interval, resampling_interval)
        tie_y = np.arange(0, lons.shape[1] * resampling_interval, resampling_interval)
        full_x = np.arange(0, full_shape[0])
        full_y = np.arange(0, full_shape[1])

        from geotiepoints import SatelliteInterpolator
        interpolator = SatelliteInterpolator((lons, lats), (tie_x, tie_y), (full_x, full_y), 2, 2,
                                             chunk_size=(1000, 500))
        new_lons, new_lats = interpolator.interpolate()
        return new_lons,new_lats

    def get_visible_dataset(self, key, h5dataset, dataset):

        dataset = self.mask_to_14_bits(dataset)
        dataset = self.calibrate(dataset, key["calibration"])
            #dataset.attrs.update(info)
            #dataset = self._mask_and_scale(dataset, h5dataset, key)

            #
        return dataset

    def mask_to_14_bits(self, dataset):
        """Mask data to 14 bits."""
        return dataset & dataset.attrs["Mask"].item()


    def calibrate(self, dataset, calibration):
        attrs = dataset.attrs
        if calibration == "counts":
            return dataset
        if calibration == "reflectance":
            calibrated = (dataset * attrs["Slope_reflectance"] + attrs["Offset_reflectance"]) * 100
        elif calibration == "radiance":
            calibrated = dataset * attrs["Slope"] + attrs["Offset"]
        missing, _ = self.get_missing_and_saturated(attrs)
        return calibrated.where(dataset < missing)

    def get_missing_and_saturated(self, attrs):
        missing_and_saturated = attrs["Bit00(LSB)-13"].item()
        mask_vals = missing_and_saturated.split(b"\n")[1:]
        missing = int(mask_vals[0].split(b":")[0].strip())
        saturation = int(mask_vals[1].split(b":")[0].strip())
        return missing, saturation

    # def _mask_and_scale(self, dataset, h5dataset, key):
    #     with xr.set_options(keep_attrs=True):
    #         if 'Mask' in h5dataset.attrs:
    #             mask_value = h5dataset.attrs['Mask'].item()
    #             dataset = dataset & mask_value
    #         if 'Bit00(LSB)-13' in h5dataset.attrs:
    #             mask_info = h5dataset.attrs['Bit00(LSB)-13'].item()
    #             mask_vals = mask_info.split(b'\n')[1:]
    #             missing = int(mask_vals[0].split(b':')[0].strip())
    #             saturation = int(mask_vals[1].split(b':')[0].strip())
    #             dataset = dataset.where(dataset < min(missing, saturation))
    #         if 'Maximum_valid_DN' in h5dataset.attrs:
    #             # dataset = dataset.where(dataset <= h5dataset.attrs['Maximum_valid_DN'].item())
    #             pass
    #         if key["name"][:2] in ['VN', 'SW', 'P1', 'P2']:
    #             if key["calibration"] == 'counts':
    #                 pass
    #             if key["calibration"] == 'radiance':
    #                 dataset = dataset * h5dataset.attrs['Slope'] + h5dataset.attrs['Offset']
    #             if key["calibration"] == 'reflectance':
    #                 # dataset = dataset * h5dataset.attrs['Slope'] + h5dataset.attrs['Offset']
    #                 # dataset *= np.pi / h5dataset.attrs['Band_weighted_TOA_solar_irradiance'] * 100
    #                 # equivalent to the two lines above
    #                 dataset = (dataset * h5dataset.attrs['Slope_reflectance']
    #                            + h5dataset.attrs['Offset_reflectance']) * 100
    #         else:
    #             dataset = dataset * h5dataset.attrs['Slope'] + h5dataset.attrs['Offset']
    #     return dataset


class H5Array(BackendArray):
    """An Hdf5-based array."""

    def __init__(self, array):
        """Initialize the array."""
        self.shape = array.shape
        self.dtype = array.dtype
        self.array = array

    def __getitem__(self, key):
        """Get a slice of the array."""
        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key):
        return self.array[key]


class SGLIBackend(BackendEntrypoint):
    """The SGLI backend."""

    def open_dataset(self, filename, *, drop_variables=None):
        """Open the dataset."""
        ds = Dataset()
        h5f = h5py.File(filename)
        h5_arr = h5f["Image_data"]["Lt_VN01"]
        chunks = dict(zip(("y", "x"), h5_arr.chunks))
        ds["Lt_VN01"] = Variable(["y", "x"],
                                 indexing.LazilyIndexedArray(H5Array(h5_arr)),
                                 encoding={"preferred_chunks": chunks})
        return ds
