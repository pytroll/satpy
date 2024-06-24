#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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

"""Reader for the FY-3D MERSI-2 L1B file format.

The files for this reader are HDF5 and come in four varieties; band data
and geolocation data, both at 250m and 1000m resolution.

This reader was tested on FY-3A/B/C MERSI-1, FY-3D MERSI-2, FY-3E MERSI-LL and FY-3G MERSI-RM data,
but should work on future platforms as well assuming no file format changes.

"""

import datetime as dt

import dask.array as da
import numpy as np
from pyspectral.blackbody import blackbody_wn_rad2temp as rad2temp

from satpy.readers.hdf5_utils import HDF5FileHandler

N_TOT_IR_CHANS_LL = 6
PLATFORMS_INSTRUMENTS = {"FY-3A": "mersi-1",
                         "FY-3B": "mersi-1",
                         "FY-3C": "mersi-1",
                         "FY-3D": "mersi-2",
                         "FY-3E": "mersi-ll",
                         "FY-3F": "mersi-3",
                         "FY-3G": "mersi-rm"}


class MERSIL1B(HDF5FileHandler):
    """MERSI-1/MERSI-2/MERSI-LL/MERSI-RM L1B file reader."""

    def _strptime(self, date_attr, time_attr):
        """Parse date/time strings."""
        date = self[date_attr]
        time = self[time_attr]  # "18:27:39.720"
        # cuts off microseconds because of unknown meaning
        # is .720 == 720 microseconds or 720000 microseconds
        return dt.datetime.strptime(date + " " + time.split(".")[0], "%Y-%m-%d %H:%M:%S")

    @property
    def start_time(self):
        """Time for first observation."""
        return self._strptime("/attr/Observing Beginning Date", "/attr/Observing Beginning Time")

    @property
    def end_time(self):
        """Time for final observation."""
        return self._strptime("/attr/Observing Ending Date", "/attr/Observing Ending Time")

    @property
    def sensor_name(self):
        """Map sensor name to Satpy 'standard' sensor names."""
        return PLATFORMS_INSTRUMENTS.get(self.platform_name)

    @property
    def platform_name(self):
        """Platform name."""
        return self["/attr/Satellite Name"]

    def get_refl_mult(self):
        """Get reflectance multiplier."""
        if self.sensor_name == "mersi-rm":
            # MERSI-RM has reflectance in the range 0-1, so we need to convert
            return 100.
        else:
            return 1.

    def _get_single_slope_intercept(self, slope, intercept, cal_index):
        try:
            # convert scalar arrays to scalar
            return slope.item(), intercept.item()
        except ValueError:
            # numpy array but has more than one element
            return slope[cal_index], intercept[cal_index]

    def _get_coefficients(self, cal_key, cal_index):
        """Get VIS calibration coeffs from calibration datasets."""
        # Only one VIS band for MERSI-LL
        coeffs = self[cal_key][cal_index] if self.sensor_name != "mersi-ll" else self[cal_key]
        slope = coeffs.attrs.pop("Slope", None)
        intercept = coeffs.attrs.pop("Intercept", None)
        if slope is not None:
            slope, intercept = self._get_single_slope_intercept(
                slope, intercept, cal_index)
            coeffs = coeffs * slope + intercept
        return coeffs

    def _get_coefficients_mersi1(self, cal_index):
        """Get VIS calibration coeffs from attributes. Only for MERSI-1 on FY-3A/B."""
        try:
            # This is found in the actual file.
            coeffs = self["/attr/VIR_Cal_Coeff"]
        except KeyError:
            # This is in the official manual.
            coeffs = self["/attr/VIS_Cal_Coeff"]
        coeffs = coeffs.reshape(19, 3)
        coeffs = coeffs[cal_index].tolist()
        return coeffs

    def _get_dn_corrections(self, data, band_index, dataset_id, attrs):
        """Use slope and intercept to get DN corrections."""
        slope = attrs.pop("Slope", None)
        intercept = attrs.pop("Intercept", None)
        if slope is not None and dataset_id.get("calibration") != "counts":
            if band_index is not None and slope.size > 1:
                slope = slope[band_index]
                intercept = intercept[band_index]
            # There's a bug in slope for MERSI-1 IR band
            slope = 0.01 if self.sensor_name == "mersi-1" and dataset_id["name"] == "5" else slope
            data = data * slope + intercept
        return data

    def get_dataset(self, dataset_id, ds_info):
        """Load data variable and metadata and calibrate if needed."""
        file_key = ds_info.get("file_key", dataset_id["name"])
        band_index = ds_info.get("band_index")
        data = self[file_key]
        data = data[band_index] if band_index is not None else data
        data = data.rename({data.dims[-2]: "y", data.dims[-1]: "x"}) if data.ndim >= 2 else data

        attrs = data.attrs.copy()  # avoid contaminating other band loading
        attrs.update(ds_info)
        if "rows_per_scan" in self.filetype_info:
            attrs.setdefault("rows_per_scan", self.filetype_info["rows_per_scan"])

        data = self._mask_data(data, dataset_id, attrs)
        data = self._get_dn_corrections(data, band_index, dataset_id, attrs)

        if dataset_id.get("calibration") == "reflectance":
            data = self._get_ref_dataset(data, ds_info)

        elif dataset_id.get("calibration") == "radiance":
            data = self._get_rad_dataset(data, ds_info, dataset_id)

        elif dataset_id.get("calibration") == "brightness_temperature":
            # Converts um^-1 (wavenumbers) and (mW/m^2)/(str/cm^-1) (radiance data)
            # to SI units m^-1, mW*m^-3*str^-1.
            wave_number = 1. / (dataset_id["wavelength"][1] / 1e6)
            # MERSI-1 doesn't have additional corrections
            calibration_index = None if self.sensor_name == "mersi-1" else ds_info["calibration_index"]
            data = self._get_bt_dataset(data, calibration_index, wave_number)

        data.attrs = attrs
        # convert bytes to str
        for key, val in attrs.items():
            # python 3 only
            if bytes is not str and isinstance(val, bytes):
                data.attrs[key] = val.decode("utf8")

        data.attrs.update({
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
        })

        return data

    def _mask_data(self, data, dataset_id, attrs):
        """Mask the data using fill_value and valid_range attributes."""
        fill_value = attrs.pop("_FillValue", np.nan) if self.platform_name in ["FY-3A", "FY-3B"] else \
            attrs.pop("FillValue", np.nan) # covered by valid_range
        valid_range = attrs.pop("valid_range", None)
        if dataset_id.get("calibration") == "counts":
            # preserve integer type of counts if possible
            attrs["_FillValue"] = fill_value
            new_fill = data.dtype.type(fill_value)
        else:
            new_fill = np.nan
        try:
            # Due to a bug in the valid_range upper limit in the 10.8(24) and 12.0(25)
            # in the HDF data, this is hardcoded here.
            valid_range[1] = 25000 if self.sensor_name == "mersi-2" and dataset_id["name"] in ["24", "25"] and \
                                      valid_range[1] == 4095 else valid_range[1]
            # Similar bug also found in MERSI-1
            valid_range[1] = 25000 if self.sensor_name == "mersi-1" and dataset_id["name"] == "5" and \
                                      valid_range[1] == 4095 else valid_range[1]
            # typically bad_values == 65535, saturated == 65534
            # dead detector == 65533
            data = data.where((data >= valid_range[0]) &
                              (data <= valid_range[1]), new_fill)
            return data
        # valid_range could be None
        except TypeError:
            return data

    def _get_ref_dataset(self, data, ds_info):
        """Get the dataset as reflectance.

        For MERSI-1/2/RM, coefficients will be as::

            Reflectance = coeffs_1 + coeffs_2 * DN + coeffs_3 * DN ** 2

        For MERSI-LL, the DN value is in radiance and the reflectance could be calculated by::

            Reflectance = Rad * pi / E0 * 100

        Here E0 represents the solar irradiance of the specific band and is the coefficient.

        """
        # Only FY-3A/B stores VIS calibration coefficients in attributes
        coeffs = self._get_coefficients_mersi1(ds_info["calibration_index"]) if self.platform_name in ["FY-3A",
            "FY-3B"] else self._get_coefficients(ds_info["calibration_key"], ds_info.get("calibration_index", None))
        data = coeffs[0] + coeffs[1] * data + coeffs[2] * data ** 2 if self.sensor_name != "mersi-ll" else \
            data * np.pi / coeffs[0] * 100

        data = data * self.get_refl_mult()
        return data

    def _get_rad_dataset(self, data, ds_info, datset_id):
        """Get the dataset as radiance.

        For MERSI-2/RM VIS bands, this could be calculated by::

            Rad = Reflectance / 100 * E0 / pi

        For MERSI-2, E0 is in the attribute "Solar_Irradiance".
        For MERSI-RM, E0 is in the calibration dataset "Solar_Irradiance".
        However we can't find the way to retrieve this value from MERSI-1.

        For MERSI-LL VIS band, it has already been stored in DN values.
        After applying slope and intercept, we just get it. And Same way for IR bands, no matter which sensor it is.

        """
        mersi_2_vis = [str(i) for i in range(1, 20)]
        mersi_rm_vis = [str(i) for i in range(1, 6)]

        if self.sensor_name == "mersi-2" and datset_id["name"] in mersi_2_vis:
            E0 = self["/attr/Solar_Irradiance"]
            rad = self._get_ref_dataset(data, ds_info) / 100 * E0[mersi_2_vis.index(datset_id["name"])] / np.pi
        elif self.sensor_name == "mersi-rm" and datset_id["name"] in mersi_rm_vis:
            E0 = self._get_coefficients("Calibration/Solar_Irradiance", mersi_rm_vis.index(datset_id["name"]))
            rad = self._get_ref_dataset(data, ds_info) / 100 * E0 / np.pi
        else:
            rad = data
        return rad

    def _get_bt_dataset(self, data, calibration_index, wave_number):
        """Get the dataset as brightness temperature.

        Apparently we don't use these calibration factors for Rad -> BT::

            coeffs = self._get_coefficients(ds_info['calibration_key'], calibration_index)
            # coefficients are per-scan, we need to repeat the values for a
            # clean alignment
            coeffs = np.repeat(coeffs, data.shape[0] // coeffs.shape[1], axis=1)
            coeffs = coeffs.rename({
                coeffs.dims[0]: 'coefficients', coeffs.dims[1]: 'y'
            })  # match data dims
            data = coeffs[0] + coeffs[1] * data + coeffs[2] * data**2 + coeffs[3] * data**3

        """
        # pass the dask array
        bt_data = rad2temp(wave_number, data.data * 1e-5)  # brightness temperature

        # old versions of pyspectral produce numpy arrays
        # new versions of pyspectral can do dask arrays
        data.data = da.from_array(bt_data, chunks=data.data.chunks) if isinstance(bt_data, np.ndarray) else bt_data

        # Some BT bands seem to have 0 in the first 10 columns
        # and it is an invalid measurement, so let's mask
        data = data.where(data != 0)

        # additional corrections from the file
        if self.sensor_name == "mersi-1":
        # https://img.nsmc.org.cn/PORTAL/NSMC/DATASERVICE/SRF/FY3C/FY3C_MERSI_SRF.rar
            corr_coeff_a = 1.0047
            corr_coeff_b = -0.8549
        elif self.sensor_name == "mersi-2":
            corr_coeff_a = float(self["/attr/TBB_Trans_Coefficient_A"][calibration_index])
            corr_coeff_b = float(self["/attr/TBB_Trans_Coefficient_B"][calibration_index])
        elif self.sensor_name == "mersi-ll":
            # MERSI-LL stores these coefficients differently
            try:
                coeffs = self["/attr/TBB_Trans_Coefficient"]
                corr_coeff_a = coeffs[calibration_index]
                corr_coeff_b = coeffs[calibration_index + N_TOT_IR_CHANS_LL]
            except KeyError:
                return data
        else:
            # MERSI-RM has no correction coefficients
            corr_coeff_a = 0

        if corr_coeff_a != 0:
            data = (data - corr_coeff_b) / corr_coeff_a if self.sensor_name != "mersi-1" else \
                data * corr_coeff_a + corr_coeff_b
        # some bands have 0 counts for the first N columns and
        # seem to be invalid data points
        data = data.where(data != 0)
        return data
