#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2022, 2023 Satpy Developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Common utilities for reading VIIRS and ATMS SDR data."""

import logging
from datetime import datetime, timedelta

import dask.array as da
import numpy as np
import xarray as xr

from satpy.readers.hdf5_utils import HDF5FileHandler

NO_DATE = datetime(1958, 1, 1)
EPSILON_TIME = timedelta(days=2)
LOG = logging.getLogger(__name__)


VIIRS_DATASET_KEYS = {'GDNBO': 'VIIRS-DNB-GEO',
                      'SVDNB': 'VIIRS-DNB-SDR',
                      'GITCO': 'VIIRS-IMG-GEO-TC',
                      'GIMGO': 'VIIRS-IMG-GEO',
                      'SVI01': 'VIIRS-I1-SDR',
                      'SVI02': 'VIIRS-I2-SDR',
                      'SVI03': 'VIIRS-I3-SDR',
                      'SVI04': 'VIIRS-I4-SDR',
                      'SVI05': 'VIIRS-I5-SDR',
                      'GMTCO': 'VIIRS-MOD-GEO-TC',
                      'GMODO': 'VIIRS-MOD-GEO',
                      'SVM01': 'VIIRS-M1-SDR',
                      'SVM02': 'VIIRS-M2-SDR',
                      'SVM03': 'VIIRS-M3-SDR',
                      'SVM04': 'VIIRS-M4-SDR',
                      'SVM05': 'VIIRS-M5-SDR',
                      'SVM06': 'VIIRS-M6-SDR',
                      'SVM07': 'VIIRS-M7-SDR',
                      'SVM08': 'VIIRS-M8-SDR',
                      'SVM09': 'VIIRS-M9-SDR',
                      'SVM10': 'VIIRS-M10-SDR',
                      'SVM11': 'VIIRS-M11-SDR',
                      'SVM12': 'VIIRS-M12-SDR',
                      'SVM13': 'VIIRS-M13-SDR',
                      'SVM14': 'VIIRS-M14-SDR',
                      'SVM15': 'VIIRS-M15-SDR',
                      'SVM16': 'VIIRS-M16-SDR',
                      'IVCDB': 'VIIRS-DualGain-Cal-IP'}
ATMS_DATASET_KEYS = {'SATMS': 'ATMS-SDR',
                     'GATMO': 'ATMS-SDR-GEO',
                     'TATMS': 'ATMS-TDR'}

DATASET_KEYS = {}
DATASET_KEYS.update(VIIRS_DATASET_KEYS)
DATASET_KEYS.update(ATMS_DATASET_KEYS)


def _get_scale_factors_for_units(factors, file_units, output_units):
    if file_units == "W cm-2 sr-1" and output_units == "W m-2 sr-1":
        LOG.debug("Adjusting scaling factors to convert '%s' to '%s'",
                  file_units, output_units)
        factors = factors * 10000.
    elif file_units == "1" and output_units == "%":
        LOG.debug("Adjusting scaling factors to convert '%s' to '%s'",
                  file_units, output_units)
        factors = factors * 100.
    else:
        raise ValueError("Don't know how to convert '{}' to '{}'".format(
            file_units, output_units))
    return factors


def _get_file_units(dataset_id, ds_info):
    """Get file units from metadata."""
    file_units = ds_info.get("file_units")
    if file_units is None:
        LOG.debug("Unknown units for file key '%s'", dataset_id)
    return file_units


class JPSS_SDR_FileHandler(HDF5FileHandler):
    """Base class for reading JPSS VIIRS & ATMS SDR HDF5 Files."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize file handler."""
        super().__init__(filename, filename_info, filetype_info, **kwargs)

    def _parse_datetime(self, datestr, timestr):
        try:
            datetime_str = datestr + timestr
        except TypeError:
            datetime_str = (str(datestr.data.compute().astype(str)) +
                            str(timestr.data.compute().astype(str)))

        time_val = datetime.strptime(datetime_str, '%Y%m%d%H%M%S.%fZ')
        if abs(time_val - NO_DATE) < EPSILON_TIME:
            # catch rare case when SDR files have incorrect date
            raise ValueError("Datetime invalid {}".format(time_val))
        return time_val

    @property
    def start_time(self):
        """Get start time."""
        date_var_path = self._get_aggr_path("start_date", "AggregateBeginningDate")
        time_var_path = self._get_aggr_path("start_time", "AggregateBeginningTime")
        return self._parse_datetime(self[date_var_path], self[time_var_path])

    @property
    def end_time(self):
        """Get end time."""
        date_var_path = self._get_aggr_path("end_date", "AggregateEndingDate")
        time_var_path = self._get_aggr_path("end_time", "AggregateEndingTime")
        return self._parse_datetime(self[date_var_path], self[time_var_path])

    @property
    def start_orbit_number(self):
        """Get start orbit number."""
        start_orbit_path = self._get_aggr_path("start_orbit", "AggregateBeginningOrbitNumber")
        return int(self[start_orbit_path])

    @property
    def end_orbit_number(self):
        """Get end orbit number."""
        end_orbit_path = self._get_aggr_path("end_orbit", "AggregateEndingOrbitNumber")
        return int(self[end_orbit_path])

    def _get_aggr_path(self, fileinfo_key, aggr_default):
        dataset_group = DATASET_KEYS[self.datasets[0]]
        default = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/' + aggr_default
        return self.filetype_info.get(fileinfo_key, default).format(dataset_group=dataset_group)

    @property
    def platform_name(self):
        """Get platform name."""
        default = '/attr/Platform_Short_Name'
        platform_path = self.filetype_info.get(
            'platform_name', default).format(**self.filetype_info)
        platform_dict = {'NPP': 'Suomi-NPP',
                         'JPSS-1': 'NOAA-20',
                         'J01': 'NOAA-20',
                         'JPSS-2': 'NOAA-21',
                         'J02': 'NOAA-21'}
        return platform_dict.get(self[platform_path], self[platform_path])

    @property
    def sensor_name(self):
        """Get sensor name."""
        dataset_group = DATASET_KEYS[self.datasets[0]]
        default = 'Data_Products/{dataset_group}/attr/Instrument_Short_Name'
        sensor_path = self.filetype_info.get(
            'sensor_name', default).format(dataset_group=dataset_group)
        return self[sensor_path].lower()

    def scale_swath_data(self, data, scaling_factors, dataset_group):
        """Scale swath data using scaling factors and offsets.

        Multi-granule (a.k.a. aggregated) files will have more than the usual two values.
        """
        rows_per_gran = self._get_rows_per_granule(dataset_group)
        factors = self._mask_and_reshape_factors(scaling_factors)
        data = self._map_and_apply_factors(data, factors, rows_per_gran)
        return data

    def scale_data_to_specified_unit(self, data, dataset_id, ds_info):
        """Get sscale and offset factors and convert/scale data to given physical unit."""
        var_path = self._generate_file_key(dataset_id, ds_info)
        dataset_group = ds_info['dataset_group']
        file_units = _get_file_units(dataset_id, ds_info)
        output_units = ds_info.get("units", file_units)

        factor_var_path = ds_info.get("factors_key", var_path + "Factors")

        factors = self.get(factor_var_path)
        factors = self._adjust_scaling_factors(factors, file_units, output_units)

        if factors is not None:
            return self.scale_swath_data(data, factors, dataset_group)

        LOG.debug("No scaling factors found for %s", dataset_id)
        return data

    @staticmethod
    def _mask_and_reshape_factors(factors):
        factors = factors.where(factors > -999, np.float32(np.nan))
        return factors.data.reshape((-1, 2)).rechunk((1, 2))  # make it so map_blocks happens per factor

    @staticmethod
    def _map_and_apply_factors(data, factors, rows_per_gran):
        # The user may have requested a different chunking scheme, but we need
        # per granule chunking right now so factor chunks map 1:1 to data chunks
        old_chunks = data.chunks
        dask_data = data.data.rechunk((tuple(rows_per_gran), data.data.chunks[1]))
        dask_data = da.map_blocks(_apply_factors, dask_data, factors,
                                  chunks=dask_data.chunks, dtype=data.dtype,
                                  meta=np.array([[]], dtype=data.dtype))
        data = xr.DataArray(dask_data.rechunk(old_chunks),
                            dims=data.dims, coords=data.coords,
                            attrs=data.attrs)
        return data

    @staticmethod
    def _scale_factors_for_units(factors, file_units, output_units):
        return _get_scale_factors_for_units(factors, file_units, output_units)

    @staticmethod
    def _get_valid_scaling_factors(factors):
        if factors is None:
            factors = np.array([1, 0], dtype=np.float32)
            factors = xr.DataArray(da.from_array(factors, chunks=1))
        else:
            factors = factors.where(factors != -999., np.float32(np.nan))
        return factors

    def _adjust_scaling_factors(self, factors, file_units, output_units):
        """Adjust scaling factors ."""
        if file_units == output_units:
            LOG.debug("File units and output units are the same (%s)", file_units)
            return factors
        factors = self._get_valid_scaling_factors(factors)
        return self._scale_factors_for_units(factors, file_units, output_units)

    @staticmethod
    def expand_single_values(var, scans):
        """Expand single valued variable to full scan lengths."""
        if scans.size == 1:
            return var
        else:
            expanded = np.repeat(var, scans)
            expanded.attrs = var.attrs
            expanded.rename({expanded.dims[0]: 'y'})
            return expanded

    def _scan_size(self, dataset_group_name):
        """Get how many rows of data constitute one scanline."""
        if 'ATM' in dataset_group_name:
            scan_size = 1
        elif 'I' in dataset_group_name:
            scan_size = 32
        else:
            scan_size = 16
        return scan_size

    def _generate_file_key(self, ds_id, ds_info, factors=False):
        var_path = ds_info.get('file_key', 'All_Data/{dataset_group}_All/{calibration}')
        calibration = {
            'radiance': 'Radiance',
            'reflectance': 'Reflectance',
            'brightness_temperature': 'BrightnessTemperature',
        }.get(ds_id.get('calibration'))
        var_path = var_path.format(calibration=calibration, dataset_group=DATASET_KEYS[ds_info['dataset_group']])
        if ds_id['name'] in ['dnb_longitude', 'dnb_latitude']:
            if self.use_tc is True:
                return var_path + '_TC'
            if self.use_tc is None and var_path + '_TC' in self.file_content:
                return var_path + '_TC'
        return var_path

    def _update_data_attributes(self, data, dataset_id, ds_info):
        file_units = _get_file_units(dataset_id, ds_info)
        output_units = ds_info.get("units", file_units)
        i = getattr(data, 'attrs', {})
        i.update(ds_info)
        i.update({
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
            "units": output_units,
            "rows_per_scan": self._scan_size(ds_info['dataset_group']),
        })
        i.update(dataset_id.to_dict())
        data.attrs.update(i)
        return data

    def _get_variable(self, var_path, **kwargs):
        return self[var_path]

    def concatenate_dataset(self, dataset_group, var_path, **kwargs):
        """Concatenate dataset."""
        scan_size = self._scan_size(dataset_group)
        scans = self._get_scans_per_granule(dataset_group)
        start_scan = 0
        data_chunks = []
        scans = xr.DataArray(scans)

        variable = self._get_variable(var_path, **kwargs)
        # check if these are single per-granule value
        if variable.size != scans.size:
            for gscans in scans.values:
                data_chunks.append(variable.isel(y=slice(start_scan,
                                                         start_scan + gscans * scan_size)))
                start_scan += gscans * scan_size
            return xr.concat(data_chunks, 'y')
        else:
            # This is not tested - Not sure this code is ever going to be used? A. Dybbroe
            # Mon Jan  2 13:31:21 2023
            return self.expand_single_values(variable, scans)

    def _get_rows_per_granule(self, dataset_group):
        scan_size = self._scan_size(dataset_group)
        scans_per_gran = self._get_scans_per_granule(dataset_group)
        return [scan_size * gran_scans for gran_scans in scans_per_gran]

    def _get_scans_per_granule(self, dataset_group):
        number_of_granules_path = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateNumberGranules'
        nb_granules_path = number_of_granules_path.format(dataset_group=DATASET_KEYS[dataset_group])
        scans = []
        for granule in range(self[nb_granules_path]):
            scans_path = 'Data_Products/{dataset_group}/{dataset_group}_Gran_{granule}/attr/N_Number_Of_Scans'
            scans_path = scans_path.format(dataset_group=DATASET_KEYS[dataset_group], granule=granule)
            scans.append(self[scans_path])
        return scans

    def mask_fill_values(self, data, ds_info):
        """Mask fill values."""
        is_floating = np.issubdtype(data.dtype, np.floating)

        if is_floating:
            # If the data is a float then we mask everything <= -999.0
            fill_max = np.float32(ds_info.pop("fill_max_float", -999.0))
            return data.where(data > fill_max, np.float32(np.nan))
        else:
            # If the data is an integer then we mask everything >= fill_min_int
            fill_min = int(ds_info.pop("fill_min_int", 65528))
            return data.where(data < fill_min, np.float32(np.nan))

    def available_datasets(self, configured_datasets=None):
        """Generate dataset info and their availablity.

        See
        :meth:`satpy.readers.file_handlers.BaseFileHandler.available_datasets`
        for details.

        """
        for is_avail, ds_info in (configured_datasets or []):
            if is_avail is not None:
                yield is_avail, ds_info
                continue
            dataset_group = [ds_group for ds_group in ds_info['dataset_groups'] if ds_group in self.datasets]
            if dataset_group:
                yield True, ds_info
            elif is_avail is None:
                yield is_avail, ds_info


def _apply_factors(data, factor_set):
    return data * factor_set[0, 0] + factor_set[0, 1]
