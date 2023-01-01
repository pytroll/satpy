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
"""
Reader for the ATMS SDR format.

A reader for Advanced Technology Microwave Sounder (ATMS) SDR data as it
e.g. comes out of the CSPP package for processing Direct Readout data.

The format is described in the JPSS COMMON DATA FORMAT CONTROL BOOK (CDFCB):

Joint Polar Satellite System (JPSS) Common Data Format Control Book -
External (CDFCB-X) Volume III - SDR/TDR Formats

(474-00001-03_JPSS-CDFCB-X-Vol-III_0124C.pdf)


https://www.nesdis.noaa.gov/about/documents-reports/jpss-technical-documents/jpss-science-documents

"""

import logging

import dask.array as da
import h5py
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.viirs_atms_sdr_utils import DATASET_KEYS, JPSS_SDR_FileHandler, _get_file_units

LOG = logging.getLogger(__name__)

ATMS_CHANNEL_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                      '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']


class ATMS_SDR_FileHandler(JPSS_SDR_FileHandler):
    """ATMS SDR HDF5 File Reader."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize file handler."""
        self.datasets = filename.split('_')[0].split('-')
        super().__init__(filename, filename_info, filetype_info, **kwargs)

    def __getitem__(self, key):
        """Get item for given key."""
        val = self.file_content[key]
        if isinstance(val, h5py.Dataset):
            dset = h5py.File(self.filename, 'r')[key]
            if dset.ndim == 3:
                dset_data = da.from_array(dset, chunks=CHUNK_SIZE)
                attrs = self._attrs_cache.get(key, dset.attrs)
                return xr.DataArray(dset_data, dims=['y', 'x', 'z'], attrs=attrs)

        return super().__getitem__(key)

    def _get_atms_channel_index(self, ch_name):
        """Get the channels array index from name."""
        try:
            return ATMS_CHANNEL_NAMES.index(ch_name)
        except ValueError:
            return None

    def _get_scaling_factors(self, file_units, output_units, factor_var_path, ch_idx):
        """Get file scaling factors and scale according to expected units."""
        if ch_idx is None:
            factors = self.get(factor_var_path)
        else:
            if ch_idx == 21:
                ch_idx = 20  # The BrightnessTemperatureFactors array is only
                # 42 long!? But there are 22 ATMS bands to be scaled! We assume
                # the scale/pffset values are the same for all bands!
                # FIXME!
            factors = self.get(factor_var_path)[ch_idx*2:ch_idx*2+2]
        factors = self._adjust_scaling_factors(factors, file_units, output_units)
        return factors

    def _generate_file_key(self, ds_id, ds_info, factors=False):
        var_path = ds_info.get('file_key', 'All_Data/{dataset_group}_All/{calibration}')
        calibration = {
            'brightness_temperature': 'BrightnessTemperature',
        }.get(ds_id.get('calibration'))
        var_path = var_path.format(calibration=calibration, dataset_group=DATASET_KEYS[ds_info['dataset_group']])
        return var_path

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

    def get_dataset(self, dataset_id, ds_info):
        """Get the dataset corresponding to *dataset_id*.

        The size of the return DataArray will be dependent on the number of
        scans actually sensed of course.

        """
        dataset_group = [ds_group for ds_group in ds_info['dataset_groups'] if ds_group in self.datasets]
        if not dataset_group:
            return

        dataset_group = dataset_group[0]
        ds_info['dataset_group'] = dataset_group
        var_path = self._generate_file_key(dataset_id, ds_info)
        factor_var_path = ds_info.get("factors_key", var_path + "Factors")

        scan_size = 1
        scans = self._get_scans_per_granule(dataset_group)
        start_scan = 0
        data_chunks = []
        scans = xr.DataArray(scans)

        ch_index = self._get_atms_channel_index(ds_info['name'])
        if ch_index is not None:
            variable = self[var_path][:, :, ch_index]
        else:
            variable = self[var_path]

        # check if these are single per-granule value
        if variable.size != scans.size:
            for gscans in scans.values:
                data_chunks.append(variable.isel(y=slice(start_scan,
                                                         start_scan + gscans * scan_size)))
                start_scan += gscans * scan_size
            data = xr.concat(data_chunks, 'y')
        else:
            data = self.expand_single_values(variable, scans)

        data = self.mask_fill_values(data, ds_info)
        file_units = _get_file_units(dataset_id, ds_info)
        output_units = ds_info.get("units", file_units)
        factors = self._get_scaling_factors(file_units, output_units, factor_var_path, ch_index)

        if factors is not None:
            data = self.scale_swath_data(data, factors, dataset_group)
        else:
            LOG.debug("No scaling factors found for %s", dataset_id)

        i = getattr(data, 'attrs', {})
        i.update(ds_info)
        i.update({
            "units": output_units,
            "platform_name": self.platform_name,
            "sensor": self.sensor_name,
            "start_orbit": self.start_orbit_number,
            "end_orbit": self.end_orbit_number,
            "rows_per_scan": self._scan_size(dataset_group),
        })
        i.update(dataset_id.to_dict())
        data.attrs.update(i)
        return data
