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
import os

import dask.array as da
import h5py
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.viirs_atms_sdr_base import DATASET_KEYS, JPSS_SDR_FileHandler

LOG = logging.getLogger(__name__)

ATMS_CHANNEL_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                      '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']


class ATMS_SDR_FileHandler(JPSS_SDR_FileHandler):
    """ATMS SDR HDF5 File Reader."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize file handler."""
        self.datasets = os.path.basename(filename).split('_')[0].split('-')
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

    def _get_scans_per_granule(self, dataset_group):
        number_of_granules_path = 'Data_Products/{dataset_group}/{dataset_group}_Aggr/attr/AggregateNumberGranules'
        nb_granules_path = number_of_granules_path.format(dataset_group=DATASET_KEYS[dataset_group])
        scans = []
        for granule in range(self[nb_granules_path]):
            scans_path = 'Data_Products/{dataset_group}/{dataset_group}_Gran_{granule}/attr/N_Number_Of_Scans'
            scans_path = scans_path.format(dataset_group=DATASET_KEYS[dataset_group], granule=granule)
            scans.append(self[scans_path])
        return scans

    def _get_variable(self, var_path, channel_index=None):
        if channel_index is not None:
            return self[var_path][:, :, channel_index]
        return super()._get_variable(var_path)

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

        ch_index = self._get_atms_channel_index(ds_info['name'])
        data = self.concatenate_dataset(dataset_group, var_path, channel_index=ch_index)
        data = self.mask_fill_values(data, ds_info)

        data = self.scale_data_to_specified_unit(data, dataset_id, ds_info)
        data = self._update_data_attributes(data, dataset_id, ds_info)

        return data
