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
"""Modis level 2 hdf-eos format reader.

Introduction
------------

The ``modis_l2`` reader reads and calibrates Modis L2 image data in hdf-eos format.
Since there are a multitude of different level 2 datasets not all of theses are implemented (yet).


Currently the reader supports:
    - m[o/y]d35_l2: cloud_mask dataset
    - some datasets in m[o/y]d06 files

To get a list of the available datasets for a given file refer to the "Load data" section in :doc:`../readers`.


Geolocation files
-----------------

Similar to the ``modis_l1b`` reader the geolocation files (mod03) for the 1km data are optional and if not
given 1km geolocations will be interpolated from the 5km geolocation contained within the file.

For the 500m and 250m data geolocation files are needed.


References:
    - Documentation about the format: https://modis-atmos.gsfc.nasa.gov/products

"""
import logging

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.hdf4_utils import from_sds
from satpy.readers.hdfeos_base import HDFEOSGeoReader

logger = logging.getLogger(__name__)


class ModisL2HDFFileHandler(HDFEOSGeoReader):
    """File handler for MODIS HDF-EOS Level 2 files.

    Includes error handling for files produced by IMAPP produced files.

    """

    def _load_all_metadata_attributes(self):
        try:
            return super()._load_all_metadata_attributes()
        except KeyError:
            return {}

    @property
    def is_imapp_mask_byte1(self):
        """Get if this file is the IMAPP 'mask_byte1' file type."""
        return "mask_byte1" in self.filetype_info["file_type"]

    @property
    def start_time(self):
        """Get the start time of the dataset."""
        try:
            return super().start_time
        except KeyError:
            try:
                return self.filename_info["start_time"]
            except KeyError:
                return self.filename_info["acquisition_time"]

    @property
    def end_time(self):
        """Get the end time of the dataset."""
        try:
            return super().end_time
        except KeyError:
            return self.start_time

    @staticmethod
    def read_geo_resolution(metadata):
        """Parse metadata to find the geolocation resolution.

        It is implemented as a staticmethod to match read_mda pattern.

        """
        try:
            return HDFEOSGeoReader.read_geo_resolution(metadata)
        except RuntimeError:
            # most L2 products are 5000m
            return 5000

    def _select_hdf_dataset(self, hdf_dataset_name, byte_dimension):
        """Load a dataset from HDF-EOS level 2 file."""
        dataset = self.sd.select(hdf_dataset_name)
        dask_arr = from_sds(dataset, chunks=CHUNK_SIZE)
        attrs = dataset.attributes()
        dims = ['y', 'x']
        if byte_dimension == 0:
            dims = ['i', 'y', 'x']
            dask_arr = dask_arr.astype(np.uint8)
        elif byte_dimension == 2:
            dims = ['y', 'x', 'i']
            dask_arr = dask_arr.astype(np.uint8)
        dataset = xr.DataArray(dask_arr, dims=dims, attrs=attrs)
        if 'i' in dataset.dims:
            # Reorder dimensions for consistency
            dataset = dataset.transpose('i', 'y', 'x')
        return dataset

    def get_dataset(self, dataset_id, dataset_info):
        """Get DataArray for specified dataset."""
        dataset_name = dataset_id['name']
        if self.is_geo_loadable_dataset(dataset_name):
            return HDFEOSGeoReader.get_dataset(self, dataset_id, dataset_info)
        dataset_name_in_file = dataset_info['file_key']
        if self.is_imapp_mask_byte1:
            dataset_name_in_file = dataset_info.get('imapp_file_key', dataset_name_in_file)

        # The dataset asked correspond to a given set of bits of the HDF EOS dataset
        if 'byte' in dataset_info and 'byte_dimension' in dataset_info:
            dataset = self._extract_and_mask_category_dataset(dataset_id, dataset_info, dataset_name_in_file)
        else:
            # No byte manipulation required
            dataset = self.load_dataset(dataset_name_in_file, dataset_info.pop("category", False))

        self._add_satpy_metadata(dataset_id, dataset)
        return dataset

    def _extract_and_mask_category_dataset(self, dataset_id, dataset_info, var_name):
        # what dimension is per-byte
        byte_dimension = None if self.is_imapp_mask_byte1 else dataset_info['byte_dimension']
        dataset = self._select_hdf_dataset(var_name, byte_dimension)
        # category products always have factor=1/offset=0 so don't apply them
        # also remove them so they don't screw up future satpy processing
        dataset.attrs.pop('scale_factor', None)
        dataset.attrs.pop('add_offset', None)
        # Don't do this byte work if we are using the IMAPP mask_byte1 file
        if self.is_imapp_mask_byte1:
            return dataset

        dataset = _extract_byte_mask(dataset,
                                     dataset_info['byte'],
                                     dataset_info['bit_start'],
                                     dataset_info['bit_count'])
        dataset = self._mask_with_quality_assurance_if_needed(dataset, dataset_info, dataset_id)
        return dataset

    def _mask_with_quality_assurance_if_needed(self, dataset, dataset_info, dataset_id):
        if not dataset_info.get('quality_assurance', False):
            return dataset

        # Get quality assurance dataset recursively
        quality_assurance_dataset_id = dataset_id.from_dict(
            dict(name='quality_assurance', resolution=1000)
        )
        quality_assurance_dataset_info = {
            'name': 'quality_assurance',
            'resolution': 1000,
            'byte_dimension': 2,
            'byte': 0,
            'bit_start': 0,
            'bit_count': 1,
            'file_key': 'Quality_Assurance'
        }
        quality_assurance = self.get_dataset(
            quality_assurance_dataset_id, quality_assurance_dataset_info
        )
        # Duplicate quality assurance dataset to create relevant filter
        duplication_factor = [int(dataset_dim / quality_assurance_dim)
                              for dataset_dim, quality_assurance_dim
                              in zip(dataset.shape, quality_assurance.shape)]
        quality_assurance = np.tile(quality_assurance, duplication_factor)
        # Replace unassured data by NaN value
        dataset = dataset.where(quality_assurance != 0, dataset.attrs["_FillValue"])
        return dataset


def _extract_byte_mask(dataset, byte_information, bit_start, bit_count):
    attrs = dataset.attrs.copy()

    if isinstance(byte_information, int):
        # Only one byte: select the byte information
        byte_dataset = dataset[byte_information, :, :]
        dataset = _bits_strip(bit_start, bit_count, byte_dataset)
    elif isinstance(byte_information, (list, tuple)) and len(byte_information) == 2:
        # Two bytes: recombine the two bytes
        byte_mask = da.map_blocks(
            _extract_two_byte_mask,
            dataset.data[byte_information[0]],
            dataset.data[byte_information[1]],
            bit_start=bit_start,
            bit_count=bit_count,
            dtype=np.uint16,
            meta=np.array((), dtype=np.uint16),
            chunks=tuple(tuple(chunk_size * 4 for chunk_size in dim_chunks) for dim_chunks in dataset.chunks[1:]),
        )
        dataset = xr.DataArray(byte_mask, dims=dataset.dims[1:])

    # Compute the final bit mask
    dataset.attrs = attrs
    return dataset


def _extract_two_byte_mask(data_a: np.ndarray, data_b: np.ndarray, bit_start: int, bit_count: int) -> np.ndarray:
    data_a = data_a.astype(np.uint16, copy=False)
    data_a = np.left_shift(data_a, 8)  # dataset_a << 8
    byte_dataset = np.bitwise_or(data_a, data_b).astype(np.uint16)
    shape = byte_dataset.shape
    # We replicate the concatenated byte with the right shape
    byte_dataset = np.repeat(np.repeat(byte_dataset, 4, axis=0), 4, axis=1)
    # All bits carry information, we update bit_start consequently
    bit_start = np.arange(16, dtype=np.uint16).reshape((4, 4))
    bit_start = np.tile(bit_start, (shape[0], shape[1]))
    return _bits_strip(bit_start, bit_count, byte_dataset)


def _bits_strip(bit_start, bit_count, value):
    """Extract specified bit from bit representation of integer value.

    Parameters
    ----------
    bit_start : int
        Starting index of the bits to extract (first bit has index 0)
    bit_count : int
        Number of bits starting from bit_start to extract
    value : int
        Number from which to extract the bits

    Returns
    -------
        int
        Value of the extracted bits

    """
    bit_mask = pow(2, bit_start + bit_count) - 1
    return np.right_shift(np.bitwise_and(value, bit_mask), bit_start)
