"""Reader for MODIS L2 products written with HDF4 files.

Documentation about the format
https://modis-atmos.gsfc.nasa.gov/products

"""
import logging

import numpy as np
import xarray as xr


from satpy import CHUNK_SIZE
from satpy.readers.hdfeos_base import HDFEOSGeoReader
from satpy.readers.hdf4_utils import from_sds

logger = logging.getLogger(__name__)


class ModisL2HDFFileHandler(HDFEOSGeoReader):

    def _select_hdf_dataset(self, hdf_dataset_name, byte_dimension):
        """Load a dataset from HDF EOS file whose information is contained in a byte in
        the given dimension.

        """
        hdf_dataset = self.sd.select(hdf_dataset_name)
        if byte_dimension == 0:
            dataset = xr.DataArray(from_sds(hdf_dataset, chunks=CHUNK_SIZE),
                                   dims=['i', 'y', 'x']).astype(np.uint8)
        elif byte_dimension == 2:
            dataset = xr.DataArray(from_sds(hdf_dataset, chunks=CHUNK_SIZE),
                                   dims=['y', 'x', 'i']).astype(np.uint8)
            # Reorder dimensions for consistency
            dataset = dataset.transpose('i', 'y', 'x')

        return dataset

    def _parse_resolution_info(self, info, resolution):
        if isinstance(info, list):
            if len(info) == 1 and isinstance(info[0], int):
                return info[0]
            # Check if the values are stored in a with resolution as a key
            if isinstance(info[0], dict):
                for elem in info:
                    try:
                        return elem[resolution]
                    except KeyError:
                        pass
            # The information doesn't concern the current resolution
            return None
        return info

    def get_dataset(self, dataset_id, dataset_info):

        dataset_name = dataset_id.name
        if dataset_name in HDFEOSGeoReader.DATASET_NAMES:
            return HDFEOSGeoReader.get_dataset(self, dataset_id, dataset_info)
        dataset_name_in_file = dataset_info['file_key']

        # The dataset asked correspond to a given set of bits of the HDF EOS dataset
        if 'byte' in dataset_info and 'byte_dimension' in dataset_info:
            byte_dimension = dataset_info['byte_dimension']  # Where the information is stored
            dataset = self._select_hdf_dataset(dataset_name_in_file, byte_dimension)

            byte_information = self._parse_resolution_info(dataset_info['byte'], dataset_id.resolution)
            # At which bit starts the information
            bit_start = self._parse_resolution_info(dataset_info['bit_start'], dataset_id.resolution)
            # How many bits store the information
            bit_count = self._parse_resolution_info(dataset_info['bit_count'], dataset_id.resolution)

            # Only one byte: select the byte information
            if isinstance(byte_information, int):
                byte_dataset = dataset[byte_information, :, :]

            # Two bytes: recombine the two bytes
            elif isinstance(byte_information, list) and len(byte_information) == 2:
                # We recombine the two bytes
                dataset_a = dataset[byte_information[0], :, :]
                dataset_b = dataset[byte_information[1], :, :]
                dataset_a = np.uint16(dataset_a)
                dataset_a = np.left_shift(dataset_a, 8)  # dataset_a << 8
                byte_dataset = np.bitwise_or(dataset_a, dataset_b).astype(np.uint16)
                shape = byte_dataset.shape
                # We replicate the concatenated byte with the right shape
                byte_dataset = np.repeat(np.repeat(byte_dataset, 4, axis=0), 4, axis=1)
                # All bits carry information, we update bit_start consequently
                bit_start = np.arange(16, dtype=np.uint16).reshape((4, 4))
                bit_start = np.tile(bit_start, (shape[0], shape[1]))

            # Compute the final bit mask
            dataset = bits_strip(bit_start, bit_count, byte_dataset)

            # Apply quality assurance filter
            if 'quality_assurance' in dataset_info:
                quality_assurance_required = self._parse_resolution_info(
                    dataset_info['quality_assurance'], dataset_id.resolution
                )
                if quality_assurance_required is True:
                    # Get quality assurance dataset recursively
                    from satpy import DatasetID
                    quality_assurance_dataset_id = DatasetID(
                        name='quality_assurance', resolution=1000
                    )
                    quality_assurance_dataset_info = {
                        'name': 'quality_assurance',
                        'resolution': [1000],
                        'byte_dimension': 2,
                        'byte': [0],
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
                    dataset[np.where(quality_assurance == 0)] = np.NaN

        # No byte manipulation required
        else:
            dataset = self.load_dataset(dataset_name)

        return dataset


def bits_strip(bit_start, bit_count, value):
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
