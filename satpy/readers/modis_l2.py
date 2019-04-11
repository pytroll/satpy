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

    def get_dataset(self, dataset_id, dataset_info):

        dataset_name = dataset_id.name
        if dataset_name in HDFEOSGeoReader.DATASET_NAMES:
            return HDFEOSGeoReader.get_dataset(self, dataset_id, dataset_info)
        dataset_name_in_file = dataset_info['file_key']

        # The dataset asked correspond to a given set of bits of the HDF EOS dataset
        if 'byte' in dataset_info and 'byte_dimension' in dataset_info:
            byte_dimension = dataset_info['byte_dimension']  # Where the information is stored
            dataset = self._select_hdf_dataset(dataset_name_in_file, byte_dimension)

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

            bit_start = dataset_info['bits'][0]
            bit_count = dataset_info['bits'][1]
            dataset = bits_strip(bit_start, bit_count, dataset[dataset_info['byte'], :, :])
        else:
            dataset = self.load_dataset(dataset_name)

        return dataset
