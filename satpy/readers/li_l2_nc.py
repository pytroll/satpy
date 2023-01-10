# Copyright (c) 2022 Satpy developers
#
# satpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# satpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with satpy.  If not, see <http://www.gnu.org/licenses/>.

"""MTG Lighting Imager (LI) L2 unified reader.

This reader supports reading all the products from the LI L2
processing level:

  * L2-LE
  * L2-LGR
  * L2-AFA
  * L2-LEF
  * L2-LFL
  * L2-AF
  * L2-AFR

"""

import logging

import dask.array as da
import numpy as np
import xarray as xr

from satpy import CHUNK_SIZE
from satpy.readers.li_base_nc import LINCFileHandler
from satpy.resample import get_area_def

logger = logging.getLogger(__name__)
LI_GRID_SHAPE = (5568, 5568)


class LIL2NCFileHandler(LINCFileHandler):
    """Implementation class for the unified LI L2 satpy reader."""

    def __init__(self, filename, filename_info, filetype_info, with_area_definition=False):
        """Initialize LIL2NCFileHandler."""
        super(LIL2NCFileHandler, self).__init__(filename, filename_info, filetype_info)

        if with_area_definition and not self.prod_in_accumulation_grid:
            logger.debug(f"The current product {filetype_info['file_desc']['product_type']} "
                         f"is not an accumulated product so it will not be regridded.")
            self.with_area_def = False
        else:
            self.with_area_def = with_area_definition

    def get_dataset(self, dataset_id, ds_info=None):
        """Get the dataset and apply gridding if requested."""
        data_array = super().get_dataset(dataset_id, ds_info)
        # variable_patterns are compiled to regex patterns
        # hence search variable name from swath_coordinate
        var_with_swath_coord = self.is_var_with_swath_coord(dataset_id)
        if var_with_swath_coord and self.with_area_def:
            data_array = self.get_array_on_fci_grid(data_array)
        return data_array

    def get_area_def(self, dsid):
        """Compute area definition for a dataset, only supported for accumulated products."""
        var_with_swath_coord = self.is_var_with_swath_coord(dsid)
        if var_with_swath_coord and self.with_area_def:
            return get_area_def('mtg_fci_fdss_2km')

        raise NotImplementedError('Area definition is not supported for accumulated products.')

    def is_var_with_swath_coord(self, dsid):
        """Check if the variable corresponding to this dataset is listed as variable with swath coordinates."""
        # since the patterns are compiled to regex we use the search() method below to find matches
        with_swath_coords = any([p.search(dsid['name']) is not None for p in self.swath_coordinates['patterns']])
        return with_swath_coords

    def get_array_on_fci_grid(self, data_array: xr.DataArray):
        """Obtain the accumulated products as a (sparse) 2-d array.

        The array has the shape of the FCI 2 km grid (5568x5568px),
        and will have an AreaDefinition attached.
        """
        # Integer values without the application of scale_factor and add_offset
        # hence no projection/index calculation.
        # Note that x and y have origin in the south-west corner of the image
        # and start with index 1.

        rows = self.get_measured_variable('y')
        cols = self.get_measured_variable('x')
        attrs = data_array.attrs

        rows, cols = da.compute(rows, cols)

        # origin is in the south-west corner, so we flip the rows (applying
        # offset of 1 implicitly)
        # And we manually offset the columns by 1 too:
        rows = (LI_GRID_SHAPE[0] - rows.astype(int))
        cols = cols.astype(int) - 1

        # Create an empyt 1-D array for the results
        flattened_result = np.nan * da.zeros((LI_GRID_SHAPE[0] * LI_GRID_SHAPE[1]), dtype=data_array.dtype)
        # Insert the data. Dask doesn't support this for more than one dimension at a time, so ...
        flattened_result[rows * LI_GRID_SHAPE[0] + cols] = data_array
        # ... reshape to final 2D grid
        data_2d = da.reshape(flattened_result, LI_GRID_SHAPE)
        xarr = xr.DataArray(da.asarray(data_2d, CHUNK_SIZE), dims=('y', 'x'))
        xarr.attrs = attrs

        return xarr
