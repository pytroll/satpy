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

"""MTG Lightning Imager (LI) Level-2 (L2) unified reader.

This reader supports reading all the products from the LI L2
processing level:

Point products:
  * L2-LE Lightning Events
  * L2-LEF Lightning Events Filtered
  * L2-LFL Lightning Flashes
  * L2-LGR Lightning Groups
Accumulated products:
  * L2-AF Accumulated Flashes
  * L2-AFA Accumulated Flash Area
  * L2-AFR Accumulated Flash Radiance

Per default, the unified LI L2 reader returns the data either as an 1-D array
or as a 2-D array depending on the product type.

Point-based products (LE, LEF, LFL, LGR) are "classic" lightning products
consisting of values with attached latitude and longitude coordinates.
Hence, these products are provided by the reader as 1-D arrays,
with a ``pyresample.geometry.SwathDefinition`` area
attribute containing the points lat-lon coordinates.

Accumulated products (AF, AFA, AFR) are the result of temporal accumulation
of events (e.g. over 30 seconds), and are gridded in the FCI 2km geostationary
projection grid, in order to facilitate the synergistic usage together with FCI.
Compared to the point products, the gridded products also give information
about the spatial extent of the lightning activity.
Hence, these products are provided by the reader as 2-D arrays in the FCI 2km
grid as per intended usage, with a ``pyresample.geometry.AreaDefinition`` area
attribute containing the grid geolocation information.
In this way, the products can directly be overlaid to FCI data.

.. note::

    L2 accumulated products retrieved from the archive
    (that have "ARC" in the filename) contain data for 20 repeat cycles (timesteps) covering
    10 minutes of sensing time. For these files, when loading the main variables
    (``accumulated_flash_area``, ``flash_accumulation``, ``flash_radiance``),
    the reader will cumulate (sum up) the data for the entire sensing period of the file.
    A solution to access easily each timestep is being worked on. See https://github.com/pytroll/satpy/issues/2878
    for possible workarounds in the meanwhile.


If needed, the accumulated products can also be accessed as 1-d array by
setting the reader kwarg ``with_area_definition=False``,
e.g.::

  scn = Scene(filenames=filenames, reader="li_l2_nc", reader_kwargs={'with_area_definition': False})

For both 1-d and 2-d products, the lat-lon coordinates of the points/grid pixels
can be accessed using e.g.
``scn['dataset_name'].attrs['area'].get_lonlats()``.

See the LI L2 Product User Guide `PUG`_ for more information.

.. _PUG: https://www-dr.eumetsat.int/media/49348

"""

import logging

import dask.array as da
import numpy as np
import xarray as xr

from satpy.readers.li_base_nc import LINCFileHandler
from satpy.resample import get_area_def
from satpy.utils import get_legacy_chunk_size

logger = logging.getLogger(__name__)
LI_GRID_SHAPE = (5568, 5568)
CHUNK_SIZE = get_legacy_chunk_size()


class LIL2NCFileHandler(LINCFileHandler):
    """Implementation class for the unified LI L2 satpy reader."""

    def __init__(self, filename, filename_info, filetype_info, with_area_definition=True):
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
            return get_area_def("mtg_fci_fdss_2km")

        raise NotImplementedError("Area definition is not supported for non-accumulated products.")

    def is_var_with_swath_coord(self, dsid):
        """Check if the variable corresponding to this dataset is listed as variable with swath coordinates."""
        # since the patterns are compiled to regex we use the search() method below to find matches
        with_swath_coords = any([p.search(dsid["name"]) is not None for p in self.swath_coordinates["patterns"]])
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

        rows = self.get_measured_variable("y")
        cols = self.get_measured_variable("x")
        attrs = data_array.attrs

        rows, cols = da.compute(rows, cols)

        # origin is in the south-west corner, so we flip the rows (applying
        # offset of 1 implicitly)
        # And we manually offset the columns by 1 too:
        rows = (LI_GRID_SHAPE[0] - rows.astype(int))
        cols = cols.astype(int) - 1

        # initialise results array with zeros
        data_2d = da.zeros((LI_GRID_SHAPE[0], LI_GRID_SHAPE[1]), dtype=data_array.dtype,
                           chunks=(LI_GRID_SHAPE[0], LI_GRID_SHAPE[1]))

        # insert the data. If a pixel has more than one entry, the values are added up (np.add.at functionality)
        data_2d = da.map_blocks(_np_add_at_wrapper, data_2d, (rows, cols), data_array,
                                dtype=data_array.dtype,
                                chunks=(LI_GRID_SHAPE[0], LI_GRID_SHAPE[1]))
        data_2d = da.where(data_2d > 0, data_2d, np.nan)

        xarr = xr.DataArray(da.asarray(data_2d, CHUNK_SIZE), dims=("y", "x"))
        xarr.attrs = attrs

        return xarr


def _np_add_at_wrapper(target_array, indices, data):
    # copy needed for correct computation in-place
    ta = target_array.copy()
    # add.at is not implemented in xarray, so we explicitly need the np.array
    np.add.at(ta, indices, data.values)
    return ta
