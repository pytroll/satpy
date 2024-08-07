# Copyright (c) 2017-2023 Satpy developers
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

r"""IASI-NG L2 reader

This reader supports reading all the products from the IASI-NG L2 processing
level:
  * IASI-L2-TWV
  * IASI-L2-CLD
  * IASI-L2-GHG
  * IASI-L2-SFC
"""

import logging
import logging
import numpy as np
import xarray as xr


from .netcdf_utils import NetCDF4FsspecFileHandler

logger = logging.getLogger(__name__)


class IASINGL2NCFileHandler(NetCDF4FsspecFileHandler):
    """Reader for IASI-NG L2 products in NetCDF format."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Initialize object."""
        super().__init__(filename, filename_info, filetype_info, **kwargs)

        # logger.info("Creating reader with infos: %s", filename_info)

    # def get_dataset(self, data_id, ds_info):
    #     """Obtain dataset."""
    #     ds = self[data_id["name"]]
    #     if "scan_lines" in ds.dims:
    #         ds = ds.rename(scan_lines="y")
    #     if "pixels" in ds.dims:
    #         ds = ds.rename(pixels="x")
    #     if "_FillValue" in ds.attrs and ds.dtype.kind == "f":
    #         with xr.set_options(keep_attrs=True):
    #             # have to inverse the logic due to https://github.com/pydata/xarray/issues/7581
    #             return xr.where(ds != ds.attrs["_FillValue"], ds, np.nan)
    #     return ds

    # def available_datasets(self, configured_datasets=None):
    #     """Get available datasets based on what's in the file.

    #     Returns all datasets in the root group.
    #     """
    #     yield from super().available_datasets(configured_datasets)
    #     common = {"file_type": "iasi_l2_cdr_nc", "resolution": 12000}
    #     for key in self.file_content:
    #         if "/" in key:  # not a dataset
    #             continue
    #         yield (True, {"name": key} | common | self[key].attrs)
