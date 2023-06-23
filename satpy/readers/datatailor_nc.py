# Copyright (c) 2023 Satpy developers
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
"""Generic reader for simplified NetCDF files from EUMETSAT Data Tailor."""

from .netcdf_utils import NetCDF4FsspecFileHandler


class DataTailorNC(NetCDF4FsspecFileHandler):
    """Reader for EUMETSAT Data Tailor simplified NetCDF output.

    The EUMETSAT Data Tailor is a tool published by EUMETSAT that allows tailoring
    satellite data to and from various file formats, among other things.
    This is a generic reader for the simplified NetCDF output format.
    For more information on the data tailor, see https://www.eumetsat.int/data-tailor

    This reader is experimental and might not yet work with all simplified
    NetCDF outputs from the data tailor.
    """

    # NB: I hope this is reasonably generic across Data Tailor products...
    _trans = {"along_track": "y", "across_track": "x"}
    _coord_dims = {"along_track", "across_track"}
    _coord_vars = ["lon", "lat"]

    def get_dataset(self, data_id, ds_info):
        """Obtain dataset."""
        da = self[data_id["name"]]
        da = da.rename({k: v for (k, v) in self._trans.items() if k in da.dims})
        return da

    def available_datasets(self, configured_datasets=None):
        """Get available datasets based on what's in the file.

        Returns all datasets in the root group.
        """
        yield from super().available_datasets(configured_datasets)
        common = {"file_type": "datatailor_simple_nc"}
        for key in self.file_content:
            if "/" in key:  # not a dataset
                continue
            # coordinates missing from metadata, add here
            yield (True, {"name": key} | common | self[key].attrs |
                         {"coordinates": self._get_coords(key)})

    def _get_coords(self, key):
        """Guess coordinates for key.

        There are no coordinates specified in the EUMETSAT Data Tailor
        simplified NetCDF files, but Satpy needs coordinates to add the
        SwathDefinition.  Make an educated guess.
        """
        if set(self[key].dims) & self._coord_dims:
            return self._coord_vars
        return []
