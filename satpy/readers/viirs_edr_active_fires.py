# Copyright (c) 2019 Satpy Developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
"""VIIRS Active Fires reader
*************************

This module implements readers for VIIRS Active Fires NetCDF and
ASCII files.
"""

from satpy.readers.netcdf_utils import NetCDF4FileHandler
from satpy.readers.file_handlers import BaseFileHandler
import dask.dataframe as dd
import xarray as xr


class VIIRSActiveFiresFileHandler(NetCDF4FileHandler):
    """NetCDF4 reader for VIIRS Active Fires
    """

    def __init__(self, filename, filename_info, filetype_info,
                 auto_maskandscale=False, xarray_kwargs=None):
        super(VIIRSActiveFiresFileHandler, self).__init__(
            filename, filename_info, filetype_info)
        self.prefix = filetype_info.get('variable_prefix')

    def get_dataset(self, dsid, dsinfo):
        """Get dataset function

        Args:
            dsid: Dataset ID
            param2: Dataset Information

        Returns:
            Dask DataArray: Data

        """

        key = dsinfo.get('file_key', dsid.name).format(variable_prefix=self.prefix)
        data = self[key]
        data.attrs.update(dsinfo)

        platform_key = {"NPP": "Suomi-NPP", "J01": "NOAA-20", "J02": "NOAA-21"}

        data.attrs["platform_name"] = platform_key.get(self.filename_info['satellite_name'].upper(), "unknown")
        data.attrs["sensor"] = "VIIRS"

        return data

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    @property
    def sensor_name(self):
        return self["sensor"]

    @property
    def platform_name(self):
        return self["platform_name"]


class VIIRSActiveFiresTextFileHandler(BaseFileHandler):
    """ASCII reader for VIIRS Active Fires
    """
    def __init__(self, filename, filename_info, filetype_info):
        """Makes sure filepath is valid and then reads data into a Dask DataFrame

        Args:
            filename: Filename
            filename_info: Filename information
            filetype_info: Filetype information
        """

        if filetype_info.get('file_type') == 'fires_text_img':
            self.file_content = dd.read_csv(filename, skiprows=15, header=None,
                                            names=["latitude", "longitude",
                                                   "T4", "Along-scan", "Along-track", "confidence_cat",
                                                   "power"])
        else:
            self.file_content = dd.read_csv(filename, skiprows=15, header=None,
                                            names=["latitude", "longitude",
                                                   "T13", "Along-scan", "Along-track", "confidence_pct",
                                                   "power"])

        super(VIIRSActiveFiresTextFileHandler, self).__init__(filename, filename_info, filetype_info)

        platform_key = {"NPP": "Suomi-NPP", "J01": "NOAA-20", "J02": "NOAA-21"}

        self.platform_name = platform_key.get(self.filename_info['satellite_name'].upper(), "unknown")

    def get_dataset(self, dsid, dsinfo):
        ds = self[dsid.name].to_dask_array(lengths=True)
        data_array = xr.DataArray(ds, dims=("y",), attrs={"platform_name": self.platform_name, "sensor": "VIIRS"})
        data_array.attrs.update(dsinfo)
        return data_array

    @property
    def start_time(self):
        return self.filename_info['start_time']

    @property
    def end_time(self):
        return self.filename_info.get('end_time', self.start_time)

    def __getitem__(self, key):
        return self.file_content[key]

    def __contains__(self, item):
        return item in self.file_content
