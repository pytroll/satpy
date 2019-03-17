from satpy.readers.netcdf_utils import NetCDF4FileHandler
from satpy.readers.file_handlers import BaseFileHandler
import os
import dask.dataframe as dd
import xarray as xr


class VIIRSActiveFiresFileHandler(NetCDF4FileHandler):

    def get_dataset(self, dsid, dsinfo):
        data = self[dsinfo.get('file_key', dsid.name)]
        data.attrs.update(dsinfo)

        data.attrs["platform_name"] = self['/attr/satellite_name']
        data.attrs["sensor"] = self['/attr/instrument_name']

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
    def __init__(self, filename, filename_info, filetype_info):
        super(VIIRSActiveFiresTextFileHandler, self).__init__(filename, filename_info, filetype_info)

        self.file_content = dd.read_csv(filename, skiprows=15, header=None,
                                        names=["latitude", "longitude",
                                               "T13", "Along-scan", "Along-track", "detection_confidence",
                                               "power"])

    def get_dataset(self, dsid, dsinfo):
        ds = self[dsid.name].to_dask_array(lengths=True)
        data_array = xr.DataArray(ds, dims=("y",), attrs={"platform_name": "unknown", "sensor": "viirs"})
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
