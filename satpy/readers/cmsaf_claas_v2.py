"""Module containing CMSAF FileHandler
"""

import pyresample.geometry
from .netcdf_utils import NetCDF4FileHandler


class Claasv2(NetCDF4FileHandler):
    def __init__(self, *args, **kwargs):
        if "cache_handle" in kwargs:
            raise TypeError(
                f"Do not pass cache_handle to {self.__class__.__name__:s} "
                "constructor please.  It must always be True.")
        super().__init__(*args, **kwargs, cache_handle=True)

    def available_datasets(self, configured_datasets=None):
        # see
        # https://satpy.readthedocs.io/en/latest/api/satpy.readers.html#satpy.readers.file_handlers.BaseFileHandler.available_datasets

        # this method should work for any NetCDF file, should it be somewhere
        # more generically available?  Perhaps in the `NetCDF4FileHandler`?

        yield from super().available_datasets(configured_datasets)
        # FIXME: instead of accessing self.file_handle, this should probably
        # use # self.file_content or something similar
        it = self.file_handle.variables.items()
        for (k, v) in it:
            if "y" not in v.dimensions:
                continue
            ds_info = {"name": k,
                       "file_type": self.filetype_info["file_type"]}
            attrs = v.__dict__.copy()
            # we don't need "special" attributes in our metadata here
            for unkey in {"_FillValue", "add_offset", "scale_factor"}:
                attrs.pop(unkey, None)
            ds_info.update(attrs)
            yield (True, ds_info)

    def get_dataset(self, dataset_id, info):
        # FIXME: This needs dimensions (x, y)
        #
        # FIXME: set start_time, end_time
        return self[dataset_id.name]

    def get_area_def(self, dataset_id):
        # FIXME: use `from_cf` in
        # https://github.com/pytroll/pyresample/pull/271 ?
        return pyresample.geometry.AreaDefinition(
                "some_area_name",
                "on-the-fly area",
                "geos",
                self["/attr/CMSAF_proj4_params"],
                self["/dimension/x"],
                self["/dimension/y"],
                self["/attr/CMSAF_area_extent"])
