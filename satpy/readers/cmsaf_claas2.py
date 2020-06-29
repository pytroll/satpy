"""Module containing CMSAF CLAAS v2 FileHandler."""

import datetime
import pyresample.geometry
from .netcdf_utils import NetCDF4FileHandler


class CLAAS2(NetCDF4FileHandler):
    """Handle CMSAF CLAAS-2 files."""
    def __init__(self, *args, **kwargs):
        """Initialise class."""
        super().__init__(*args, **kwargs, cache_handle=False,
                         auto_maskandscale=True)

    @property
    def start_time(self):
        """Get start time from file."""

        # datetime module can't handle timezone identifier
        return datetime.datetime.fromisoformat(
                self["/attr/time_coverage_start"].rstrip("Z"))

    @property
    def end_time(self):
        """Get end time from file."""
        return datetime.datetime.fromisoformat(
                self["/attr/time_coverage_end"].rstrip("Z"))

    def available_datasets(self, configured_datasets=None):
        """Yield a collection of available datasets.

        Return a generator that will yield the datasets available in the loaded
        files.  See docstring in parent class for specification details.
        """

        # this method should work for any (CF-conform) NetCDF file, should it
        # be somewhere more generically available?  Perhaps in the
        # `NetCDF4FileHandler`?

        yield from super().available_datasets(configured_datasets)
        data_vars = [k for k in self.file_content
                     if k + "/dimensions" in self.file_content]
        for k in data_vars:
            # if it doesn't have a y-dimension we're not interested
            if "y" not in self.file_content[k + "/dimensions"]:
                continue
            ds_info = self._get_dsinfo(k)
            yield (True, ds_info)

    def _get_dsinfo(self, var):
        """Get metadata for variable.

        Return metadata dictionary for variable ``var``.
        """
        ds_info = {"name": var,
                   "file_type": self.filetype_info["file_type"]}
        # attributes for this data variable
        attrs = {k[len(f"{k:s}/attr")+1]: v
                 for (k, v) in self.file_content.items()
                 if k.startswith(f"{k:s}/attr")}
        # we don't need "special" attributes in our metadata here
        for unkey in {"_FillValue", "add_offset", "scale_factor"}:
            attrs.pop(unkey, None)
        return ds_info

    def get_dataset(self, dataset_id, info):
        ds = self[dataset_id.name]
        if "time" in ds.dims:
            return ds.squeeze(["time"])
        else:
            return ds

    def get_area_def(self, dataset_id):
        return pyresample.geometry.AreaDefinition(
                "some_area_name",
                "on-the-fly area",
                "geos",
                self["/attr/CMSAF_proj4_params"],
                self["/dimension/x"],
                self["/dimension/y"],
                self["/attr/CMSAF_area_extent"])
