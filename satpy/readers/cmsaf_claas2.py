"""Module containing CMSAF CLAAS v2 FileHandler."""

import datetime

from satpy.resample import get_area_def

from .netcdf_utils import NetCDF4FileHandler


def _is_georef_offset_present(date):
    # Reference: Product User Manual, section 3.
    # https://doi.org/10.5676/EUM_SAF_CM/CLAAS/V002_01
    return date < datetime.date(2017, 12, 6)


def _adjust_area_to_match_shifted_data(area):
    # Reference:
    # https://github.com/pytroll/satpy/wiki/SEVIRI-georeferencing-offset-correction
    offset = area.pixel_size_x / 2
    llx, lly, urx, ury = area.area_extent
    new_extent = [llx + offset, lly - offset, urx + offset, ury - offset]
    return area.copy(area_extent=new_extent)


FULL_DISK = get_area_def("msg_seviri_fes_3km")
FULL_DISK_WITH_OFFSET = _adjust_area_to_match_shifted_data(FULL_DISK)


class CLAAS2(NetCDF4FileHandler):
    """Handle CMSAF CLAAS-2 files."""

    grid_size = 3636

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
        """Get the dataset."""
        ds = self[dataset_id['name']]
        if "time" in ds.dims:
            return ds.squeeze(["time"])

        return ds

    def get_area_def(self, dataset_id):
        """Get the area definition."""
        return self._get_subset_of_full_disk()

    def _get_subset_of_full_disk(self):
        """Get subset of the full disk.

        CLAAS products are provided on a grid that is slightly smaller
        than the full disk (excludes most of the space pixels).
        """
        full_disk = self._get_full_disk()
        offset = int((full_disk.width - self.grid_size) // 2)
        return full_disk[offset:-offset, offset:-offset]

    def _get_full_disk(self):
        if _is_georef_offset_present(self.start_time.date()):
            return FULL_DISK_WITH_OFFSET
        return FULL_DISK
