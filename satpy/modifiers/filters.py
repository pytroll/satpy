"""Tests for image filters."""
import logging

import xarray as xr

from satpy.modifiers import ModifierBase

logger = logging.getLogger(__name__)


class Median(ModifierBase):
    """Apply a median filter to the band."""

    def __init__(self, median_filter_params, **kwargs):
        """Create the instance.

        Args:
            median_filter_params: The arguments to pass to dask-image's median_filter function. For example, {size: 3}
                                  makes give the median filter a kernel of size 3.

        """
        self.median_filter_params = median_filter_params
        super().__init__(**kwargs)

    def __call__(self, arrays, **info):
        """Get the median filtered band."""
        from dask_image.ndfilters import median_filter

        data = arrays[0]
        logger.debug(f"Apply median filtering with parameters {self.median_filter_params}.")
        res = xr.DataArray(median_filter(data.data, **self.median_filter_params),
                           dims=data.dims, attrs=data.attrs, coords=data.coords)
        self.apply_modifier_info(data, res)
        return res
