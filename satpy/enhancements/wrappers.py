# Copyright (c) 2017-2025 Satpy developers
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
"""Context managers for enhancements."""

import logging
from functools import wraps

import dask.array as da
import numpy as np
import xarray as xr

LOG = logging.getLogger(__name__)

def exclude_alpha(func):
    """Exclude the alpha channel from the DataArray before further processing."""

    @wraps(func)
    def wrapper(data, **kwargs):
        bands = data.coords["bands"].values
        exclude = ["A"] if "A" in bands else []
        band_data = data.sel(bands=[b for b in bands
                                    if b not in exclude])
        band_data = func(band_data, **kwargs)

        attrs = data.attrs
        attrs.update(band_data.attrs)
        # combine the new data with the excluded data
        new_data = xr.concat([band_data, data.sel(bands=exclude)],
                             dim="bands")
        data.data = new_data.sel(bands=bands).data
        data.attrs = attrs
        return data

    return wrapper


def on_separate_bands(func):
    """Apply `func` one band of the DataArray at a time.

    If this decorator is to be applied along with `on_dask_array`, this decorator has to be applied first, eg::

        @on_separate_bands
        @on_dask_array
        def my_enhancement_function(data):
            ...


    """

    @wraps(func)
    def wrapper(data, **kwargs):
        attrs = data.attrs
        data_arrs = []
        for idx, band in enumerate(data.coords["bands"].values):
            band_data = func(data.sel(bands=[band]), index=idx, **kwargs)
            data_arrs.append(band_data)
            # we assume that the func can add attrs
            attrs.update(band_data.attrs)
        data.data = xr.concat(data_arrs, dim="bands").data
        data.attrs = attrs
        return data

    return wrapper


def on_dask_array(func):
    """Pass the underlying dask array to *func* instead of the xarray.DataArray."""

    @wraps(func)
    def wrapper(data, **kwargs):
        dims = data.dims
        coords = data.coords
        d_arr = func(data.data, **kwargs)
        return xr.DataArray(d_arr, dims=dims, coords=coords)

    return wrapper


def using_map_blocks(func):
    """Run the provided function using :func:`dask.array.map_blocks`.

    This means dask will call the provided function with a single chunk
    as a numpy array.
    """

    @wraps(func)
    def wrapper(data, **kwargs):
        return da.map_blocks(func, data, meta=np.array((), dtype=data.dtype), dtype=data.dtype, chunks=data.chunks,
                             **kwargs)

    return on_dask_array(wrapper)
