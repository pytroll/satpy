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
"""CF encoding."""
import logging

import numpy as np
import xarray as xr
from xarray.coding.times import CFDatetimeCoder

logger = logging.getLogger(__name__)


def _set_default_chunks(encoding, dataset):
    """Update encoding to preserve current dask chunks.

    Existing user-defined chunks take precedence.
    """
    for var_name, variable in dataset.variables.items():
        if variable.chunks:
            chunks = tuple(
                np.stack([variable.data.chunksize,
                          variable.shape]).min(axis=0)
            )  # Chunksize may not exceed shape
            encoding.setdefault(var_name, {})
            encoding[var_name].setdefault("chunksizes", chunks)
    return encoding


def _set_default_fill_value(encoding, dataset):
    """Set default fill values.

    Avoid _FillValue attribute being added to coordinate variables
    (https://github.com/pydata/xarray/issues/1865).
    """
    coord_vars = []
    for data_array in dataset.values():
        coord_vars.extend(set(data_array.dims).intersection(data_array.coords))
    for coord_var in coord_vars:
        encoding.setdefault(coord_var, {})
        encoding[coord_var].update({"_FillValue": None})
    return encoding


def _set_default_time_encoding(encoding, dataset):
    """Set default time encoding.

    Make sure time coordinates and bounds have the same units.
    Default is xarray's CF datetime encoding, which can be overridden
    by user-defined encoding.
    """
    if "time" in dataset:
        try:
            dtnp64 = dataset["time"].data[0]
        except IndexError:
            dtnp64 = dataset["time"].data

        default = CFDatetimeCoder().encode(xr.DataArray(dtnp64))
        time_enc = {"units": default.attrs["units"], "calendar": default.attrs["calendar"]}
        time_enc.update(encoding.get("time", {}))
        bounds_enc = {"units": time_enc["units"],
                      "calendar": time_enc["calendar"],
                      "_FillValue": None}
        encoding["time"] = time_enc
        encoding["time_bnds"] = bounds_enc  # FUTURE: Not required anymore with xarray-0.14+
    return encoding


def _update_encoding_dataset_names(encoding, dataset, numeric_name_prefix):
    """Ensure variable names of the encoding dictionary account for numeric_name_prefix.

    A lot of channel names in satpy starts with a digit.
    When preparing CF-compliant datasets, these channels are prefixed with numeric_name_prefix.

    If variables names in the encoding dictionary are numeric digits, their name is prefixed
    with numeric_name_prefix
    """
    for var_name in list(dataset.variables):
        if not numeric_name_prefix or not var_name.startswith(numeric_name_prefix):
            continue
        orig_var_name = var_name.replace(numeric_name_prefix, "")
        if orig_var_name in encoding:
            encoding[var_name] = encoding.pop(orig_var_name)
    return encoding


def update_encoding(dataset, to_engine_kwargs, numeric_name_prefix="CHANNEL_"):
    """Update encoding.

    Preserve dask chunks, avoid fill values in coordinate variables and make sure that
    time & time bounds have the same units.
    """
    other_to_engine_kwargs = to_engine_kwargs.copy()
    encoding = other_to_engine_kwargs.pop("encoding", {}).copy()
    encoding = _update_encoding_dataset_names(encoding, dataset, numeric_name_prefix)
    encoding = _set_default_chunks(encoding, dataset)
    encoding = _set_default_fill_value(encoding, dataset)
    encoding = _set_default_time_encoding(encoding, dataset)
    return encoding, other_to_engine_kwargs
