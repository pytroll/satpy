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
"""Utility to generate a CF-compliant DataArray."""
import logging
import warnings

from satpy.cf.attrs import preprocess_attrs
from satpy.cf.coords import add_xy_coords_attrs, set_cf_time_info

logger = logging.getLogger(__name__)


def _handle_data_array_name(original_name, numeric_name_prefix):
    if original_name[0].isdigit():
        if numeric_name_prefix:
            new_name = numeric_name_prefix + original_name
        else:
            warnings.warn(
                f"Invalid NetCDF dataset name: {original_name} starts with a digit.",
                stacklevel=5
            )
            new_name = original_name  # occurs when numeric_name_prefix = '', None or False
    else:
        new_name = original_name
    return original_name, new_name


def _preprocess_data_array_name(dataarray, numeric_name_prefix, include_orig_name):
    """Change the DataArray name by prepending numeric_name_prefix if the name is a digit."""
    original_name = None
    named_has_changed = False
    dataarray = dataarray.copy()
    if "name" in dataarray.attrs:
        original_name = dataarray.attrs.pop("name")
        original_name, new_name = _handle_data_array_name(original_name, numeric_name_prefix)
        dataarray = dataarray.rename(new_name)
        named_has_changed = original_name != new_name

    if named_has_changed and include_orig_name:
        dataarray.attrs["original_name"] = original_name
    return dataarray


def make_cf_data_array(dataarray,
                       epoch=None,
                       flatten_attrs=False,
                       exclude_attrs=None,
                       include_orig_name=True,
                       numeric_name_prefix="CHANNEL_"):
    """Make the xr.DataArray CF-compliant.

    Args:
        dataarray (xr.DataArray): The data array to be made CF-compliant.
        epoch (str, optional): Reference time for encoding of time coordinates.
            If None, the default reference time is defined using `from satpy.cf.coords import EPOCH`.
        flatten_attrs (bool, optional): If True, flatten dict-type attributes. Defaults to False.
        exclude_attrs (list, optional): List of dataset attributes to be excluded. Defaults to None.
        include_orig_name (bool, optional): Include the original dataset name in the netcdf variable attributes.
            Defaults to True.
        numeric_name_prefix (str, optional): Prepend dataset name with this if starting with a digit.
            Defaults to ``"CHANNEL_"``.

    Returns:
        xr.DataArray: A CF-compliant xr.DataArray.
    """
    dataarray = _preprocess_data_array_name(dataarray=dataarray,
                                            numeric_name_prefix=numeric_name_prefix,
                                            include_orig_name=include_orig_name)
    dataarray = preprocess_attrs(data_arr=dataarray,
                                 flatten_attrs=flatten_attrs,
                                 exclude_attrs=exclude_attrs)
    dataarray = add_xy_coords_attrs(dataarray)
    if "time" in dataarray.coords:
        dataarray = set_cf_time_info(dataarray, epoch=epoch)
    return dataarray
