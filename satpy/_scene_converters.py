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
"""Helper functions for converting the Scene object to some other object."""
from __future__ import annotations

from typing import Iterable

import xarray as xr

from .scene import Scene

try:
    from datatree import DataTree
except ImportError:
    DataTree = None


def to_xarray_datatree(
        data_arrays: Scene | Iterable[xr.DataArray],
        group_keys: Iterable[str] = ("platform_name", "sensor"),
        **kwargs
) -> DataTree:
    """Convert this Scene into an Xarray DataTree object."""
    if DataTree is None:
        raise ImportError("Missing 'xarray-datatree' library required for DataTree conversion")

    tree = DataTree()
    for data_arr in data_arrays:
        leaf_node = _generate_leaf_node_for_data_array(tree, data_arr, group_keys)
        var_name = _data_array_to_variable_name(data_arr)
        leaf_node[var_name] = data_arr
    return tree


def _generate_leaf_node_for_data_array(
        root_node: DataTree,
        data_arr: xr.DataArray,
        group_keys: Iterable[str]
) -> DataTree:
    current_node = root_node
    for group_key in group_keys:
        group_id = _data_array_to_group_id(data_arr, group_key)
        current_node = _get_or_create_child_node(current_node, group_id)
    return current_node


def _data_array_to_group_id(data_arr: xr.DataArray, group_key: str) -> str:
    group_id = data_arr.attrs[group_key]
    if group_key == "sensor" and isinstance(group_id, set):
        group_id = "-".join(sorted(group_id))
    return group_id


def _get_or_create_child_node(current_node: DataTree, node_id: str) -> DataTree:
    if node_id in current_node:
        return current_node[node_id]
    return DataTree(parent=current_node, name=node_id)


def _data_array_to_variable_name(data_arr: xr.DataArray) -> str:
    return data_arr.attrs["name"]
