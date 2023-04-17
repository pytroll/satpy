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
import xarray as xr

from .scene import Scene

try:
    from datatree import DataTree
except ImportError:
    DataTree = None


def to_xarray_datatree(scn: Scene, **kwargs) -> DataTree:
    """Convert this Scene into an Xarray DataTree object."""
    if DataTree is None:
        raise ImportError("Missing 'xarray-datatree' library required for DataTree conversion")

    datasets = {}
    for data_arr in scn:
        sensor_id = data_arr.attrs["sensor"]
        if isinstance(sensor_id, set):
            sensor_id = "-".join(sorted(sensor_id))
        group_id = data_arr.attrs["platform_name"] + "/" + sensor_id
        if group_id not in datasets:
            datasets[group_id] = xr.Dataset({
                data_arr.attrs["name"]: data_arr,
            })
    tree = DataTree.from_dict(datasets)
    return tree
