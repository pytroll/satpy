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
from .scene import Scene

try:
    from datatree import DataTree
except ImportError:
    DataTree = None


def to_xarray_datatree(scn: Scene, **kwargs) -> DataTree:
    """Convert this Scene into an Xarray DataTree object."""
    if DataTree is None:
        raise ImportError("Missing 'xarray-datatree' library required for DataTree conversion")

    tree = DataTree()
    return tree
